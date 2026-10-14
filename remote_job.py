from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
from typing import TYPE_CHECKING, Any

from job_utils import (
    Deadline, DeadlineExceeded, JobError, LLAMA_COMMIT, MAX_WORKER_REQUEST_BYTES, atomic_write,
    commit_sha, completion_text, error_text, model_filename, number,
    object_value, positive_int, redact, run_command, sha256_file,
    stop_process, write_json,
)

if TYPE_CHECKING:
    import requests


def log(root: Path, message: str, token: str = "") -> None:
    with (root / "remote.log").open("a", encoding="utf-8") as file:
        file.write(redact(message, (token,)) + "\n")


def gpu_details(expected: str, deadline: Deadline) -> dict[str, Any]:
    output = run_command(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        deadline, context="GPU verification",
    ).decode("utf-8")
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise JobError("The instance must expose exactly one NVIDIA GPU.")
    try:
        name, memory = rows[0].rsplit(",", 1)
        memory_mib = number(float(memory.strip()), "GPU memory", positive=True)
    except ValueError as error:
        raise JobError("Could not parse nvidia-smi GPU details.") from error
    if not 24_000 <= memory_mib:
        raise JobError("The remote GPU has less than the required 24 GB VRAM.")

    def canonical(value: str) -> str:
        value = re.sub(r"\b(nvidia|geforce|tesla)\b", "", value, flags=re.IGNORECASE)
        value = re.sub(r"[-_ ]\d+gb(?:\b|_)", "", value, flags=re.IGNORECASE)
        return re.sub(r"[^a-z0-9]", "", value.lower())

    if canonical(expected) not in canonical(name) or not canonical(expected):
        raise JobError("The instance GPU does not match the selected offer.")
    return {"gpu_name": name.strip(), "gpu_vram_mib": memory_mib}


def logged_command(args: list[str], root: Path, deadline: Deadline, context: str) -> None:
    log(root, context)
    with (root / "remote.log").open("ab") as file:
        with subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=file, stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process:
            try:
                process.wait(timeout=deadline.remaining())
            except subprocess.TimeoutExpired as error:
                stop_process(process)
                raise DeadlineExceeded(f"Deadline exceeded during {context}.") from error
            except BaseException:
                stop_process(process)
                raise
    if process.returncode:
        raise JobError(f"{context} failed (exit {process.returncode}); see remote.log.")
    deadline.remaining()


def bootstrap(root: Path, request: dict[str, Any], deadline: Deadline) -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise JobError("The CUDA template must provide Python 3.10 or newer.")
    expected_gpu = request.get("gpu_name")
    if not isinstance(expected_gpu, str):
        raise JobError("Expected GPU name is missing.")
    gpu = gpu_details(expected_gpu, deadline)
    if shutil.which("nvcc") is None:
        raise JobError("The template must include the CUDA development toolkit (nvcc).")
    missing = [tool for tool in ("git", "cmake", "g++") if shutil.which(tool) is None]
    if missing:
        if os.geteuid() != 0 or shutil.which("apt-get") is None:
            raise JobError(f"The CUDA template is missing build tools: {', '.join(missing)}.")
        logged_command(["apt-get", "update", "-qq"], root, deadline, "Updating build dependencies")
        logged_command(
            ["apt-get", "install", "-y", "-qq", "git", "cmake", "build-essential"],
            root, deadline, "Installing missing build dependencies",
        )
    virtualenv = root / "venv"
    logged_command(
        [sys.executable, "-m", "venv", str(virtualenv)],
        root, deadline, "Creating remote Python environment",
    )
    logged_command(
        [str(virtualenv / "bin/python"), "-m", "pip", "install", "--quiet", "-r",
         str(root / "requirements.txt")],
        root, deadline, "Installing pinned remote dependencies",
    )
    source = root / "llama.cpp"
    source.mkdir()
    for args, context in (
        (["git", "-C", str(source), "init", "-q"], "Initializing llama.cpp source"),
        (["git", "-C", str(source), "remote", "add", "origin",
          "https://github.com/ggml-org/llama.cpp.git"], "Configuring llama.cpp source"),
        (["git", "-C", str(source), "fetch", "-q", "--depth", "1", "origin", LLAMA_COMMIT],
         "Fetching pinned llama.cpp commit"),
        (["git", "-C", str(source), "checkout", "-q", "--detach", "FETCH_HEAD"],
         "Checking out pinned llama.cpp commit"),
    ):
        logged_command(args, root, deadline, context)
    build = source / "build"
    logged_command(
        ["cmake", "-S", str(source), "-B", str(build), "-DGGML_CUDA=ON",
         "-DCMAKE_CUDA_ARCHITECTURES=native", "-DCMAKE_BUILD_TYPE=Release",
         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
         "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_APP=OFF",
         "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_OPENSSL=OFF"],
        root, deadline, "Configuring CUDA llama.cpp",
    )
    logged_command(
        ["cmake", "--build", str(build), "--target", "llama-server",
         "--parallel", str(min(os.cpu_count() or 1, 8))],
        root, deadline, "Building CUDA llama-server",
    )
    binary = build / "bin/llama-server"
    if not binary.is_file():
        raise JobError("CUDA build did not produce llama-server.")
    log(root, "CUDA bootstrap complete.")
    return {"status": "ready", "llama_commit": LLAMA_COMMIT, **gpu}


def download_model(
    root: Path, model: dict[str, Any], token: str, deadline: Deadline,
) -> Path:
    from huggingface_hub import hf_hub_download

    repo = model.get("repo")
    if not isinstance(repo, str) or not repo:
        raise JobError("Model repository is missing.")
    filename = model_filename(model.get("filename"))
    revision = commit_sha(model.get("revision"))
    expected_size = positive_int(model.get("size_bytes"), "Model size")
    if shutil.disk_usage(root).free < expected_size + 2 * 1024 ** 3:
        raise JobError("Insufficient free disk for the model and inference artifacts.")
    log(root, "Downloading the selected GGUF inside the instance.", token)
    try:
        downloaded = hf_hub_download(
            repo_id=repo, filename=filename, revision=revision, token=token,
            local_dir=root / "models", etag_timeout=deadline.timeout(),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise JobError(f"GGUF download failed: {error_text(error, (token,))}") from error
    if not isinstance(downloaded, str):
        raise JobError("Hugging Face did not return a downloaded model path.")
    path = Path(downloaded).resolve()
    if path != (root / "models" / filename).resolve():
        raise JobError("Downloaded model is outside the requested job directory.")
    if path.stat().st_size != expected_size:
        raise JobError("Downloaded GGUF size does not match authenticated metadata.")
    with path.open("rb") as file:
        if file.read(4) != b"GGUF":
            raise JobError("Downloaded file is not a GGUF model.")
    expected_hash = model.get("sha256")
    if expected_hash is not None:
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_hash):
            raise JobError("Invalid model SHA-256 metadata.")
        if sha256_file(path, deadline) != expected_hash:
            raise JobError("Downloaded GGUF SHA-256 does not match authenticated metadata.")
    deadline.remaining()
    log(root, "GGUF download and integrity checks complete.", token)
    return path


def wait_for_server(
    process: subprocess.Popen[bytes], url: str, deadline: Deadline,
    http: requests.Session,
) -> None:
    import requests

    while True:
        if process.poll() is not None:
            raise JobError("llama-server exited before model readiness; see remote.log.")
        timeout = deadline.timeout(5)
        try:
            response = http.get(
                f"{url}/health", timeout=(timeout / 2, timeout / 2),
                allow_redirects=False,
            )
        except (requests.ConnectionError, requests.Timeout):
            deadline.sleep(1)
            continue
        if response.status_code == 200:
            try:
                health = object_value(response.json(), "Model health")
            except ValueError as error:
                raise JobError("llama-server returned invalid health JSON.") from error
            if health.get("status") != "ok":
                raise JobError("llama-server did not confirm healthy model readiness.")
            return
        if response.status_code != 503:
            raise JobError(f"llama-server health check failed: HTTP {response.status_code}.")
        deadline.sleep(1)


def server_args(binary: Path, model: Path, context_size: int, port: int) -> list[str]:
    # The pinned runtime maps loader INFO messages to trace verbosity.
    return [
        str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(context_size), "--parallel", "1", "--n-gpu-layers", "all",
        "--fit", "off", "--no-context-shift", "--alias", "batch-model",
        "--offline", "--log-colors", "off", "--verbosity", "4",
    ]


def wait_for_gpu_offload(
    process: subprocess.Popen[bytes], path: Path, offset: int, deadline: Deadline,
) -> None:
    end = deadline.clock() + min(5, deadline.remaining())
    diagnostics = ""
    with path.open("rb") as diagnostics_file:
        diagnostics_file.seek(offset)
        while True:
            if process.poll() is not None:
                raise JobError("llama-server exited before GPU offload verification; see remote.log.")
            diagnostics += diagnostics_file.read().decode("utf-8", errors="replace")
            offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", diagnostics)
            if offload is not None:
                loaded, total = int(offload[1]), int(offload[2])
                if loaded <= 0 or loaded != total:
                    raise JobError(
                        f"llama-server offloaded {loaded}/{total} layers; full GPU offload is required."
                    )
                return
            remaining = end - deadline.clock()
            if remaining <= 0:
                raise JobError("llama-server did not report GPU layer-offload counts; see remote.log.")
            # The runtime's asynchronous logger can lag behind its health response.
            deadline.sleep(min(0.1, remaining))


def infer(root: Path, request: dict[str, Any], deadline: Deadline) -> dict[str, Any]:
    import requests

    token = request.get("hf_token")
    prompt = request.get("prompt")
    expected_gpu = request.get("gpu_name")
    if not isinstance(token, str) or not token.strip():
        raise JobError("Hugging Face credential is missing from the inference request.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise JobError("Prompt must not be empty.")
    if not isinstance(expected_gpu, str):
        raise JobError("Expected GPU name is missing.")
    context_size = positive_int(request.get("context_size"), "Context size")
    max_tokens = positive_int(request.get("max_tokens"), "Output token limit")
    temperature = number(request.get("temperature"), "Temperature")
    if max_tokens >= context_size or not 0 <= temperature <= 2:
        raise JobError("Output limit must be below context size; temperature must be between 0 and 2.")
    gpu = gpu_details(expected_gpu, deadline)
    model = download_model(
        root, object_value(request.get("model"), "Model configuration"), token, deadline,
    )
    binary = root / "llama.cpp/build/bin/llama-server"
    if not binary.is_file():
        raise JobError("The CUDA llama-server binary is missing; bootstrap must run first.")
    with socket.socket() as socket_handle:
        socket_handle.bind(("127.0.0.1", 0))
        port = socket_handle.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    environment = {
        key: value for key, value in os.environ.items()
        if "TOKEN" not in key.upper() and "API_KEY" not in key.upper()
    }
    log(root, "Loading the GGUF into the CUDA runtime.", token)
    with (root / "remote.log").open("ab") as output:
        log_offset = output.tell()
        process = subprocess.Popen(
            server_args(binary, model, context_size, port),
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True,
        )
        http = requests.Session()
        http.trust_env = False
        try:
            wait_for_server(process, url, deadline, http)
            wait_for_gpu_offload(process, root / "remote.log", log_offset, deadline)
            remaining = deadline.remaining()
            connect_timeout = min(5, remaining / 2)
            try:
                response = http.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model": "batch-model", "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
                    },
                    timeout=(connect_timeout, remaining - connect_timeout),
                    allow_redirects=False,
                )
            except requests.RequestException as error:
                raise JobError(f"Model request failed: {error_text(error, (token,))}") from error
            try:
                completion = object_value(response.json(), "Completion response")
            except ValueError as error:
                raise JobError("Model returned invalid completion JSON.") from error
            if response.status_code != 200:
                detail = redact(str(completion.get("error", "Inference rejected")), (token,))
                raise JobError(f"Model request failed: HTTP {response.status_code}, {detail[:1000]}")
            content = completion_text(completion)
            deadline.remaining()
            atomic_write(root / "response.txt", content.encode("utf-8"))
            write_json(root / "completion.json", completion)
            result = {
                "status": "completed", "llama_commit": LLAMA_COMMIT, **gpu,
                "response_sha256": sha256_file(root / "response.txt", deadline),
                "completion_sha256": sha256_file(root / "completion.json", deadline),
                "finish_reason": completion["choices"][0]["finish_reason"],
                "usage": completion.get("usage"),
            }
            log(root, "Inference complete; response artifacts are ready for retrieval.", token)
            return result
        finally:
            try:
                http.close()
            finally:
                stop_process(process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VastAI remote GGUF worker")
    parser.add_argument("stage", choices=("bootstrap", "infer"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    request: dict[str, Any] = {}

    def cancelled(signum: int, frame: object) -> None:
        raise JobError(f"Remote worker cancelled (signal {signum}).")

    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    try:
        data = sys.stdin.buffer.read(MAX_WORKER_REQUEST_BYTES + 1)
        if len(data) > MAX_WORKER_REQUEST_BYTES:
            raise JobError("Inference request exceeds the 8 MiB protocol limit.")
        request = object_value(json.loads(data), "Worker request")
        deadline = Deadline(number(request.get("remaining_seconds"), "Remaining time", positive=True))
        result = (
            bootstrap(root, request, deadline) if args.stage == "bootstrap"
            else infer(root, request, deadline)
        )
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (JobError, OSError, ValueError, RuntimeError) as error:
        token = request.get("hf_token")
        message = error_text(error, (token,) if isinstance(token, str) else ())
        print(json.dumps({"status": "failed", "error": message}))
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
