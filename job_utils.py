from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit


LLAMA_COMMIT = "8345f333951c661d166b00e6f9362e553768f292"
LLAMA_RELEASE = "v0.6.0"
# Allow JSON escaping of a 1 MiB prompt plus job metadata.
MAX_WORKER_REQUEST_BYTES = 8 * 1024 * 1024


class JobError(Exception):
    """An actionable workflow failure, safe to display after redaction."""


class DeadlineExceeded(JobError):
    pass


class Deadline:
    def __init__(
        self,
        seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise JobError("Timeout must be a positive finite number.")
        self.clock = clock
        self.sleeper = sleeper
        self.end = clock() + seconds

    def remaining(self) -> float:
        remaining = self.end - self.clock()
        if remaining <= 0:
            raise DeadlineExceeded("Job deadline exceeded.")
        return remaining

    def timeout(self, maximum: float = 30) -> float:
        return min(maximum, self.remaining())

    def sleep(self, seconds: float) -> None:
        self.sleeper(min(seconds, self.remaining()))
        self.remaining()


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    def clean_url(match: re.Match[str]) -> str:
        try:
            parts = urlsplit(match.group())
            host = parts.hostname or ""
            if ":" in host:
                host = f"[{host}]"
            if parts.port:
                host += f":{parts.port}"
            return urlunsplit((parts.scheme, host, parts.path, "", ""))
        except ValueError:
            return "[REDACTED URL]"

    text = re.sub(r"https?://[^\s<>\"')]+", clean_url, text)
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|access[_-]?token|authorization|hf_token)"
        r"[\"']?\s*[:=]\s*[\"']?)[^\s,\"'}&]+",
        r"\1[REDACTED]",
        text,
    )
    return text


def error_text(error: BaseException, secrets: Iterable[str] = ()) -> str:
    return redact(f"{type(error).__name__}: {error}", secrets)


def object_value(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JobError(f"{context} must be a JSON object.")
    return value


def number(value: object, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JobError(f"{context} must be a number.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise JobError(f"{context} must be {'positive and ' if positive else ''}finite.")
    return result


def positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JobError(f"{context} must be a positive integer.")
    return value


def model_filename(value: object, *, allow_split: bool = False) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise JobError("Model filename must be a relative Hugging Face GGUF path.")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise JobError("Model filename must not be absolute or traverse directories.")
    basename = value.rsplit("/", 1)[-1].lower()
    if not basename.endswith(".gguf"):
        raise JobError("Select an exact .gguf filename.")
    if not allow_split and re.search(r"-\d+-of-\d+\.gguf$", basename):
        raise JobError("Split GGUF files are not supported; select a standalone model.")
    if re.match(r"^(mmproj|mtp|vocab|adapter|lora)([-_.]|$)", basename) or re.search(
        r"[-_.]lora\.gguf$", basename
    ):
        raise JobError("Select a standalone text model, not a sidecar or adapter.")
    return value


def model_filenames(value: object) -> list[str]:
    filename = model_filename(value, allow_split=True)
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})(\.gguf)", filename, re.IGNORECASE)
    if match is None:
        if re.search(r"-\d+-of-\d+\.gguf$", filename, re.IGNORECASE):
            raise JobError("Split GGUF filenames must use five-digit shard indices.")
        return [filename]
    first, count = int(match[2]), int(match[3])
    if first != 1 or not 2 <= count <= 128:
        raise JobError("Select shard 00001 from a complete GGUF set of 2 to 128 shards.")
    return [f"{match[1]}-{index:05d}-of-{count:05d}{match[4]}" for index in range(1, count + 1)]


def commit_sha(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{40}", value):
        raise JobError("Model revision must resolve to a full commit SHA.")
    return value


def completion_text(value: object) -> str:
    completion = object_value(value, "Completion")
    choices = completion.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise JobError("Model must return exactly one completion choice.")
    choice = object_value(choices[0], "Completion choice")
    message = object_value(choice.get("message"), "Assistant message")
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, str) or not content.strip():
        raise JobError("Model returned no non-empty assistant text.")
    if choice.get("finish_reason") not in ("stop", "length"):
        raise JobError("Model returned an unsupported or unfinished completion.")
    return content


def sha256_file(path: Path, deadline: Deadline) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            deadline.remaining()
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    atomic_write(
        path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        return object_value(json.loads(path.read_text(encoding="utf-8")), str(path.name))
    except (ValueError, UnicodeError) as error:
        raise JobError(f"Invalid JSON in {path.name}.") from error


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_command(
    args: list[str],
    deadline: Deadline,
    *,
    input_data: bytes | None = None,
    secrets: Iterable[str] = (),
    context: str,
) -> bytes:
    timeout = deadline.remaining()
    with subprocess.Popen(
        args,
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(input=input_data, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            stop_process(process)
            raise DeadlineExceeded(f"Deadline exceeded during {context}.") from error
        except BaseException:
            stop_process(process)
            raise
    if process.returncode:
        detail = redact(stderr.decode("utf-8", errors="replace")[-4000:], secrets).strip()
        raise JobError(f"{context} failed (exit {process.returncode}). {detail}")
    deadline.remaining()
    return stdout
