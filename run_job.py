from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import sys
import tempfile
from typing import Any, Callable, Iterator
import uuid

from job_utils import (
    Deadline, DeadlineExceeded, JobError, LLAMA_COMMIT, LLAMA_RELEASE, atomic_write,
    commit_sha, completion_text, error_text, model_filename, number, object_value,
    positive_int, read_json, redact, run_command, sha256_file, write_json,
)
from vast_client import (
    ApiError, DEFAULT_TEMPLATE_HASH, Offer, Template, VastClient, choose_offer,
)


ROOT = Path(__file__).resolve().parent
CLEANUP_SECONDS = 300
MAX_PROMPT_BYTES = 1024 * 1024
RUN_ID_PATTERN = r"\d{8}T\d{6}Z-[a-f0-9]{12}"


@contextmanager
def defer_cancellation() -> Iterator[None]:
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


@dataclass(frozen=True)
class Credentials:
    vast: str = field(repr=False)
    hugging_face: str = field(repr=False)

    @property
    def secrets(self) -> tuple[str, str]:
        return self.vast, self.hugging_face


@dataclass(frozen=True)
class Settings:
    model_repo: str
    model_file: str
    prompt: str = field(repr=False)
    credentials: Credentials = field(repr=False)
    output_dir: Path
    state_root: Path
    model_revision: str = "main"
    template_hash: str = DEFAULT_TEMPLATE_HASH
    max_hourly_price: float = 4
    timeout_minutes: float = 180
    gpu_policy: str = "prefer-a100"
    disk_gb: int = 64
    context_size: int = 4096
    max_tokens: int = 1024
    temperature: float = 0.7

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.model_repo):
            raise JobError("Model repository must be a Hugging Face owner/repository ID.")
        model_filename(self.model_file)
        if not self.model_revision.strip():
            raise JobError("Model revision must not be empty.")
        if not self.prompt.strip() or len(self.prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise JobError("Prompt must be non-empty UTF-8 text of at most 1 MiB.")
        number(self.max_hourly_price, "Maximum hourly price", positive=True)
        number(self.timeout_minutes, "Timeout minutes", positive=True)
        positive_int(self.disk_gb, "Disk allocation")
        positive_int(self.context_size, "Context size")
        positive_int(self.max_tokens, "Maximum output tokens")
        temperature = number(self.temperature, "Temperature")
        if self.max_tokens >= self.context_size or not 0 <= temperature <= 2:
            raise JobError("Output tokens must be below context size; temperature must be between 0 and 2.")
        if self.gpu_policy not in ("prefer-a100", "a100-only"):
            raise JobError("GPU policy must be prefer-a100 or a100-only.")
        if not self.credentials.vast.strip() or not self.credentials.hugging_face.strip():
            raise JobError("Both VastAI and Hugging Face credentials are required.")


def credentials_from_file(path: Path, *, require_hf: bool = True) -> Credentials:
    from dotenv import dotenv_values

    if not path.is_file():
        raise JobError(f"Environment file not found: {path}.")
    values = dotenv_values(path)

    def value(name: str, required: bool) -> str:
        result = os.environ[name] if name in os.environ else values.get(name)
        if not isinstance(result, str) or not result.strip():
            if required:
                raise JobError(f"{name} is missing from the environment or selected .env.")
            return ""
        return result.strip()

    return Credentials(
        value("VAST_AI_API_KEY", True),
        value("HUGGING_FACE_ACCESS_TOKEN", require_hf),
    )


def check_writable(path: Path, *, create: bool) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = path
    while not directory.exists():
        directory = directory.parent
    if not directory.is_dir():
        raise JobError(f"Output/state path is not a directory: {directory}.")
    with tempfile.TemporaryFile(dir=directory) as file:
        file.write(b"write-check")
        file.flush()


def inspect_model(settings: Settings, deadline: Deadline) -> dict[str, Any]:
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

    try:
        info = HfApi(token=settings.credentials.hugging_face).model_info(
            settings.model_repo, revision=settings.model_revision, timeout=deadline.timeout(),
        )
        revision = commit_sha(info.sha)
        metadata = get_hf_file_metadata(
            hf_hub_url(settings.model_repo, settings.model_file, revision=revision),
            token=settings.credentials.hugging_face, timeout=deadline.timeout(),
        )
        if commit_sha(metadata.commit_hash) != revision:
            raise JobError("Hugging Face returned a different file revision than requested.")
        size = positive_int(metadata.size, "GGUF file size")
    except (OSError, ValueError, RuntimeError) as error:
        raise JobError(
            "Hugging Face file access failed. Check the token, filename, and model license: "
            + error_text(error, settings.credentials.secrets)
        ) from error
    etag = metadata.etag
    sha256 = etag if isinstance(etag, str) and re.fullmatch(r"[a-f0-9]{64}", etag) else None
    deadline.remaining()
    return {
        "repo": settings.model_repo, "filename": settings.model_file,
        "revision": revision, "size_bytes": size, "sha256": sha256,
    }


def preflight(
    settings: Settings, client: VastClient, deadline: Deadline,
    model_lookup: Callable[[Settings, Deadline], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Template, Offer]:
    settings.validate()
    model = (model_lookup or inspect_model)(settings, deadline)
    template = client.template(settings.template_hash, deadline)
    required_disk = math.ceil(template.recommended_disk + model["size_bytes"] / 1_000_000_000 + 5)
    if settings.disk_gb < required_disk:
        raise JobError(
            f"Allocate at least {required_disk} GB disk for the template, GGUF, and build artifacts."
        )
    offers = client.offers(
        template, settings.disk_gb, settings.max_hourly_price,
        settings.timeout_minutes * 60 + CLEANUP_SECONDS, deadline,
    )
    return model, template, choose_offer(offers, settings.gpu_policy, settings.max_hourly_price)


def preview(settings: Settings, model: dict[str, Any], template: Template, offer: Offer) -> dict[str, Any]:
    return {
        "dry_run": True, "model": model,
        "template": {"hash": template.hash_id, "image": template.image},
        "selected_offer": offer.summary(),
        "estimated_model_download_cost_usd": model["size_bytes"] / 1_000_000_000
        * offer.download_cost_per_gb,
        "output_dir": str(settings.output_dir),
        "limits": {
            "max_hourly_price_usd": settings.max_hourly_price,
            "timeout_minutes": settings.timeout_minutes,
            "cleanup_seconds": CLEANUP_SECONDS, "disk_gb": settings.disk_gb,
            "context_size": settings.context_size, "max_tokens": settings.max_tokens,
        },
        "note": "Read-only checks only. No instance was rented. Bandwidth charges are additional.",
    }


@dataclass
class RunState:
    directory: Path
    output: Path
    data: dict[str, Any]
    credentials: Credentials = field(repr=False)

    @classmethod
    def create(cls, settings: Settings) -> RunState:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
        settings.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory = settings.state_root / run_id
        directory.mkdir(mode=0o700)
        output = settings.output_dir / run_id
        output.mkdir(mode=0o700)
        state = cls(directory, output, {
            "run_id": run_id, "label": f"gguf-batch-{run_id}", "output_dir": str(output),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "job_status": "running", "cleanup_status": "not_required",
            "creation_outcome": "not_attempted", "instance_id": None, "warnings": [],
        }, settings.credentials)
        state.persist()
        return state

    @classmethod
    def load(cls, run_id: str, root: Path, credentials: Credentials) -> RunState:
        if not re.fullmatch(RUN_ID_PATTERN, run_id):
            raise JobError("Invalid run ID; use the ID printed by the workflow.")
        directory = root / run_id
        data = read_json(directory / "state.json")
        if data.get("run_id") != run_id or data.get("label") != f"gguf-batch-{run_id}":
            raise JobError("Recovery manifest does not match the requested run.")
        if data.get("creation_outcome") not in ("not_attempted", "rejected", "unknown", "confirmed"):
            raise JobError("Recovery manifest has an invalid allocation status.")
        if data.get("instance_id") is not None:
            positive_int(data["instance_id"], "Recorded instance ID")
        ids = data.get("instance_ids", [])
        if not isinstance(ids, list):
            raise JobError("Recovery instance IDs must be a list.")
        for instance_id in ids:
            positive_int(instance_id, "Recorded instance ID")
        has_id = data.get("instance_id") is not None or bool(ids)
        if data["creation_outcome"] == "confirmed" and not has_id:
            raise JobError("Confirmed allocation is missing its recorded instance ID.")
        if data["creation_outcome"] in ("not_attempted", "rejected") and has_id:
            raise JobError("Recovery manifest has contradictory allocation records.")
        warnings = data.get("warnings")
        if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
            raise JobError("Recovery warnings must be a list of strings.")
        output_value = data.get("output_dir")
        if not isinstance(output_value, str):
            raise JobError("Recovery manifest is missing its output directory.")
        output = Path(output_value)
        if not output.is_absolute() or output.name != run_id:
            raise JobError("Recovery output directory must belong to the requested run.")
        return cls(directory, output, data, credentials)

    def persist(self) -> None:
        write_json(self.directory / "state.json", self.data)
        write_json(self.output / "metadata.json", self.data)

    def log(self, message: str) -> None:
        clean = redact(message, self.credentials.secrets)
        print(clean, file=sys.stderr)
        try:
            descriptor = os.open(self.output / "workflow.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as file:
                file.write(datetime.now(timezone.utc).isoformat() + " " + clean + "\n")
        except OSError as error:
            warning = "Could not persist workflow log: " + error_text(error, self.credentials.secrets)
            print(warning, file=sys.stderr)
            self.data["warnings"].append(warning)


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    @property
    def target(self) -> str:
        return f"root@{self.host}"

    def scp_path(self, path: str) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"root@{host}:{path}"


def endpoint_for(instance: dict[str, Any]) -> Endpoint | None:
    ports = instance.get("ports")
    direct = ports.get("22/tcp") if isinstance(ports, dict) else None
    if direct:
        if not isinstance(direct, list) or not direct:
            raise JobError("VastAI returned invalid direct SSH ports.")
        host = instance.get("public_ipaddr")
        port = object_value(direct[0], "SSH port mapping").get("HostPort")
    else:
        host, port = instance.get("ssh_host"), instance.get("ssh_port")
    if not host or port in (None, 0, "0"):
        return None
    if not isinstance(host, str):
        raise JobError("VastAI SSH host must be a string.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
            raise JobError("VastAI returned an invalid SSH host.")
    if isinstance(port, bool) or not isinstance(port, (str, int)) or not str(port).isdigit():
        raise JobError("VastAI returned an invalid SSH port.")
    port_number = int(port)
    if not 1 <= port_number <= 65535:
        raise JobError("VastAI SSH port is outside the valid range.")
    return Endpoint(host, port_number)


class HostKeyError(JobError):
    pass


class SSHTransport:
    def __init__(self, state: RunState, endpoint: Endpoint) -> None:
        self.state = state
        self.endpoint = endpoint
        self.remote_dir = f"/workspace/gguf-batch-{state.data['run_id']}"

    def options(self, deadline: Deadline) -> list[str]:
        return [
            "-F", "/dev/null", "-i", str(self.state.directory / "id_ed25519"),
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={self.state.directory / 'known_hosts'}",
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ConnectionAttempts=1",
            "-o", f"ConnectTimeout={max(1, math.ceil(deadline.timeout(10)))}",
        ]

    def command(self, command: str, deadline: Deadline, payload: bytes | None = None) -> bytes:
        try:
            return run_command(
                ["ssh", "-T", *self.options(deadline), "-p", str(self.endpoint.port),
                 self.endpoint.target, command],
                deadline, input_data=payload, secrets=self.state.credentials.secrets,
                context="SSH remote command",
            )
        except JobError as error:
            if "Host key verification failed" in str(error) or "HOST IDENTIFICATION HAS CHANGED" in str(error):
                raise HostKeyError("SSH host key changed; refusing to connect.") from error
            raise

    def probe(self, deadline: Deadline) -> None:
        self.command("true", deadline)

    def upload(self, deadline: Deadline) -> None:
        self.command(f"mkdir -m 700 -p {shlex.quote(self.remote_dir)}", deadline)
        run_command(
            ["scp", "-q", *self.options(deadline), "-P", str(self.endpoint.port),
             str(ROOT / "remote_job.py"), str(ROOT / "job_utils.py"),
             str(ROOT / "requirements.txt"), self.endpoint.scp_path(self.remote_dir + "/")],
            deadline, secrets=self.state.credentials.secrets, context="Uploading remote worker",
        )

    def invoke(self, stage: str, request: dict[str, Any], deadline: Deadline) -> dict[str, Any]:
        if stage not in ("bootstrap", "infer"):
            raise JobError("Invalid remote worker stage.")
        interpreter = "python3" if stage == "bootstrap" else f"{self.remote_dir}/venv/bin/python"
        payload = {**request, "remaining_seconds": deadline.remaining()}
        command = (
            f"{shlex.quote(interpreter)} {shlex.quote(self.remote_dir + '/remote_job.py')} {stage}"
        )
        raw = self.command(
            command, deadline, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
        )
        try:
            result = object_value(json.loads(raw), "Remote worker result")
        except (ValueError, UnicodeError) as error:
            raise JobError("Remote worker returned invalid JSON; see remote.log.") from error
        expected = "ready" if stage == "bootstrap" else "completed"
        if result.get("status") != expected or result.get("llama_commit") != LLAMA_COMMIT:
            raise JobError("Remote worker did not confirm the expected pinned inference stage.")
        return result

    def fetch(self, filename: str, destination: Path, deadline: Deadline) -> None:
        if filename not in ("response.txt", "completion.json", "remote.log"):
            raise JobError("Invalid remote artifact name.")
        run_command(
            ["scp", "-q", *self.options(deadline), "-P", str(self.endpoint.port),
             self.endpoint.scp_path(f"{self.remote_dir}/{filename}"), str(destination)],
            deadline, secrets=self.state.credentials.secrets, context=f"Retrieving {filename}",
        )


def wait_ready(
    client: VastClient, state: RunState, deadline: Deadline,
    transport_factory: Callable[[RunState, Endpoint], SSHTransport],
) -> SSHTransport:
    state.log("Waiting for the instance and SSH readiness.")
    while True:
        instance = client.instance(state.data["instance_id"], deadline)
        if instance is not None:
            if instance.get("label") != state.data["label"]:
                raise JobError("Created instance label does not match this run.")
            status = instance.get("actual_status")
            if status in ("exited", "offline", "unknown", "error", "destroyed", "deleted"):
                raise JobError(f"VastAI instance entered a failed state: {status}.")
            if status == "running":
                endpoint = endpoint_for(instance)
                if endpoint is not None:
                    transport = transport_factory(state, endpoint)
                    try:
                        transport.probe(deadline)
                    except (HostKeyError, DeadlineExceeded):
                        raise
                    except JobError as error:
                        state.log("SSH not ready: " + error_text(error, state.credentials.secrets))
                    else:
                        state.data["ssh"] = {"host": endpoint.host, "port": endpoint.port}
                        known_hosts = state.directory / "known_hosts"
                        if known_hosts.exists():
                            known_hosts.chmod(0o600)
                            state.data["ssh_fingerprint"] = run_command(
                                ["ssh-keygen", "-l", "-f", str(known_hosts)],
                                deadline, context="Recording SSH host fingerprint",
                            ).decode().strip()
                        state.persist()
                        return transport
        deadline.sleep(3)


def reconcile_creation(client: VastClient, state: RunState, deadline: Deadline) -> int:
    end = deadline.clock() + min(30, deadline.remaining())
    while True:
        matches = client.labeled_instances(state.data["label"], deadline)
        if matches:
            ids = [positive_int(row.get("id"), "Reconciled instance ID") for row in matches]
            state.data["instance_ids"] = ids
            if len(ids) != 1:
                raise JobError("Creation reconciliation found multiple run-owned instances.")
            state.data["creation_reconciled"] = True
            return ids[0]
        if deadline.clock() >= end:
            raise JobError("Instance creation outcome is uncertain; no duplicate rental was attempted.")
        deadline.sleep(2)


def retrieve(
    transport: SSHTransport, state: RunState, report: dict[str, Any], deadline: Deadline,
) -> None:
    response_part = state.output / "response.txt.part"
    completion_part = state.output / "completion.json.part"
    try:
        for filename, path, checksum_key in (
            ("response.txt", response_part, "response_sha256"),
            ("completion.json", completion_part, "completion_sha256"),
        ):
            checksum = report.get(checksum_key)
            if not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum):
                raise JobError("Remote result is missing a valid artifact SHA-256.")
            transport.fetch(filename, path, deadline)
            if sha256_file(path, deadline) != checksum:
                raise JobError(f"Retrieved {filename} failed its integrity check.")
        completion = read_json(completion_part)
        expected = completion_text(completion).encode("utf-8")
        if response_part.read_bytes() != expected:
            raise JobError("Retrieved response does not match the structured model completion.")
        atomic_write(state.output / "completion.json", completion_part.read_bytes())
        atomic_write(state.output / "response.txt", expected)
        state.data["finish_reason"] = completion["choices"][0]["finish_reason"]
        state.data["usage"] = completion.get("usage")
        if state.data["finish_reason"] == "length":
            warning = "Response reached the output token limit and may be truncated."
            state.data["warnings"].append(warning)
            state.log(warning)
    finally:
        response_part.unlink(missing_ok=True)
        completion_part.unlink(missing_ok=True)


def fetch_diagnostics(transport: SSHTransport, state: RunState, deadline: Deadline) -> None:
    part = state.output / "remote.log.part"
    try:
        transport.fetch("remote.log", part, deadline)
        content = redact(part.read_text(encoding="utf-8", errors="replace"), state.credentials.secrets)
        atomic_write(state.output / "remote.log", content.encode())
    finally:
        part.unlink(missing_ok=True)


def cleanup_owned(client: VastClient, state: RunState, deadline: Deadline) -> None:
    if state.data["creation_outcome"] in ("not_attempted", "rejected"):
        state.data["cleanup_status"] = "not_required"
    else:
        ids = set(state.data.get("instance_ids", []))
        if state.data.get("instance_id") is not None:
            ids.add(positive_int(state.data["instance_id"], "Recorded instance ID"))
        if state.data["creation_outcome"] == "unknown":
            matches = client.labeled_instances(state.data["label"], deadline)
            ids.update(positive_int(row.get("id"), "Reconciled instance ID") for row in matches)
            if not ids:
                raise JobError(
                    "Creation outcome is still uncertain; no matching instance is currently visible. "
                    "Keep the recovery manifest and retry cleanup before another rental."
                )
        state.data["instance_ids"] = sorted(ids)
        for instance_id in sorted(ids):
            requested = False
            while True:
                try:
                    instance = client.instance(instance_id, deadline)
                    if instance is None:
                        break
                    if instance.get("label") != state.data["label"]:
                        raise JobError(f"Refusing to destroy unrelated instance {instance_id}: label mismatch.")
                    if not requested:
                        state.log(f"Destroying run-owned instance {instance_id}.")
                        client.destroy(instance_id, deadline)
                        requested = True
                        continue
                except ApiError as error:
                    if not error.retryable:
                        raise
                    state.log("Retrying teardown: " + error_text(error, state.credentials.secrets))
                deadline.sleep(2)
        remaining = client.labeled_instances(state.data["label"], deadline)
        if remaining:
            raise JobError("Run-owned instances are still visible after teardown; retry cleanup.")
        state.data["cleanup_status"] = "confirmed"
    for filename in ("id_ed25519", "id_ed25519.pub"):
        (state.directory / filename).unlink(missing_ok=True)
    state.data["cleanup_confirmed_at"] = datetime.now(timezone.utc).isoformat()


def run_workflow(
    settings: Settings,
    client: VastClient,
    *,
    deadline: Deadline | None = None,
    model_lookup: Callable[[Settings, Deadline], dict[str, Any]] | None = None,
    transport_factory: Callable[[RunState, Endpoint], SSHTransport] | None = None,
) -> RunState:
    settings.validate()
    check_writable(settings.output_dir, create=True)
    check_writable(settings.state_root, create=True)
    for executable in ("ssh", "scp", "ssh-keygen"):
        if shutil.which(executable) is None:
            raise JobError(f"Required local tool is missing: {executable}.")
    deadline = deadline or Deadline(settings.timeout_minutes * 60)
    transport_factory = transport_factory or SSHTransport
    model, template, offer = preflight(settings, client, deadline, model_lookup)
    state = RunState.create(settings)
    state.data.update({
        "model": model, "template": {"hash": template.hash_id, "image": template.image},
        "offer": offer.summary(), "llama_release": LLAMA_RELEASE, "llama_commit": LLAMA_COMMIT,
        "settings": {
            "max_hourly_price_usd": settings.max_hourly_price,
            "timeout_minutes": settings.timeout_minutes, "gpu_policy": settings.gpu_policy,
            "disk_gb": settings.disk_gb, "context_size": settings.context_size,
            "max_tokens": settings.max_tokens, "temperature": settings.temperature,
            "prompt_sha256": hashlib.sha256(settings.prompt.encode()).hexdigest(),
        },
    })
    transport: SSHTransport | None = None
    failure: str | None = None
    cleanup_failure: str | None = None
    reporting_failure: str | None = None
    succeeded = False
    creation_attempted = False
    state.log(f"Run {state.data['run_id']}: selected {offer.gpu_name} at ${offer.hourly_price:.4f}/hour.")
    try:
        state.persist()
        key = state.directory / "id_ed25519"
        run_command(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", state.data["label"], "-f", str(key)],
            deadline, context="Generating run-specific SSH identity",
        )
        public_key = key.with_suffix(".pub").read_text().strip()
        refreshed = client.offers(
            template, settings.disk_gb, settings.max_hourly_price,
            deadline.remaining() + CLEANUP_SECONDS, deadline, offer_id=offer.id,
        )
        offer = choose_offer(refreshed, settings.gpu_policy, settings.max_hourly_price)
        state.data["offer"] = offer.summary()
        state.data["creation_outcome"] = "unknown"
        state.data["creation_started_at"] = datetime.now(timezone.utc).isoformat()
        state.persist()
        deadline.remaining()
        creation_attempted = True
        try:
            instance_id = client.create(offer, template, settings.disk_gb, state.data["label"], deadline)
        except ApiError as error:
            if not error.ambiguous:
                state.data["creation_outcome"] = "rejected"
                raise
            state.log("Creation response was uncertain; reconciling the unique run label.")
            instance_id = reconcile_creation(client, state, deadline)
        state.data["instance_id"] = instance_id
        state.data["creation_outcome"] = "confirmed"
        state.data["cleanup_status"] = "pending"
        state.persist()
        deadline.remaining()
        client.attach_key(instance_id, public_key, deadline)
        transport = wait_ready(client, state, deadline, transport_factory)
        state.log("Uploading the worker and building the pinned CUDA runtime.")
        transport.upload(deadline)
        transport.invoke("bootstrap", {"gpu_name": offer.gpu_name}, deadline)
        state.log("Downloading the GGUF and generating the response remotely.")
        report = transport.invoke("infer", {
            "model": model, "hf_token": settings.credentials.hugging_face,
            "prompt": settings.prompt, "gpu_name": offer.gpu_name,
            "context_size": settings.context_size, "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
        }, deadline)
        gpu_name = report.get("gpu_name")
        gpu_memory = number(report.get("gpu_vram_mib"), "Reported GPU VRAM", positive=True)
        if not isinstance(gpu_name, str) or not gpu_name.strip() or gpu_memory < 24_000:
            raise JobError("Remote worker did not report a valid >=24 GB NVIDIA GPU.")
        state.data["remote_gpu"] = {"gpu_name": gpu_name, "gpu_vram_mib": gpu_memory}
        state.log("Retrieving and validating the model response.")
        retrieve(transport, state, report, deadline)
        try:
            fetch_diagnostics(transport, state, deadline)
        except (HostKeyError, DeadlineExceeded):
            raise
        except (JobError, OSError) as error:
            warning = "Remote diagnostics could not be retrieved: " + error_text(
                error, settings.credentials.secrets,
            )
            state.data["warnings"].append(warning)
            state.log(warning)
        deadline.remaining()
        succeeded = True
    except (JobError, OSError, ValueError) as error:
        failure = error_text(error, settings.credentials.secrets)
        state.log("Job failed: " + failure)
    finally:
        active_error = sys.exc_info()[1]
        if active_error is not None and failure is None:
            failure = error_text(active_error, settings.credentials.secrets)
        if not creation_attempted:
            state.data["creation_outcome"] = "not_attempted"
        state.data["job_status"] = "completed" if succeeded else "failed"
        if failure:
            state.data["error"] = failure
        cleanup_deadline = Deadline(CLEANUP_SECONDS)
        with defer_cancellation():
            if not succeeded and transport is not None:
                try:
                    fetch_diagnostics(
                        transport, state, Deadline(min(30, cleanup_deadline.remaining())),
                    )
                except (JobError, OSError) as error:
                    warning = "Failure diagnostics unavailable: " + error_text(
                        error, settings.credentials.secrets,
                    )
                    state.data["warnings"].append(warning)
                    state.log(warning)
            try:
                cleanup_owned(client, state, cleanup_deadline)
            except (JobError, OSError, ValueError) as error:
                cleanup_failure = error_text(error, settings.credentials.secrets)
                state.data["cleanup_status"] = "failed"
                state.data["cleanup_error"] = cleanup_failure
                state.log("Cleanup failed: " + cleanup_failure)
            state.data["finished_at"] = datetime.now(timezone.utc).isoformat()
            try:
                state.persist()
            except OSError as error:
                reporting_failure = error_text(error, settings.credentials.secrets)
                state.log("Could not persist final run metadata: " + reporting_failure)
    if failure or cleanup_failure or reporting_failure:
        message = failure or "Response retrieved, but the workflow could not complete."
        if cleanup_failure:
            message += f"\nCleanup failed: {cleanup_failure}"
        if reporting_failure:
            message += f"\nMetadata persistence failed: {reporting_failure}"
        message += (
            f"\nRun: {state.data['run_id']}; instance: {state.data.get('instance_id')}."
            f"\nOutput: {state.output}"
        )
        if cleanup_failure:
            message += f"\nRecover with: python run_job.py cleanup --run-id {state.data['run_id']}"
        raise JobError(message)
    return state


def parser_for_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a GGUF prompt on a temporary VastAI GPU.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Rent, infer, retrieve, and destroy.")
    run.add_argument("--model-repo", required=True, help="Hugging Face owner/repository.")
    run.add_argument("--model-file", required=True, help="Exact standalone GGUF filename.")
    prompt = run.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Prompt text; use quotes to preserve shell characters.")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file.")
    run.add_argument("--model-revision", default="main")
    run.add_argument("--template-hash", default=DEFAULT_TEMPLATE_HASH)
    run.add_argument("--env-file", type=Path, default=ROOT / ".env")
    run.add_argument("--output-dir", type=Path, default=ROOT / "modeloutput")
    run.add_argument("--max-hourly-price", type=float, default=4.0)
    run.add_argument("--timeout-minutes", type=float, default=180)
    run.add_argument("--gpu-policy", choices=("prefer-a100", "a100-only"), default="prefer-a100")
    run.add_argument("--disk-gb", type=int, default=64)
    run.add_argument("--context-size", type=int, default=4096)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--dry-run", action="store_true", help="Only validate and query metadata/offers.")
    cleanup = commands.add_parser("cleanup", help="Destroy only resources owned by a recorded run.")
    cleanup.add_argument("--run-id", required=True)
    cleanup.add_argument("--env-file", type=Path, default=ROOT / ".env")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_for_cli().parse_args(argv)
    credentials: Credentials | None = None
    client: VastClient | None = None

    def cancelled(signum: int, frame: object) -> None:
        raise JobError(f"Workflow cancelled (signal {signum}).")

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, cancelled)
    try:
        credentials = credentials_from_file(args.env_file.resolve(), require_hf=args.command == "run")
        client = VastClient(credentials.vast)
        if args.command == "cleanup":
            state = RunState.load(args.run_id, ROOT / ".vastai-runs", credentials)
            try:
                with defer_cancellation():
                    cleanup_owned(client, state, Deadline(CLEANUP_SECONDS))
            except (JobError, OSError, ValueError) as error:
                state.data["cleanup_status"] = "failed"
                state.data["cleanup_error"] = error_text(error, credentials.secrets)
                try:
                    state.persist()
                except OSError as recording:
                    print("Could not persist recovery status: " + error_text(recording, credentials.secrets),
                          file=sys.stderr)
                raise
            state.data.pop("cleanup_error", None)
            state.persist()
            print(f"Cleanup confirmed for run {args.run_id}.")
            return 0
        if args.prompt_file is not None:
            if args.prompt_file.stat().st_size > MAX_PROMPT_BYTES:
                raise JobError("Prompt file exceeds the 1 MiB limit.")
            prompt = args.prompt_file.read_text(encoding="utf-8")
        else:
            prompt = args.prompt
        settings = Settings(
            args.model_repo, args.model_file, prompt, credentials,
            args.output_dir.resolve(), ROOT / ".vastai-runs",
            args.model_revision, args.template_hash, args.max_hourly_price,
            args.timeout_minutes, args.gpu_policy, args.disk_gb,
            args.context_size, args.max_tokens, args.temperature,
        )
        settings.validate()
        if args.dry_run:
            check_writable(settings.output_dir, create=False)
            check_writable(settings.state_root, create=False)
            model, template, offer = preflight(settings, client, Deadline(settings.timeout_minutes * 60))
            print(json.dumps(preview(settings, model, template, offer), indent=2, ensure_ascii=False))
            return 0
        state = run_workflow(settings, client)
        print(f"Response: {state.output / 'response.txt'}")
        print(f"Instance destruction confirmed. Metadata: {state.output / 'metadata.json'}")
        return 0
    except (JobError, OSError, ValueError) as error:
        secrets = credentials.secrets if credentials else ()
        print(error_text(error, secrets), file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
