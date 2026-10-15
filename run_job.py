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
from typing import Any, Callable, Iterable, Iterator
import uuid

from agent_utils import (
    AgentOptions, MAX_INPUT_BYTES, MAX_VERIFICATION_BYTES, command_succeeded,
    extract_archive, inspect_directory, pack_directory,
    task_env_name, validate_manifest,
)
from job_utils import (
    Deadline, DeadlineExceeded, JobError, LLAMA_COMMIT, LLAMA_RELEASE, MAX_WORKER_REQUEST_BYTES, atomic_write,
    commit_sha, completion_text, error_text, model_filenames, number, object_value,
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
    task: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def secrets(self) -> tuple[str, ...]:
        return self.vast, self.hugging_face, *self.task.values()


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
    agent: AgentOptions | None = None
    max_bandwidth_price: float | None = None
    offer_id: int | None = None
    gpu_count: int = 1

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.model_repo):
            raise JobError("Model repository must be a Hugging Face owner/repository ID.")
        model_filenames(self.model_file)
        positive_int(self.gpu_count, "GPU count")
        if not self.model_revision.strip():
            raise JobError("Model revision must not be empty.")
        if not self.prompt.strip() or len(self.prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise JobError("Prompt must be non-empty UTF-8 text of at most 1 MiB.")
        number(self.max_hourly_price, "Maximum hourly price", positive=True)
        number(self.timeout_minutes, "Timeout minutes", positive=True)
        if self.max_bandwidth_price is not None and number(
            self.max_bandwidth_price, "Maximum bandwidth price",
        ) < 0:
            raise JobError("Maximum bandwidth price must be non-negative.")
        positive_int(self.disk_gb, "Disk allocation")
        positive_int(self.context_size, "Context size")
        positive_int(self.max_tokens, "Maximum output tokens")
        temperature = number(self.temperature, "Temperature")
        if self.max_tokens >= self.context_size or not 0 <= temperature <= 2:
            raise JobError("Output tokens must be below context size; temperature must be between 0 and 2.")
        if self.gpu_policy not in ("prefer-a100", "a100-only"):
            raise JobError("GPU policy must be prefer-a100 or a100-only.")
        if self.offer_id is not None:
            positive_int(self.offer_id, "Requested offer ID")
        if not self.credentials.vast.strip() or not self.credentials.hugging_face.strip():
            raise JobError("Both VastAI and Hugging Face credentials are required.")
        if self.agent is not None:
            self.agent.validate()
            if self.agent.verification_code is not None and any(
                secret and secret in self.agent.verification_code for secret in self.credentials.secrets
            ):
                raise JobError("Verification script must not embed credential values.")
            if len(self.credentials.task) > 20:
                raise JobError("At most 20 task environment variables may be forwarded.")
            for name, value in self.credentials.task.items():
                task_env_name(name)
                if (
                    not isinstance(value, str) or not value.strip() or len(value.encode()) > 4096
                    or value in (self.credentials.vast, self.credentials.hugging_face)
                ):
                    raise JobError(f"Task variable {name} is invalid or reuses a protected credential.")


def credentials_from_file(
    path: Path, *, require_hf: bool = True, task_names: Iterable[str] = (),
) -> Credentials:
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

    names = list(task_names)
    if len(set(names)) != len(names):
        raise JobError("Task environment names must not be duplicated.")
    task = {task_env_name(name): value(name, True) for name in names}
    return Credentials(
        value("VAST_AI_API_KEY", True),
        value("HUGGING_FACE_ACCESS_TOKEN", require_hf),
        task,
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
        names = model_filenames(settings.model_file)
        if len(names) > 1:
            available = {item.rfilename for item in info.siblings}
            if not set(names) <= available:
                raise JobError("The selected split GGUF set is incomplete in the resolved repository revision.")
        files = []
        for name in names:
            metadata = get_hf_file_metadata(
                hf_hub_url(settings.model_repo, name, revision=revision),
                token=settings.credentials.hugging_face, timeout=deadline.timeout(),
            )
            if commit_sha(metadata.commit_hash) != revision:
                raise JobError("Hugging Face returned a different file revision than requested.")
            size = positive_int(metadata.size, "GGUF file size")
            etag = metadata.etag
            digest = etag if isinstance(etag, str) and re.fullmatch(r"[a-f0-9]{64}", etag) else None
            if len(names) > 1 and digest is None:
                raise JobError("Each split GGUF shard must provide authenticated SHA-256 metadata.")
            files.append({"filename": name, "size_bytes": size, "sha256": digest})
    except (OSError, ValueError, RuntimeError) as error:
        raise JobError(
            "Hugging Face file access failed. Check the token, filename, and model license: "
            + error_text(error, settings.credentials.secrets)
        ) from error
    deadline.remaining()
    return {
        "repo": settings.model_repo, "filename": settings.model_file,
        "revision": revision, "size_bytes": sum(item["size_bytes"] for item in files),
        "sha256": files[0]["sha256"] if len(files) == 1 else None,
        **({"files": files} if len(files) > 1 else {}),
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
        settings.timeout_minutes * 60 + CLEANUP_SECONDS, deadline, offer_id=settings.offer_id,
        gpu_count=settings.gpu_count,
    )
    return model, template, select_offer(settings, offers, model_size_bytes=model["size_bytes"])


def select_offer(settings: Settings, offers: list[Offer], *, model_size_bytes: int = 0) -> Offer:
    offers = [offer for offer in offers if offer.gpu_count == settings.gpu_count]
    if settings.offer_id is not None:
        offers = [offer for offer in offers if offer.id == settings.offer_id]
        if not offers:
            raise JobError(f"Requested offer {settings.offer_id} is unavailable or ineligible.")
    if settings.max_bandwidth_price is not None:
        offers = [
            offer for offer in offers
            if max(offer.download_cost_per_gb, offer.upload_cost_per_gb) <= settings.max_bandwidth_price
        ]
        if not offers:
            raise JobError(
                f"No offers meet the USD {settings.max_bandwidth_price:g}/GB bandwidth cap."
            )
    offers = [
        offer for offer in offers
        if offer.gpu_ram_mb * offer.gpu_count * 1_000_000 >= model_size_bytes
    ]
    if not offers:
        raise JobError("No eligible offer has enough aggregate GPU memory for the model weights.")
    return choose_offer(offers, settings.gpu_policy, settings.max_hourly_price)


def preview(settings: Settings, model: dict[str, Any], template: Template, offer: Offer) -> dict[str, Any]:
    return {
        "dry_run": True, "mode": "agent" if settings.agent is not None else "run", "model": model,
        "template": {"hash": template.hash_id, "image": template.image},
        "selected_offer": offer.summary(),
        "estimated_model_download_cost_usd": model["size_bytes"] / 1_000_000_000
        * offer.download_cost_per_gb,
        "output_dir": str(settings.output_dir),
        "limits": {
            "max_hourly_price_usd": settings.max_hourly_price,
            "requested_offer_id": settings.offer_id,
            "gpu_count": settings.gpu_count,
            "timeout_minutes": settings.timeout_minutes,
            "cleanup_seconds": CLEANUP_SECONDS, "disk_gb": settings.disk_gb,
            "context_size": settings.context_size, "max_tokens": settings.max_tokens,
            "max_bandwidth_price_per_gb_usd": settings.max_bandwidth_price,
            **({"agent": settings.agent.public()} if settings.agent is not None else {}),
        },
        "task_environment_names": sorted(settings.credentials.task),
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
            "mode": "agent" if settings.agent is not None else "run",
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

    def upload_agent(self, inputs: Path | None, deadline: Deadline) -> None:
        files = [ROOT / name for name in ("agent_job.py", "agent_tools.py", "agent_utils.py")]
        if inputs is not None:
            files.append(inputs)
        run_command(
            ["scp", "-q", *self.options(deadline), "-P", str(self.endpoint.port),
             *(str(path) for path in files), self.endpoint.scp_path(self.remote_dir + "/")],
            deadline, secrets=self.state.credentials.secrets, context="Uploading agent tools and inputs",
        )

    def invoke(self, stage: str, request: dict[str, Any], deadline: Deadline) -> dict[str, Any]:
        if stage not in ("bootstrap", "infer", "agent"):
            raise JobError("Invalid remote worker stage.")
        interpreter = "python3" if stage == "bootstrap" else f"{self.remote_dir}/venv/bin/python"
        payload = {**request, "remaining_seconds": deadline.remaining()}
        command = (
            f"{shlex.quote(interpreter)} {shlex.quote(self.remote_dir + '/remote_job.py')} {stage}"
        )
        if stage == "bootstrap":
            command = (
                "if ! command -v python3 >/dev/null 2>&1; then "
                "if [ \"$(id -u)\" != 0 ] || ! command -v apt-get >/dev/null 2>&1; then "
                "printf '%s\\n' 'JobError: Python 3 is missing and cannot be installed.' >&2; exit 1; fi; "
                "apt-get update -qq </dev/null >&2 && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv "
                "</dev/null >&2 || exit 1; fi; " + command
            )
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > MAX_WORKER_REQUEST_BYTES:
            raise JobError("Remote request exceeds the 8 MiB protocol limit.")
        raw = self.command(command, deadline, encoded)
        try:
            result = object_value(json.loads(raw), "Remote worker result")
        except (ValueError, UnicodeError) as error:
            raise JobError("Remote worker returned invalid JSON; see remote.log.") from error
        expected = {"ready"} if stage == "bootstrap" else {"completed", "failed"} if stage == "agent" else {"completed"}
        if result.get("status") not in expected or result.get("llama_commit") != LLAMA_COMMIT:
            raise JobError("Remote worker did not confirm the expected pinned inference stage.")
        return result

    def fetch(self, filename: str, destination: Path, deadline: Deadline) -> None:
        if filename not in (
            "response.txt", "completion.json", "remote.log", "agent-report.json",
            "agent-transcript.json", "artifacts.tar",
        ):
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


def checked_fetch(
    transport: SSHTransport, filename: str, part: Path, checksum: object, deadline: Deadline,
) -> None:
    if not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum):
        raise JobError(f"Remote agent report has no valid SHA-256 for {filename}.")
    transport.fetch(filename, part, deadline)
    if sha256_file(part, deadline) != checksum:
        raise JobError(f"Retrieved {filename} failed its integrity check.")


def retrieve_agent(
    transport: SSHTransport, state: RunState, options: AgentOptions, deadline: Deadline,
    expected_report: dict[str, Any] | None = None,
) -> None:
    report_part = state.output / "agent-report.json.part"
    transcript_part = state.output / "agent-transcript.json.part"
    archive_part = state.output / "artifacts.tar.part"
    response_part = state.output / "response.txt.part"
    try:
        if expected_report is not None:
            checked_fetch(
                transport, "agent-report.json", report_part,
                expected_report.get("agent_report_sha256"), deadline,
            )
        else:
            transport.fetch("agent-report.json", report_part, deadline)
        report = read_json(report_part)
        if any(secret and secret.encode() in report_part.read_bytes() for secret in state.credentials.secrets):
            raise JobError("Agent report contains a protected credential value.")
        if expected_report is not None and report != {
            key: value for key, value in expected_report.items() if key != "agent_report_sha256"
        }:
            raise JobError("Agent report does not match the worker result.")
        if report.get("status") not in ("completed", "failed") or report.get("llama_commit") != LLAMA_COMMIT:
            raise JobError("Agent report has an invalid outcome or runtime revision.")
        steps = report.get("steps")
        if (
            isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= options.max_steps
            or report.get("max_steps") != options.max_steps
        ):
            raise JobError("Agent report has invalid step counts.")
        checked_fetch(
            transport, "agent-transcript.json", transcript_part, report.get("transcript_sha256"), deadline,
        )
        transcript = read_json(transcript_part)
        if any(secret and secret.encode() in transcript_part.read_bytes() for secret in state.credentials.secrets):
            raise JobError("Agent transcript contains a protected credential value.")
        if (
            transcript.get("version") != 1
            or transcript.get("prompt_sha256") != state.data["settings"]["prompt_sha256"]
            or not isinstance(transcript.get("steps"), list) or len(transcript["steps"]) != steps
        ):
            raise JobError("Agent transcript does not match the task and reported decisions.")
        atomic_write(state.output / "agent-report.json", report_part.read_bytes())
        atomic_write(state.output / "agent-transcript.json", transcript_part.read_bytes())
        state.data["agent"] = {
            "status": report["status"], "steps": steps,
            "environment_names": report.get("environment_names"),
        }
        archive = report.get("archive")
        if archive is not None:
            bundle = object_value(archive, "Agent archive")
            maximum = options.max_artifact_bytes
            manifest = validate_manifest(bundle.get("manifest"), maximum)
            checked_fetch(transport, "artifacts.tar", archive_part, bundle.get("sha256"), deadline)
            if archive_part.stat().st_size != bundle.get("size_bytes"):
                raise JobError("Agent archive size does not match its report.")
            destination = state.output / "artifacts"
            if destination.exists():
                files, _ = inspect_directory(destination, maximum)
                if {name for _, name, _ in files} != set(manifest):
                    raise JobError("Previously retrieved agent artifacts do not match the manifest.")
                for path, name, size in files:
                    if size != manifest[name]["size_bytes"] or sha256_file(path, deadline) != manifest[name]["sha256"]:
                        raise JobError("Previously retrieved agent artifacts failed integrity checks.")
            else:
                with tempfile.TemporaryDirectory(dir=state.output, prefix=".artifacts-") as temporary:
                    unpacked = Path(temporary) / "files"
                    unpacked.mkdir(mode=0o700)
                    extract_archive(
                        archive_part, unpacked, bundle["manifest"], deadline, maximum=maximum,
                    )
                    for path, name, _ in inspect_directory(unpacked, maximum)[0]:
                        data = path.read_bytes()
                        if any(secret and (secret in name or secret.encode() in data)
                               for secret in state.credentials.secrets):
                            raise JobError("Retrieved workspace contains a protected credential value.")
                    os.replace(unpacked, destination)
            state.data["agent"]["artifacts"] = len(manifest)
            state.data["agent"]["excluded_paths"] = bundle["manifest"]["excluded_paths"]
        elif report["status"] == "completed":
            raise JobError("Completed agent did not provide its workspace archive.")
        else:
            warning = "Agent workspace could not be collected: " + str(report.get("artifact_error", "No archive"))
            state.data["warnings"].append(warning)
            state.log(warning)
        summary = report.get("final_response")
        if options.verification_code is not None:
            if report.get("verification_sha256") != options.public()["verification_sha256"]:
                raise JobError("Agent did not use the caller's verification script snapshot.")
            if report["status"] == "completed":
                verification = object_value(report.get("verification"), "Agent verification")
                uid = verification.get("uid")
                if (
                    not command_succeeded(verification)
                    or verification.get("sandbox_verified") is not True
                    or isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0
                ):
                    raise JobError("Completed agent has no successful non-root task verification.")
                final_step = object_value(transcript["steps"][-1], "Final agent decision") if steps else {}
                final_action = object_value(final_step.get("action"), "Final agent action")
                final_observation = object_value(final_step.get("observation"), "Final agent observation")
                if (
                    final_action.get("action") != "finish"
                    or final_observation.get("finished") is not True
                    or final_observation.get("verification") != verification
                ):
                    raise JobError("Agent verification does not match the final execution observation.")
            state.data["agent"]["verification_sha256"] = report["verification_sha256"]
            state.data["agent"]["verification"] = report.get("verification")
        if summary is not None:
            if not isinstance(summary, str) or not summary.strip():
                raise JobError("Agent final response is invalid.")
            checked_fetch(
                transport, "response.txt", response_part, report.get("response_sha256"), deadline,
            )
            if response_part.read_bytes() != summary.encode():
                raise JobError("Agent final response does not match its structured report.")
            atomic_write(state.output / "response.txt", response_part.read_bytes())
        elif report["status"] == "completed":
            raise JobError("Completed agent did not provide a final response.")
        if report["status"] == "failed" and (
            not isinstance(report.get("error"), str) or not report["error"].strip()
        ):
            raise JobError("Failed agent did not provide an explicit error.")
        state.data["agent_artifacts_retrieved"] = True
    finally:
        for part in (report_part, transcript_part, archive_part, response_part):
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
            "requested_offer_id": settings.offer_id,
            "gpu_count": settings.gpu_count,
            "timeout_minutes": settings.timeout_minutes, "gpu_policy": settings.gpu_policy,
            "disk_gb": settings.disk_gb, "context_size": settings.context_size,
            "max_tokens": settings.max_tokens, "temperature": settings.temperature,
            "max_bandwidth_price_per_gb_usd": settings.max_bandwidth_price,
            "prompt_sha256": hashlib.sha256(settings.prompt.encode()).hexdigest(),
            **({
                "agent": settings.agent.public(), "task_environment_names": sorted(settings.credentials.task),
            } if settings.agent is not None else {}),
        },
    })
    transport: SSHTransport | None = None
    failure: str | None = None
    cleanup_failure: str | None = None
    reporting_failure: str | None = None
    succeeded = False
    creation_attempted = False
    input_bundle: dict[str, Any] | None = None
    state.log(f"Run {state.data['run_id']}: selected {offer.gpu_name} at ${offer.hourly_price:.4f}/hour.")
    try:
        if settings.agent is not None and settings.agent.input_dir is not None:
            input_bundle = pack_directory(
                settings.agent.input_dir, state.directory / "inputs.tar", deadline,
                maximum=MAX_INPUT_BYTES, secrets=settings.credentials.secrets, input_files=True,
            )
            state.data["inputs"] = input_bundle
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
            gpu_count=settings.gpu_count,
        )
        offer = select_offer(settings, refreshed, model_size_bytes=model["size_bytes"])
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
        if settings.agent is not None:
            transport.upload_agent(state.directory / "inputs.tar" if input_bundle is not None else None, deadline)
        transport.invoke("bootstrap", {
            "gpu_name": offer.gpu_name, "gpu_count": settings.gpu_count,
            **({"agent": True} if settings.agent is not None else {}),
        }, deadline)
        state.log("Downloading the GGUF and running the agent remotely." if settings.agent is not None
                  else "Downloading the GGUF and generating the response remotely.")
        request = {
            "model": model, "hf_token": settings.credentials.hugging_face,
            "prompt": settings.prompt, "gpu_name": offer.gpu_name,
            "gpu_count": settings.gpu_count,
            "context_size": settings.context_size, "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
        }
        if settings.agent is not None:
            request.update({
                "agent": {
                    **settings.agent.public(), "verification_code": settings.agent.verification_code,
                },
                "task_env": settings.credentials.task, "inputs": input_bundle,
            })
        report = transport.invoke("agent" if settings.agent is not None else "infer", request, deadline)
        gpu_name = report.get("gpu_name")
        gpu_memory = number(report.get("gpu_vram_mib"), "Reported GPU VRAM", positive=True)
        if not isinstance(gpu_name, str) or not gpu_name.strip() or gpu_memory < 24_000:
            raise JobError("Remote worker did not report a valid >=24 GB NVIDIA GPU.")
        state.data["remote_gpu"] = {"gpu_name": gpu_name, "gpu_vram_mib": gpu_memory}
        if settings.gpu_count > 1:
            count = positive_int(report.get("gpu_count"), "Reported GPU count")
            devices = report.get("gpus")
            if count != settings.gpu_count or not isinstance(devices, list) or len(devices) != count:
                raise JobError("Remote worker did not confirm every selected GPU.")
            checked = []
            for item in devices:
                device = object_value(item, "Reported GPU")
                memory = number(device.get("gpu_vram_mib"), "Reported per-GPU VRAM", positive=True)
                if device.get("gpu_name") != gpu_name or memory < 24_000:
                    raise JobError("Remote GPU inventory does not match the selected homogeneous GPU group.")
                checked.append({"gpu_name": gpu_name, "gpu_vram_mib": memory})
            total_memory = number(report.get("gpu_vram_total_mib"), "Reported total GPU VRAM", positive=True)
            if total_memory != sum(item["gpu_vram_mib"] for item in checked):
                raise JobError("Remote aggregate GPU VRAM does not match the per-device inventory.")
            state.data["remote_gpu"].update({
                "gpu_count": count, "gpu_vram_total_mib": total_memory, "gpus": checked,
            })
        if settings.agent is not None and report["status"] == "failed":
            raise JobError("Remote agent failed: " + str(report["error"]))
        state.log("Retrieving and validating agent outputs." if settings.agent is not None
                  else "Retrieving and validating the model response.")
        if settings.agent is not None:
            retrieve_agent(transport, state, settings.agent, deadline, report)
        else:
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
                if settings.agent is not None and not state.data.get("agent_artifacts_retrieved"):
                    try:
                        retrieve_agent(
                            transport, state, settings.agent,
                            Deadline(min(45, cleanup_deadline.remaining())),
                        )
                    except (JobError, OSError, ValueError) as error:
                        warning = "Partial agent artifacts unavailable: " + error_text(error, settings.credentials.secrets)
                        state.data["warnings"].append(warning)
                        state.log(warning)
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
    parser = argparse.ArgumentParser(description="Run a GGUF prompt or action-capable agent on a temporary VastAI GPU.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "agent"):
        run = commands.add_parser(name, help="Rent, infer, retrieve, and destroy." if name == "run"
                                  else "Run remote actions, retrieve artifacts, and destroy.")
        run.add_argument("--model-repo", required=True, help="Hugging Face owner/repository.")
        run.add_argument("--model-file", required=True, help="Standalone GGUF or shard 00001 of a complete split set.")
        prompt = run.add_mutually_exclusive_group(required=True)
        prompt.add_argument("--prompt", help="Prompt text; use quotes to preserve shell characters.")
        prompt.add_argument("--prompt-file", type=Path, help="UTF-8 task/prompt file, including Markdown.")
        run.add_argument("--model-revision", default="main")
        run.add_argument("--template-hash", default=DEFAULT_TEMPLATE_HASH)
        run.add_argument("--offer-id", type=int,
                         help="Select one marketplace offer without bypassing compatibility or price checks.")
        run.add_argument("--env-file", type=Path, default=ROOT / ".env")
        run.add_argument("--output-dir", type=Path, default=ROOT / "modeloutput")
        run.add_argument("--max-hourly-price", type=float, default=4.0)
        run.add_argument("--max-bandwidth-price", type=float,
                         help="Optional advertised USD/GB cap for both ingress and egress.")
        run.add_argument("--timeout-minutes", type=float, default=180)
        run.add_argument("--gpu-policy", choices=("prefer-a100", "a100-only"), default="prefer-a100")
        run.add_argument("--gpu-count", type=int, default=1, help="Exact number of GPUs in one rented instance.")
        run.add_argument("--disk-gb", type=int, default=64)
        run.add_argument("--context-size", type=int, default=4096 if name == "run" else 16384)
        run.add_argument("--max-tokens", type=int, default=1024 if name == "run" else 2048)
        run.add_argument("--temperature", type=float, default=0.7 if name == "run" else 0)
        run.add_argument("--dry-run", action="store_true", help="Only validate and query metadata/offers.")
        if name == "agent":
            run.add_argument("--input-dir", type=Path, help="Optional small, credential-free workspace snapshot.")
            run.add_argument("--verify-script", type=Path,
                             help="Snapshot a trusted standard-library Python verifier; require exit 0 before finish.")
            run.add_argument("--agent-env", action="append", default=[], metavar="NAME",
                             help="Forward only this selected task variable from the chosen .env/environment.")
            run.add_argument("--max-steps", type=int, default=30)
            run.add_argument("--command-timeout-seconds", type=float, default=120)
            run.add_argument("--max-artifact-mb", type=int, default=64)
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
        credentials = credentials_from_file(
            args.env_file.resolve(), require_hf=args.command != "cleanup",
            task_names=args.agent_env if args.command == "agent" else (),
        )
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
        agent_options = None
        if args.command == "agent":
            verification_code = None
            if args.verify_script is not None:
                if args.verify_script.is_symlink() or not args.verify_script.is_file():
                    raise JobError("Verification script must be a regular file, not a symlink.")
                with args.verify_script.open("rb") as source:
                    verification_code = source.read(MAX_VERIFICATION_BYTES + 1).decode("utf-8")
            agent_options = AgentOptions(
                args.max_steps, args.command_timeout_seconds, args.max_artifact_mb * 1024 ** 2,
                args.input_dir.absolute() if args.input_dir is not None else None,
                verification_code=verification_code,
            )
        settings = Settings(
            args.model_repo, args.model_file, prompt, credentials,
            args.output_dir.resolve(), ROOT / ".vastai-runs",
            args.model_revision, args.template_hash, args.max_hourly_price,
            args.timeout_minutes, args.gpu_policy, args.disk_gb,
            args.context_size, args.max_tokens, args.temperature,
            agent_options,
            args.max_bandwidth_price,
            args.offer_id,
            args.gpu_count,
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
        if settings.agent is not None:
            print(f"Agent files: {state.output / 'artifacts'}; transcript: {state.output / 'agent-transcript.json'}")
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
