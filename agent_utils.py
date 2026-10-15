from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
from typing import Any, Iterable

from job_utils import Deadline, JobError, number, object_value, positive_int, sha256_file


MAX_INPUT_BYTES = 64 * 1024 ** 2
MAX_ARCHIVE_FILES = 5000
TOOL_OUTPUT_BYTES = 16 * 1024
MAX_VERIFICATION_BYTES = 16 * 1024
ARTIFACT_EXCLUDES = frozenset({
    "node_modules", ".git", ".venv", "venv", "__pycache__", ".cache", ".npm",
})


@dataclass(frozen=True)
class AgentOptions:
    max_steps: int = 30
    command_timeout_seconds: float = 120
    max_artifact_bytes: int = 64 * 1024 ** 2
    input_dir: Path | None = None
    verification_code: str | None = field(default=None, repr=False)

    def validate(self) -> None:
        if positive_int(self.max_steps, "Agent step limit") > 200:
            raise JobError("Agent step limit must not exceed 200.")
        timeout = number(self.command_timeout_seconds, "Command timeout", positive=True)
        if timeout > 600:
            raise JobError("Command timeout must not exceed 600 seconds.")
        if positive_int(self.max_artifact_bytes, "Artifact byte limit") > 1024 ** 3:
            raise JobError("Artifact limit must not exceed 1 GiB.")
        if self.input_dir is not None:
            inspect_directory(self.input_dir, MAX_INPUT_BYTES, input_files=True)
        if self.verification_code is not None:
            if (
                not isinstance(self.verification_code, str) or not self.verification_code.strip()
                or len(self.verification_code.encode("utf-8")) > MAX_VERIFICATION_BYTES
                or "\0" in self.verification_code
            ):
                raise JobError("Verification script must be non-empty UTF-8 Python of at most 16 KiB.")
            try:
                compile(self.verification_code, "<agent verification>", "exec")
            except SyntaxError as error:
                raise JobError("Verification script has invalid Python syntax.") from error

    def public(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "command_timeout_seconds": self.command_timeout_seconds,
            "max_artifact_bytes": self.max_artifact_bytes,
            "tool_output_bytes": TOOL_OUTPUT_BYTES,
            "has_input_directory": self.input_dir is not None,
            "verification_sha256": (
                hashlib.sha256(self.verification_code.encode("utf-8")).hexdigest()
                if self.verification_code is not None else None
            ),
        }


def command_succeeded(result: dict[str, Any]) -> bool:
    exit_code = result.get("exit_code")
    return (
        "error" not in result and isinstance(exit_code, int) and not isinstance(exit_code, bool)
        and exit_code == 0 and result.get("timed_out") is False
    )


def task_env_name(name: str) -> str:
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
        raise JobError("Task environment names must be uppercase identifiers.")
    if name in {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "ENV", "BASH_ENV",
        "TOKIO_WORKER_THREADS",
    } or name.startswith(
        ("VAST", "HF_", "HUGGING", "LLAMA", "PYTHON", "LD_", "DYLD_")
    ):
        raise JobError(f"{name} is reserved and cannot be forwarded to agent tools.")
    return name


def relative_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise JobError("Workspace paths must be non-empty relative POSIX paths.")
    path = PurePosixPath(value)
    if (
        path.is_absolute() or path.as_posix() != value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise JobError("Workspace paths must not be absolute, traverse directories, or contain controls.")
    return value


def credential_filename(name: str) -> bool:
    lower = name.lower()
    return (
        lower == ".env" or (lower.startswith(".env.") and lower != ".env.example")
        or lower in {".ssh", ".aws", ".azure", ".gnupg", "credentials", "id_rsa", "id_ed25519"}
        or lower.endswith((".pem", ".key", ".p12", ".pfx", ".kdbx"))
    )


def inspect_directory(
    directory: Path, maximum: int, *, input_files: bool = False,
    excludes: Iterable[str] = (),
) -> tuple[list[tuple[Path, str, int]], list[str]]:
    if directory.is_symlink() or not directory.is_dir():
        raise JobError("Input/workspace directory must be a real directory, not a symlink.")
    directory = directory.resolve()
    excluded_names = frozenset(excludes)
    files: list[tuple[Path, str, int]] = []
    excluded: list[str] = []
    total = 0
    entries = 0

    def visit(parent: Path, depth: int = 0) -> None:
        nonlocal total, entries
        if depth > 64:
            raise JobError("Workspace directory nesting cannot exceed 64 levels.")
        for path in sorted(parent.iterdir()):
            entries += 1
            if entries > MAX_ARCHIVE_FILES * 2:
                raise JobError("Workspace contains too many directory entries.")
            name = relative_name(path.relative_to(directory).as_posix())
            if path.name in excluded_names:
                excluded.append(name)
                continue
            if credential_filename(path.name) or (
                input_files and path.name in ARTIFACT_EXCLUDES
            ):
                raise JobError(f"Refusing credential/dependency input or artifact path: {name}.")
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise JobError(f"Workspace archives cannot contain symlinks: {name}.")
            if stat.S_ISDIR(info.st_mode):
                visit(path, depth + 1)
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise JobError(f"Workspace archives cannot contain hard-linked files: {name}.")
                total += info.st_size
                if total > maximum:
                    raise JobError(f"Workspace files exceed the {maximum}-byte limit.")
                files.append((path, name, info.st_size))
                if len(files) > MAX_ARCHIVE_FILES:
                    raise JobError(f"Workspace archives cannot exceed {MAX_ARCHIVE_FILES} files.")
            else:
                raise JobError(f"Workspace archives cannot contain special files: {name}.")

    visit(directory)
    return files, excluded


def pack_directory(
    directory: Path, archive: Path, deadline: Deadline, *, maximum: int,
    secrets: Iterable[str] = (), input_files: bool = False,
    excludes: Iterable[str] = (),
) -> dict[str, Any]:
    files, excluded = inspect_directory(
        directory, maximum, input_files=input_files, excludes=excludes,
    )
    secret_values = tuple(secret for secret in secrets if secret)
    sentinels = [secret.encode("utf-8") for secret in secret_values]
    overlap = max((len(secret) for secret in sentinels), default=1) - 1
    manifest: list[dict[str, Any]] = []
    if any(secret in name for name in excluded for secret in secret_values):
        raise JobError("Credential content detected in an excluded workspace path.")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent, prefix=".bundle-", delete=False) as output:
            temporary = Path(output.name)
            with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as bundle:
                for path, name, size in files:
                    deadline.remaining()
                    if any(secret in name for secret in secret_values):
                        raise JobError("Credential content detected in a workspace filename.")
                    digest = hashlib.sha256()
                    count = 0
                    trailing = b""
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(descriptor, "rb") as source, \
                         tempfile.SpooledTemporaryFile(max_size=1024 ** 2) as snapshot:
                        while chunk := source.read(1024 ** 2):
                            deadline.remaining()
                            count += len(chunk)
                            if count > size:
                                raise JobError(f"Workspace file changed during snapshot: {name}.")
                            scanned = trailing + chunk
                            if any(secret in scanned for secret in sentinels):
                                raise JobError(f"Credential content detected in workspace file: {name}.")
                            trailing = scanned[-overlap:] if overlap else b""
                            digest.update(chunk)
                            snapshot.write(chunk)
                        if count != size:
                            raise JobError(f"Workspace file changed during snapshot: {name}.")
                        snapshot.seek(0)
                        member = tarfile.TarInfo(name)
                        member.size = count
                        member.mode = 0o600
                        bundle.addfile(member, snapshot)
                    manifest.append({
                        "path": name, "size_bytes": count, "sha256": digest.hexdigest(),
                    })
                if len(json.dumps(manifest, ensure_ascii=False).encode()) > 1024 ** 2:
                    raise JobError("Workspace file manifest exceeds the 1 MiB limit.")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, archive)
        return {
            "sha256": sha256_file(archive, deadline),
            "size_bytes": archive.stat().st_size,
            "manifest": {
                "files": manifest, "total_bytes": sum(item["size_bytes"] for item in manifest),
                "excluded_paths": excluded,
            },
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_manifest(value: object, maximum: int) -> dict[str, dict[str, Any]]:
    manifest = object_value(value, "Workspace manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) > MAX_ARCHIVE_FILES:
        raise JobError("Workspace manifest has an invalid file list.")
    result: dict[str, dict[str, Any]] = {}
    total = 0
    for raw in files:
        row = object_value(raw, "Workspace file")
        name = relative_name(row.get("path"))
        size = row.get("size_bytes")
        checksum = row.get("sha256")
        if (
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            or not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum)
            or name in result
        ):
            raise JobError("Workspace manifest contains invalid or duplicate file metadata.")
        total += size
        if total > maximum:
            raise JobError("Workspace manifest exceeds the configured artifact limit.")
        result[name] = row
    if manifest.get("total_bytes") != total:
        raise JobError("Workspace manifest total does not match its files.")
    excluded = manifest.get("excluded_paths")
    if not isinstance(excluded, list) or any(not isinstance(name, str) for name in excluded):
        raise JobError("Workspace manifest has invalid excluded paths.")
    for name in excluded:
        relative_name(name)
    return result


def extract_archive(
    archive: Path, destination: Path, manifest: object, deadline: Deadline, *, maximum: int,
) -> None:
    expected = validate_manifest(manifest, maximum)
    if archive.stat().st_size > maximum + MAX_ARCHIVE_FILES * 2048 + 10240:
        raise JobError("Workspace archive exceeds its bounded transfer size.")
    if not destination.is_dir() or destination.is_symlink() or any(destination.iterdir()):
        raise JobError("Archive destination must be an empty private directory.")
    seen: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:") as bundle:
            for member in bundle:
                deadline.remaining()
                name = relative_name(member.name)
                if (
                    not member.isfile() or member.issparse()
                    or name not in expected or name in seen
                    or member.size != expected[name]["size_bytes"]
                ):
                    raise JobError("Workspace archive has unexpected, linked, or duplicate entries.")
                seen.add(name)
                path = destination / name
                parent = destination
                for part in name.split("/")[:-1]:
                    parent = parent / part
                    parent.mkdir(exist_ok=True, mode=0o700)
                source = bundle.extractfile(member)
                if source is None:
                    raise JobError("Workspace archive file could not be read.")
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                digest = hashlib.sha256()
                with source, os.fdopen(descriptor, "wb") as output:
                    while chunk := source.read(1024 ** 2):
                        deadline.remaining()
                        digest.update(chunk)
                        output.write(chunk)
                if digest.hexdigest() != expected[name]["sha256"]:
                    raise JobError(f"Workspace artifact failed its integrity check: {name}.")
    except tarfile.TarError as error:
        raise JobError("Workspace archive is malformed.") from error
    if seen != set(expected):
        raise JobError("Workspace archive is missing expected files.")
