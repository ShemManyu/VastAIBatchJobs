from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import selectors
import stat
import subprocess
import sys
import time
from typing import Any


class ToolError(Exception):
    pass


def path_parts(value: object, *, allow_root: bool = False) -> list[str]:
    if allow_root and value == ".":
        return []
    if not isinstance(value, str) or not value or "\\" in value:
        raise ToolError("Use a relative workspace path.")
    if (
        PurePosixPath(value).is_absolute() or PurePosixPath(value).as_posix() != value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ToolError("Absolute paths, traversal, and control characters are forbidden.")
    return value.split("/")


def directory_descriptor(workspace: Path, parts: list[str], *, create: bool = False) -> int:
    descriptor = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def file_action(workspace: Path, action: dict[str, Any], maximum: int) -> dict[str, Any]:
    kind = action["action"]
    parts = path_parts(action.get("path", "."), allow_root=kind == "list_files")
    parent = directory_descriptor(
        workspace, parts if kind == "list_files" else parts[:-1], create=kind == "write_file",
    )
    try:
        if kind == "list_files":
            names = sorted(os.listdir(parent))
            return {"entries": names[:200], "truncated": len(names) > 200}
        if kind == "write_file":
            content = action.get("content")
            if not isinstance(content, str) or len(content.encode()) > 1024 ** 2:
                raise ToolError("write_file content must be UTF-8 text of at most 1 MiB.")
            descriptor = os.open(
                parts[-1], os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600, dir_fd=parent,
            )
            with os.fdopen(descriptor, "wb") as output:
                info = os.fstat(output.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ToolError("write_file only accepts unlinked regular files.")
                os.ftruncate(output.fileno(), 0)
                output.write(content.encode())
            return {"path": action["path"], "bytes_written": len(content.encode())}
        offset = action.get("offset", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolError("read_file offset must be a non-negative byte offset.")
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ToolError("read_file only accepts regular files.")
            if offset > info.st_size:
                raise ToolError("read_file offset exceeds the file size.")
            source.seek(offset)
            data = source.read(maximum + 1)
        truncated = len(data) > maximum
        data = data[:maximum]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            if truncated and error.end == len(data) and error.reason == "unexpected end of data":
                data = data[:error.start]
                text = data.decode("utf-8")
            else:
                raise ToolError("read_file requires UTF-8 text; use a script to inspect binary files.") from error
        return {
            "path": action["path"], "content": text, "offset": offset,
            "bytes_read": len(data), "truncated": truncated,
            "next_offset": offset + len(data) if truncated else None,
        }
    finally:
        os.close(parent)


def command_action(
    workspace: Path, action: dict[str, Any], maximum: int, timeout: float,
) -> dict[str, Any]:
    argv = action.get("argv")
    if (
        not isinstance(argv, list) or not 1 <= len(argv) <= 128
        or any(not isinstance(item, str) or "\0" in item or len(item.encode()) > 16384 for item in argv)
        or not argv[0] or sum(len(item.encode()) for item in argv) > 65536
    ):
        raise ToolError("run requires a bounded, non-empty argv list of strings, not a shell string.")
    if Path(argv[0]).name in {"bash", "sh", "dash", "zsh", "ksh", "fish"}:
        for item in argv[1:]:
            if re.search(r"\$\{[^}]*@P\}|\$\{!|\beval\b", item):
                raise ToolError("Obfuscated shell expansion and eval are forbidden.")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    counts = {"stdout": 0, "stderr": 0}
    end = time.monotonic() + timeout
    timed_out = False
    with subprocess.Popen(
        argv, cwd=workspace, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) as process, selectors.DefaultSelector() as selector:
        if process.stdout is None or process.stderr is None:
            raise ToolError("Command output pipes are unavailable.")
        for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        while selector.get_map():
            remaining = end - time.monotonic()
            if remaining <= 0:
                timed_out = True
                process.kill()
                break
            for key, _ in selector.select(min(0.1, remaining)):
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                counts[name] += len(chunk)
                buffers[name].extend(chunk[:max(0, maximum - len(buffers[name]))])
        if timed_out:
            process.wait(timeout=5)
        else:
            remaining = end - time.monotonic()
            try:
                process.wait(timeout=max(0, remaining))
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait(timeout=5)
        return {
            "exit_code": process.returncode, "timed_out": timed_out,
            **{name: data.decode("utf-8", errors="replace") for name, data in buffers.items()},
            **{name + "_truncated": counts[name] > maximum for name in buffers},
        }


def execute(
    workspace: Path, action: dict[str, Any], *, maximum: int = 16384, timeout: float = 120,
) -> dict[str, Any]:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 4 <= maximum <= 16384:
        raise ToolError("Tool output limit must be between 4 and 16,384 bytes.")
    if (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout) or not 0 < timeout <= 600
    ):
        raise ToolError("Tool timeout must be positive and at most 600 seconds.")
    kind = action.get("action")
    if kind == "run":
        return command_action(workspace, action, maximum, timeout)
    if kind in ("read_file", "write_file", "list_files"):
        return file_action(workspace, action, maximum)
    raise ToolError("Unknown action; use run, read_file, write_file, or list_files.")


def main() -> int:
    sandbox_verified = False
    try:
        data = sys.stdin.buffer.read(2 * 1024 ** 2 + 1)
        if len(data) > 2 * 1024 ** 2:
            raise ToolError("Tool request is too large.")
        request = json.loads(data)
        expected_uid = request["uid"]
        if not isinstance(expected_uid, int) or expected_uid <= 0 or os.geteuid() != expected_uid:
            raise ToolError("Tool identity must be the expected non-root user.")
        status = Path("/proc/self/status").read_text()
        if not re.search(r"^NoNewPrivs:\s+1$", status, re.MULTILINE) or not re.search(
            r"^CapEff:\s+0+$", status, re.MULTILINE
        ):
            raise ToolError("Tools require no-new-privileges and zero effective capabilities.")
        sandbox_verified = True
        timeout = request["timeout"]
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ToolError("Tool timeout is invalid.")
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        # V8 reserves several GiB of virtual address space even for a small heap.
        resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 ** 3, 16 * 1024 ** 3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 ** 2, 64 * 1024 ** 2))
        resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(timeout) + 1, math.ceil(timeout) + 1))
        result = execute(
            Path(request["workspace"]), request["action"], timeout=timeout,
            maximum=request["max_output_bytes"],
        )
        result["uid"] = expected_uid
        result["sandbox_verified"] = True
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ToolError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({
            "error": f"{type(error).__name__}: {error}",
            "uid": os.geteuid(), "sandbox_verified": sandbox_verified,
        }))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
