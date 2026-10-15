import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from agent_utils import (
    AgentOptions, ARTIFACT_EXCLUDES, MAX_VERIFICATION_BYTES, extract_archive, inspect_directory,
    pack_directory, task_env_name, validate_manifest,
)
from job_utils import Deadline, JobError


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.archive = self.root / "bundle.tar"
        self.destination = self.root / "unpacked"
        self.destination.mkdir(mode=0o700)

    def test_snapshot_round_trip_preserves_bytes_hashes_and_private_permissions(self):
        (self.workspace / "scripts").mkdir()
        data = "print('caf\u00e9')\n".encode()
        (self.workspace / "scripts/job.py").write_bytes(data)
        (self.workspace / "empty").write_bytes(b"")
        info = pack_directory(self.workspace, self.archive, Deadline(10), maximum=len(data))
        self.assertEqual(info["manifest"]["total_bytes"], len(data))
        self.assertEqual(info["sha256"], hashlib.sha256(self.archive.read_bytes()).hexdigest())
        extract_archive(self.archive, self.destination, info["manifest"], Deadline(10), maximum=len(data))
        self.assertEqual((self.destination / "scripts/job.py").read_bytes(), data)
        self.assertEqual((self.destination / "empty").read_bytes(), b"")
        self.assertEqual((self.destination / "scripts").stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.destination / "scripts/job.py").stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.archive.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(JobError):
            pack_directory(self.workspace, self.root / "too-small.tar", Deadline(10), maximum=len(data) - 1)

    def test_dependency_exclusions_are_explicit_and_do_not_return_their_contents(self):
        (self.workspace / "node_modules").mkdir()
        (self.workspace / "node_modules/key.txt").write_text("not returned")
        (self.workspace / "report.md").write_text("report")
        info = pack_directory(self.workspace, self.archive, Deadline(10), maximum=10, excludes=ARTIFACT_EXCLUDES)
        self.assertEqual(info["manifest"]["excluded_paths"], ["node_modules"])
        self.assertEqual([row["path"] for row in info["manifest"]["files"]], ["report.md"])
        with self.assertRaises(JobError):
            inspect_directory(self.workspace, 100, input_files=True)

    def test_secret_filenames_content_and_chunk_boundary_matches_are_rejected(self):
        (self.workspace / ".env").write_text("sensitive")
        with self.assertRaises(JobError):
            pack_directory(self.workspace, self.archive, Deadline(10), maximum=2 * 1024 ** 2)
        (self.workspace / ".env").unlink()
        secret = "API_KEY_SENTINEL_MUST_NOT_LEAK"
        (self.workspace / "binary").write_bytes(b"x" * (1024 ** 2 - 3) + secret.encode() + b"\0")
        with self.assertRaises(JobError):
            pack_directory(self.workspace, self.archive, Deadline(10), maximum=2 * 1024 ** 2, secrets=(secret,))
        self.assertFalse(self.archive.exists())
        self.assertEqual(list(self.root.glob(".bundle-*")), [])
        (self.workspace / "binary").unlink()
        (self.workspace / (secret + ".txt")).write_text("safe content")
        with self.assertRaises(JobError):
            pack_directory(self.workspace, self.archive, Deadline(10), maximum=100, secrets=iter((secret,)))

    def test_source_symlinks_hardlinks_and_special_files_are_rejected(self):
        original = self.root / "outside"
        original.write_text("outside")
        for kind in ("symlink", "hardlink", "fifo"):
            path = self.workspace / "bad"
            if kind == "symlink":
                path.symlink_to(original)
            elif kind == "hardlink":
                os.link(original, path)
            else:
                os.mkfifo(path)
            with self.subTest(kind=kind), self.assertRaises(JobError):
                inspect_directory(self.workspace, 100)
            path.unlink()

    def crafted_archive(self, name, *, kind=tarfile.REGTYPE, duplicate=False):
        data = b"data"
        with tarfile.open(self.archive, "w") as bundle:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            member.linkname = "../outside"
            bundle.addfile(member, io.BytesIO(data) if member.size else None)
            if duplicate:
                bundle.addfile(member, io.BytesIO(data))
        return {
            "files": [{"path": "safe", "size_bytes": 4, "sha256": hashlib.sha256(data).hexdigest()}],
            "total_bytes": 4, "excluded_paths": [],
        }

    def test_untrusted_archive_paths_links_and_duplicates_never_escape_destination(self):
        for name, kind, duplicate in (
            ("../escape", tarfile.REGTYPE, False), ("/absolute", tarfile.REGTYPE, False),
            ("safe", tarfile.SYMTYPE, False), ("safe", tarfile.LNKTYPE, False),
            ("safe", tarfile.REGTYPE, True),
        ):
            with self.subTest(name=name, kind=kind, duplicate=duplicate):
                destination = self.root / f"test-{len(list(self.root.iterdir()))}"
                destination.mkdir()
                manifest = self.crafted_archive(name, kind=kind, duplicate=duplicate)
                with self.assertRaises(JobError):
                    extract_archive(self.archive, destination, manifest, Deadline(10), maximum=100)
                self.assertFalse((self.root / "escape").exists())

    def test_manifest_and_content_integrity_are_not_just_message_checks(self):
        (self.workspace / "file").write_bytes(b"correct")
        info = pack_directory(self.workspace, self.archive, Deadline(10), maximum=100)
        changed = json.loads(json.dumps(info["manifest"]))
        changed["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(JobError):
            extract_archive(self.archive, self.destination, changed, Deadline(10), maximum=100)
        for variant in (
            {**info["manifest"], "total_bytes": 0},
            {**info["manifest"], "files": info["manifest"]["files"] * 2},
            {**info["manifest"], "files": [{**info["manifest"]["files"][0], "path": "../bad"}]},
        ):
            with self.assertRaises(JobError):
                validate_manifest(variant, 100)


class AgentConfigurationTests(unittest.TestCase):
    def test_verification_snapshot_is_bounded_valid_python_and_only_its_hash_is_public(self):
        code = "print('CALLER_VERIFIER_SOURCE')\n"
        options = AgentOptions(verification_code=code)
        options.validate()
        self.assertEqual(options.public()["verification_sha256"], hashlib.sha256(code.encode()).hexdigest())
        self.assertNotIn(code, repr(options))
        self.assertNotIn("CALLER_VERIFIER_SOURCE", json.dumps(options.public()))
        AgentOptions(verification_code="pass\n" + "#" * (MAX_VERIFICATION_BYTES - 5)).validate()
        for invalid in ("", " \n", "pass\n" + "#" * (MAX_VERIFICATION_BYTES - 4), "print(", "pass\0", 42):
            with self.subTest(code=repr(invalid)[:80]), self.assertRaises(JobError):
                AgentOptions(verification_code=invalid).validate()

    def test_agent_limits_are_bounded_and_exact_boundaries_are_accepted(self):
        AgentOptions(200, 600, 1024 ** 3).validate()
        for options in (
            AgentOptions(201), AgentOptions(0), AgentOptions(command_timeout_seconds=601),
            AgentOptions(command_timeout_seconds=float("nan")), AgentOptions(max_artifact_bytes=0),
            AgentOptions(max_artifact_bytes=1024 ** 3 + 1),
        ):
            with self.subTest(options=options), self.assertRaises(JobError):
                options.validate()

    def test_task_env_names_cannot_override_identity_loader_or_controller_credentials(self):
        self.assertEqual(task_env_name("ALCHEMY_API_KEY"), "ALCHEMY_API_KEY")
        self.assertEqual(task_env_name("ETHERSCAN_API_KEY"), "ETHERSCAN_API_KEY")
        for name in ("VAST_AI_API_KEY", "HF_TOKEN", "HUGGING_FACE_ACCESS_TOKEN", "PATH", "HOME",
                     "LD_PRELOAD", "PYTHONPATH", "LLAMA_API_KEY", "BASH_ENV", "invalid-name"):
            with self.subTest(name=name), self.assertRaises(JobError):
                task_env_name(name)


if __name__ == "__main__":
    unittest.main()
