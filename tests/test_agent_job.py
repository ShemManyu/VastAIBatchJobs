from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from agent_job import ACTION_SCHEMA, Sandbox, bootstrap_tools, parse_action, run_agent, scrub_values
from agent_tools import ToolError, execute
from agent_utils import AgentOptions, MAX_VERIFICATION_BYTES
from job_utils import Deadline, DeadlineExceeded, JobError
from remote_job import ModelRuntime, agent, model_runtime, server_args
from tests.fakes import HF_TOKEN, Response, VAST_TOKEN


TASK_KEY = "ALCHEMY_TEST_KEY_MUST_NOT_LEAK"
MODEL_KEY = "MODEL_SERVER_TEST_KEY_MUST_NOT_LEAK"


def decision(action, finish_reason="stop"):
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(action)},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


class LocalSandbox(Sandbox):
    def __post_init__(self):
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.home = self.root / "home"
        self.helper = self.root / "unused-helper.py"
        self.actions = []
        self.stops = 0

    def execute(self, action, deadline):
        self.actions.append(action)
        environment = {"PATH": os.defpath, **(self.task_env if action["action"] == "run" else {})}
        try:
            with patch.dict(os.environ, environment, clear=True):
                result = execute(
                    self.workspace, action, timeout=min(self.options.command_timeout_seconds, deadline.remaining()),
                )
        except (ToolError, OSError) as error:
            result = {"error": str(error)}
        return scrub_values(result, self.secrets)

    def stop_children(self):
        self.stops += 1


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.options = AgentOptions(max_steps=10, command_timeout_seconds=5)
        self.sandbox = LocalSandbox(self.root, self.options, {"ALCHEMY_API_KEY": TASK_KEY}, (TASK_KEY,))
        self.model_requests = []
        self.responses = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.headers.get("Authorization") != f"Bearer {MODEL_KEY}":
                    self.send_response(401)
                    self.end_headers()
                    self.wfile.write(b'{"error":"unauthorized"}')
                    return
                if self.path == "/apply-template":
                    payload = {"prompt": json.dumps(body["messages"])}
                elif self.path == "/tokenize":
                    payload = {"tokens": [1] * len(body["content"].split())}
                elif self.path == "/v1/chat/completions":
                    owner.model_requests.append(body)
                    payload = owner.responses.pop(0)
                else:
                    raise AssertionError("Unexpected model route")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        import requests

        self.http = requests.Session()
        self.http.trust_env = False
        self.http.headers["Authorization"] = f"Bearer {MODEL_KEY}"
        self.addCleanup(self.http.close)
        self.runtime = ModelRuntime(
            self.http, f"http://127.0.0.1:{self.server.server_port}",
            {"gpu_name": "TEST_GPU", "gpu_vram_mib": 40960},
            16384, 2048, 0, (HF_TOKEN, TASK_KEY, MODEL_KEY),
        )

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def run_actions(self, actions, *, options=None):
        self.responses = [decision(action) for action in actions]
        return run_agent(
            self.root, "Create and execute a script, then report its real result.",
            options or self.options, self.sandbox, self.runtime.complete, Deadline(60),
            self.runtime.gpu, (HF_TOKEN, TASK_KEY, MODEL_KEY),
        )

    def test_http_model_loop_actually_creates_executes_and_reads_a_script(self):
        script = "import json\nfrom pathlib import Path\nvalue = 6 * 7\nPath('value.json').write_text(json.dumps({'value': value}))\nprint(value)\n"
        report = self.run_actions([
            {"action": "write_file", "path": "compute.py", "content": script},
            {"action": "run", "argv": [sys.executable, "compute.py"]},
            {"action": "read_file", "path": "value.json"},
            {"action": "finish", "summary": "The executed script returned 42."},
        ])
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["steps"], 4)
        self.assertEqual(json.loads((self.sandbox.workspace / "value.json").read_text()), {"value": 42})
        self.assertEqual((self.root / "response.txt").read_text(), "The executed script returned 42.")
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        self.assertEqual(transcript["steps"][1]["observation"]["exit_code"], 0)
        self.assertEqual(transcript["steps"][1]["observation"]["stdout"], "42\n")
        observation = json.loads(self.model_requests[2]["messages"][-1]["content"].split("\n", 1)[1])
        self.assertEqual(observation["stdout"], "42\n")
        self.assertEqual(self.model_requests[0]["response_format"], {"type": "json_object", "schema": ACTION_SCHEMA})
        self.assertEqual({row["path"] for row in report["archive"]["manifest"]["files"]}, {"compute.py", "value.json"})
        self.assertNotIn(TASK_KEY, json.dumps(self.model_requests))
        self.assertNotIn(MODEL_KEY, json.dumps(self.model_requests))
        self.assertNotIn(HF_TOKEN, json.dumps(report))

    def test_tool_failure_is_observed_and_a_later_model_action_can_correct_it(self):
        report = self.run_actions([
            {"action": "run", "argv": [sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"]},
            {"action": "write_file", "path": "fixed.py", "content": "print(42)\n"},
            {"action": "run", "argv": [sys.executable, "fixed.py"]},
            {"action": "finish", "summary": "The corrected script returned 42."},
        ])
        self.assertEqual(report["status"], "completed")
        first_observation = json.loads(self.model_requests[1]["messages"][-1]["content"].split("\n", 1)[1])
        self.assertEqual(first_observation["exit_code"], 7)
        self.assertEqual(first_observation["stdout"], "bad\n")
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        self.assertEqual(transcript["steps"][2]["observation"]["stdout"], "42\n")

    def test_finish_verifier_executes_the_snapshot_and_rejects_incorrect_output_until_fixed(self):
        code = (
            "import json\nfrom pathlib import Path\n"
            "if json.loads(Path('value.json').read_text()).get('value') != 42:\n"
            "    raise ValueError('Expected the computed value 42.')\n"
            "print('Verified the actual value 42.')\n"
        )
        report = self.run_actions([
            {"action": "write_file", "path": "value.json", "content": '{"value":7}'},
            {"action": "write_file", "path": "verification.py", "content": "print('fake success')\n"},
            {"action": "finish", "summary": "The incorrect output is ready."},
            {"action": "write_file", "path": "value.json", "content": '{"value":42}'},
            {"action": "finish", "summary": "The verified output is ready."},
        ], options=AgentOptions(verification_code=code))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["steps"], 5)
        self.assertEqual(report["verification"]["exit_code"], 0)
        self.assertEqual(report["verification"]["stdout"], "Verified the actual value 42.\n")
        self.assertEqual((self.root / "response.txt").read_text(), "The verified output is ready.")
        observation = json.loads(self.model_requests[3]["messages"][-1]["content"].split("\n", 1)[1])
        self.assertIn("Finish rejected", observation["error"])
        self.assertNotEqual(observation["verification"]["exit_code"], 0)
        self.assertIn("Expected the computed value 42.", observation["verification"]["stderr"])
        verifier_calls = [action for action in self.sandbox.actions if action["action"] == "run"]
        self.assertEqual(len(verifier_calls), 2)
        self.assertTrue(all(action["argv"][-1] == code for action in verifier_calls))
        self.assertTrue(all("-I" in action["argv"] and "-S" in action["argv"] for action in verifier_calls))
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        self.assertNotIn("finished", transcript["steps"][2]["observation"])
        self.assertEqual(transcript["steps"][-1]["observation"]["verification"], report["verification"])

    def test_permanently_failed_verification_never_publishes_a_success_response(self):
        report = self.run_actions([
            {"action": "write_file", "path": "partial.txt", "content": "incomplete evidence"},
            {"action": "finish", "summary": "Claimed success."},
        ], options=AgentOptions(max_steps=2, verification_code="raise ValueError('missing verified source')"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["steps"], 2)
        self.assertNotEqual(report["verification"]["exit_code"], 0)
        self.assertFalse((self.root / "response.txt").exists())
        self.assertEqual([row["path"] for row in report["archive"]["manifest"]["files"]], ["partial.txt"])

    def test_maximum_size_verifier_really_executes_through_the_argv_limited_tool(self):
        header = "print('maximum verifier executed')\n"
        code = header + "#" * (MAX_VERIFICATION_BYTES - len(header.encode()))
        report = self.run_actions([
            {"action": "finish", "summary": "The maximum-size verifier passed."},
        ], options=AgentOptions(max_steps=1, verification_code=code))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["verification"]["exit_code"], 0)
        self.assertEqual(report["verification"]["stdout"], "maximum verifier executed\n")
        self.assertEqual(len(self.sandbox.actions[0]["argv"][-1].encode()), MAX_VERIFICATION_BYTES)

    def test_finish_requires_a_completed_verifier_not_a_timeout_or_success_shaped_error(self):
        for verification in (
            {"exit_code": 0, "timed_out": True}, {"exit_code": 0},
            {"exit_code": False, "timed_out": False}, {"exit_code": 0.0, "timed_out": False},
            {"exit_code": 0, "timed_out": False, "error": "tool failed"},
        ):
            with self.subTest(verification=verification), patch.object(
                self.sandbox, "execute", return_value=verification,
            ):
                report = self.run_actions([
                    {"action": "finish", "summary": "Claimed success."},
                ], options=AgentOptions(max_steps=1, verification_code="print('verification')"))
                self.assertEqual(report["status"], "failed")
                self.assertFalse((self.root / "response.txt").exists())

    def test_five_consecutive_failures_stop_exactly_and_preserve_recovery_evidence(self):
        report = self.run_actions([
            {"action": "run", "argv": [sys.executable, "-c", "import sys; sys.exit(7)"]}
            for _ in range(5)
        ])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["steps"], 5)
        self.assertEqual(len(self.model_requests), 5)
        self.assertEqual(len(self.sandbox.actions), 5)
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        self.assertIn("recovery_hint", transcript["steps"][2]["observation"])
        self.assertFalse((self.root / "response.txt").exists())

    def test_step_limit_is_exact_and_preserves_partial_scripts_and_execution_evidence(self):
        report = self.run_actions([
            {"action": "write_file", "path": "partial.py", "content": "print(7)\n"},
            {"action": "run", "argv": [sys.executable, "partial.py"]},
        ], options=AgentOptions(max_steps=2))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["steps"], 2)
        self.assertEqual(len(self.model_requests), 2)
        self.assertFalse((self.root / "response.txt").exists())
        self.assertEqual([row["path"] for row in report["archive"]["manifest"]["files"]], ["partial.py"])
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        self.assertEqual(transcript["steps"][1]["observation"]["stdout"], "7\n")

    def test_token_limited_json_decision_never_executes_even_when_its_json_is_valid(self):
        self.responses = [decision({"action": "run", "argv": [sys.executable, "-c", "print(42)"]}, "length")]
        report = run_agent(
            self.root, "Do not execute incomplete decisions.", self.options, self.sandbox,
            self.runtime.complete, Deadline(60), self.runtime.gpu, (TASK_KEY,),
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["steps"], 1)
        self.assertEqual(self.sandbox.actions, [])
        self.assertFalse((self.root / "response.txt").exists())

    def test_finish_rejects_credential_files_then_allows_a_safe_correction(self):
        report = self.run_actions([
            {"action": "run", "argv": [sys.executable, "-c",
             "import os; from pathlib import Path; Path('unsafe.txt').write_text(os.environ['ALCHEMY_API_KEY'])"]},
            {"action": "finish", "summary": "Finished."},
            {"action": "write_file", "path": "unsafe.txt", "content": "safe output"},
            {"action": "finish", "summary": "The credential-free output is ready."},
        ])
        self.assertEqual(report["status"], "completed")
        transcript = (self.root / "agent-transcript.json").read_text()
        self.assertNotIn(TASK_KEY, transcript)
        self.assertIn("Finish rejected", transcript)
        self.assertEqual((self.sandbox.workspace / "unsafe.txt").read_text(), "safe output")
        self.assertEqual(report["steps"], 4)

    def test_invalid_action_fields_do_not_reach_any_tool(self):
        report = self.run_actions([{"action": "run", "argv": ["echo", "bad"], "unexpected": True}])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(self.sandbox.actions, [])
        self.assertEqual(report["steps"], 1)

    def test_model_deadline_failure_preserves_the_failed_report(self):
        def expired(messages, schema, deadline):
            raise DeadlineExceeded("controlled model timeout")

        report = run_agent(
            self.root, "task", self.options, self.sandbox, expired,
            Deadline(60), self.runtime.gpu, (TASK_KEY,),
        )
        self.assertEqual(report["status"], "failed")
        self.assertIn("controlled model timeout", report["error"])
        self.assertTrue((self.root / "agent-report.json").is_file())
        self.assertGreater(self.sandbox.stops, 0)


class RuntimeAndIsolationTests(unittest.TestCase):
    def test_remote_agent_uses_the_snapshot_and_rejects_a_missing_or_changed_verifier(self):
        code = "print('REMOTE_VERIFICATION_SNAPSHOT')\n"
        options = AgentOptions(verification_code=code)
        request = {
            "agent": {**options.public(), "verification_code": code},
            "task_env": {}, "prompt": "task", "hf_token": HF_TOKEN,
        }
        runtime = SimpleNamespace(complete=MagicMock(), gpu={}, secrets=(HF_TOKEN,))

        @contextmanager
        def local_runtime(*args, **kwargs):
            yield runtime

        with tempfile.TemporaryDirectory() as directory, \
             patch("remote_job.model_runtime", side_effect=local_runtime) as model, \
             patch("agent_job.Sandbox") as sandbox, \
             patch("agent_job.run_agent", return_value={"status": "completed", "steps": 1}) as loop:
            agent(Path(directory), request, Deadline(10))
            self.assertEqual(loop.call_args.args[2].verification_code, code)
            self.assertEqual(sandbox.call_args.args[1].verification_code, code)
            for changed in (None, "print('changed')\n"):
                with self.subTest(changed=changed), self.assertRaises(JobError):
                    agent(Path(directory), {
                        **request, "agent": {**request["agent"], "verification_code": changed},
                    }, Deadline(10))
            self.assertEqual(model.call_count, 1)

    def test_token_budget_checks_the_measured_boundary_before_generation(self):
        for limit, allowed in ((5, True), (4, False)):
            with self.subTest(limit=limit):
                http = MagicMock()
                http.post.side_effect = [
                    Response({"prompt": "rendered"}), Response({"tokens": [1, 2, 3]}),
                    Response(decision({"action": "finish", "summary": "done"})),
                ]
                runtime = ModelRuntime(http, "http://127.0.0.1", {}, limit, 2, 0, ())
                if allowed:
                    runtime.complete([{"role": "user", "content": "task"}], ACTION_SCHEMA, Deadline(10))
                    self.assertEqual(http.post.call_count, 3)
                else:
                    with self.assertRaises(JobError):
                        runtime.complete([{"role": "user", "content": "task"}], ACTION_SCHEMA, Deadline(10))
                    self.assertEqual(http.post.call_count, 2)
                self.assertFalse(http.post.call_args.kwargs["allow_redirects"])

    def test_protected_server_uses_a_private_key_file_not_a_key_argument(self):
        args = server_args(Path("/server"), Path("/model"), 16384, 1234, Path("/private/api-key"))
        self.assertEqual(args[args.index("--api-key-file") + 1], "/private/api-key")
        self.assertNotIn("--api-key", args)
        self.assertIn("--no-agent", args)
        self.assertIn("--no-webui", args)
        self.assertEqual(args[args.index("--verbosity") + 1], "4")

    def test_private_server_key_is_removed_even_when_process_startup_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "llama.cpp/build/bin/llama-server"
            binary.parent.mkdir(parents=True)
            binary.write_text("FIXTURE")
            request = {
                "hf_token": HF_TOKEN, "prompt": "task", "gpu_name": "A100",
                "model": {}, "context_size": 16384, "max_tokens": 2048, "temperature": 0,
            }
            with patch("remote_job.gpu_details", return_value={"gpu_name": "A100"}), \
                 patch("remote_job.download_model", return_value=root / "model.gguf"), \
                 patch("remote_job.subprocess.Popen", side_effect=OSError("controlled startup failure")), \
                 patch("requests.Session") as session:
                with self.assertRaises(OSError):
                    with model_runtime(root, request, Deadline(10), agent=True):
                        self.fail("Server should not start")
                session.return_value.close.assert_called_once()
            self.assertFalse((root / "llama-api-key").exists())

    def test_node_checksum_failure_never_installs_or_executes_the_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stream = MagicMock()
            stream.__enter__.return_value = stream
            stream.read.side_effect = [b"incorrect binary archive", b""]
            with patch("agent_job.platform.system", return_value="Linux"), \
                 patch("agent_job.platform.machine", return_value="x86_64"), \
                 patch("agent_job.os.geteuid", return_value=0), \
                 patch("agent_job.shutil.which", return_value="/fixture/tool"), \
                 patch("agent_job.NODE_DIRECTORY", root / "not-installed"), \
                 patch("agent_job.urllib.request.urlopen", return_value=stream), \
                 patch("agent_job.run_command") as command:
                with self.assertRaises(JobError):
                    bootstrap_tools(root, Deadline(10))
                command.assert_not_called()
                self.assertFalse((root / "not-installed").exists())

    def sandbox_call(self, result, action=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "worker"
            root.mkdir(mode=0o700)
            sandbox = Sandbox(root, AgentOptions(), {"ALCHEMY_API_KEY": TASK_KEY}, (TASK_KEY, HF_TOKEN))
            sandbox.uid = 2001
            process = MagicMock()
            process.__enter__.return_value = process
            process.communicate.return_value = (json.dumps(result).encode(), b"")
            process.returncode = 0
            with patch("agent_job.pwd.getpwuid", return_value=SimpleNamespace(pw_gid=2002)), \
                 patch("agent_job.subprocess.Popen", return_value=process) as spawn, \
                 patch.object(sandbox, "stop_children") as stop, \
                 patch.dict(os.environ, {"VAST_AI_API_KEY": VAST_TOKEN, "HF_TOKEN": HF_TOKEN}):
                output = sandbox.execute(action or {"action": "run", "argv": ["node", "task.cjs"]}, Deadline(10))
            return output, spawn.call_args, process.communicate.call_args, stop.call_count

    def test_tools_drop_privileges_capabilities_groups_and_protected_environment(self):
        result, spawn, communicate, stops = self.sandbox_call({
            "uid": 2001, "sandbox_verified": True, "exit_code": 0, "stdout": TASK_KEY, "stderr": "",
        })
        args = spawn.args[0]
        for option in ("--reuid=2001", "--regid=2002", "--clear-groups", "--no-new-privs",
                       "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", "-I", "-S"):
            self.assertIn(option, args)
        self.assertEqual(spawn.kwargs["env"]["ALCHEMY_API_KEY"], TASK_KEY)
        self.assertNotIn("HF_TOKEN", spawn.kwargs["env"])
        self.assertNotIn("VAST_AI_API_KEY", spawn.kwargs["env"])
        self.assertNotIn(TASK_KEY, " ".join(args))
        self.assertNotIn(TASK_KEY, communicate.kwargs["input"].decode())
        self.assertEqual(spawn.kwargs["umask"], 0o077)
        self.assertEqual(result["stdout"], "[REDACTED]")
        self.assertEqual(stops, 1)

    def test_file_tools_do_not_receive_even_the_approved_task_credentials(self):
        _, spawn, _, _ = self.sandbox_call(
            {"uid": 2001, "sandbox_verified": True, "entries": []},
            {"action": "list_files", "path": "."},
        )
        self.assertNotIn("ALCHEMY_API_KEY", spawn.kwargs["env"])

    def test_missing_or_root_isolation_proof_is_an_error_not_a_fallback(self):
        for result in (
            {"uid": 0, "sandbox_verified": True}, {"uid": 2001, "sandbox_verified": False},
            {"uid": 2001},
        ):
            with self.subTest(result=result), self.assertRaises(JobError):
                self.sandbox_call(result)

    def test_incomplete_or_unknown_actions_are_rejected_before_execution(self):
        for action in (
            {"action": "finish", "summary": ""}, {"action": "unknown"},
            {"action": "write_file", "path": "file"}, {"action": "run", "argv": [], "extra": True},
        ):
            with self.subTest(action=action), self.assertRaises(JobError):
                parse_action(decision(action))


if __name__ == "__main__":
    unittest.main()
