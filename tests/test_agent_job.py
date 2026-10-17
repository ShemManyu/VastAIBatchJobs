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

from agent_job import (
    ACTION_SCHEMA, HISTORY_MARKER, Sandbox, agent_uid, bootstrap_tools, compact_history,
    history_observation, parse_action, run_agent, scrub_values,
)
from agent_tools import ToolError, execute
from agent_utils import AgentOptions, MAX_VERIFICATION_BYTES, TOOL_OUTPUT_BYTES
from job_utils import ContextLimitError, Deadline, DeadlineExceeded, JobError
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

    def test_measured_context_overflow_compacts_then_recalls_the_original_observation(self):
        self.runtime.context_size = 4800
        payload = "detail " * 1700
        report = self.run_actions([
            {"action": "write_file", "path": "source.txt", "content": payload},
            {"action": "read_file", "path": "source.txt"},
            {"action": "read_history", "step": 2},
            {"action": "finish", "summary": "The original source observation was recalled."},
        ])
        self.assertEqual(report["status"], "completed", report.get("error"))
        self.assertEqual(report["steps"], 4)
        self.assertGreater(report["context_compactions"], 0)
        self.assertEqual([action["action"] for action in self.sandbox.actions], ["write_file", "read_file"])
        self.assertEqual((self.sandbox.workspace / "source.txt").read_text(), payload)
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        original = transcript["steps"][1]
        recalled = json.loads(transcript["steps"][2]["observation"]["content"])
        self.assertEqual(recalled, original)
        self.assertEqual(original["observation"]["content"], payload[:TOOL_OUTPUT_BYTES])
        self.assertEqual(len(transcript["context_compactions"]), report["context_compactions"])
        initial = self.model_requests[0]["messages"]
        for request in self.model_requests:
            messages = request["messages"]
            self.assertEqual(messages[0], initial[0])
            self.assertEqual(messages[1]["content"].split(HISTORY_MARKER)[0], initial[1]["content"])
            self.assertLessEqual(
                len(json.dumps(messages).split()) + self.runtime.max_tokens,
                self.runtime.context_size,
            )
        self.assertIn(HISTORY_MARKER, self.model_requests[-1]["messages"][1]["content"])

    def test_context_compaction_preserves_failed_verification_and_requires_a_real_fix(self):
        self.runtime.context_size = 4800
        payload = "detail " * 1700
        code = (
            "import json\nfrom pathlib import Path\n"
            "if json.loads(Path('value.json').read_text())['value'] != 42:\n"
            "    raise ValueError('actual value must be 42')\n"
            "print('actual corrected value verified')\n"
        )
        report = self.run_actions([
            {"action": "write_file", "path": "value.json", "content": '{"value":7}'},
            {"action": "finish", "summary": "Incorrect output."},
            {"action": "write_file", "path": "source.txt", "content": payload},
            {"action": "read_file", "path": "source.txt"},
            {"action": "write_file", "path": "value.json", "content": '{"value":42}'},
            {"action": "finish", "summary": "The corrected output passed the actual verifier."},
        ], options=AgentOptions(verification_code=code))
        self.assertEqual(report["status"], "completed", report.get("error"))
        self.assertGreater(report["context_compactions"], 0)
        self.assertEqual(report["verification"]["stdout"], "actual corrected value verified\n")
        compacted = self.model_requests[4]["messages"]
        ledger = json.loads(compacted[1]["content"].split(HISTORY_MARKER)[1].split("\nDetails ")[0])
        rejected = next(row for row in ledger if row["step"] == 2)
        self.assertIn("Finish rejected", rejected["error"])
        self.assertNotEqual(rejected["verification"]["exit_code"], 0)
        self.assertIn("actual value must be 42", rejected["verification"]["stderr"])
        latest = json.loads(compacted[-1]["content"].split("\n", 1)[1])
        self.assertEqual(latest["content"], payload[:TOOL_OUTPUT_BYTES])
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        self.assertNotEqual(transcript["steps"][1]["observation"]["verification"]["exit_code"], 0)
        self.assertEqual(json.loads((self.sandbox.workspace / "value.json").read_text()), {"value": 42})

    def test_unfit_original_task_is_not_discarded_or_sent_to_generation(self):
        self.runtime.context_size = 10
        self.runtime.max_tokens = 2
        report = self.run_actions([{"action": "finish", "summary": "must not run"}])
        self.assertEqual(report["status"], "failed")
        self.assertIn("ContextLimitError", report["error"])
        self.assertEqual(report["context_compactions"], 0)
        self.assertEqual(self.model_requests, [])
        self.assertEqual(self.sandbox.actions, [])
        self.assertFalse((self.root / "response.txt").exists())

    def test_reasoning_only_completion_is_preserved_redacted_and_never_executed(self):
        completion = {
            "choices": [{"message": {"role": "assistant", "content": "",
                                     "reasoning_content": "still thinking " + TASK_KEY},
                         "finish_reason": "length"}],
            "usage": {"completion_tokens": 2048},
        }
        self.responses = [completion]
        report = run_agent(
            self.root, "Do the task.", self.options, self.sandbox,
            self.runtime.complete, Deadline(60), self.runtime.gpu, (TASK_KEY,),
        )
        self.assertEqual(report["status"], "failed")
        self.assertIn("no non-empty assistant text", report["error"])
        self.assertEqual(self.sandbox.actions, [])
        raw = (self.root / "agent-transcript.json").read_text()
        self.assertNotIn(TASK_KEY, raw)
        saved = json.loads(raw)["steps"][0]
        self.assertEqual(saved["completion"]["choices"][0]["finish_reason"], "length")
        self.assertEqual(saved["completion"]["usage"]["completion_tokens"], 2048)
        self.assertIn("still thinking", saved["completion"]["choices"][0]["message"]["reasoning_content"])
        self.assertNotIn("action", saved)
        self.assertFalse((self.root / "response.txt").exists())

    def test_unfit_latest_observation_fails_without_dropping_it_or_reexecuting_tools(self):
        self.runtime.context_size = 2000
        self.runtime.max_tokens = 128
        (self.sandbox.workspace / "large.txt").write_text("detail " * 5000)
        report = self.run_actions([
            {"action": "run", "argv": [sys.executable, "-c", "print('executed once')"]},
            {"action": "read_file", "path": "large.txt"},
            {"action": "finish", "summary": "must not run"},
        ])
        self.assertEqual(report["status"], "failed")
        self.assertIn("ContextLimitError", report["error"])
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["context_compactions"], 1)
        self.assertEqual(len(self.model_requests), 2)
        self.assertEqual(len(self.sandbox.actions), 2)
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        self.assertEqual(transcript["steps"][0]["observation"]["stdout"], "executed once\n")
        self.assertEqual(len(transcript["steps"][1]["observation"]["content"].encode()), TOOL_OUTPUT_BYTES)
        self.assertFalse((self.root / "response.txt").exists())

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
        self.assertTrue(all(action["argv"][4] == code for action in verifier_calls))
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
        self.assertEqual(len(self.sandbox.actions[0]["argv"][4].encode()), MAX_VERIFICATION_BYTES)

    def test_verifier_gets_real_current_commands_not_workspace_success_claims(self):
        code = (
            "import json,sys\nfrom pathlib import Path\n"
            "commands=json.loads(Path(sys.argv[1]).read_text())\n"
            "assert commands and commands[-1]['observation']['exit_code']==0, 'No actual successful command'\n"
            "assert commands[-1]['observation']['stdout']=='42\\n', 'No actual computed output'\n"
            "print(json.dumps(commands))\n"
        )
        report = self.run_actions([
            {"action": "write_file", "path": "verification-context.json", "content": '[{"claimed_success":true}]'},
            {"action": "run", "argv": [sys.executable, "-c", "raise SystemExit(7)"]},
            {"action": "finish", "summary": "Prior JSON says it worked."},
            {"action": "run", "argv": [sys.executable, "-c", "print(6*7)"]},
            {"action": "finish", "summary": "The actual current command computed 42."},
        ], options=AgentOptions(verification_code=code))
        self.assertEqual(report["status"], "completed", report.get("error"))
        self.assertEqual(report["steps"], 5)
        commands = json.loads(report["verification"]["stdout"])
        self.assertEqual([row["step"] for row in commands], [2, 4])
        self.assertEqual([row["observation"]["exit_code"] for row in commands], [7, 0])
        self.assertEqual(commands[-1]["action"]["argv"], [sys.executable, "-c", "print(6*7)"])
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        self.assertNotEqual(transcript["steps"][2]["observation"]["verification"]["exit_code"], 0)
        self.assertEqual(
            json.loads((self.sandbox.workspace / "verification-context.json").read_text()),
            [{"claimed_success": True}],
        )
        self.assertFalse((self.sandbox.helper.parent / "verification-context.json").exists())

    def test_command_context_is_redacted_read_only_and_removed_after_verification(self):
        code = (
            "import json,sys\nfrom pathlib import Path\n"
            "path=Path(sys.argv[1])\n"
            "assert path.stat().st_mode & 0o222 == 0, 'Context must not be writable'\n"
            f"assert {TASK_KEY!r} not in path.read_text(), 'Context must redact command arguments too'\n"
            "commands=json.loads(path.read_text())\n"
            "assert commands[0]['observation']['stdout']=='[REDACTED]\\n'\n"
            "print('private command evidence verified')\n"
        )
        report = self.run_actions([
            {"action": "run", "argv": [
                sys.executable, "-c", f"print({TASK_KEY!r})",
            ]},
            {"action": "finish", "summary": "Actual output was checked without exposing the task key."},
        ], options=AgentOptions(verification_code=code))
        self.assertEqual(report["status"], "completed", report.get("error"))
        self.assertEqual(report["verification"]["stdout"], "private command evidence verified\n")
        self.assertNotIn(TASK_KEY, (self.root / "agent-transcript.json").read_text())
        self.assertFalse((self.sandbox.helper.parent / "verification-context.json").exists())

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

    def test_identical_finish_failures_stop_after_five_despite_successful_intervening_edits(self):
        actions = []
        for version in range(6):
            actions.extend([
                {"action": "write_file", "path": "report.md", "content": f"Incorrect revision {version}"},
                {"action": "finish", "summary": "Claimed correction."},
            ])
        report = self.run_actions(actions, options=AgentOptions(
            max_steps=12, verification_code="raise SystemExit('Named modifiers must be none.')",
        ))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["steps"], 10)
        self.assertEqual(len(self.model_requests), 10)
        self.assertIn("five identical finish rejections", report["error"])
        self.assertFalse((self.root / "response.txt").exists())
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        rejected = transcript["steps"][1::2]
        self.assertEqual(
            [step["observation"]["repeated_finish_rejections"] for step in rejected], [1, 2, 3, 4, 5],
        )
        self.assertIn("recovery_hint", rejected[2]["observation"])
        self.assertTrue(all(step["observation"]["verification"]["exit_code"] == 1 for step in rejected))
        self.assertEqual((self.sandbox.workspace / "report.md").read_text(), "Incorrect revision 4")
        self.assertEqual([row["path"] for row in report["archive"]["manifest"]["files"]], ["report.md"])

    def test_changed_verifier_failure_resets_repeat_count_and_allows_real_correction(self):
        code = (
            "from pathlib import Path\nvalue = Path('value.txt').read_text()\n"
            "if value != '42': raise SystemExit('Incorrect value: ' + value)\n"
            "print('Verified 42.')\n"
        )
        actions = []
        for value in ["first"] * 4 + ["second", "42"]:
            actions.extend([
                {"action": "write_file", "path": "value.txt", "content": value},
                {"action": "finish", "summary": "The corrected value was verified."},
            ])
        report = self.run_actions(actions, options=AgentOptions(max_steps=12, verification_code=code))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["steps"], 12)
        self.assertEqual(report["verification"]["stdout"], "Verified 42.\n")
        transcript = json.loads((self.root / "agent-transcript.json").read_text())
        self.assertEqual(
            [step["observation"]["repeated_finish_rejections"] for step in transcript["steps"][1:10:2]],
            [1, 2, 3, 4, 1],
        )
        self.assertTrue(transcript["steps"][-1]["observation"]["finished"])

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
    def test_agent_uid_uses_an_unused_high_mapped_uid_instead_of_shared_system_identity(self):
        with patch("agent_job.Path.read_text", return_value="0 0 4294967295\n"), \
             patch("agent_job.uuid.uuid4", side_effect=[SimpleNamespace(int=7), SimpleNamespace(int=8)]), \
             patch("agent_job.pwd.getpwuid", side_effect=[SimpleNamespace(pw_uid=1000007), KeyError]) as lookup:
            self.assertEqual(agent_uid(), 1000008)
        self.assertEqual([call.args[0] for call in lookup.call_args_list], [1000007, 1000008])

    def test_agent_uid_stays_inside_restricted_mapping_and_rejects_root_only_namespace(self):
        with patch("agent_job.Path.read_text", return_value="0 100000 65536\n"), \
             patch("agent_job.uuid.uuid4", return_value=SimpleNamespace(int=7)), \
             patch("agent_job.pwd.getpwuid", side_effect=KeyError):
            self.assertEqual(agent_uid(), 10007)
        with patch("agent_job.Path.read_text", return_value="0 100000 1\n"), self.assertRaises(JobError):
            agent_uid()

    def test_prepare_requests_exact_uid_without_large_login_logs_and_checks_native_async_runtime(self):
        for exit_code in (0, -6):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "worker"
                root.mkdir()
                (root / "agent_tools.py").write_text("# fixture helper\n")
                sandbox = Sandbox(root, AgentOptions(), {}, ())
                native = {
                    "exit_code": exit_code, "timed_out": False,
                    "stdout": "agent-runtime-ok\n" if exit_code == 0 else "",
                }
                with patch("agent_job.platform.system", return_value="Linux"), \
                     patch("agent_job.os.geteuid", return_value=0), \
                     patch("agent_job.agent_uid", return_value=1000007), \
                     patch("agent_job.run_command") as command, \
                     patch("agent_job.pwd.getpwnam", return_value=SimpleNamespace(pw_uid=1000007, pw_gid=2002)), \
                     patch("agent_job.os.chown"), \
                     patch.object(sandbox, "execute", side_effect=[{"entries": []}, native]) as execute:
                    if exit_code == 0:
                        sandbox.prepare(Deadline(10))
                        self.assertEqual(sandbox.uid, 1000007)
                    else:
                        with self.assertRaisesRegex(JobError, "Node async runtime probe failed"):
                            sandbox.prepare(Deadline(10))
                argv = command.call_args.args[0]
                self.assertEqual(argv[argv.index("--uid") + 1], "1000007")
                self.assertIn("--no-log-init", argv)
                self.assertEqual(execute.call_args_list[-1].args[0]["argv"][:2], ["node", "-e"])

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
                    with self.assertRaises(ContextLimitError) as raised:
                        runtime.complete([{"role": "user", "content": "task"}], ACTION_SCHEMA, Deadline(10))
                    self.assertEqual(raised.exception.input_tokens, 3)
                    self.assertEqual(raised.exception.output_tokens, 2)
                    self.assertEqual(raised.exception.context_size, 4)
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

    def test_native_worker_pool_is_bounded_despite_larger_parent_environment(self):
        with patch.dict(os.environ, {"TOKIO_WORKER_THREADS": "128"}):
            for action in (
                {"action": "run", "argv": ["node", "task.cjs"]},
                {"action": "list_files", "path": "."},
            ):
                with self.subTest(action=action):
                    _, spawn, _, _ = self.sandbox_call(
                        {"uid": 2001, "sandbox_verified": True, "exit_code": 0},
                        action,
                    )
                    self.assertEqual(spawn.kwargs["env"].get("TOKIO_WORKER_THREADS"), "2")

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


class HistoryTests(unittest.TestCase):
    def record(self, action=None, observation=None):
        action = action or {"action": "run", "argv": ["python3", "script.py"]}
        return {
            "step": 1, "action": action, "completion": decision(action),
            "observation": observation or {
                "exit_code": 7, "timed_out": False, "stdout": "original output",
            },
        }

    def test_history_pages_are_exact_bounded_utf8_and_redacted_before_offsets(self):
        record = self.record(observation={
            "exit_code": 0, "timed_out": False, "stdout": "\u00e9" * 20000 + TASK_KEY,
        })
        pages = []
        offset = 0
        while True:
            result = history_observation([record], {"step": 1, "offset": offset}, (TASK_KEY,))
            self.assertNotIn("error", result)
            self.assertLessEqual(result["bytes_read"], TOOL_OUTPUT_BYTES)
            self.assertEqual(result["bytes_read"], len(result["content"].encode()))
            self.assertEqual(result["offset"], offset)
            pages.append(result["content"])
            if not result["truncated"]:
                break
            self.assertGreater(result["next_offset"], offset)
            offset = result["next_offset"]
        data = "".join(pages)
        self.assertGreater(len(pages), 1)
        self.assertEqual(json.loads(data), scrub_values(record, (TASK_KEY,)))
        self.assertNotIn(TASK_KEY, data)
        self.assertEqual(result["total_bytes"], len(data.encode()))
        self.assertIsNone(result["next_offset"])

    def test_invalid_future_and_out_of_range_history_reads_are_observed_errors(self):
        for action in (
            {"step": True}, {"step": 0}, {"step": -1}, {"step": 2}, {"step": "1"},
            {"step": 1, "offset": True}, {"step": 1, "offset": -1},
            {"step": 1, "offset": 1.5}, {"step": 1, "offset": 100000},
        ):
            with self.subTest(action=action):
                self.assertIn("error", history_observation([self.record()], action, ()))

    def test_ledger_redacts_before_preview_truncation_and_never_mutates_original_steps(self):
        action = {"action": "run", "argv": ["x" * 280 + TASK_KEY]}
        record = self.record(action)
        recent = self.record({"action": "read_file", "path": "source.sol"})
        initial = [{"role": "system", "content": "original system"},
                   {"role": "user", "content": "original task"}]
        messages = compact_history(initial, [record, recent], 1, (TASK_KEY,))
        self.assertEqual(messages[0], initial[0])
        self.assertEqual(messages[1]["content"].split(HISTORY_MARKER)[0], "original task")
        self.assertEqual(initial[1]["content"], "original task")
        self.assertNotIn("ALCHEMY_TEST", messages[1]["content"])
        self.assertIn("[REDACTED]", messages[1]["content"])
        self.assertEqual(record["action"]["argv"], ["x" * 280 + TASK_KEY])
        self.assertEqual(messages[-2]["content"], decision(recent["action"])["choices"][0]["message"]["content"])
        self.assertEqual(json.loads(messages[-1]["content"].split("\n", 1)[1]), recent["observation"])

    def test_history_action_is_in_the_schema_and_rejects_extra_fields(self):
        action = {"action": "read_history", "step": 1, "offset": 0}
        self.assertEqual(parse_action(decision(action)), action)
        self.assertTrue(any(
            row["properties"]["action"]["const"] == "read_history" for row in ACTION_SCHEMA["oneOf"]
        ))
        with self.assertRaises(JobError):
            parse_action(decision({**action, "path": "../other-run"}))


if __name__ == "__main__":
    unittest.main()
