from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_utils import AgentOptions
from job_utils import Deadline, JobError, LLAMA_COMMIT
from run_job import Credentials, Endpoint, RunState, Settings, SSHTransport, main, run_workflow
from tests.fakes import ANSWER, Clock, FakeAgentTransport, FakeClient, HF_TOKEN, MODEL, VAST_TOKEN
from vast_client import Offer


TASK_KEY = "ALCHEMY_TEST_KEY_MUST_NOT_LEAK"


class AgentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.settings = Settings(
            "unit/model", "model-Q4_K_M.gguf", "# Execute scripts\nA literal task.",
            Credentials(VAST_TOKEN, HF_TOKEN, {"ALCHEMY_API_KEY": TASK_KEY}),
            self.root / "output", self.root / ".vastai-runs",
            context_size=16384, max_tokens=2048, temperature=0, agent=AgentOptions(),
        )
        self.transports = []

    def key_command(self, args, deadline, **kwargs):
        if args[0] != "ssh-keygen":
            raise AssertionError("Unexpected local executable")
        key = Path(args[args.index("-f") + 1])
        key.write_text("PRIVATE_TEST_FIXTURE")
        key.with_suffix(".pub").write_text("ssh-ed25519 PUBLIC_TEST_FIXTURE")
        return b""

    def transport(self, state, endpoint, client, fail=""):
        transport = FakeAgentTransport(state, endpoint, client, fail=fail)
        self.transports.append(transport)
        return transport

    def run_workflow(self, client=None, fail="", settings=None):
        client = client or FakeClient()
        with patch("run_job.run_command", side_effect=self.key_command), \
             patch("run_job.shutil.which", return_value="/fixture/tool"), \
             patch("run_job.Deadline", side_effect=lambda seconds: Clock().deadline(seconds)), \
             redirect_stderr(io.StringIO()):
            return run_workflow(
                settings or self.settings, client, model_lookup=lambda settings, deadline: dict(MODEL),
                transport_factory=lambda state, endpoint: self.transport(state, endpoint, client, fail),
            )

    def latest(self):
        files = list(self.settings.state_root.glob("*/state.json"))
        path = max(files, key=lambda file: file.stat().st_mtime_ns)
        data = json.loads(path.read_text())
        return data, Path(data["output_dir"])

    def test_agent_retrieves_scripts_outputs_transcript_and_summary_before_teardown(self):
        client = FakeClient()
        state = self.run_workflow(client)
        self.assertEqual(state.data["mode"], "agent")
        self.assertEqual(state.data["job_status"], "completed")
        self.assertEqual(state.data["cleanup_status"], "confirmed")
        self.assertEqual((state.output / "response.txt").read_text(), ANSWER)
        self.assertEqual((state.output / "artifacts/scripts/task.py").read_text(), "print(42)\n")
        self.assertEqual(json.loads((state.output / "artifacts/result.json").read_text()), {"value": 42})
        self.assertEqual(state.data["agent"]["steps"], 3)
        self.assertLess(client.events.index("fetch:artifacts.tar"), client.events.index("destroy:101"))
        self.assertEqual(client.created, 1)
        self.assertEqual(client.destroyed, 1)
        self.assertFalse(client.exists)
        self.assertFalse((state.directory / "id_ed25519").exists())
        self.assertEqual(list(state.output.glob("*.part")), [])
        self.assertEqual((state.output / "artifacts").stat().st_mode & 0o777, 0o700)
        for path in state.output.rglob("*"):
            if path.is_file():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_selected_task_keys_go_only_in_agent_stdin_not_metadata_or_bootstrap(self):
        state = self.run_workflow()
        bootstrap, agent = self.transports[-1].requests
        self.assertEqual(bootstrap[0], "bootstrap")
        self.assertEqual(bootstrap[1], {"gpu_name": "A100 SXM4", "gpu_count": 1, "agent": True})
        self.assertEqual(agent[0], "agent")
        self.assertEqual(agent[1]["task_env"], {"ALCHEMY_API_KEY": TASK_KEY})
        self.assertEqual(agent[1]["hf_token"], HF_TOKEN)
        self.assertNotIn(VAST_TOKEN, json.dumps(agent))
        for path in state.output.rglob("*"):
            if path.is_file():
                content = path.read_bytes()
                for secret in (TASK_KEY, HF_TOKEN, VAST_TOKEN):
                    self.assertNotIn(secret.encode(), content)
        self.assertNotIn(TASK_KEY, (state.directory / "state.json").read_text())

    def test_input_directory_is_snapshotted_and_integrity_checked_before_use(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        (inputs / "data.csv").write_text("name,value\nsample,42\n")
        settings = replace(self.settings, agent=AgentOptions(input_dir=inputs))
        state = self.run_workflow(settings=settings)
        self.assertEqual((state.output / "artifacts/data.csv").read_text(), "name,value\nsample,42\n")
        self.assertEqual(self.transports[-1].input_archive, state.directory / "inputs.tar")
        self.assertEqual(state.data["inputs"]["manifest"]["files"][0]["path"], "data.csv")

    def test_bandwidth_quote_changes_are_rejected_before_rental(self):
        client = FakeClient()
        client.offer = replace(client.offer, download_cost_per_gb=0.01, upload_cost_per_gb=0.01)
        original = client.offers

        def changed(template, disk_gb, max_price, rental_seconds, deadline, *, offer_id=None, gpu_count=1):
            rows = original(template, disk_gb, max_price, rental_seconds, deadline,
                            offer_id=offer_id, gpu_count=gpu_count)
            return [replace(row, download_cost_per_gb=0.010001) for row in rows] if offer_id else rows

        with patch.object(client, "offers", side_effect=changed), self.assertRaises(JobError):
            self.run_workflow(client, settings=replace(self.settings, max_bandwidth_price=0.01))
        self.assertEqual(client.created, 0)
        self.assertEqual(client.destroyed, 0)
        self.assertIn("revalidate", client.events)

    def test_bandwidth_boundary_is_inclusive_and_applies_to_both_directions(self):
        from run_job import select_offer

        settings = replace(self.settings, max_bandwidth_price=0.01)
        accepted = Offer(1, "A100", 40960, 0.8, 1, 0.01, 0.01)
        for rejected in (
            replace(accepted, id=2, hourly_price=0.1, download_cost_per_gb=0.010001),
            replace(accepted, id=3, hourly_price=0.1, upload_cost_per_gb=0.010001),
        ):
            self.assertEqual(select_offer(settings, [rejected, accepted]).id, accepted.id)
            with self.assertRaises(JobError):
                select_offer(settings, [rejected])

    def test_secret_input_files_and_protected_variable_aliases_fail_without_rental(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        (inputs / "innocent.txt").write_text(TASK_KEY)
        client = FakeClient()
        with self.assertRaises(JobError):
            self.run_workflow(client, settings=replace(self.settings, agent=AgentOptions(input_dir=inputs)))
        self.assertEqual(client.created, 0)
        self.assertEqual(client.destroyed, 0)
        for value in (VAST_TOKEN, HF_TOKEN):
            settings = replace(self.settings, credentials=Credentials(VAST_TOKEN, HF_TOKEN, {"TASK_KEY": value}))
            with self.assertRaises(JobError):
                self.run_workflow(client, settings=settings)
        self.assertEqual(client.created, 0)

    def test_verifier_cannot_embed_any_controller_or_task_credential_before_rental(self):
        client = FakeClient()
        for secret in (VAST_TOKEN, HF_TOKEN, TASK_KEY):
            options = AgentOptions(verification_code=f"print({secret!r})\n")
            with self.subTest(secret=secret), self.assertRaises(JobError):
                self.run_workflow(client, settings=replace(self.settings, agent=options))
        self.assertEqual(client.created, 0)

    def test_completed_agent_cannot_skip_the_requested_finish_verification(self):
        options = AgentOptions(verification_code="print('verified')\n")
        original = FakeAgentTransport.invoke

        def omitted(transport, stage, request, deadline):
            if stage == "agent":
                request = {**request, "agent": {**request["agent"], "verification_code": None}}
            return original(transport, stage, request, deadline)

        client = FakeClient()
        with patch.object(FakeAgentTransport, "invoke", omitted), self.assertRaises(JobError):
            self.run_workflow(client, settings=replace(self.settings, agent=options))
        metadata, output = self.latest()
        self.assertEqual(metadata["job_status"], "failed")
        self.assertEqual(metadata["cleanup_status"], "confirmed")
        self.assertFalse((output / "response.txt").exists())
        self.assertTrue((output / "artifacts/scripts/task.py").is_file())
        self.assertEqual(client.destroyed, 1)

    def test_reported_verifier_failure_or_unverified_identity_cannot_become_workflow_success(self):
        options = AgentOptions(verification_code="print('verified')\n")
        original = FakeAgentTransport.invoke
        for invalid in (
            None, {"exit_code": 1}, {"timed_out": True}, {"uid": 0},
            {"uid": True}, {"sandbox_verified": False},
        ):
            with self.subTest(invalid=invalid):
                def forged(transport, stage, request, deadline):
                    report = original(transport, stage, request, deadline)
                    if stage == "agent":
                        report = {key: value for key, value in report.items() if key != "agent_report_sha256"}
                        report["verification"] = (
                            {**report["verification"], **invalid} if invalid is not None else None
                        )
                        transport.files["agent-report.json"] = (json.dumps(report) + "\n").encode()
                        return {
                            **report,
                            "agent_report_sha256": hashlib.sha256(transport.files["agent-report.json"]).hexdigest(),
                        }
                    return report

                client = FakeClient()
                with patch.object(FakeAgentTransport, "invoke", forged), self.assertRaises(JobError):
                    self.run_workflow(client, settings=replace(self.settings, agent=options))
                metadata, output = self.latest()
                self.assertEqual(metadata["job_status"], "failed")
                self.assertEqual(metadata["cleanup_status"], "confirmed")
                self.assertFalse((output / "response.txt").exists())
                self.assertTrue((output / "artifacts/scripts/task.py").is_file())
                self.assertEqual(client.destroyed, 1)

    def test_agent_stage_and_transfer_failures_still_confirm_destruction(self):
        for failure in ("upload-agent", "bootstrap", "agent", "agent-archive"):
            with self.subTest(failure=failure):
                client = FakeClient()
                with self.assertRaises(JobError):
                    self.run_workflow(client, fail=failure)
                metadata, output = self.latest()
                self.assertEqual(metadata["job_status"], "failed")
                self.assertEqual(metadata["cleanup_status"], "confirmed")
                self.assertEqual(client.destroyed, 1)
                self.assertFalse(client.exists)
                self.assertEqual(list(output.glob("*.part")), [])
                self.assertFalse((output / "response.txt").exists())

    def test_failed_agent_returns_partial_scripts_and_execution_output_but_exits_nonzero(self):
        client = FakeClient()
        with self.assertRaises(JobError):
            self.run_workflow(client, fail="agent-failed")
        metadata, output = self.latest()
        self.assertEqual(metadata["agent"]["status"], "failed")
        self.assertEqual(metadata["job_status"], "failed")
        self.assertEqual(metadata["cleanup_status"], "confirmed")
        self.assertTrue((output / "artifacts/scripts/task.py").is_file())
        transcript = json.loads((output / "agent-transcript.json").read_text())
        self.assertEqual(transcript["steps"][1]["observation"]["stdout"], "42\n")
        self.assertFalse((output / "response.txt").exists())
        self.assertEqual(client.destroyed, 1)

    def test_primary_agent_failure_is_not_replaced_by_artifact_retrieval_failure(self):
        client = FakeClient()
        with patch("run_job.retrieve_agent", side_effect=JobError("synthetic artifact transfer failure")):
            with self.assertRaisesRegex(JobError, "Remote agent failed"):
                self.run_workflow(client, fail="agent-failed")
        metadata, _ = self.latest()
        self.assertIn("Remote agent failed", metadata["error"])
        self.assertNotIn("synthetic artifact transfer failure", metadata["error"])
        self.assertTrue(any("synthetic artifact transfer failure" in item for item in metadata["warnings"]))
        self.assertEqual(client.destroyed, 1)
        self.assertFalse(client.exists)

    def test_cleanup_failure_preserves_all_agent_artifacts_and_requires_recovery(self):
        client = FakeClient(fail="cleanup")
        with self.assertRaises(JobError):
            self.run_workflow(client)
        metadata, output = self.latest()
        self.assertEqual(metadata["job_status"], "completed")
        self.assertEqual(metadata["cleanup_status"], "failed")
        self.assertTrue((output / "artifacts/scripts/task.py").exists())
        self.assertEqual((output / "response.txt").read_text(), ANSWER)
        self.assertTrue(client.exists)

    def test_agent_cli_reads_markdown_and_only_requested_environment_variables(self):
        env = self.root / ".env"
        env.write_text(
            f"VAST_AI_API_KEY={VAST_TOKEN}\nHUGGING_FACE_ACCESS_TOKEN={HF_TOKEN}\n"
            f"ALCHEMY_API_KEY={TASK_KEY}\nETHERSCAN_API_KEY=UNSELECTED_KEY\n"
        )
        task = self.root / "task.md"
        task.write_text(self.settings.prompt)
        verifier = self.root / "verify.py"
        verifier_code = "print('CALLER_VERIFIER_SOURCE')\n"
        verifier.write_text(verifier_code)
        client = FakeClient()
        stdout = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch("run_job.ROOT", self.root))
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(patch("run_job.VastClient", return_value=client))
            stack.enter_context(patch("run_job.inspect_model", return_value=dict(MODEL)))
            stack.enter_context(patch("run_job.SSHTransport",
                                      side_effect=lambda state, endpoint: self.transport(state, endpoint, client)))
            stack.enter_context(patch("run_job.run_command", side_effect=self.key_command))
            stack.enter_context(patch("run_job.shutil.which", return_value="/fixture/tool"))
            stack.enter_context(patch("run_job.Deadline", side_effect=lambda seconds: Clock().deadline(seconds)))
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(io.StringIO()))
            result = main([
                "agent", "--model-repo", "unit/model", "--model-file", "model-Q4_K_M.gguf",
                "--prompt-file", str(task), "--agent-env", "ALCHEMY_API_KEY",
                "--verify-script", str(verifier),
            ])
        self.assertEqual(result, 0)
        request = self.transports[-1].requests[-1][1]
        self.assertEqual(request["prompt"], self.settings.prompt)
        self.assertEqual(request["task_env"], {"ALCHEMY_API_KEY": TASK_KEY})
        self.assertEqual(request["context_size"], 16384)
        self.assertEqual(request["max_tokens"], 2048)
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["agent"]["verification_code"], verifier_code)
        metadata, output = self.latest()
        self.assertEqual(
            metadata["settings"]["agent"]["verification_sha256"], hashlib.sha256(verifier_code.encode()).hexdigest(),
        )
        self.assertEqual(metadata["agent"]["verification"]["exit_code"], 0)
        self.assertNotIn(verifier_code, (output / "metadata.json").read_text())
        self.assertNotIn("UNSELECTED_KEY", json.dumps(request))
        self.assertIn("Agent files:", stdout.getvalue())
        self.assertTrue(client.closed)

    def test_agent_ssh_payload_and_protocol_are_private_and_distinguish_explicit_failure(self):
        state = RunState(
            self.root, self.root, {"run_id": "20261015T000000Z-aaaaaaaaaaaa"},
            self.settings.credentials,
        )
        transport = SSHTransport(state, Endpoint("unit.invalid", 2222))
        report = {"status": "failed", "llama_commit": LLAMA_COMMIT, "error": "Explicit agent failure."}
        with patch("run_job.run_command", return_value=json.dumps(report).encode()) as command:
            result = transport.invoke("agent", {"task_env": {"ALCHEMY_API_KEY": TASK_KEY}}, Deadline(10))
        self.assertEqual(result["status"], "failed")
        self.assertNotIn(TASK_KEY, " ".join(command.call_args.args[0]))
        self.assertEqual(json.loads(command.call_args.kwargs["input_data"])["task_env"], {"ALCHEMY_API_KEY": TASK_KEY})


if __name__ == "__main__":
    unittest.main()
