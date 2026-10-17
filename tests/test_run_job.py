from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from job_utils import JobError, MAX_WORKER_REQUEST_BYTES
import run_job
from run_job import (
    Credentials, Endpoint, RunState, Settings, SSHTransport, cleanup_owned,
    credentials_from_file, endpoint_for, inspect_model, main, run_workflow,
)
from tests.fakes import (
    ANSWER, Clock, FakeClient, FakeTransport, HF_TOKEN, MODEL, VAST_TOKEN,
)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.credentials = Credentials(VAST_TOKEN, HF_TOKEN)
        self.settings = Settings(
            "unit/model", "model-Q4_K_M.gguf", "Literal 'quotes'; $(not-a-command)\nnext caf\u00e9",
            self.credentials, self.base / "modeloutput", self.base / ".vastai-runs",
        )
        self.transports = []
        self.stderr = io.StringIO()

    def key_command(self, args, deadline, **kwargs):
        self.assertEqual(args[0], "ssh-keygen")
        self.assertNotIn(VAST_TOKEN, " ".join(args))
        self.assertNotIn(HF_TOKEN, " ".join(args))
        if "-l" in args:
            return b"256 SHA256:FIXTURE\n"
        key = Path(args[args.index("-f") + 1])
        key.write_text("NOT_A_REAL_PRIVATE_KEY")
        key.chmod(0o600)
        key.with_suffix(".pub").write_text("ssh-ed25519 TEST_PUBLIC_KEY fixture")
        return b""

    def execute(self, client=None, *, fail="", deadline=None, settings=None, finish_reason="stop"):
        client = client or FakeClient()

        def transport_factory(state, endpoint):
            transport = FakeTransport(state, endpoint, client, fail=fail, finish_reason=finish_reason)
            self.transports.append(transport)
            return transport

        with ExitStack() as stack:
            stack.enter_context(patch("run_job.run_command", side_effect=self.key_command))
            stack.enter_context(patch("run_job.shutil.which", return_value="/usr/bin/fixture"))
            stack.enter_context(patch("run_job.Deadline", side_effect=lambda seconds: Clock().deadline(seconds)))
            stack.enter_context(redirect_stderr(self.stderr))
            return run_workflow(
                settings or self.settings, client, deadline=deadline,
                model_lookup=lambda settings, deadline: dict(MODEL),
                transport_factory=transport_factory,
            )

    def latest(self):
        path = sorted(self.settings.state_root.glob("*/state.json"),
                      key=lambda path: path.stat().st_mtime_ns)[-1]
        return json.loads(path.read_text()), Path(json.loads(path.read_text())["output_dir"])

    def test_success_retrieves_exact_response_before_confirmed_destruction(self):
        client = FakeClient()
        state = self.execute(client)
        self.assertEqual((state.output / "response.txt").read_bytes(), ANSWER.encode())
        self.assertEqual(state.data["job_status"], "completed")
        self.assertEqual(state.data["cleanup_status"], "confirmed")
        self.assertEqual(state.data["instance_id"], 101)
        self.assertEqual(client.created, 1)
        self.assertEqual(client.destroyed, 1)
        self.assertFalse(client.exists)
        self.assertLess(client.events.index("fetch:completion.json"), client.events.index("destroy:101"))
        self.assertIn("attach:101", client.events)
        self.assertFalse((state.directory / "id_ed25519").exists())
        self.assertFalse((state.directory / "id_ed25519.pub").exists())
        self.assertEqual(state.directory.stat().st_mode & 0o777, 0o700)
        self.assertEqual(state.output.stat().st_mode & 0o777, 0o700)
        self.assertEqual(state.data["usage"]["completion_tokens"], 18)
        for path in state.output.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_multi_gpu_count_is_revalidated_persisted_and_sent_to_both_worker_stages(self):
        client = FakeClient()
        client.offer = replace(client.offer, gpu_count=8)
        with patch.object(client, "offers", wraps=client.offers) as search:
            state = self.execute(client, settings=replace(self.settings, gpu_count=8))
        self.assertEqual([call.kwargs["gpu_count"] for call in search.call_args_list], [8, 8])
        self.assertEqual(state.data["settings"]["gpu_count"], 8)
        self.assertEqual(state.data["offer"]["gpu_count"], 8)
        self.assertEqual(state.data["remote_gpu"]["gpu_count"], 8)
        self.assertEqual(state.data["remote_gpu"]["gpu_vram_total_mib"], 327680)
        self.assertEqual(len(state.data["remote_gpu"]["gpus"]), 8)
        self.assertEqual([payload["gpu_count"] for _, payload in self.transports[-1].requests], [8, 8])
        self.assertFalse(client.exists)

    def test_unconfirmed_or_inconsistent_multi_gpu_inventory_fails_with_cleanup(self):
        original = FakeTransport.invoke
        for malformed in (
            {"gpu_count": 4},
            {"gpus": []},
            {"gpu_vram_total_mib": 40960},
            {"gpus": [{"gpu_name": "RTX 3090", "gpu_vram_mib": 40960}] * 8},
        ):
            with self.subTest(malformed=malformed):
                client = FakeClient()
                client.offer = replace(client.offer, gpu_count=8)

                def forged(transport, stage, request, deadline):
                    report = original(transport, stage, request, deadline)
                    if stage == "infer":
                        report.update(malformed)
                    return report

                with patch.object(FakeTransport, "invoke", forged), self.assertRaises(JobError):
                    self.execute(client, settings=replace(self.settings, gpu_count=8))
                self.assertEqual(client.destroyed, 1)
                self.assertFalse(client.exists)

    def test_changed_gpu_count_or_insufficient_aggregate_weights_capacity_cannot_be_rented(self):
        from run_job import select_offer

        client = FakeClient()
        with self.assertRaises(JobError):
            self.execute(client, settings=replace(self.settings, gpu_count=8))
        self.assertEqual(client.created, 0)
        with self.assertRaises(JobError):
            select_offer(self.settings, [client.offer], model_size_bytes=50_000_000_000)

    def test_credentials_stay_out_of_artifacts_and_vast_key_stays_local(self):
        state = self.execute()
        payloads = self.transports[-1].requests
        self.assertEqual(payloads[0][0], "bootstrap")
        self.assertNotIn(HF_TOKEN, json.dumps(payloads[0]))
        inference = payloads[1][1]
        self.assertEqual(inference["hf_token"], HF_TOKEN)
        self.assertEqual(inference["prompt"], self.settings.prompt)
        self.assertNotIn(VAST_TOKEN, json.dumps(payloads))
        for path in (*state.output.iterdir(), state.directory / "state.json"):
            text = path.read_text()
            self.assertTrue(VAST_TOKEN not in text and HF_TOKEN not in text, "Credential sentinel leaked")
        self.assertNotIn(VAST_TOKEN, self.stderr.getvalue())
        self.assertNotIn(HF_TOKEN, self.stderr.getvalue())
        self.assertNotIn(HF_TOKEN, repr(self.settings))
        self.assertNotIn(VAST_TOKEN, repr(self.credentials))

    def test_every_provisioned_stage_failure_still_destroys_the_instance(self):
        for stage in ("upload", "bootstrap", "infer", "download", "cancel", "transfer", "checksum", "completion"):
            with self.subTest(stage=stage):
                client = FakeClient()
                with self.assertRaises(JobError):
                    self.execute(client, fail=stage)
                metadata, output = self.latest()
                self.assertEqual(client.created, 1)
                self.assertEqual(client.destroyed, 1)
                self.assertFalse(client.exists)
                self.assertEqual(metadata["job_status"], "failed")
                self.assertEqual(metadata["cleanup_status"], "confirmed")
                self.assertFalse((output / "response.txt").exists())
                self.assertEqual(list(output.glob("*.part")), [])
                self.assertNotIn(HF_TOKEN, json.dumps(metadata))

    def test_key_attachment_and_offline_failures_do_not_wait_indefinitely(self):
        for failure in ("attach", "offline"):
            with self.subTest(failure=failure):
                client = FakeClient(fail=failure)
                with self.assertRaises(JobError):
                    self.execute(client)
                self.assertEqual(client.destroyed, 1)
                self.assertFalse(client.exists)
                self.assertNotIn("infer", client.events)

    def test_cleanup_failure_preserves_response_and_actionable_recovery_state(self):
        client = FakeClient(fail="cleanup")
        with self.assertRaises(JobError) as caught:
            self.execute(client)
        metadata, output = self.latest()
        self.assertEqual((output / "response.txt").read_text(), ANSWER)
        self.assertEqual(metadata["job_status"], "completed")
        self.assertEqual(metadata["cleanup_status"], "failed")
        self.assertEqual(metadata["instance_id"], 101)
        self.assertTrue(client.exists)
        self.assertIn("cleanup --run-id", str(caught.exception))
        self.assertIn("101", str(caught.exception))

    def test_primary_and_cleanup_errors_are_both_preserved(self):
        with self.assertRaises(JobError) as caught:
            self.execute(FakeClient(fail="cleanup"), fail="infer")
        self.assertIn("Fixture infer failure", str(caught.exception))
        self.assertIn("Fixture deletion refused", str(caught.exception))
        self.assertNotIn(HF_TOKEN, str(caught.exception))

    def test_creation_response_loss_is_reconciled_without_another_rental(self):
        client = FakeClient(fail="ambiguous")
        state = self.execute(client)
        self.assertEqual(client.created, 1)
        self.assertEqual(client.destroyed, 1)
        self.assertTrue(state.data["creation_reconciled"])
        self.assertEqual(state.data["cleanup_status"], "confirmed")

    def test_unknown_allocation_is_never_reported_as_successful_cleanup(self):
        client = FakeClient(fail="ambiguous-empty")
        with self.assertRaises(JobError):
            self.execute(client)
        metadata, _ = self.latest()
        self.assertEqual(client.created, 1)
        self.assertEqual(client.destroyed, 0)
        self.assertEqual(metadata["creation_outcome"], "unknown")
        self.assertEqual(metadata["cleanup_status"], "failed")
        self.assertIn("uncertain", metadata["cleanup_error"])

    def test_rejected_rental_does_not_claim_an_allocated_instance(self):
        client = FakeClient(fail="rejected")
        with self.assertRaises(JobError):
            self.execute(client)
        metadata, _ = self.latest()
        self.assertEqual(metadata["creation_outcome"], "rejected")
        self.assertEqual(metadata["cleanup_status"], "not_required")
        self.assertEqual(client.destroyed, 0)

    def test_instance_id_is_persisted_even_when_creation_consumes_the_deadline(self):
        clock = Clock()
        client = FakeClient()
        client.clock = clock
        with self.assertRaises(JobError):
            self.execute(client, deadline=clock.deadline(10))
        metadata, _ = self.latest()
        self.assertEqual(metadata["instance_id"], 101)
        self.assertEqual(client.destroyed, 1)
        self.assertEqual(metadata["cleanup_status"], "confirmed")
        self.assertNotIn("attach:101", client.events)

    def test_invalid_inputs_and_empty_marketplace_never_create_an_instance(self):
        for changes in ({"prompt": ""}, {"max_hourly_price": float("nan")},
                        {"model_file": "../model.gguf"}, {"timeout_minutes": 0},
                        {"max_tokens": 4096}, {"disk_gb": 1}):
            with self.subTest(changes=changes):
                client = FakeClient()
                with self.assertRaises(JobError):
                    self.execute(client, settings=replace(self.settings, **changes))
                self.assertEqual(client.created, 0)
        client = FakeClient(fail="no-offers")
        with self.assertRaises(JobError):
            self.execute(client)
        self.assertEqual(client.created, 0)

    def test_requested_offer_restricts_selection_and_never_falls_back(self):
        client = FakeClient()
        requested = replace(client.offer, id=20, hourly_price=2)
        settings = replace(self.settings, offer_id=20)
        self.assertEqual(run_job.select_offer(settings, [client.offer, requested]), requested)
        with self.assertRaises(JobError):
            run_job.select_offer(settings, [client.offer])

    def test_requested_offer_is_queried_and_revalidated_before_creation(self):
        client = FakeClient()
        requested = replace(client.offer, id=20, hourly_price=2)
        with patch.object(client, "offers", return_value=[client.offer, requested]) as search:
            state = self.execute(client, settings=replace(self.settings, offer_id=20))
        self.assertEqual([call.kwargs["offer_id"] for call in search.call_args_list], [20, 20])
        self.assertEqual(state.data["offer"]["offer_id"], 20)
        self.assertEqual(state.data["settings"]["requested_offer_id"], 20)
        self.assertEqual(client.created, 1)
        self.assertEqual(client.destroyed, 1)

    def test_invalid_or_ineligible_requested_offers_never_rent(self):
        for value in (0, -1, True, "20", 1.5):
            with self.subTest(offer_id=value):
                client = FakeClient()
                with self.assertRaises(JobError):
                    self.execute(client, settings=replace(self.settings, offer_id=value))
                self.assertEqual(client.created, 0)
        client = FakeClient()
        valid = replace(client.offer, upload_cost_per_gb=0.01)
        settings = replace(
            self.settings, offer_id=20, gpu_policy="a100-only",
            max_hourly_price=1, max_bandwidth_price=0.01,
        )
        variants = (
            replace(valid, id=20, hourly_price=1.0001),
            replace(valid, id=20, gpu_ram_mb=23999),
            replace(valid, id=20, gpu_name="RTX 3090"),
            replace(valid, id=20, upload_cost_per_gb=0.010001),
        )
        for requested in variants:
            with self.subTest(requested=requested):
                with patch.object(client, "offers", return_value=[valid, requested]):
                    with self.assertRaises(JobError):
                        self.execute(client, settings=settings)
                self.assertEqual(client.created, 0)

    def test_download_cap_boundary_combines_with_bidirectional_cap(self):
        client = FakeClient()
        valid = replace(client.offer, download_cost_per_gb=.00261, upload_cost_per_gb=.00521)
        settings = replace(self.settings, max_download_price=.00261, max_bandwidth_price=.00521)
        self.assertEqual(run_job.select_offer(settings, [valid]), valid)
        with patch.object(client, "offers", return_value=[valid]):
            state = self.execute(client, settings=settings)
        self.assertEqual(state.data["settings"]["max_download_price_per_gb_usd"], .00261)
        for offer in (
            replace(valid, download_cost_per_gb=.002610001),
            replace(valid, upload_cost_per_gb=.005210001),
        ):
            with self.subTest(offer=offer), self.assertRaises(JobError):
                run_job.select_offer(settings, [offer])
        free = replace(valid, download_cost_per_gb=0)
        self.assertEqual(run_job.select_offer(replace(settings, max_download_price=0), [free]), free)
        self.assertEqual(run_job.select_offer(self.settings, [valid]), valid)

    def test_download_cap_is_revalidated_before_creation(self):
        for download_price in (.002610001, .00521):
            with self.subTest(download_price=download_price):
                client = FakeClient()
                valid = replace(client.offer, download_cost_per_gb=.00261, upload_cost_per_gb=.00521)
                increased = replace(valid, download_cost_per_gb=download_price)
                settings = replace(self.settings, max_download_price=.00261, max_bandwidth_price=.00521)
                with patch.object(client, "offers", side_effect=[[valid], [increased]]) as search:
                    with self.assertRaises(JobError):
                        self.execute(client, settings=settings)
                self.assertEqual(search.call_count, 2)
                self.assertEqual(client.created, 0)

    def test_invalid_download_prices_never_rent(self):
        for value in (-.001, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                client = FakeClient()
                with self.assertRaises(JobError):
                    self.execute(client, settings=replace(self.settings, max_download_price=value))
                self.assertEqual(client.created, 0)

    def test_hugging_face_access_denial_happens_before_rental(self):
        client = FakeClient()
        with patch("run_job.shutil.which", return_value="/fixture/tool"), \
             patch("huggingface_hub.HfApi") as api, \
             patch("huggingface_hub.get_hf_file_metadata",
                   side_effect=OSError(f"denied https://unit.invalid/api?token={HF_TOKEN}")):
            api.return_value.model_info.return_value = SimpleNamespace(sha="a" * 40)
            with self.assertRaises(JobError) as caught:
                run_workflow(self.settings, client)
        self.assertEqual(client.created, 0)
        self.assertEqual(client.events, [])
        self.assertNotIn(HF_TOKEN, str(caught.exception))

    def test_limited_output_is_reported_as_truncated(self):
        state = self.execute(finish_reason="length")
        self.assertEqual(state.data["finish_reason"], "length")
        self.assertTrue(any("truncated" in warning for warning in state.data["warnings"]))

    def test_cleanup_is_repeatable_and_refuses_an_unrelated_label(self):
        client = FakeClient()
        state = self.execute(client)
        loaded = RunState.load(state.data["run_id"], self.settings.state_root, self.credentials)
        with redirect_stderr(self.stderr):
            cleanup_owned(client, loaded, Clock().deadline(300))
        self.assertEqual(client.destroyed, 1)
        client.exists = True
        client.label = "unrelated"
        with self.assertRaises(JobError), redirect_stderr(self.stderr):
            cleanup_owned(client, loaded, Clock().deadline(300))
        self.assertEqual(client.destroyed, 1)

    def test_recovery_rejects_path_traversal(self):
        with self.assertRaises(JobError):
            RunState.load("../outside", self.settings.state_root, self.credentials)

    def test_recovery_rejects_malformed_or_contradictory_allocation_records(self):
        state = self.execute()
        original = dict(state.data)
        variants = (
            {"instance_id": None, "instance_ids": [], "creation_outcome": "confirmed"},
            {"instance_id": True}, {"instance_ids": "101"}, {"instance_ids": [-1]},
            {"warnings": "not a list"}, {"creation_outcome": "invalid"},
            {"creation_outcome": "not_attempted"},
        )
        for changes in variants:
            with self.subTest(changes=changes):
                state.data = {**original, **changes}
                state.persist()
                with self.assertRaises(JobError):
                    RunState.load(state.data["run_id"], self.settings.state_root, self.credentials)

    def test_failure_journaling_creation_intent_does_not_imply_an_attempted_rental(self):
        client = FakeClient()
        original = RunState.persist
        failed = False

        def persist(state):
            nonlocal failed
            if state.data["creation_outcome"] == "unknown" and not failed:
                failed = True
                raise OSError("Fixture creation intent could not be written.")
            original(state)

        with patch.object(RunState, "persist", persist):
            with self.assertRaises(JobError):
                self.execute(client)
        metadata, _ = self.latest()
        self.assertEqual(client.created, 0)
        self.assertEqual(client.destroyed, 0)
        self.assertEqual(metadata["creation_outcome"], "not_attempted")
        self.assertEqual(metadata["cleanup_status"], "not_required")

    def test_missing_remote_gpu_evidence_fails_before_publishing_a_response(self):
        original = FakeTransport.invoke

        def incomplete(transport, stage, request, deadline):
            result = original(transport, stage, request, deadline)
            if stage == "infer":
                result.pop("gpu_vram_mib")
            return result

        client = FakeClient()
        with patch.object(FakeTransport, "invoke", incomplete):
            with self.assertRaises(JobError):
                self.execute(client)
        _, output = self.latest()
        self.assertFalse((output / "response.txt").exists())
        self.assertEqual(client.destroyed, 1)

    def test_cleanup_command_needs_only_vast_credentials_and_defers_cancellation(self):
        client = FakeClient()
        state = self.execute(client)
        env = self.base / ".env"
        env.write_text(f"VAST_AI_API_KEY={VAST_TOKEN}\n")
        client.exists = True
        original = client.destroy

        def interrupted_destroy(instance_id, deadline):
            signal.raise_signal(signal.SIGINT)
            original(instance_id, deadline)

        with patch("run_job.ROOT", self.base), patch.dict(os.environ, {}, clear=True), \
             patch("run_job.VastClient", return_value=client), \
             patch.object(client, "destroy", side_effect=interrupted_destroy), \
             patch("run_job.Deadline", side_effect=lambda seconds: Clock().deadline(seconds)), \
             redirect_stderr(self.stderr), redirect_stdout(io.StringIO()):
            result = main(["cleanup", "--run-id", state.data["run_id"]])
        self.assertEqual(result, 0)
        self.assertFalse(client.exists)

    def test_main_runs_a_mocked_end_to_end_job_with_default_repository_output(self):
        client = FakeClient()
        stdout = io.StringIO()
        env_file = self.base / ".env"
        env_file.write_text(f"VAST_AI_API_KEY={VAST_TOKEN}\nHUGGING_FACE_ACCESS_TOKEN={HF_TOKEN}\n")
        with ExitStack() as stack:
            stack.enter_context(patch("run_job.ROOT", self.base))
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(patch("run_job.VastClient", return_value=client))
            stack.enter_context(patch("run_job.inspect_model", return_value=dict(MODEL)))
            stack.enter_context(patch("run_job.SSHTransport",
                                      side_effect=lambda state, endpoint: FakeTransport(state, endpoint, client)))
            stack.enter_context(patch("run_job.run_command", side_effect=self.key_command))
            stack.enter_context(patch("run_job.shutil.which", return_value="/usr/bin/fixture"))
            stack.enter_context(patch("run_job.Deadline", side_effect=lambda seconds: Clock().deadline(seconds)))
            stack.enter_context(redirect_stderr(self.stderr))
            stack.enter_context(redirect_stdout(stdout))
            result = main(["run", "--model-repo", "unit/model", "--model-file", "model-Q4_K_M.gguf",
                           "--prompt", self.settings.prompt])
        self.assertEqual(result, 0)
        self.assertTrue(client.closed)
        responses = list((self.base / "modeloutput").glob("*/response.txt"))
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].read_text(), ANSWER)
        self.assertIn(str(responses[0]), stdout.getvalue())

    def test_main_dry_run_creates_no_resources_or_persistent_run_directories(self):
        client = FakeClient()
        stdout = io.StringIO()
        env = self.base / ".env"
        env.write_text(f"VAST_AI_API_KEY={VAST_TOKEN}\nHUGGING_FACE_ACCESS_TOKEN={HF_TOKEN}\n")
        with patch("run_job.ROOT", self.base), patch.dict(os.environ, {}, clear=True), \
             patch("run_job.VastClient", return_value=client), \
             patch("run_job.inspect_model", return_value=dict(MODEL)), redirect_stdout(stdout):
            result = main(["run", "--model-repo", "unit/model", "--model-file", "model-Q4_K_M.gguf",
                           "--prompt", "test", "--dry-run"])
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(stdout.getvalue())["dry_run"])
        self.assertEqual(client.created, 0)
        self.assertFalse((self.base / ".vastai-runs").exists())
        self.assertFalse((self.base / "modeloutput").exists())

    def test_cli_offer_override_controls_the_actual_dry_run_selection(self):
        client = FakeClient()
        requested = replace(client.offer, id=20, hourly_price=2,
                            download_cost_per_gb=.00261, upload_cost_per_gb=.00521)
        stdout = io.StringIO()
        env = self.base / ".env"
        env.write_text(f"VAST_AI_API_KEY={VAST_TOKEN}\nHUGGING_FACE_ACCESS_TOKEN={HF_TOKEN}\n")
        with patch("run_job.ROOT", self.base), patch.dict(os.environ, {}, clear=True), \
             patch("run_job.VastClient", return_value=client), \
             patch.object(client, "offers", return_value=[client.offer, requested]) as search, \
             patch("run_job.inspect_model", return_value=dict(MODEL)), redirect_stdout(stdout):
            result = main(["run", "--model-repo", "unit/model", "--model-file", "model-Q4_K_M.gguf",
                           "--prompt", "test", "--offer-id", "20", "--max-download-price", ".00261",
                           "--max-bandwidth-price", ".00521", "--dry-run"])
        self.assertEqual(result, 0)
        preview = json.loads(stdout.getvalue())
        self.assertEqual(preview["selected_offer"]["offer_id"], 20)
        self.assertEqual(preview["limits"]["requested_offer_id"], 20)
        self.assertEqual(preview["limits"]["max_download_price_per_gb_usd"], .00261)
        self.assertEqual(search.call_args.kwargs["offer_id"], 20)
        self.assertEqual(client.created, 0)
        self.assertFalse((self.base / ".vastai-runs").exists())

    def test_cli_prompt_file_and_relative_overrides_resolve_against_the_caller(self):
        client = FakeClient()
        caller = self.base / "caller"
        caller.mkdir()
        (caller / "credentials.env").write_text(
            f"VAST_AI_API_KEY={VAST_TOKEN}\nHUGGING_FACE_ACCESS_TOKEN={HF_TOKEN}\n"
        )
        (caller / "prompt.txt").write_text(self.settings.prompt, encoding="utf-8")
        transport = []
        original_cwd = Path.cwd()

        def factory(state, endpoint):
            worker = FakeTransport(state, endpoint, client)
            transport.append(worker)
            return worker

        try:
            os.chdir(caller)
            with ExitStack() as stack:
                stack.enter_context(patch("run_job.ROOT", self.base))
                stack.enter_context(patch.dict(os.environ, {}, clear=True))
                stack.enter_context(patch("run_job.VastClient", return_value=client))
                stack.enter_context(patch("run_job.inspect_model", return_value=dict(MODEL)))
                stack.enter_context(patch("run_job.SSHTransport", side_effect=factory))
                stack.enter_context(patch("run_job.run_command", side_effect=self.key_command))
                stack.enter_context(patch("run_job.shutil.which", return_value="/fixture/tool"))
                stack.enter_context(patch("run_job.Deadline", side_effect=lambda seconds: Clock().deadline(seconds)))
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(self.stderr))
                result = main(["run", "--model-repo", "unit/model", "--model-file", "model-Q4_K_M.gguf",
                               "--prompt-file", "prompt.txt", "--env-file", "credentials.env",
                               "--output-dir", "results"])
        finally:
            os.chdir(original_cwd)
        self.assertEqual(result, 0)
        self.assertEqual(transport[-1].requests[1][1]["prompt"], self.settings.prompt)
        responses = list((caller / "results").glob("*/response.txt"))
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].read_text(), ANSWER)
        self.assertFalse((self.base / "modeloutput").exists())
        self.assertTrue(self.settings.state_root.exists())

    def test_real_sigterm_cancels_the_job_and_sigint_during_teardown_is_deferred(self):
        client = FakeClient()
        env = self.base / ".env"
        env.write_text(f"VAST_AI_API_KEY={VAST_TOKEN}\nHUGGING_FACE_ACCESS_TOKEN={HF_TOKEN}\n")

        class InterruptingTransport(FakeTransport):
            def invoke(self, stage, request, deadline):
                if stage == "infer":
                    signal.raise_signal(signal.SIGTERM)
                return super().invoke(stage, request, deadline)

        original_destroy = client.destroy

        def interrupted_destroy(instance_id, deadline):
            signal.raise_signal(signal.SIGINT)
            original_destroy(instance_id, deadline)

        before = signal.getsignal(signal.SIGTERM)
        with ExitStack() as stack:
            stack.enter_context(patch("run_job.ROOT", self.base))
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(patch("run_job.VastClient", return_value=client))
            stack.enter_context(patch("run_job.inspect_model", return_value=dict(MODEL)))
            stack.enter_context(patch("run_job.SSHTransport",
                                      side_effect=lambda state, endpoint:
                                      InterruptingTransport(state, endpoint, client)))
            stack.enter_context(patch.object(client, "destroy", side_effect=interrupted_destroy))
            stack.enter_context(patch("run_job.run_command", side_effect=self.key_command))
            stack.enter_context(patch("run_job.shutil.which", return_value="/usr/bin/fixture"))
            stack.enter_context(patch("run_job.Deadline", side_effect=lambda seconds: Clock().deadline(seconds)))
            stack.enter_context(redirect_stderr(self.stderr))
            result = main(["run", "--model-repo", "unit/model", "--model-file", "model-Q4_K_M.gguf",
                           "--prompt", "cancel test"])
        self.assertEqual(result, 1)
        self.assertEqual(client.destroyed, 1)
        self.assertFalse(client.exists)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)


class ConfigAndTransportTests(unittest.TestCase):
    def test_valid_prompt_byte_limit_fits_the_escaped_worker_protocol(self):
        settings = Settings("unit/model", "model.gguf", "\0" * run_job.MAX_PROMPT_BYTES,
                            Credentials(VAST_TOKEN, HF_TOKEN), Path("/unused"), Path("/unused/state"))
        settings.validate()
        payload = {"prompt": settings.prompt, "hf_token": HF_TOKEN, "model": MODEL}
        self.assertLessEqual(len(json.dumps(payload, ensure_ascii=False).encode()), MAX_WORKER_REQUEST_BYTES)

    def test_file_credentials_and_process_precedence_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(f"VAST_AI_API_KEY={VAST_TOKEN}\nHUGGING_FACE_ACCESS_TOKEN={HF_TOKEN}\n")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(credentials_from_file(path), Credentials(VAST_TOKEN, HF_TOKEN))
            with patch.dict(os.environ, {"VAST_AI_API_KEY": ""}, clear=True):
                with self.assertRaises(JobError):
                    credentials_from_file(path)

    def test_missing_file_or_credentials_fail_without_global_credential_fallback(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            with self.assertRaises(JobError):
                credentials_from_file(path)
            path.write_text(f"VAST_AI_API_KEY={VAST_TOKEN}\n")
            with self.assertRaises(JobError):
                credentials_from_file(path)
            self.assertEqual(credentials_from_file(path, require_hf=False), Credentials(VAST_TOKEN, ""))

    def test_hugging_face_access_is_checked_on_the_actual_file_without_download(self):
        settings = Settings("unit/model", "model-Q4_K_M.gguf", "test", Credentials(VAST_TOKEN, HF_TOKEN),
                            Path("/unused/output"), Path("/unused/state"))
        metadata = SimpleNamespace(commit_hash="a" * 40, size=16, etag="b" * 64)
        with patch("huggingface_hub.HfApi") as api, \
             patch("huggingface_hub.get_hf_file_metadata", return_value=metadata) as head, \
             patch("huggingface_hub.hf_hub_download") as download:
            api.return_value.model_info.return_value = SimpleNamespace(sha="a" * 40)
            result = inspect_model(settings, Clock().deadline(30))
        self.assertEqual(result["revision"], "a" * 40)
        self.assertEqual(result["size_bytes"], 16)
        self.assertEqual(head.call_args.kwargs["token"], HF_TOKEN)
        self.assertIn("/resolve/" + "a" * 40 + "/", head.call_args.args[0])
        download.assert_not_called()

    def test_complete_split_set_checks_every_file_at_one_revision_and_sums_all_sizes(self):
        names = [f"Q3/model-0000{index}-of-00003.gguf" for index in (1, 2, 3)]
        settings = Settings("unit/model", names[0], "test", Credentials(VAST_TOKEN, HF_TOKEN),
                            Path("/unused/output"), Path("/unused/state"))
        metadata = [SimpleNamespace(commit_hash="a" * 40, size=index * 16, etag=str(index) * 64)
                    for index in (1, 2, 3)]
        with patch("huggingface_hub.HfApi") as api, \
             patch("huggingface_hub.get_hf_file_metadata", side_effect=metadata) as head, \
             patch("huggingface_hub.hf_hub_download") as download:
            api.return_value.model_info.return_value = SimpleNamespace(
                sha="a" * 40, siblings=[SimpleNamespace(rfilename=name) for name in names],
            )
            result = inspect_model(settings, Clock().deadline(30))
        self.assertEqual(result["size_bytes"], 96)
        self.assertIsNone(result["sha256"])
        self.assertEqual([item["filename"] for item in result["files"]], names)
        self.assertEqual([item["sha256"] for item in result["files"]], [str(index) * 64 for index in (1, 2, 3)])
        self.assertEqual(head.call_count, 3)
        self.assertTrue(all("/resolve/" + "a" * 40 + "/" in call.args[0] for call in head.call_args_list))
        download.assert_not_called()

    def test_missing_split_member_or_mismatched_member_revision_fails_without_download(self):
        names = [f"model-0000{index}-of-00002.gguf" for index in (1, 2)]
        settings = Settings("unit/model", names[0], "test", Credentials(VAST_TOKEN, HF_TOKEN),
                            Path("/unused/output"), Path("/unused/state"))
        with patch("huggingface_hub.HfApi") as api, \
             patch("huggingface_hub.get_hf_file_metadata") as head:
            api.return_value.model_info.return_value = SimpleNamespace(
                sha="a" * 40, siblings=[SimpleNamespace(rfilename=names[0])],
            )
            with self.assertRaises(JobError):
                inspect_model(settings, Clock().deadline(30))
            head.assert_not_called()
            api.return_value.model_info.return_value.siblings.append(SimpleNamespace(rfilename=names[1]))
            head.side_effect = [
                SimpleNamespace(commit_hash="a" * 40, size=16, etag="b" * 64),
                SimpleNamespace(commit_hash="c" * 40, size=16, etag="b" * 64),
            ]
            with self.assertRaises(JobError):
                inspect_model(settings, Clock().deadline(30))

    def test_file_access_failure_is_explicit_and_sanitized(self):
        settings = Settings("unit/model", "model.gguf", "test", Credentials(VAST_TOKEN, HF_TOKEN),
                            Path("/unused/output"), Path("/unused/state"))
        with patch("huggingface_hub.HfApi") as api, \
             patch("huggingface_hub.get_hf_file_metadata",
                   side_effect=OSError(f"https://unit.invalid/api?token={HF_TOKEN}")):
            api.return_value.model_info.return_value = SimpleNamespace(sha="a" * 40)
            with self.assertRaises(JobError) as caught:
                inspect_model(settings, Clock().deadline(30))
        self.assertNotIn(HF_TOKEN, str(caught.exception))

    def test_direct_and_proxy_ssh_endpoints_are_validated(self):
        self.assertEqual(endpoint_for({
            "ports": {"22/tcp": [{"HostPort": "2222"}]}, "public_ipaddr": "127.0.0.1",
        }), Endpoint("127.0.0.1", 2222))
        self.assertEqual(endpoint_for({"ssh_host": "ssh1.vast.ai", "ssh_port": 3333}),
                         Endpoint("ssh1.vast.ai", 3333))
        self.assertIsNone(endpoint_for({"ssh_host": None, "ssh_port": None}))
        with self.assertRaises(JobError):
            endpoint_for({"ssh_host": "host;not-a-command", "ssh_port": 2222})
        with self.assertRaises(JobError):
            endpoint_for({"ssh_host": "host", "ssh_port": 70000})

    def test_ssh_payload_is_stdin_data_not_a_shell_command_or_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = RunState(root, root, {"run_id": "20261009T000000Z-aaaaaaaaaaaa", "warnings": []},
                             Credentials(VAST_TOKEN, HF_TOKEN))
            transport = SSHTransport(state, Endpoint("unit.invalid", 2222))
            result = {"status": "completed", "llama_commit": run_job.LLAMA_COMMIT}
            prompt = "Literal 'quotes'; $(not-a-command)\nnext"
            with patch("run_job.run_command", return_value=json.dumps(result).encode()) as command:
                transport.invoke("infer", {"prompt": prompt, "hf_token": HF_TOKEN}, Clock().deadline(20))
            arguments = command.call_args.args[0]
            self.assertNotIn(HF_TOKEN, " ".join(arguments))
            self.assertNotIn(prompt, " ".join(arguments))
            payload = json.loads(command.call_args.kwargs["input_data"])
            self.assertEqual(payload["prompt"], prompt)
            self.assertEqual(payload["hf_token"], HF_TOKEN)
            self.assertEqual(payload["remaining_seconds"], 20)
            self.assertNotIn(VAST_TOKEN, json.dumps(payload))
            self.assertIn("StrictHostKeyChecking=accept-new", arguments)
            self.assertNotIn("apt-get", " ".join(arguments))

    def test_scp_connection_failures_retry_only_idempotent_transfers(self):
        for operation in ("worker", "agent", "fetch"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = RunState(root, root, {"run_id": "20261009T000000Z-aaaaaaaaaaaa", "warnings": []},
                                 Credentials(VAST_TOKEN, HF_TOKEN))
                transport = SSHTransport(state, Endpoint("unit.invalid", 2222))
                clock = Clock()
                deadline = clock.deadline(20)
                attempts = []

                def execute(args, deadline, **kwargs):
                    if args[0] == "scp":
                        attempts.append(args)
                        self.assertEqual(kwargs["secrets"], (VAST_TOKEN, HF_TOKEN))
                        if len(attempts) < 3:
                            raise JobError("scp: Connection closed")
                    return b""

                with patch("run_job.run_command", side_effect=execute), patch.object(state, "log") as log:
                    if operation == "worker":
                        transport.upload(deadline)
                    elif operation == "agent":
                        transport.upload_agent(None, deadline)
                    else:
                        transport.fetch("remote.log", root / "remote.log.part", deadline)
                self.assertEqual(len(attempts), 3)
                self.assertTrue(all(args == attempts[0] for args in attempts))
                self.assertEqual(log.call_count, 2)
                self.assertEqual(clock.value, 3)
                self.assertIn("StrictHostKeyChecking=accept-new", attempts[0])

    def test_scp_retries_are_bounded_and_permanent_failures_stop_immediately(self):
        for message, count, error_type in (
            ("scp: Connection reset by peer", 3, JobError),
            ("Permission denied (publickey). scp: Connection closed", 1, JobError),
            ("scp: No such file or directory", 1, JobError),
            ("Host key verification failed. scp: Connection closed", 1, run_job.HostKeyError),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = RunState(root, root, {"run_id": "20261009T000000Z-aaaaaaaaaaaa", "warnings": []},
                                 Credentials(VAST_TOKEN, HF_TOKEN))
                transport = SSHTransport(state, Endpoint("unit.invalid", 2222))
                with patch("run_job.run_command", side_effect=JobError(message)) as command, \
                     patch.object(state, "log"):
                    with self.assertRaises(error_type):
                        transport.fetch("remote.log", root / "remote.log.part", Clock().deadline(20))
                self.assertEqual(command.call_count, count)

    def test_scp_retry_cannot_extend_deadline_or_replay_remote_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = RunState(root, root, {"run_id": "20261009T000000Z-aaaaaaaaaaaa", "warnings": []},
                             Credentials(VAST_TOKEN, HF_TOKEN))
            transport = SSHTransport(state, Endpoint("unit.invalid", 2222))
            clock = Clock()
            with patch("run_job.run_command", side_effect=JobError("scp: Connection closed")) as command, \
                 patch.object(state, "log"):
                with self.assertRaises(run_job.DeadlineExceeded):
                    transport.fetch("remote.log", root / "remote.log.part", clock.deadline(.5))
                self.assertEqual(command.call_count, 1)
            self.assertEqual(clock.value, .5)
            with patch("run_job.run_command", side_effect=JobError("Connection closed")) as command:
                with self.assertRaises(JobError):
                    transport.command("owned-worker-stage", Clock().deadline(20))
                self.assertEqual(command.call_count, 1)

    def test_help_does_not_read_credentials_or_contact_providers(self):
        with patch("run_job.credentials_from_file") as credentials, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        credentials.assert_not_called()

    def test_missing_remote_python_bootstrap_keeps_request_stdin_and_stdout_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries = root / "bin"
            binaries.mkdir()
            state = RunState(root, root, {"run_id": "20261009T000000Z-aaaaaaaaaaaa", "warnings": []},
                             Credentials(VAST_TOKEN, HF_TOKEN))
            transport = SSHTransport(state, Endpoint("unit.invalid", 2222))
            stub = root / "python-stub"
            handler = (
                "import json,sys; request=json.load(sys.stdin); "
                f"print(json.dumps({{'status':'ready','llama_commit':{run_job.LLAMA_COMMIT!r},"
                "'received':request}))"
            )
            stub.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(handler)}\n")
            stub.chmod(0o700)
            identity = binaries / "id"
            identity.write_text("#!/bin/sh\nprintf '%s\\n' 0\n")
            identity.chmod(0o700)
            installer = binaries / "apt-get"
            installer.write_text(
                "#!/bin/sh\nprintf '%s\\n' APT_LOG_MUST_NOT_POLLUTE_JSON\n"
                "if read unexpected; then exit 91; fi\n"
                "if [ \"$1\" = install ]; then "
                f"/bin/cp {shlex.quote(str(stub))} {shlex.quote(str(binaries / 'python3'))} && "
                f"/bin/chmod 700 {shlex.quote(str(binaries / 'python3'))}; fi\n"
            )
            installer.chmod(0o700)
            observations = []

            def execute(command, deadline, input_data):
                process = subprocess.run(
                    ["/bin/sh", "-c", command], input=input_data, capture_output=True,
                    env={"PATH": str(binaries)}, check=True,
                )
                observations.append(process)
                return process.stdout

            with patch.object(transport, "command", side_effect=execute):
                result = transport.invoke("bootstrap", {"hf_token": HF_TOKEN}, Clock().deadline(20))
            self.assertEqual(result["received"]["hf_token"], HF_TOKEN)
            self.assertEqual(result["received"]["remaining_seconds"], 20)
            self.assertNotIn(b"APT_LOG", observations[0].stdout)
            self.assertIn(b"APT_LOG_MUST_NOT_POLLUTE_JSON", observations[0].stderr)

            identity.write_text("#!/bin/sh\nprintf '%s\\n' 1000\n")
            with patch.object(transport, "command", side_effect=execute):
                result = transport.invoke("bootstrap", {"hf_token": HF_TOKEN}, Clock().deadline(20))
            self.assertEqual(result["received"]["hf_token"], HF_TOKEN)
            self.assertEqual(observations[-1].stderr, b"")

            (binaries / "python3").unlink()
            with patch.object(transport, "command", side_effect=execute), \
                 self.assertRaises(subprocess.CalledProcessError) as caught:
                transport.invoke("bootstrap", {}, Clock().deadline(20))
            self.assertIn(b"Python 3 is missing and cannot be installed", caught.exception.stderr)

            identity.write_text("#!/bin/sh\nprintf '%s\\n' 0\n")
            installer.unlink()
            with patch.object(transport, "command", side_effect=execute), \
                 self.assertRaises(subprocess.CalledProcessError) as caught:
                transport.invoke("bootstrap", {}, Clock().deadline(20))
            self.assertIn(b"Python 3 is missing and cannot be installed", caught.exception.stderr)

            installer.write_text("#!/bin/sh\nexit 37\n")
            installer.chmod(0o700)
            with patch.object(transport, "command", side_effect=execute), \
                 self.assertRaises(subprocess.CalledProcessError) as caught:
                transport.invoke("bootstrap", {}, Clock().deadline(20))
            self.assertEqual(caught.exception.stdout, b"")
            self.assertFalse((binaries / "python3").exists())
