from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from job_utils import Deadline, DeadlineExceeded, JobError, LLAMA_COMMIT
import remote_job
from remote_job import (
    bootstrap, download_model, gpu_details, infer, server_args,
    wait_for_gpu_offload, wait_for_server,
)
from tests.fakes import ANSWER, Clock, HF_TOKEN, VAST_TOKEN


MODEL_BYTES = b"GGUF_SYNTHETIC_TEST_FIXTURE"


class WorkerFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.model = {
            "repo": "unit/model", "filename": "model-Q4_K_M.gguf", "revision": "a" * 40,
            "size_bytes": len(MODEL_BYTES), "sha256": hashlib.sha256(MODEL_BYTES).hexdigest(),
        }

    def download(self, **kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(MODEL_BYTES)
        return str(path)


class WorkerTests(WorkerFixture):
    def split_fixture(self):
        files = []
        bodies = {}
        for index in (1, 2, 3):
            name = f"Q3/model-0000{index}-of-00003.gguf"
            body = MODEL_BYTES + bytes([index])
            bodies[name] = body
            files.append({"filename": name, "size_bytes": len(body),
                          "sha256": hashlib.sha256(body).hexdigest()})
        model = {**self.model, "filename": files[0]["filename"], "files": files,
                 "size_bytes": sum(item["size_bytes"] for item in files), "sha256": None}

        def download(**kwargs):
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bodies[kwargs["filename"]])
            return str(path)

        return model, download

    def test_download_uses_explicit_remote_path_revision_and_token(self):
        with patch("huggingface_hub.hf_hub_download", side_effect=self.download) as download:
            path = download_model(self.root, self.model, HF_TOKEN, Deadline(30))
        self.assertEqual(path, self.root / "models/model-Q4_K_M.gguf")
        self.assertEqual(path.read_bytes(), MODEL_BYTES)
        self.assertEqual(download.call_args.kwargs["revision"], "a" * 40)
        self.assertEqual(download.call_args.kwargs["token"], HF_TOKEN)
        self.assertEqual(download.call_args.kwargs["local_dir"], self.root / "models")
        self.assertNotIn(VAST_TOKEN, json.dumps({key: str(value) for key, value in download.call_args.kwargs.items()}))

    def test_split_download_verifies_all_members_and_returns_first_shard(self):
        model, transfer = self.split_fixture()
        with patch("huggingface_hub.hf_hub_download", side_effect=transfer) as download:
            first = download_model(self.root, model, HF_TOKEN, Deadline(30))
        self.assertEqual(first, self.root / "models" / model["filename"])
        self.assertEqual([call.kwargs["filename"] for call in download.call_args_list],
                         [item["filename"] for item in model["files"]])
        for item in model["files"]:
            path = self.root / "models" / item["filename"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])

    def test_corrupt_later_shard_does_not_pass_first_file_integrity(self):
        model, transfer = self.split_fixture()
        model["files"][1]["sha256"] = "0" * 64
        with patch("huggingface_hub.hf_hub_download", side_effect=transfer) as download:
            with self.assertRaisesRegex(JobError, "SHA-256"):
                download_model(self.root, model, HF_TOKEN, Deadline(30))
        self.assertEqual(download.call_count, 2)

    def test_incomplete_reordered_or_unsigned_shards_fail_before_transfer(self):
        model, _ = self.split_fixture()
        for members in (model["files"][:-1], list(reversed(model["files"])),
                        [{**item, "sha256": None} for item in model["files"]]):
            with self.subTest(members=members), patch("huggingface_hub.hf_hub_download") as download:
                with self.assertRaises(JobError):
                    download_model(self.root, {**model, "files": members}, HF_TOKEN, Deadline(30))
                download.assert_not_called()

    def test_split_free_disk_check_uses_the_whole_set_not_just_first_file(self):
        model, _ = self.split_fixture()
        with patch("remote_job.shutil.disk_usage") as disk, patch("huggingface_hub.hf_hub_download") as download:
            disk.return_value.free = model["size_bytes"] + 2 * 1024 ** 3 - 1
            with self.assertRaises(JobError):
                download_model(self.root, model, HF_TOKEN, Deadline(30))
            download.assert_not_called()

    def test_download_size_hash_magic_and_path_are_not_silently_trusted(self):
        variants = (
            {**self.model, "size_bytes": len(MODEL_BYTES) + 1},
            {**self.model, "sha256": "0" * 64},
        )
        for model in variants:
            with self.subTest(model=model), \
                 patch("huggingface_hub.hf_hub_download", side_effect=self.download):
                with self.assertRaises(JobError):
                    download_model(self.root, model, HF_TOKEN, Deadline(30))
        bad = self.root / "outside.gguf"
        bad.write_bytes(MODEL_BYTES)
        with patch("huggingface_hub.hf_hub_download", return_value=str(bad)):
            with self.assertRaises(JobError):
                download_model(self.root, self.model, HF_TOKEN, Deadline(30))

        def not_gguf(**kwargs):
            result = self.download(**kwargs)
            Path(result).write_bytes(b"NOTG" + MODEL_BYTES[4:])
            return result

        with patch("huggingface_hub.hf_hub_download", side_effect=not_gguf):
            with self.assertRaises(JobError):
                download_model(self.root, self.model, HF_TOKEN, Deadline(30))

    def test_insufficient_disk_fails_before_download(self):
        with patch("remote_job.shutil.disk_usage") as disk, \
             patch("huggingface_hub.hf_hub_download") as download:
            disk.return_value.free = len(MODEL_BYTES)
            with self.assertRaises(JobError):
                download_model(self.root, self.model, HF_TOKEN, Deadline(30))
            download.assert_not_called()

    def test_download_error_does_not_leak_signed_url_or_token(self):
        with patch("huggingface_hub.hf_hub_download",
                   side_effect=RuntimeError(f"failed https://unit.invalid/path?token={HF_TOKEN}&Signature=x")):
            with self.assertRaises(JobError) as caught:
                download_model(self.root, self.model, HF_TOKEN, Deadline(30))
        self.assertNotIn(HF_TOKEN, str(caught.exception))
        self.assertNotIn("Signature", str(caught.exception))

    def test_actual_gpu_and_memory_must_match_the_offer(self):
        with patch("remote_job.run_command", return_value=b"NVIDIA A100-SXM4-40GB, 40536\n"):
            self.assertEqual(gpu_details("A100 SXM4", Deadline(10))["gpu_vram_mib"], 40536)
        with patch("remote_job.run_command", return_value=b"NVIDIA A100 80GB PCIe, 81920\n"):
            self.assertEqual(gpu_details("A100_PCIE_80GB", Deadline(10))["gpu_vram_mib"], 81920)
        with patch("remote_job.run_command", return_value=b"Quadro RTX 8000, 49152\n"):
            self.assertEqual(gpu_details("Q RTX 8000", Deadline(10))["gpu_vram_mib"], 49152)
        for output in (
            b"RTX 3090, 24576\n", b"NVIDIA A100-SXM4-40GB, 23999\n",
            b"NVIDIA A100-SXM4-40GB, nan\n", b"NVIDIA A100-SXM4-40GB, inf\n",
            b"one, 40960\ntwo, 40960\n",
        ):
            with self.subTest(output=output), patch("remote_job.run_command", return_value=output):
                with self.assertRaises(JobError):
                    gpu_details("A100 SXM4", Deadline(10))

    def test_every_allocated_gpu_is_verified_with_per_device_and_total_memory(self):
        output = b"NVIDIA A100-SXM4-40GB, 40536\n" * 8
        with patch("remote_job.run_command", return_value=output):
            gpu = gpu_details("A100 SXM4", Deadline(10), 8)
        self.assertEqual(gpu["gpu_count"], 8)
        self.assertEqual(gpu["gpu_vram_mib"], 40536)
        self.assertEqual(gpu["gpu_vram_total_mib"], 324288)
        self.assertEqual(len(gpu["gpus"]), 8)
        for incorrect in (
            output[:-len(b"NVIDIA A100-SXM4-40GB, 40536\n")],
            output.replace(b"40536", b"23999", 1),
            output.replace(b"NVIDIA A100-SXM4-40GB", b"RTX 3090", 1),
        ):
            with self.subTest(output=incorrect), patch("remote_job.run_command", return_value=incorrect):
                with self.assertRaises(JobError):
                    gpu_details("A100 SXM4", Deadline(10), 8)

    def test_cuda_bootstrap_pins_source_and_compiles_only_the_required_target(self):
        commands = []

        def capture(args, root, deadline, context):
            commands.append(args)
            if args[:2] == ["cmake", "--build"]:
                binary = root / "llama.cpp/build/bin/llama-server"
                binary.parent.mkdir(parents=True)
                binary.write_text("SYNTHETIC_BINARY_FIXTURE")

        with patch("remote_job.gpu_details", return_value={"gpu_name": "A100", "gpu_vram_mib": 40536}), \
             patch("remote_job.shutil.which", return_value="/fixture/tool"), \
             patch("remote_job.logged_command", side_effect=capture):
            result = bootstrap(self.root, {"gpu_name": "A100"}, Deadline(30))
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["llama_commit"], LLAMA_COMMIT)
        self.assertTrue(any(LLAMA_COMMIT in args and "fetch" in args for args in commands))
        configure = next(args for args in commands if args[:2] == ["cmake", "-S"])
        self.assertIn("-DGGML_CUDA=ON", configure)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=native", configure)
        build = next(args for args in commands if args[:2] == ["cmake", "--build"])
        self.assertEqual(build[build.index("--target") + 1], "llama-server")

    def test_template_without_cuda_toolkit_is_rejected_before_setup(self):
        with patch("remote_job.gpu_details", return_value={"gpu_name": "A100"}), \
             patch("remote_job.shutil.which", return_value=None), \
             patch("remote_job.logged_command") as setup:
            with self.assertRaises(JobError):
                bootstrap(self.root, {"gpu_name": "A100"}, Deadline(30))
            setup.assert_not_called()

    def test_bare_cuda_template_installs_venv_before_creating_the_environment(self):
        commands = []

        def capture(args, root, deadline, context):
            commands.append(args)
            if args[:2] == ["cmake", "--build"]:
                binary = root / "llama.cpp/build/bin/llama-server"
                binary.parent.mkdir(parents=True)
                binary.write_text("SYNTHETIC_BINARY_FIXTURE")

        with patch("remote_job.gpu_details", return_value={"gpu_name": "A100"}), \
             patch("remote_job.shutil.which", return_value="/fixture/tool"), \
             patch("remote_job.importlib.util.find_spec", return_value=None), \
             patch("remote_job.os.geteuid", return_value=0), \
             patch("remote_job.logged_command", side_effect=capture):
            bootstrap(self.root, {"gpu_name": "A100"}, Deadline(30))
        install = next(args for args in commands if args[:2] == ["apt-get", "install"])
        self.assertEqual(install[-1], "python3-venv")
        environment = next(args for args in commands if "-m" in args and "venv" in args)
        self.assertLess(commands.index(install), commands.index(environment))

    def test_missing_venv_without_install_privilege_fails_explicitly(self):
        with patch("remote_job.gpu_details", return_value={"gpu_name": "A100"}), \
             patch("remote_job.shutil.which", return_value="/fixture/tool"), \
             patch("remote_job.importlib.util.find_spec", return_value=None), \
             patch("remote_job.os.geteuid", return_value=1000), \
             patch("remote_job.logged_command") as commands:
            with self.assertRaisesRegex(JobError, "Python venv/ensurepip"):
                bootstrap(self.root, {"gpu_name": "A100"}, Deadline(30))
            commands.assert_not_called()

    def test_server_is_loopback_gpu_only_and_has_explicit_generation_capacity(self):
        args = server_args(Path("/fixture/server"), Path("/fixture/model.gguf"), 4096, 12345)
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertEqual(args[args.index("--n-gpu-layers") + 1], "all")
        self.assertEqual(args[args.index("--fit") + 1], "off")
        self.assertEqual(args[args.index("--parallel") + 1], "1")
        self.assertEqual(args[args.index("--ctx-size") + 1], "4096")
        self.assertIn("--no-context-shift", args)
        self.assertIn("--offline", args)
        self.assertEqual(args[args.index("--verbosity") + 1], "4")

    def test_server_readiness_waits_for_loading_but_rejects_failed_or_missing_health(self):
        process = MagicMock()
        process.poll.return_value = None
        http = MagicMock()
        loading = MagicMock(status_code=503)
        ready = MagicMock(status_code=200)
        ready.json.return_value = {"status": "ok"}
        http.get.side_effect = [loading, ready]
        clock = Clock()
        wait_for_server(process, "http://127.0.0.1:1234", clock.deadline(10), http)
        self.assertEqual(clock.value, 1)
        self.assertEqual(http.get.call_count, 2)
        http.get.side_effect = None
        http.get.return_value = ready
        ready.json.return_value = {}
        with self.assertRaises(JobError):
            wait_for_server(process, "http://127.0.0.1:1234", Deadline(10), http)
        process.poll.return_value = 1
        with self.assertRaises(JobError):
            wait_for_server(process, "http://127.0.0.1:1234", Deadline(10), http)

    def test_offload_verification_waits_for_async_logs_from_only_the_current_server(self):
        path = self.root / "remote.log"
        path.write_text("old server: offloaded 33/33 layers to GPU\n")
        offset = path.stat().st_size
        process = MagicMock()
        process.poll.return_value = None
        clock = Clock()
        deadline = clock.deadline(10)
        flushed = False

        def flush_later(seconds):
            nonlocal flushed
            clock.sleep(seconds)
            if clock.value >= 0.2 and not flushed:
                with path.open("ab") as file:
                    file.write(b"0.01.897.000 I load_tensors: offloaded 33/33 layers to GPU\n")
                flushed = True

        deadline.sleeper = flush_later
        wait_for_gpu_offload(process, path, offset, deadline)
        self.assertTrue(flushed)
        self.assertEqual(clock.value, 0.2)

    def test_missing_zero_or_partial_offload_never_passes_gpu_verification(self):
        path = self.root / "remote.log"
        process = MagicMock()
        process.poll.return_value = None
        for message in (
            "model loaded\n", "offloaded 0/33 layers to GPU\n",
            "offloaded 16/33 layers to GPU\n", "offloaded 0/0 layers to GPU\n",
        ):
            with self.subTest(message=message):
                path.write_text(message)
                clock = Clock()
                with self.assertRaises(JobError):
                    wait_for_gpu_offload(process, path, 0, clock.deadline(10))
                self.assertEqual(clock.value, 5 if message == "model loaded\n" else 0)

    def test_offload_wait_is_bounded_by_job_deadline_and_server_liveness(self):
        path = self.root / "remote.log"
        path.write_text("")
        process = MagicMock()
        process.poll.return_value = None
        clock = Clock()
        with self.assertRaises(DeadlineExceeded):
            wait_for_gpu_offload(process, path, 0, clock.deadline(0.2))
        self.assertEqual(clock.value, 0.2)
        process.poll.return_value = 1
        with self.assertRaises(JobError):
            wait_for_gpu_offload(process, path, 0, Deadline(10))


class LocalHttpInferenceTests(WorkerFixture):
    def setUp(self):
        super().setUp()
        binary = self.root / "llama.cpp/build/bin/llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_text("SYNTHETIC_BINARY_FIXTURE")
        self.requests = []
        self.processes = []
        self.servers = []
        self.threads = []
        self.spawn_arguments = None
        self.spawn_environment = None
        self.layers = "33/33"
        self.properties_status = 200
        self.properties = {"total_slots": 1, "default_generation_settings": {"n_ctx": 4096}}
        self.context_checks = []
        self.http_status = 200
        self.http_response = {
            "choices": [{"message": {"role": "assistant", "content": ANSWER}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }

    def spawn(self, args, **kwargs):
        self.spawn_arguments = args
        self.spawn_environment = kwargs["env"]
        verbosity = int(args[args.index("--verbosity") + 1]) if "--verbosity" in args else 3
        diagnostics = [
            f"0.00.115.121 I cmn common_param: verbosity = {verbosity}",
            "0.00.121.871 I srv load_model: loading model '/fixture/model.gguf'",
        ]
        if verbosity >= 4:
            diagnostics.append(f"0.01.897.000 I load_tensors: offloaded {self.layers} layers to GPU")
        diagnostics.extend([
            "0.01.962.488 I srv llama_server: model loaded",
            "0.01.962.495 I srv llama_server: listening on http://127.0.0.1",
        ])
        kwargs["stdout"].write(("\n".join(diagnostics) + "\n").encode())
        kwargs["stdout"].flush()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/props":
                    owner.context_checks.append(self.path)
                    self.send_response(owner.properties_status)
                    content = json.dumps(owner.properties).encode()
                else:
                    self.send_response(200)
                    content = b'{"status":"ok"}'
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(content)

            def do_POST(self):
                payload = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append((self.path, json.loads(payload), dict(self.headers)))
                self.send_response(owner.http_status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(owner.http_response).encode())

            def log_message(self, format, *args):
                pass

        port = int(args[args.index("--port") + 1])
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        self.servers.append(server)
        self.threads.append(thread)
        process = MagicMock()
        process.poll.return_value = None
        self.processes.append(process)
        return process

    def stop(self, process):
        for server, thread in zip(self.servers, self.threads):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        process.poll.return_value = 0

    def execute_inference(self, context_size=4096):
        request = {
            "hf_token": HF_TOKEN, "model": self.model, "gpu_name": "A100 SXM4",
            "prompt": "Literal 'quotes'; $(not-a-command)\nnext caf\u00e9",
            "context_size": context_size, "max_tokens": 1024, "temperature": 0.7,
        }
        with patch("remote_job.gpu_details",
                   return_value={"gpu_name": "NVIDIA A100-SXM4-40GB", "gpu_vram_mib": 40536}), \
             patch("huggingface_hub.hf_hub_download", side_effect=self.download), \
             patch("remote_job.subprocess.Popen", side_effect=self.spawn), \
             patch("remote_job.stop_process", side_effect=self.stop) as stop, \
             patch.dict(os.environ, {
                 "HF_TOKEN": HF_TOKEN, "VAST_AI_API_KEY": VAST_TOKEN,
                 "HTTP_PROXY": "http://invalid-proxy.invalid:3128", "NO_PROXY": "",
             }):
            try:
                return infer(self.root, request, Clock().deadline(30))
            finally:
                stop.assert_called_once()
                self.assertTrue(all(not thread.is_alive() for thread in self.threads))

    def test_local_http_path_writes_exact_text_without_credentials_or_proxy_egress(self):
        result = self.execute_inference()
        self.assertEqual(result["status"], "completed")
        self.assertEqual((self.root / "response.txt").read_bytes(), ANSWER.encode())
        self.assertEqual(result["response_sha256"], hashlib.sha256(ANSWER.encode()).hexdigest())
        path, body, headers = self.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["messages"], [{"role": "user", "content":
                                           "Literal 'quotes'; $(not-a-command)\nnext caf\u00e9"}])
        self.assertEqual(body["max_tokens"], 1024)
        self.assertNotIn(HF_TOKEN, json.dumps(body))
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("HF_TOKEN", self.spawn_environment)
        self.assertNotIn("VAST_AI_API_KEY", self.spawn_environment)
        self.assertNotIn(HF_TOKEN, " ".join(self.spawn_arguments))

    def test_training_context_cap_is_rejected_before_sending_a_prompt(self):
        self.properties["default_generation_settings"]["n_ctx"] = 32768
        with self.assertRaisesRegex(JobError, "32768.*65536"):
            self.execute_inference(context_size=65536)
        self.assertEqual(self.context_checks, ["/props"])
        self.assertEqual(self.requests, [])
        self.assertFalse((self.root / "response.txt").exists())

    def test_actual_larger_context_is_verified_before_generation(self):
        self.properties["default_generation_settings"]["n_ctx"] = 65536
        result = self.execute_inference(context_size=65536)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.context_checks, ["/props"])
        self.assertEqual(len(self.requests), 1)
        self.assertIn("single-slot context verified: 65536 tokens", (self.root / "remote.log").read_text())

    def test_context_properties_errors_missing_capacity_and_extra_slots_never_generate(self):
        for status, properties in (
            (500, self.properties),
            (200, {}),
            (200, {"total_slots": 1, "default_generation_settings": {"n_ctx": True}}),
            (200, {"total_slots": 2, "default_generation_settings": {"n_ctx": 4096}}),
        ):
            with self.subTest(status=status, properties=properties):
                self.properties_status, self.properties = status, properties
                with self.assertRaises(JobError):
                    self.execute_inference()
                self.assertEqual(self.requests, [])
                self.servers.clear()
                self.threads.clear()

    def test_pinned_runtime_trace_logging_confirms_gpu_before_generating(self):
        result = self.execute_inference()
        self.assertEqual(result["status"], "completed")
        self.assertIn("offloaded 33/33 layers to GPU", (self.root / "remote.log").read_text())
        self.assertEqual(len(self.requests), 1)
        self.assertEqual((self.root / "response.txt").read_text(), ANSWER)

    def test_cpu_or_partial_offload_cannot_pass_as_gpu_execution(self):
        for layers in ("0/33", "16/33", ""):
            with self.subTest(layers=layers):
                self.layers = layers
                self.servers.clear()
                self.threads.clear()
                with self.assertRaises(JobError):
                    self.execute_inference()
                self.assertEqual(self.requests, [])
                self.assertFalse((self.root / "response.txt").exists())

    def test_stale_gpu_log_cannot_validate_the_current_cpu_server(self):
        (self.root / "remote.log").write_text("old job: offloaded 33/33 layers to GPU\n")
        self.layers = "0/33"
        with self.assertRaises(JobError):
            self.execute_inference()
        self.assertEqual(self.requests, [])

    def test_inference_http_error_is_not_success_shaped(self):
        self.http_status = 400
        self.http_response = {"error": {"message": "context is too long"}}
        with self.assertRaises(JobError):
            self.execute_inference()
        self.assertFalse((self.root / "response.txt").exists())

    def test_empty_or_nonassistant_reply_is_rejected(self):
        self.http_response = {"choices": [{"message": {"role": "assistant", "content": ""},
                                          "finish_reason": "stop"}]}
        with self.assertRaises(JobError):
            self.execute_inference()
        self.assertFalse((self.root / "response.txt").exists())
