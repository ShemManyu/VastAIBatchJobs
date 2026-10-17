import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from job_utils import (
    Deadline, DeadlineExceeded, JobError, atomic_write, completion_text,
    model_filename, model_filenames, number, reasoning_token_budget, redact, run_command, write_json,
)
from tests.fakes import ANSWER, Clock, HF_TOKEN, VAST_TOKEN


class UtilityTests(unittest.TestCase):
    def test_deadline_exact_boundary_and_sleep_are_measurable(self):
        clock = Clock()
        deadline = clock.deadline(10)
        self.assertEqual(deadline.remaining(), 10)
        clock.sleep(9)
        self.assertEqual(deadline.timeout(20), 1)
        with self.assertRaises(DeadlineExceeded):
            deadline.sleep(20)
        self.assertEqual(clock.value, 10)

    def test_nonfinite_or_invalid_limits_fail(self):
        for value in (float("nan"), float("inf"), True, "4"):
            with self.subTest(value=value), self.assertRaises(JobError):
                number(value, "price", positive=True)
        for value in (0, -1, float("nan")):
            with self.subTest(value=value), self.assertRaises(JobError):
                Deadline(value)

    def test_paths_and_sidecars_are_rejected(self):
        for name in ("/model.gguf", "../model.gguf", "dir/../model.gguf", "a//model.gguf",
                     "model.bin", "mmproj-model.gguf", "adapter.gguf", "lora.gguf", "model-lora.gguf",
                     "model-00001-of-00002.gguf"):
            with self.subTest(name=name), self.assertRaises(JobError):
                model_filename(name)
        self.assertEqual(model_filename("subdir/8B-Q4_K_M.gguf"), "subdir/8B-Q4_K_M.gguf")
        self.assertEqual(model_filename("floral-8b.gguf"), "floral-8b.gguf")

    def test_reasoning_budget_reserves_output_and_rejects_invalid_types(self):
        for value in (None, 0, 1, 1023):
            self.assertEqual(reasoning_token_budget(value, 1024), value)
        for value in (True, False, -1, 1024, 1025, 1.0, "512"):
            with self.subTest(value=value), self.assertRaises(JobError):
                reasoning_token_budget(value, 1024)

    def test_split_model_names_are_a_complete_ordered_safe_set(self):
        self.assertEqual(model_filenames("quant/model-00001-of-00003.gguf"), [
            "quant/model-00001-of-00003.gguf",
            "quant/model-00002-of-00003.gguf",
            "quant/model-00003-of-00003.gguf",
        ])
        self.assertEqual(model_filenames("model.gguf"), ["model.gguf"])
        for name in (
            "model-00002-of-00003.gguf", "model-00001-of-00001.gguf",
            "model-00001-of-00129.gguf", "model-1-of-3.gguf",
            "../model-00001-of-00003.gguf", "mmproj-model-00001-of-00003.gguf",
        ):
            with self.subTest(name=name), self.assertRaises(JobError):
                model_filenames(name)

    def test_error_redaction_removes_secrets_and_signed_url_parameters(self):
        text = (
            f"failed https://user:password@unit.invalid/path?api_key={HF_TOKEN}&Signature=other "
            f"Authorization: Bearer {VAST_TOKEN}"
        )
        clean = redact(text, (HF_TOKEN, VAST_TOKEN))
        self.assertNotIn(HF_TOKEN, clean)
        self.assertNotIn(VAST_TOKEN, clean)
        self.assertNotIn("Signature", clean)
        self.assertNotIn("password", clean)
        self.assertIn("https://unit.invalid/path", clean)
        self.assertEqual(
            redact(f"https://[::1]:9/path?token={HF_TOKEN}&Signature=x", (HF_TOKEN,)),
            "https://[::1]:9/path",
        )

    def test_atomic_files_are_complete_and_private(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "response.txt"
            atomic_write(path, ANSWER.encode())
            self.assertEqual(path.read_bytes(), ANSWER.encode())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(folder).glob("*.part")), [])
            write_json(Path(folder) / "metadata.json", {"response": ANSWER})
            self.assertEqual(json.loads((Path(folder) / "metadata.json").read_text())["response"], ANSWER)

    def test_completion_requires_real_assistant_text_and_finish_reason(self):
        choice = {"message": {"role": "assistant", "content": ANSWER}, "finish_reason": "stop"}
        self.assertEqual(completion_text({"choices": [choice]}), ANSWER)
        for value in ({}, {"choices": []}, {"choices": [{"message": None}]},
                      {"choices": [{"message": {"role": "user", "content": "not an answer"},
                                    "finish_reason": "stop"}]}):
            with self.subTest(value=value), self.assertRaises(JobError):
                completion_text(value)

    def test_subprocess_timeout_actually_terminates_the_owned_process(self):
        processes = []
        real_popen = subprocess.Popen

        def capture(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        with patch("job_utils.subprocess.Popen", side_effect=capture):
            with self.assertRaises(DeadlineExceeded):
                run_command(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    Deadline(0.15), context="fixture timeout",
                )
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertLess(processes[0].returncode, 0)

    def test_subprocess_error_text_is_sanitized(self):
        script = f"import sys; sys.stderr.write('https://unit.invalid/api?token={HF_TOKEN}'); sys.exit(2)"
        with self.assertRaises(JobError) as caught:
            run_command([sys.executable, "-c", script], Deadline(10),
                        context="fixture command", secrets=(HF_TOKEN,))
        self.assertNotIn(HF_TOKEN, str(caught.exception))
        self.assertIn("exit 2", str(caught.exception))
