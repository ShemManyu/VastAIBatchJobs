import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from agent_tools import ToolError, execute


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_real_file_actions_preserve_literal_text_and_complete_replacements(self):
        text = "caf\u00e9 'quotes'; $(not-a-command)\n"
        result = execute(self.root, {"action": "write_file", "path": "scripts/task.py", "content": text})
        self.assertEqual(result["bytes_written"], len(text.encode()))
        self.assertEqual(execute(self.root, {"action": "read_file", "path": "scripts/task.py"})["content"], text)
        execute(self.root, {"action": "write_file", "path": "scripts/task.py", "content": "short"})
        self.assertEqual((self.root / "scripts/task.py").read_bytes(), b"short")
        self.assertEqual(execute(self.root, {"action": "list_files", "path": "."})["entries"], ["scripts"])

    def test_file_actions_reject_traversal_absolute_paths_and_symlink_parents(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "alias").symlink_to(outside, target_is_directory=True)
        for path in ("../outside", "/etc/passwd", "a/../b", "a//b", "a\\b", "alias/new.txt"):
            with self.subTest(path=path), self.assertRaises((ToolError, OSError)):
                execute(self.root, {"action": "write_file", "path": path, "content": "bad"})
        self.assertEqual(list(outside.iterdir()), [])

    def test_linked_and_special_files_are_not_read_or_overwritten(self):
        original = self.root / "original"
        original.write_text("unchanged")
        symlink_target = self.root / "symlink-target"
        symlink_target.write_text("also unchanged")
        (self.root / "symlink").symlink_to(symlink_target)
        os.link(original, self.root / "hardlink")
        os.mkfifo(self.root / "fifo")
        for path in ("symlink", "hardlink", "fifo"):
            for action in ("read_file", "write_file"):
                with self.subTest(path=path, action=action), self.assertRaises((ToolError, OSError)):
                    execute(self.root, {"action": action, "path": path, "content": "bad"})
        self.assertEqual(original.read_text(), "unchanged")
        self.assertEqual(symlink_target.read_text(), "also unchanged")

    def test_utf8_pagination_reports_real_byte_boundaries(self):
        (self.root / "text").write_text("abcd\u00e9wxyz", encoding="utf-8")
        first = execute(self.root, {"action": "read_file", "path": "text"}, maximum=5)
        self.assertEqual(first["content"], "abcd")
        self.assertEqual(first["next_offset"], 4)
        second = execute(self.root, {"action": "read_file", "path": "text", "offset": first["next_offset"]}, maximum=5)
        self.assertEqual(second["content"], "\u00e9wxy")
        self.assertEqual(second["next_offset"], 9)
        third = execute(self.root, {"action": "read_file", "path": "text", "offset": 9}, maximum=5)
        self.assertEqual(third["content"], "z")
        self.assertFalse(third["truncated"])
        with self.assertRaises(ToolError):
            execute(self.root, {"action": "read_file", "path": "text", "offset": 100})

    def test_cli_argv_is_literal_data_not_shell_interpolation(self):
        literal = "$(not-a-command); 'quotes'\ncaf\u00e9"
        result = execute(self.root, {
            "action": "run", "argv": [sys.executable, "-c", "import json,sys; print(json.dumps(sys.argv[1:]))", literal],
        })
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(json.loads(result["stdout"]), [literal])
        self.assertFalse(result["timed_out"])

    def test_real_script_execution_and_nonzero_exit_are_observed(self):
        (self.root / "task.py").write_text("import sys\nprint(6 * 7)\nprint('failure detail', file=sys.stderr)\nsys.exit(7)\n")
        result = execute(self.root, {"action": "run", "argv": [sys.executable, "task.py"]})
        self.assertEqual(result["stdout"], "42\n")
        self.assertEqual(result["stderr"], "failure detail\n")
        self.assertEqual(result["exit_code"], 7)

    def test_command_output_is_bounded_without_blocking_a_noisy_process(self):
        result = execute(self.root, {
            "action": "run", "argv": [sys.executable, "-c", "import sys; sys.stdout.write('x'*100000); sys.stderr.write('y'*100000)"],
        }, maximum=1024)
        self.assertEqual(result["stdout"], "x" * 1024)
        self.assertEqual(result["stderr"], "y" * 1024)
        self.assertTrue(result["stdout_truncated"])
        self.assertTrue(result["stderr_truncated"])
        self.assertEqual(result["exit_code"], 0)

    def test_command_wall_clock_timeout_kills_the_actual_script(self):
        start = time.monotonic()
        result = execute(self.root, {
            "action": "run", "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
        }, timeout=0.1)
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["exit_code"], 0)
        self.assertLess(time.monotonic() - start, 3)

    def test_invalid_actions_limits_and_obfuscated_shell_expansions_fail(self):
        for action in (
            {"action": "unknown"}, {"action": "run", "argv": "echo bad"},
            {"action": "run", "argv": []}, {"action": "run", "argv": ["echo", "\0"]},
            {"action": "run", "argv": ["bash", "-c", "${x@P}"]},
            {"action": "run", "argv": ["sh", "-c", "eval unsafe"]},
        ):
            with self.subTest(action=action), self.assertRaises(ToolError):
                execute(self.root, action)
        for limit in (0, -1, True, 20000):
            with self.assertRaises(ToolError):
                execute(self.root, {"action": "list_files"}, maximum=limit)


if __name__ == "__main__":
    unittest.main()
