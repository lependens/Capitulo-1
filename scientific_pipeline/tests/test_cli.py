import json
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from siar_science.cli import main
from siar_science.pipeline import json_bytes
from tests_support import dataset_fixture, temporary_directory


class CLITests(unittest.TestCase):
    def test_module_help_smoke(self):
        result = subprocess.run([sys.executable, "-m", "siar_science", "--help"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("validate", "build-dataset", "calculate-pm", "calculate-hargreaves", "evaluate", "run-baselines"):
            self.assertIn(command, result.stdout)

    def test_synthetic_cli_success_failure_and_build_gate(self):
        with temporary_directory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root = Path(directory)
            path = root/"synthetic.json"
            path.write_bytes(json_bytes(dataset_fixture()))
            self.assertEqual(main(["run-baselines", "--dataset", str(path), "--output", str(root/"run")]), 0)
            self.assertEqual(main(["run-baselines", "--dataset", str(path), "--output", str(root/"run")]), 2)
            handoff = root/"not-authorized.json"
            handoff.write_text(json.dumps({"authorized": False}), encoding="utf-8")
            self.assertEqual(main(["build-dataset", "--handoff", str(handoff), "--output", str(root/"never-built.json")]), 2)
            self.assertFalse((root/"never-built.json").exists())
