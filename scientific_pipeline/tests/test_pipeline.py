from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from siar_science.pipeline import calculate, execute_run, json_bytes
from tests_support import dataset_fixture, temporary_directory


class PipelineTests(unittest.TestCase):
    def test_independent_models_no_mutation_no_reference_dependency(self):
        dataset = dataset_fixture()
        original = deepcopy(dataset)
        result = calculate(dataset)
        self.assertEqual(dataset, original)
        self.assertEqual(result["audit_source_repo_sha"], original["audit_source_repo_sha"])
        self.assertEqual(result["source_repo_sha"], original["source_repo_sha"])
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(result["missing_days"], original["missing_days"])
        rows = result["rows"]
        self.assertAlmostEqual(rows[0]["ET0_FAO56"], 3.9, delta=.05)
        self.assertEqual(rows[0]["values"]["TempMedia"], 99)
        self.assertEqual(rows[0]["model_diagnostics"]["PM_FAO56_V1"]["tmean_calc_c"], 16.9)
        self.assertIsNone(rows[1]["ET0_FAO56"])
        self.assertIsNotNone(rows[1]["ET0_Hargreaves"])
        self.assertIsNotNone(rows[2]["ET0_FAO56"])
        self.assertIn("wind_height_assumed_2m", rows[0]["scientific_flags"])

    def test_full_synthetic_run_manifest_hashes_and_cohorts(self):
        with temporary_directory() as directory:
            root = Path(directory)
            path = root/"synthetic.json"
            content = json_bytes(dataset_fixture())
            path.write_bytes(content)
            manifest = execute_run(path, root/"run")
            self.assertEqual(manifest["status"], "SUCCESS")
            self.assertEqual(manifest["dataset_sha256"], hashlib.sha256(content).hexdigest())
            for key in ("audit_source_repo_sha", "source_repo_sha"):
                self.assertEqual(manifest[key], dataset_fixture()[key])
                predictions = json.loads((root/"run/predictions.json").read_text(encoding="utf-8"))
                self.assertEqual(predictions[key], manifest[key])
                scores = json.loads((root/"run/metrics.json").read_text(encoding="utf-8"))
                self.assertTrue(all(score[key] == manifest[key] for score in scores))
            self.assertEqual(manifest["assumptions"]["wind_height_assumption"], "2 m / SIAR convention")
            self.assertEqual(manifest["counts"]["PM_FAO56_V1"]["calculated"], 2)
            self.assertEqual(manifest["counts"]["HS_RA_V1"]["calculated"], 3)
            self.assertEqual(manifest["counts"]["PM_FAO56_V1"]["evaluated"], 1)
            self.assertEqual(manifest["counts"]["HS_RA_V1"]["evaluated"], 2)
            for artifact in manifest["outputs"].values():
                self.assertEqual(hashlib.sha256((root/"run"/artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])
            self.assertEqual(path.read_bytes(), content)
            with self.assertRaisesRegex(ValueError, "already exists"):
                execute_run(path, root/"run")

    def test_independent_stages_and_evaluate_same_result(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = root/"synthetic.json"
            source.write_bytes(json_bytes(dataset_fixture()))
            for command, target in (("calculate-pm", "pm"), ("calculate-hargreaves", "hs"), ("run-baselines", "all")):
                self.assertEqual(execute_run(source, root/target, command)["status"], "SUCCESS")
            manifest = execute_run(source, root/"evaluation", "evaluate", prediction_paths=[root/"pm/predictions.json", root/"hs/predictions.json"])
            self.assertEqual(manifest["status"], "SUCCESS")
            self.assertEqual(len(manifest["prediction_inputs"]), 2)
            for key in ("audit_source_repo_sha", "source_repo_sha"):
                self.assertEqual(manifest[key], dataset_fixture()[key])
            self.assertIn("source_files_sha256", manifest["prediction_inputs"][0]["producer"]["pipeline"])
            self.assertEqual((root/"evaluation/metrics.json").read_bytes(), (root/"all/metrics.json").read_bytes())
            single = execute_run(source, root/"single", "evaluate", prediction_paths=[root/"pm/predictions.json"])
            single_scores = json.loads((root/"single/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(single["cohorts"]["paired_evaluation"]["n"], 0)
            self.assertTrue(all(r["n"] == 0 for r in single_scores if r["cohort"] == "paired"))
            altered = json.loads((root/"pm/predictions.json").read_text(encoding="utf-8"))
            altered["producer"]["parameters"]["PM_FAO56_V1"]["albedo"] = .25
            (root/"wrong-parameters.json").write_bytes(json_bytes(altered))
            rejected = execute_run(source, root/"parameter-mismatch", "evaluate", prediction_paths=[root/"wrong-parameters.json"])
            self.assertEqual(rejected["status"], "FAILED")
            self.assertIn("parameters", rejected["error"])
            self.assertIn("sha256", rejected["prediction_inputs"][0])
            altered = json.loads((root/"pm/predictions.json").read_text(encoding="utf-8"))
            altered["audit_source_repo_sha"] = "f"*40
            (root/"wrong-audit-identity.json").write_bytes(json_bytes(altered))
            rejected = execute_run(source, root/"audit-mismatch", "evaluate", prediction_paths=[root/"wrong-audit-identity.json"])
            self.assertEqual(rejected["status"], "FAILED")
            self.assertIn("repository provenance", rejected["error"])
            # Changing only dataset identity still invalidates predictions linkage.
            data = dataset_fixture()
            data["dataset_id"] += "-changed"
            source.write_bytes(json_bytes(data))
            bad = execute_run(source, root/"bad", "evaluate", prediction_paths=[root/"pm/predictions.json"])
            self.assertEqual(bad["status"], "FAILED")
            self.assertIn("different dataset", bad["error"])

    def test_validation_does_not_calculate_and_failed_runs_remain_explicit(self):
        with temporary_directory() as directory:
            root = Path(directory)
            path = root/"synthetic.json"
            path.write_bytes(json_bytes(dataset_fixture()))
            with patch("siar_science.pipeline.fao56_daily", side_effect=ValueError("controlled failure")):
                self.assertEqual(execute_run(path, root/"validation", "validate")["status"], "SUCCESS")
                failed = execute_run(path, root/"failure")
            self.assertEqual(failed["status"], "FAILED")
            self.assertEqual(failed["counts"]["PM_FAO56_V1"]["failed"], 2)
            self.assertTrue((root/"failure/manifest.json").is_file())
            path.write_text("{}", encoding="utf-8")
            invalid = execute_run(path, root/"invalid")
            self.assertEqual(invalid["status"], "FAILED")
            self.assertEqual(invalid["outputs"], {})

    def test_protected_outputs_and_positive_height_evidence_rejected(self):
        with temporary_directory() as directory:
            root = Path(directory)
            path = root/"synthetic.json"
            path.write_bytes(json_bytes(dataset_fixture()))
            for target in ("datos_siar_baleares", "raw", "RAW"):
                with self.assertRaisesRegex(ValueError, "reference"):
                    execute_run(path, root/target/"run")
            data = dataset_fixture()
            data["stations"]["SYNTH"]["wind_height_m"] = 10
            with self.assertRaisesRegex(ValueError, "wind-height"):
                calculate(data)
            self.assertIsNotNone(calculate(data, methods=("hargreaves",))["rows"][0]["ET0_Hargreaves"])
