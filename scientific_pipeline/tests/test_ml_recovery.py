"""Recovery-specific ordering and non-training tests using controlled fixtures."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from siar_science import ml_recovery
from siar_science.pipeline import json_bytes
from siar_science.serialization import digest
from tests_support import temporary_directory


class MLRecoveryTests(unittest.TestCase):
    def _frozen_source(self):
        selected = [{"model_id": "ML_LINEAR__FEATURE_SET_T", "candidate_id": "linear-t",
                     "algorithm": "LINEAR", "feature_set": "FEATURE_SET_T", "features": ["tmin", "tmax"], "parameters": {}}]
        protocol = {"development_period": ["2004-01-01", "2020-12-31"], "test_period": ["2021-01-01", "2025-12-31"],
                    "feature_sets": ["FEATURE_SET_T"], "test_status": "retrospective_locked"}
        bundle = {"models": {"linear-t": object()}}
        return (Path("/frozen"), Path("/frozen/.staging"), {"selected": selected}, "a" * 64,
                {}, "b" * 64, [], "c" * 64, bundle, "d" * 64, protocol, "e" * 64)

    def _identity(self):
        return {"dataset_id": "fixture", "dataset_sha256": digest(b"fixture dataset"), "audit_source_repo_sha": "a" * 40,
                "source_repo_sha": "b" * 40, "quality_post": {}, "source_data_hashes": {}, "units": {},
                "catalog": {}, "protocol_sha256": "e" * 64, "protocol_file_sha256": "e" * 64,
                "ahc_test_manifest_sha256": "c" * 64, "split_assignment_sha256": "9" * 64}

    def test_journal_is_durable_before_test_loader_and_no_fit_or_selection(self):
        with temporary_directory() as directory:
            root = Path(directory)
            (root / "protocol").write_text("{}", encoding="utf-8")
            (root / "dataset").write_bytes(b"fixture dataset")
            (root / "ahc").write_bytes(b"fixture ahc")
            def load_test(*args, **kwargs):
                journal = [json.loads(line)["state"] for line in (root / "run" / "recovery_journal.jsonl").read_text().splitlines()]
                self.assertEqual(journal[-2:], ["PRE_TEST_VERIFIED", "TEST_ACCESS_INTENT_RECORDED"])
                return {}, [], self._frozen_source()[-2], self._identity()
            with patch.object(ml_recovery, "_validate_frozen_source", return_value=self._frozen_source()), \
                 patch.object(ml_recovery, "_file_sha256", return_value="0" * 64), \
                 patch.object(ml_recovery, "_context_for_split", side_effect=load_test), \
                 patch.object(ml_recovery, "_test_predictions", return_value=[]), \
                 patch.object(ml_recovery, "score_predictions", return_value=[]), \
                 patch.object(ml_recovery, "_write_recovery_outputs") as writer:
                result = ml_recovery.execute_ml_test_recovery(root / "source", root / "dataset", root / "ahc", root / "protocol", root / "run")
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["stage"], "retrospective_recovery_evaluation")
            self.assertEqual(result["prior_attempt_status"], "TEST_ACCESS_UNKNOWN")
            writer.assert_called_once()
            states = [json.loads(line)["state"] for line in (root / "run" / "recovery_journal.jsonl").read_text().splitlines()]
            self.assertEqual(states, ["RECOVERY_STARTED", "PRE_TEST_VERIFIED", "TEST_ACCESS_INTENT_RECORDED", "TEST_OPENED", "TEST_EVALUATION_COMPLETED", "SUCCESS"])
            self.assertFalse(hasattr(ml_recovery, "_fit_full_development"))
            self.assertFalse(hasattr(ml_recovery, "_frozen_selection"))
            manifest_path = root / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            self.assertEqual(digest(manifest_path.read_bytes()), (root / "run" / "manifest.sha256").read_text().split()[0])
            for entry in manifest["outputs"].values():
                self.assertEqual(digest((root / "run" / entry["path"]).read_bytes()), entry["sha256"])

    def test_test_loader_interruption_leaves_durable_failed_journal(self):
        with temporary_directory() as directory:
            root = Path(directory)
            (root / "protocol").write_text("{}", encoding="utf-8")
            (root / "dataset").write_bytes(b"fixture dataset")
            (root / "ahc").write_bytes(b"fixture ahc")
            def interrupted(*args, **kwargs):
                states = [json.loads(line)["state"] for line in (root / "run" / "recovery_journal.jsonl").read_text().splitlines()]
                self.assertEqual(states[-1], "TEST_ACCESS_INTENT_RECORDED")
                raise RuntimeError("synthetic interruption")
            with patch.object(ml_recovery, "_validate_frozen_source", return_value=self._frozen_source()), \
                 patch.object(ml_recovery, "_file_sha256", return_value="0" * 64), \
                 patch.object(ml_recovery, "_context_for_split", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    ml_recovery.execute_ml_test_recovery(root / "source", root / "dataset", root / "ahc", root / "protocol", root / "run")
            manifest = json.loads((root / "run" / "manifest.json").read_bytes())
            self.assertEqual(manifest["status"], "FAILED")
            self.assertFalse(manifest["test_inspected"])
            states = [json.loads(line)["state"] for line in (root / "run" / "recovery_journal.jsonl").read_text().splitlines()]
            self.assertEqual(states[-1], "FAILED")
            self.assertIn("TEST_ACCESS_INTENT_RECORDED", states)

    def test_frozen_artifact_failure_never_calls_test_loader(self):
        with temporary_directory() as directory:
            root = Path(directory)
            with patch.object(ml_recovery, "_validate_frozen_source", side_effect=ValueError("selection hash mismatch")), \
                 patch.object(ml_recovery, "_context_for_split") as loader:
                with self.assertRaisesRegex(ValueError, "selection hash mismatch"):
                    ml_recovery.execute_ml_test_recovery(root / "source", root / "dataset", root / "ahc", root / "protocol", root / "run")
            loader.assert_not_called()
            manifest = json.loads((root / "run" / "manifest.json").read_bytes())
            self.assertEqual(manifest["status"], "FAILED")
            self.assertEqual(manifest["outputs"], {})
            states = [json.loads(line)["state"] for line in (root / "run" / "recovery_journal.jsonl").read_text().splitlines()]
            self.assertEqual(states, ["RECOVERY_STARTED", "FAILED"])

    def test_output_path_is_append_only(self):
        with temporary_directory() as directory:
            root = Path(directory)
            (root / "existing").mkdir()
            with self.assertRaises(ValueError):
                ml_recovery.execute_ml_test_recovery(root / "source", root / "dataset", root / "ahc", root / "protocol", root / "existing")


if __name__ == "__main__":
    unittest.main()
