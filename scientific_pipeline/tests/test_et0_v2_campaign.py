import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from siar_science.et0_v2 import load_protocol
from siar_science.et0_v2_campaign import _fit_cell, read_json, sha_file, shard_keys
from siar_science.et0_v2_models import fit_ahc, make_cell, selections
from siar_science.et0_v2_retrospective import CONTRAST, evaluate, retrospective_metrics, write_csv


ROOT = Path(__file__).resolve().parents[2]
FAMILY = "HG_RS_V1"
KEY = ("IB02", FAMILY, "annual")


def small_rows():
    return [{"date": f"{year}-01-{day:02d}", "station_code": "IB02",
             "inputs": {FAMILY: [10 + day, 9 + day]}, "target": day * 2.,
             "baselines": {FAMILY: {"estimate": float(day)}}}
            for year in range(2017, 2020) for day in range(1, 8)]


class V2ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, _ = load_protocol(ROOT)
        cls.ml = read_json(ROOT / cls.protocol["ml"]["protocol_path"])

    def test_outer_ahc_coefficients_depend_only_on_train_and_use_common_keys(self):
        rows = small_rows()
        cell = make_cell(rows, KEY, self.protocol)
        ahc = fit_ahc(cell, cell.plans)
        self.assertEqual(ahc["groups"]["annual"]["coefficient"], 2)
        self.assertTrue(all(item["rmse"] == 0 and item["n"] == 7 for item in ahc["folds"]))
        mutated = [{**row, "target": 1000.} if row["date"].startswith("2017") else row for row in rows]
        other = make_cell(mutated, KEY, self.protocol)
        held = fit_ahc(other, other.plans)["folds"][0]
        self.assertEqual(held["coefficients"], ahc["folds"][0]["coefficients"])
        self.assertEqual(held["keys_sha256"], ahc["folds"][0]["keys_sha256"])
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT"):
            make_cell([*rows, {**rows[0], "date": "2020-01-01"}], KEY, self.protocol)

    def test_candidate_resume_checksums_and_a_e_remain_separate(self):
        ann = {**self.protocol["ann"], "architectures": [[2], [3]],
               "early_stopping": {**self.protocol["ann"]["early_stopping"], "max_epochs": 2}}
        cell = make_cell(small_rows(), KEY, {**self.protocol, "ann": ann})
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "A"
            self.assertEqual(_fit_cell("ANN_A", cell, path, self.ml, "pilot", 1), (1, 0, False))
            checkpoint = path / "candidates" / "ann-2.json"
            previous = sha_file(checkpoint)
            self.assertEqual(_fit_cell("ANN_A", cell, path, self.ml, "pilot", None), (1, 1, True))
            self.assertEqual(sha_file(checkpoint), previous)
            other = Path(tmp) / "E"
            self.assertEqual(_fit_cell("ANN_E", cell, other, self.ml, "pilot", None), (2, 0, True))
            self.assertIn("ANN_A.joblib", read_json(path / "frozen.json")["models"])
            self.assertIn("ANN_E.joblib", read_json(other / "frozen.json")["models"])
            checkpoint.write_text("corrupted", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum|hash"):
                _fit_cell("ANN_A", cell, path, self.ml, "pilot", None)

    def test_retrospective_unopened_without_freeze_and_coverage_separate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Freeze"):
                evaluate(ROOT, root / "capsule", root, root / "never-open.json",
                         expected_freeze_sha="irrelevant", output=root / "results")
            self.assertFalse((root / "retrospective-access.json").exists())
            row = {"station_code": "IB02", "date": "2020-01-01", "target": 2.0, **dict.fromkeys(CONTRAST, 3.0)}
            others = {**row, "date": "2020-01-02", "ANN_E": None}
            table = list(retrospective_metrics([row, others], CONTRAST,
                                               {("station", "IB02"): 4, ("station", "IB09"): 0}))
            formula, ann_e = table[0], table[4]
            self.assertEqual(formula["common_keys_sha256"], ann_e["common_keys_sha256"])
            self.assertEqual((formula["n"], formula["bias"], formula["coverage_calendar"]), (1, 1., .5))
            self.assertEqual((ann_e["n"], ann_e["coverage_calendar"]), (1, .25))
            self.assertIsNone(table[-1]["coverage_calendar"])
            write_csv(root / "pred.csv.gz", [row])
            self.assertIn("2020-01-01", gzip.decompress((root / "pred.csv.gz").read_bytes()).decode())

    def test_shards_disjoint_and_ml_principal_is_one_of_frozen_algorithm_winners(self):
        keys = list(range(105))
        parts = [shard_keys(keys, (i, 4)) for i in range(4)]
        self.assertEqual(sorted(key for group in parts for key in group), keys)
        with self.assertRaisesRegex(ValueError, "Invalid V2 shard"):
            shard_keys(keys, (4, 4))
        cell = make_cell(small_rows(), KEY, self.protocol)
        algorithms = self.protocol["ml"]["complexity_rank"]
        summaries = [{"algorithm": algorithm, "candidate_id": algorithm,
                      "folds": [{"year": year, "rmse": 1 + index} for year in range(2017, 2020)]}
                     for index, algorithm in enumerate(algorithms)]
        principal, winners = selections("ML", cell, summaries)
        self.assertEqual(principal["candidate_id"], algorithms[0])
        self.assertIn(principal["candidate_id"], {item["candidate_id"] for item in winners.values()})
