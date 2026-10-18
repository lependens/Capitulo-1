import math
import unittest
from copy import deepcopy
from siar_science.metrics import evaluate, metrics
from tests_support import dataset_fixture


class MetricsTests(unittest.TestCase):
    def test_zero_reference_signed_bias_and_negative_r2(self):
        score = metrics([0, 2], [1, 0])
        self.assertEqual(score, dict(n=2, mae=1.5, rmse=math.sqrt(2.5), bias=-.5, r2=-1.5))

    def test_empty_constant_single_and_nonfinite(self):
        self.assertEqual(metrics([], []), dict(n=0, mae=None, rmse=None, bias=None, r2=None))
        self.assertIsNone(metrics([2, 2], [2, 2])["r2"])
        self.assertIsNone(metrics([2], [2])["r2"])
        self.assertEqual(metrics([0, None, float("inf"), 2], [1, 3, 4, float("nan")])["n"], 1)
        with self.assertRaises(ValueError):
            metrics([0], [])
        self.assertTrue(all(r["n"] == 0 and r["mae"] is None for r in evaluate([])))

    def test_own_paired_and_pooled_global(self):
        rows = dataset_fixture()["rows"]
        rows[0].update(ET0_FAO56=1, ET0_Hargreaves=3)
        rows[1].update(ET0_FAO56=None, ET0_Hargreaves=4)
        rows[1]["station_code"] = "SECOND"
        scores = evaluate(rows)
        global_scores = {(r["cohort"], r["method"]): r for r in scores if r["group"] == "global"}
        self.assertEqual(global_scores["own", "ET0_FAO56"]["n"], 1)
        self.assertEqual(global_scores["own", "ET0_Hargreaves"]["n"], 2)
        self.assertEqual(global_scores["own", "ET0_Hargreaves"]["rmse"], math.sqrt(6.5))
        self.assertEqual(global_scores["paired", "ET0_Hargreaves"]["n"], 1)
        self.assertEqual(global_scores["paired", "ET0_FAO56"]["n"], 1)

    def test_december_season_year_and_invalid_reference(self):
        row = deepcopy(dataset_fixture()["rows"][0])
        row.update(date="2001-12-01", ET0_FAO56=1, ET0_Hargreaves=1)
        scores = evaluate([row])
        self.assertTrue(any(s["group"] == "season_year" and s["key"] == "2002-DJF" for s in scores))
        row["values"]["EtPMon"] = -1
        self.assertTrue(all(s["n"] == 0 for s in evaluate([row])))
