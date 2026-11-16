"""Controlled mathematical and full three-model regression/export fixtures."""
import csv
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
import math
from pathlib import Path
import unittest

from siar_science.cli import main
from siar_science.metrics import evaluate
from siar_science.pipeline import calculate, execute_run, json_bytes
from siar_science.quality import eligibility
from siar_science.valiantzas import estimated_solar_radiation, valiantzas
from tests_support import dataset_fixture, temporary_directory


def fixture():
    data = dataset_fixture()
    for row in data["rows"]:
        row["values"]["HumedadMedia"] = 70
    return data


class ValiantzasMathTests(unittest.TestCase):
    def test_composition_and_expanded_coefficient(self):
        for rh in (0, 30, 70, 100):
            for ra in (0, 15, 40):
                for lo, hi in ((10, 25), (20, 20), (-25, -20)):
                    actual = valiantzas(tmin_c=lo, tmax_c=hi, rhmean_pct=rh, ra_mj_m2_day=ra)
                    self.assertEqual(actual, valiantzas(tmin_c=lo, tmax_c=hi, rhmean_pct=rh, ra_mj_m2_day=ra))
                    expected = .001861704 * ra * (hi-lo)**.3 * (1.001-rh/100)**.2 * ((hi+lo)/2+17.8)
                    self.assertAlmostEqual(actual, expected, places=13)
        # Independent hand-calculation: TD=1 and RH=.1% make both powers 1.
        self.assertAlmostEqual(valiantzas(tmin_c=19.5, tmax_c=20.5, rhmean_pct=.1,
                                         ra_mj_m2_day=40), 2.814896448, places=12)

    def test_boundaries_units_and_no_clipping(self):
        base = dict(tmin_c=10, tmax_c=25, rhmean_pct=100, ra_mj_m2_day=30)
        self.assertGreater(valiantzas(**base), 0)
        self.assertEqual(valiantzas(**dict(base, tmax_c=10)), 0)
        self.assertEqual(valiantzas(**dict(base, ra_mj_m2_day=0)), 0)
        self.assertLess(valiantzas(**dict(base, tmin_c=-25, tmax_c=-20)), 0)
        a = estimated_solar_radiation(**base)
        self.assertAlmostEqual(valiantzas(**base), .0135*.408*a*(17.5+17.8))
        self.assertAlmostEqual(valiantzas(**dict(base, ra_mj_m2_day=60)), 2*valiantzas(**base))

    def test_invalid_inputs(self):
        base = dict(tmin_c=10, tmax_c=25, rhmean_pct=70, ra_mj_m2_day=30)
        for key, value in (("tmin_c", -273.15), ("tmax_c", 9), ("rhmean_pct", -1),
                           ("rhmean_pct", 101), ("ra_mj_m2_day", -1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                valiantzas(**dict(base, **{key: value}))
        for key in base:
            for value in (None, True, float("inf"), float("nan"), "20"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    valiantzas(**dict(base, **{key: value}))


class ClassicalIntegrationTests(unittest.TestCase):
    def test_independent_eligibility_inputs_and_no_mutation(self):
        data = fixture()
        original = deepcopy(data)
        result = calculate(data, methods=("pm", "hargreaves", "valiantzas"))
        self.assertEqual(data, original)
        self.assertIsNone(result["rows"][1]["ET0_FAO56"])
        self.assertIsNotNone(result["rows"][1]["ET0_Valiantzas"])
        row, station = data["rows"][0], data["stations"]["SYNTH"]
        for key in ("VelViento", "Radiacion", "HumedadMin", "HumedadMax", "TempMedia", "EtPMon"):
            row["values"][key] = None
        station["altitude_m"] = None
        station["wind_height_m"] = 10
        self.assertEqual(eligibility(row, station, "VALIANTZAS_V1"), [])
        self.assertIsNotNone(calculate(data, ("valiantzas",))["rows"][0]["ET0_Valiantzas"])
        row["values"]["HumedadMedia"] = None
        self.assertIn("missing_or_nonfinite:HumedadMedia", eligibility(row, station, "valiantzas"))
        self.assertEqual(eligibility(row, station, "hargreaves"), [])
        row["values"]["HumedadMedia"] = 101
        self.assertIn("humidity_out_of_range:HumedadMedia", eligibility(row, station, "valiantzas"))
        station["latitude_deg"] = 90
        self.assertIn("invalid_latitude", eligibility(row, station, "valiantzas"))

    def test_all_cohorts_are_explicit_and_pm_hs_backward_compatible(self):
        data = fixture()
        result = calculate(data, tuple(("pm", "hargreaves", "valiantzas")))
        result["rows"][2]["ET0_Valiantzas"] = None
        scores = evaluate(result["rows"], ("ET0_FAO56", "ET0_Hargreaves", "ET0_Valiantzas"), classical=True)
        global_scores = {(s["cohort"], s["method"]): s for s in scores if s["group"] == "global"}
        self.assertEqual(global_scores["own", "ET0_Valiantzas"]["n"], 2)
        self.assertEqual(global_scores["PM_HS_VAL", "ET0_FAO56"]["n"], 1)
        self.assertEqual(global_scores["HS_VAL", "ET0_Hargreaves"]["n"], 2)
        self.assertNotIn(("HS_VAL", "ET0_FAO56"), global_scores)
        default = evaluate(result["rows"])
        self.assertTrue(all(s["cohort"] in ("own", "paired") for s in default))
        before = calculate(data)
        for a, b in zip(before["rows"], result["rows"]):
            self.assertEqual(a["ET0_FAO56"], b["ET0_FAO56"])
            self.assertEqual(a["ET0_Hargreaves"], b["ET0_Hargreaves"])

    def test_cli_staged_merge_export_hashes_null_zero_and_provenance(self):
        with temporary_directory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            data = fixture()
            source = root/"data.json"
            original = json_bytes(data)
            source.write_bytes(original)
            for command, target in (("run-baselines", "pmhs"), ("calculate-valiantzas", "val"),
                                    ("run-classical-baselines", "all")):
                self.assertEqual(main([command, "--dataset", str(source), "--output", str(root/target)]), 0)
            result = execute_run(source, root/"merged", "evaluate-classical", prediction_paths=[
                root/"pmhs/predictions.json", root/"val/predictions.json"])
            self.assertEqual(result["status"], "SUCCESS", result.get("error"))
            self.assertEqual(result["counts"]["VALIANTZAS_V1"]["calculated"], 3)
            self.assertEqual(result["cohorts"]["paired_evaluation"]["n"], 1)
            self.assertEqual(result["cohorts"]["PM_HS_VAL:evaluation"]["n"], 1)
            self.assertEqual((root/"merged/metrics.json").read_bytes(), (root/"all/metrics.json").read_bytes())
            self.assertEqual(len(result["prediction_inputs"]), 2)
            for artifact in result["outputs"].values():
                self.assertEqual(hashlib.sha256((root/"merged"/artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])
            csv_path = root/"merged/classical-baselines.csv"
            self.assertTrue(csv_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["EtPMon"], "0")
            self.assertEqual(rows[0]["evaluated_VAL"], "true")
            self.assertEqual(rows[1]["ET0_PM_FAO56_V1"], "")
            self.assertEqual(rows[1]["eligible_VAL"], "true")
            self.assertEqual(rows[2]["evaluated_VAL"], "false")
            self.assertEqual(json.loads(rows[0]["provenance"]), data["rows"][0]["provenance"])
            for row in rows:
                for key in ("audit_source_repo_sha", "source_repo_sha", "dataset_id"):
                    self.assertEqual(row[key], data[key])
                self.assertEqual(row["dataset_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(source.read_bytes(), original)
            rejected = execute_run(source, root/"incomplete", "evaluate-classical", prediction_paths=[root/"val/predictions.json"])
            self.assertEqual(rejected["status"], "FAILED")
            self.assertFalse((root/"incomplete/classical-baselines.csv").exists())
