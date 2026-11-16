"""Reference arithmetic, independent eligibility, four-way lineage and CSV."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from siar_science.cli import main
from siar_science.hargreaves_rs import hargreaves_rs
from siar_science.metrics import evaluate, FOUR_FIELDS, FOUR_COHORTS
from siar_science.pipeline import calculate, execute_run, json_bytes
from siar_science.quality import eligibility
from tests_support import dataset_fixture, temporary_directory


def fixture():
    data = dataset_fixture()
    for row in data["rows"]:
        row["values"]["HumedadMedia"] = 70
    return data


class HargreavesRsMathTests(unittest.TestCase):
    def test_hand_reference_units_and_determinism(self):
        self.assertAlmostEqual(hargreaves_rs(tmean_c=20, solar_radiation_mj_m2_day=20), 4.164048, places=12)
        for t in (-20, 0, 20, 40):
            for rs in (0, 1, 20, 35):
                inputs = dict(tmean_c=t, solar_radiation_mj_m2_day=rs)
                value = hargreaves_rs(**inputs)
                self.assertEqual(value, hargreaves_rs(**inputs))
                self.assertAlmostEqual(value, .005508*rs*(t+17.8), places=12)
                self.assertAlmostEqual(hargreaves_rs(**dict(inputs, solar_radiation_mj_m2_day=rs*2)), value*2)

    def test_zero_radiation_and_signed_cold_regime(self):
        self.assertEqual(hargreaves_rs(tmean_c=20, solar_radiation_mj_m2_day=0), 0)
        self.assertEqual(hargreaves_rs(tmean_c=-17.8, solar_radiation_mj_m2_day=20), 0)
        self.assertAlmostEqual(hargreaves_rs(tmean_c=-20, solar_radiation_mj_m2_day=20), -.242352)

    def test_invalid_domains_nonfinite_and_overflow(self):
        base = dict(tmean_c=20, solar_radiation_mj_m2_day=20)
        for key in base:
            for value in (None, True, "20", float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    hargreaves_rs(**dict(base, **{key:value}))
        for t, rs in ((-273.15,20),(-274,20),(20,-1),(1e308,1e308)):
            with self.assertRaises(ValueError):
                hargreaves_rs(tmean_c=t, solar_radiation_mj_m2_day=rs)


class HargreavesRsIntegrationTests(unittest.TestCase):
    def test_only_temperature_extremes_and_rs_are_required(self):
        data = fixture()
        row, station = data["rows"][0], data["stations"]["SYNTH"]
        for key in ("TempMedia","HumedadMedia","HumedadMin","HumedadMax","VelViento","EtPMon"):
            row["values"][key] = None
        station.update(altitude_m=None, wind_height_m=10)
        self.assertEqual(eligibility(row, station, "HG_RS_V1"), [])
        self.assertEqual(eligibility(row, {}, "hargreaves_rs"), [])  # no latitude input to this equation
        original = deepcopy(data)
        with patch("siar_science.pipeline.extraterrestrial_radiation", side_effect=AssertionError("HG-Rs must not calculate Ra")):
            result = calculate(data, ("hargreaves_rs",))
        self.assertEqual(data, original)
        self.assertIsNotNone(result["rows"][0]["ET0_Hargreaves_Rs"])
        self.assertIn("STOP:wind_height_not_2m", result["rows"][0]["scientific_flags"])
        self.assertIsNone(result["rows"][1]["ET0_Hargreaves_Rs"])
        row["values"]["Radiacion"] = -1
        self.assertIn("negative:Radiacion", eligibility(row, station, "hargreaves_rs"))
        row["values"].update(Radiacion=20, TempMin=-274)
        self.assertIn("temperature_domain:TempMin", eligibility(row, station, "hargreaves_rs"))
        row["values"].update(TempMin=30, TempMax=20)
        self.assertIn("temperature_extremes_order", eligibility(row, station, "hargreaves_rs"))

    def test_zero_range_is_not_zero_radiation_and_original_models_unchanged(self):
        data = fixture()
        before = calculate(data, ("pm","hargreaves","valiantzas"))
        after = calculate(data, ("pm","hargreaves","valiantzas","hargreaves_rs"))
        for a,b in zip(before["rows"],after["rows"]):
            for field in FOUR_FIELDS[:3]:
                self.assertEqual(a[field],b[field])
            for method in ("pm","hargreaves","valiantzas"):
                self.assertEqual(a["eligibility"][method],b["eligibility"][method])
        row=data["rows"][0]
        row["values"].update(TempMin=20, TempMax=20, Radiacion=20)
        result=calculate(data,("hargreaves","valiantzas","hargreaves_rs"))["rows"][0]
        self.assertEqual(result["ET0_Hargreaves"],0)
        self.assertEqual(result["ET0_Valiantzas"],0)
        self.assertAlmostEqual(result["ET0_Hargreaves_Rs"],4.164048)

    def test_explicit_four_way_cohorts_and_preserved_three_way_metrics(self):
        rows=calculate(fixture(),("pm","hargreaves","valiantzas","hargreaves_rs"))["rows"]
        old=evaluate(rows, FOUR_FIELDS[:3],classical=True)
        new=evaluate(rows, FOUR_FIELDS,four=True)
        preserved=[r for r in new if r["cohort"] in {"own","PM_HS","PM_VAL","HS_VAL","PM_HS_VAL"} and r["method"] in FOUR_FIELDS[:3]]
        self.assertEqual(old,preserved)
        global_scores=[r for r in new if r["group"]=="global"]
        self.assertEqual({r["cohort"] for r in global_scores},{"own",*FOUR_COHORTS})
        for cohort in FOUR_COHORTS:
            selected=[r for r in global_scores if r["cohort"]==cohort]
            self.assertEqual({r["method"] for r in selected},set(FOUR_COHORTS[cohort]))
            self.assertEqual(len({r["n"] for r in selected}),1)
        self.assertEqual(next(r["n"] for r in global_scores if r["method"]=="ET0_Hargreaves_Rs" and r["cohort"]=="own"),1)

    def test_cli_stages_csv_null_zero_provenance_and_parameter_rejection(self):
        with temporary_directory() as directory, redirect_stdout(StringIO()):
            root=Path(directory)
            data=fixture()
            data["rows"][2]["values"].update(Radiacion=0,EtPMon=0)
            original=json_bytes(data)
            source=root/"dataset.json"
            source.write_bytes(original)
            for command,target in (("run-classical-baselines","three"),("calculate-hargreaves-rs","rs"),("run-four-baselines","four")):
                self.assertEqual(main([command,"--dataset",str(source),"--output",str(root/target)]),0)
            paths=[root/"three/predictions.json",root/"rs/predictions.json"]
            result=execute_run(source,root/"merged","evaluate-four",prediction_paths=paths)
            self.assertEqual(result["status"],"SUCCESS",result.get("error"))
            self.assertEqual(result["counts"]["HG_RS_V1"]["calculated"],2)
            self.assertEqual(result["counts"]["HG_RS_V1"]["evaluated"],2)
            self.assertEqual(result["cohorts"]["PM_HS_VAL_HGRS:evaluation"]["n"],2)
            self.assertEqual((root/"merged/metrics.json").read_bytes(),(root/"four/metrics.json").read_bytes())
            self.assertEqual(source.read_bytes(),original)
            for entry in result["outputs"].values():
                self.assertEqual(hashlib.sha256((root/"merged"/entry["path"]).read_bytes()).hexdigest(),entry["sha256"])
            with (root/"merged/classical-baselines.csv").open(encoding="utf-8-sig",newline="") as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[1]["ET0_HG_RS_V1"],"")
            self.assertEqual(rows[1]["eligible_HGRS"],"false")
            self.assertEqual(float(rows[2]["ET0_HG_RS_V1"]),0)
            self.assertEqual(rows[2]["evaluated_HGRS"],"true")
            for row in rows:
                for key in ("dataset_id","source_repo_sha","audit_source_repo_sha"):
                    self.assertEqual(row[key],data[key])
            with (root/"three/classical-baselines.csv").open(encoding="utf-8-sig",newline="") as stream:
                self.assertNotIn("ET0_HG_RS_V1",next(csv.reader(stream)))
            incomplete=execute_run(source,root/"incomplete","evaluate-four",prediction_paths=paths[:1])
            self.assertEqual(incomplete["status"],"FAILED")
            self.assertFalse((root/"incomplete/classical-baselines.csv").exists())
            legacy=execute_run(source,root/"wrong-command","evaluate-classical",prediction_paths=paths)
            self.assertEqual(legacy["status"],"FAILED")
            artifact=json.loads(paths[1].read_bytes())
            artifact["producer"]["parameters"]["HG_RS_V1"]["coefficient"] = .0023
            (root/"wrong.json").write_bytes(json_bytes(artifact))
            rejected=execute_run(source,root/"rejected","evaluate-four",prediction_paths=[paths[0],root/"wrong.json"])
            self.assertEqual(rejected["status"],"FAILED")
            self.assertIn("parameters",rejected["error"])
