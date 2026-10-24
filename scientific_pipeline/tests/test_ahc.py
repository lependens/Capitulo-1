"""Controlled AHC arithmetic, support, anti-leakage and staged CLI regression."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import csv
from datetime import date
from io import StringIO
import json
from pathlib import Path
import unittest

from siar_science.ahc import train_analysis, validate_selection, apply_validated, closeout_analysis
from siar_science.ahc_pipeline import execute_ahc
from siar_science.calibration import (EMPIRICAL, fit_scale, fit_station, coefficient_for,
                                    partition, support, validate_protocol, block_stability)
from siar_science.cli import main
from siar_science.pipeline import calculate, execute_run
from siar_science.serialization import digest, json_bytes
from tests_support import dataset_fixture, temporary_directory


def protocol():
    p = json.loads((Path(__file__).parents[1]/"experiments/ahc-v1.json").read_bytes())
    p.update(dataset_id="CONTROLLED_ONLY", require_clean_upstream=False)
    p["splits"] = {"TRAIN": ["2001-01-01", "2001-12-31"], "VALIDATION": ["2002-01-01", "2002-12-31"], "TEST": ["2003-01-01", "2003-12-31"]}
    p["training_stability_blocks"] = [[2001, 2001]]
    p["post_test_descriptive_blocks"] = [[2001, 2001], [2002, 2002], [2003, 2003]]
    p["support"] = dict(minimum_fit_n=4, minimum_fit_years=1, qualified_year_n=4, qualified_year_season_n=1,
                        minimum_season_fit_n=1, minimum_month_fit_n=1, minimum_group_years=1,
                        minimum_season_year_n=1, minimum_month_year_n=1)
    p["selection"]["station_minimum_validation_n"] = 1
    return p


def controlled_rows(vary_seasons=False):
    result = []
    for year in (2001, 2002, 2003):
        for i, month in enumerate((1, 4, 7, 10)):
            x = (4, 5, 6, 5)[i] if vary_seasons else 5
            row = dict(station_code="SYNTH", date=f"{year}-{month:02}-15", observation_status="OBSERVED",
                quality_status="PASS", quality_flags=[], scientific_flags=[], provenance={"source_file": "synthetic"}, source_data_sha="b"*64,
                EtPMon=10, ET0_FAO56=10, ET0_Hargreaves=x, ET0_Hargreaves_Rs=x, ET0_Valiantzas=x,
                eligibility={m: dict(eligible=True, reasons=[]) for m in ("pm", "hargreaves", "hargreaves_rs", "valiantzas")})
            result.append(row)
    return result


class CalibrationMathTests(unittest.TestCase):
    def test_manual_estimands_and_least_squares_property(self):
        pairs = [(1, 2), (2, 2)]
        a = fit_scale(pairs)["coefficient"]
        self.assertEqual(a, 1.2)
        self.assertEqual(fit_scale(pairs, "MEAN_RATIO")["coefficient"], 1.5)
        self.assertEqual(fit_scale(pairs, "RATIO_SUMS")["coefficient"], 4/3)
        objective = lambda c: sum((y-c*x)**2 for x, y in pairs)
        self.assertLess(objective(a), objective(1.5))
        self.assertLess(objective(a), objective(a-.01))
        self.assertLess(objective(a), objective(a+.01))
        self.assertAlmostEqual(sum(x*(y-a*x) for x, y in pairs), 0)

    def test_zeros_empty_invalid_and_no_epsilon(self):
        pairs = [(0, 3), (2, 0)]
        self.assertEqual(fit_scale(pairs)["coefficient"], 0)
        self.assertEqual(fit_scale(pairs)["zero_predictors"], 1)
        self.assertEqual(fit_scale(pairs, "MEAN_RATIO")["n_used"], 1)
        self.assertEqual(fit_scale(pairs, "RATIO_SUMS")["coefficient"], 1.5)
        self.assertIsNone(fit_scale([])["coefficient"])
        self.assertIsNone(fit_scale([(0, 0), (0, 3)])["coefficient"])
        self.assertAlmostEqual(fit_scale([(1e-12, 1)])["coefficient"], 1e12)
        for pairs in ([(-1, 2)], [(1, -1)], [(True, 1)], [(None, 1)], [(float("nan"), 1)], [(1, float("inf"))], [(1e308, 1e308)], [(1e-200, 1)]):
            with self.subTest(pairs=pairs), self.assertRaises((ValueError, OverflowError)):
                fit_scale(pairs)

    def test_protocol_and_exact_global_cuts(self):
        p = json.loads((Path(__file__).parents[1]/"experiments/ahc-v1.json").read_bytes())
        validate_protocol(p)
        for day, split in (("2015-12-31", "TRAIN"), ("2016-01-01", "VALIDATION"),
                           ("2020-12-31", "VALIDATION"), ("2021-01-01", "TEST"), ("2026-01-01", "OUTSIDE_EXPERIMENT")):
            self.assertEqual(partition(day, p), split)
        for mutate in (lambda q: q.update(estimator="MEAN_RATIO"), lambda q: q.update(reference="EtPMon"),
                       lambda q: q["splits"]["VALIDATION"].__setitem__(0, "2015-12-31"),
                       lambda q: q["splits"]["TRAIN"].reverse(),
                       lambda q: q["training_stability_blocks"].append([2021, 2025]),
                       lambda q: q["post_test_descriptive_blocks"].append([2025, 2026]),
                       lambda q: q["support"].update(minimum_fit_n=0.5),
                       lambda q: q["selection"].update(relative_validation_rmse_improvement=float("nan")),
                       lambda q: q["selection"].update(refit_train_plus_validation=True)):
            q = deepcopy(p); mutate(q)
            with self.assertRaises(ValueError):
                validate_protocol(q)

    def test_support_partial_year_and_short_series(self):
        p = protocol(); rows = controlled_rows()
        self.assertTrue(support(rows[:4], p["support"])["robust"])
        self.assertFalse(support(rows[:3], p["support"])["robust"])
        extra = deepcopy(rows[0]); extra["date"] = "2001-02-01"
        records = fit_station(rows+[extra], "SYNTH", "HS_RA_V1", p)
        self.assertEqual(records[0]["estimate"]["n_input"], 5)  # retain partial/unbalanced observations
        official = json.loads((Path(__file__).parents[1]/"experiments/ahc-v1.json").read_bytes())
        self.assertFalse(support(rows, official["support"])["robust"])

    def test_lookup_rejects_climatology_other_reference_and_duplicates(self):
        c = train_analysis(controlled_rows(), ["SYNTH"], protocol())["coefficients"][0]
        args = ("SYNTH", "HS_RA_V1", "2002-01-15")
        self.assertEqual(coefficient_for([c], *args)["coefficient"], 2)
        for key, value in (("kind", "AHC_CLIMATOLOGY"), ("reference_id", "EtPMon"), ("fit_split", "TEST")):
            with self.assertRaises(ValueError):
                coefficient_for([dict(c, **{key:value})], *args)
        with self.assertRaises(ValueError):
            coefficient_for([c, c], *args)


class ChronologyTests(unittest.TestCase):
    def test_training_and_selection_cannot_depend_on_future(self):
        p = protocol(); rows = controlled_rows(vary_seasons=True)
        training = train_analysis(rows, ["SYNTH"], p)
        self.assertEqual(training["status"], "PASS")
        altered = deepcopy(rows)
        for row in altered:
            if row["date"] >= "2002-01-01":
                row.update(ET0_FAO56=99999, EtPMon=1, ET0_Hargreaves=0, ET0_Hargreaves_Rs=123, ET0_Valiantzas=7)
        self.assertEqual(training, train_analysis(altered, ["SYNTH"], p))
        selection = validate_selection(rows, training, p)
        self.assertTrue(all(s["selected_granularity"] == "seasonal" for s in selection["selections"]))
        altered = deepcopy(rows)
        for row in altered:
            if row["date"] >= "2003-01-01":
                row.update(ET0_FAO56=99999, EtPMon=1, ET0_Hargreaves=0)
        self.assertEqual(selection, validate_selection(altered, training, p))
        self.assertFalse(selection["coefficients_refitted"])

    def test_constant_or_tied_validation_keeps_annual(self):
        p = protocol(); rows = controlled_rows()
        training = train_analysis(rows, ["SYNTH"], p)
        selection = validate_selection(rows, training, p)
        self.assertTrue(all(s["selected_granularity"] == "annual" for s in selection["selections"]))

    def test_stop_temporal_instability_and_reference_sensitivity(self):
        p = protocol(); p["splits"] = {"TRAIN":["2001-01-01","2002-12-31"],"VALIDATION":["2003-01-01","2003-12-31"],"TEST":["2004-01-01","2004-12-31"]}
        p["training_stability_blocks"] = [[2001,2001],[2002,2002]]
        rows = controlled_rows()
        for row in rows:
            if row["date"].startswith("2002"):
                row.update(ET0_Hargreaves=.5,ET0_Hargreaves_Rs=.5,ET0_Valiantzas=.5)
        result = train_analysis(rows,["SYNTH"],p)
        self.assertEqual(result["status"],"STOP")
        self.assertTrue(any(r["reason"]=="STOP_TEMPORAL_INSTABILITY" for r in result["stop_reasons"]))
        rows=controlled_rows()
        for row in rows[:4]:
            row["EtPMon"]=40
        result=train_analysis(rows,["SYNTH"],protocol())
        self.assertTrue(any(r["reason"]=="STOP_REFERENCE_DISAGREEMENT" for r in result["stop_reasons"]))

    def test_no_target_needed_for_prediction_no_short_coefficient_fallback(self):
        p=protocol(); rows=controlled_rows()
        training=train_analysis(rows,["SYNTH"],p)
        selection=validate_selection(rows,training,p)
        rows[-1]["ET0_FAO56"]=None
        result=apply_validated(rows,training["coefficients"],p,selection["selections"])
        self.assertEqual(result[-1]["predictions"]["HS_RA_AHC_VALIDATED"],10)
        self.assertIsNone(result[-1]["predictions"]["PM_FAO56_V1"])
        coefficients=deepcopy(training["coefficients"])
        coefficients[0].update(deployable=False,status="SHORT_SERIES_DESCRIPTIVE")
        self.assertIsNone(apply_validated(rows,coefficients,p,selection["selections"])[0]["predictions"]["HS_RA_AHC_VALIDATED"])
        with self.assertRaises(ValueError):
            fit_station(rows,"SYNTH","HS_RA_V1",p,split="TEST")

    def test_climatology_requires_test_and_excludes_future_rows(self):
        p=protocol(); rows=controlled_rows()
        with self.assertRaises(ValueError):
            closeout_analysis(rows,["SYNTH"],p)
        before=closeout_analysis(rows,["SYNTH"],p,test_completed=True)
        future=deepcopy(rows[-1]); future.update(date="2004-01-01",ET0_FAO56=99999)
        after=closeout_analysis(rows+[future],["SYNTH"],p,test_completed=True)
        self.assertEqual(before["climatology"],after["climatology"])
        self.assertFalse(after["out_of_sample_claim"])


class AHCPipelineTests(unittest.TestCase):
    def test_synthetic_cli_all_stages_hashes_views_and_tamper_guard(self):
        with temporary_directory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root=Path(directory); dataset=dataset_fixture(); rows=[]
            for year in (2001,2002,2003):
                for month in (1,4,7,10):
                    row=deepcopy(dataset["rows"][0]); day=f"{year}-{month:02}-15"
                    row.update(date=day); row["values"].update(Fecha=day,HumedadMedia=70)
                    row["provenance"]["source_row"]=len(rows)+2; rows.append(row)
            dataset.update(rows=rows,missing_days=[])
            computed = {r["date"]: r["ET0_FAO56"] for r in calculate(dataset, ("pm",))["rows"]}
            for row in dataset["rows"]:
                row["values"]["EtPMon"] = computed[row["date"]]
            source=root/"dataset.json"; original=json_bytes(dataset); source.write_bytes(original)
            baseline=execute_run(source,root/"baseline","run-four-baselines")
            self.assertEqual(baseline["status"],"SUCCESS")
            p=protocol(); p.update(dataset_id=dataset["dataset_id"],dataset_sha256=digest(original),baseline_manifest_sha256=digest((root/"baseline/manifest.json").read_bytes()))
            config=root/"protocol.json"; config.write_bytes(json_bytes(p))
            common=["--dataset",str(source),"--baseline-manifest",str(root/"baseline/manifest.json"),"--protocol",str(config)]
            self.assertEqual(main(["ahc-train",*common,"--output",str(root/"train")]),0)
            previous=root/"train/manifest.json"
            premature=execute_ahc(source,root/"baseline/manifest.json",config,root/"premature","closeout",parent_manifest=previous)
            self.assertEqual(premature["status"],"FAILED")
            for stage in ("validate","test","closeout"):
                target=root/stage
                self.assertEqual(main(["ahc-"+stage,*common,"--parent-manifest",str(previous),"--output",str(target)]),0)
                previous=target/"manifest.json"
                m=json.loads(previous.read_bytes())
                for entry in m["outputs"].values():
                    self.assertEqual(digest((target/entry["path"]).read_bytes()),entry["sha256"])
                self.assertEqual(m["source_repo_sha"],dataset["source_repo_sha"])
                self.assertEqual(m["audit_source_repo_sha"],dataset["audit_source_repo_sha"])
            with (root/"test/daily_predictions.csv").open(encoding="utf-8-sig",newline="") as stream:
                daily=list(csv.DictReader(stream))
            self.assertEqual(len(daily),12)
            self.assertEqual({r["split"] for r in daily},{"TRAIN","VALIDATION","TEST"})
            self.assertTrue(all("HS_RA_AHC_VALIDATED" in r for r in daily))
            self.assertNotIn("Radiacion",daily[0])  # thin view, no needless raw duplication
            view=json.loads((root/"test/view-manifest.json").read_bytes())
            self.assertEqual(view["schema_version"],"science-view-v1")
            self.assertFalse(view["science_recalculation_allowed"])
            schema=json.loads((root/"test/view_schemas.json").read_bytes())
            self.assertEqual(next(c["type"] for c in schema["daily_predictions.csv"]["columns"] if c["name"]=="EtPMon"),"float64_nullable")
            with (root/"test/monthly_metrics.csv").open(encoding="utf-8-sig",newline="") as stream:
                monthly=list(csv.DictReader(stream))
            self.assertTrue(all(r["group"] in ("year_month","station_year_month") for r in monthly))
            self.assertTrue(all(r["model_instance_id"].startswith("model-") for r in monthly))
            self.assertTrue((root/"train/split_assignments.csv").is_file())
            self.assertEqual(source.read_bytes(),original)
            with self.assertRaisesRegex(ValueError,"already exists"):
                execute_ahc(source,root/"baseline/manifest.json",config,root/"train","train")
            (root/"train/training.json").write_text("{}",encoding="utf-8")
            tamper=execute_ahc(source,root/"baseline/manifest.json",config,root/"tampered","validate",parent_manifest=root/"train/manifest.json")
            self.assertEqual(tamper["status"],"FAILED")
            self.assertIn("hash mismatch",tamper["error"])
