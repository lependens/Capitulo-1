"""Tests for the forensic experiment, independent of the pending pipeline."""
import math
import csv
import json
import unittest
from unittest.mock import patch
from collections import defaultdict
from datetime import date
from pathlib import Path

from units_offline import HYPOTHESES, Summary, calculate, latitude, load_legacy_functions, parse_row, run
from report_units import render


class ForensicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.legacy = load_legacy_functions(cls.repo / "legacy/et0/PM2.py")
        cls.sample = dict(TempMedia=12.515, TempMax=19.09, TempMin=6.471,
                          HumedadMax=96.6, HumedadMin=42.18,
                          VelViento=1.908, Radiacion=19.999, EtPMon=3.205964)

    def test_preserved_ib04_example_and_alternative(self):
        predictions = {h: calculate(self.sample, date(2004, 4, 1), 39.5, 100,
                                   *factors, "legacy_pm2", self.legacy)
                       for h, factors in HYPOTHESES.items()}
        self.assertAlmostEqual(predictions["Rad1_Wind1"]["et0"], 3.214339259752004, places=12)
        self.assertAlmostEqual(predictions["Rad2_Wind1"]["et0"], 1.27, delta=.02)
        self.assertAlmostEqual(predictions["Rad2_Wind1"]["rs"], 19.999*86400/1_000_000)
        self.assertAlmostEqual(predictions["Rad1_Wind2"]["wind"], 1.908*1000/3600)

    def test_zero_reference_included_signed_bias_and_rmse(self):
        summary = Summary()
        for observed, predicted in ((0, 1), (2, 0)):
            summary.add(observed, dict(et0=predicted, rs=1, wind=1, ratio=.5, rn=1, rnl=1))
        result = summary.result()
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["bias"], -.5)
        self.assertEqual(result["mae"], 1.5)
        self.assertEqual(result["rmse"], math.sqrt(2.5))

    def test_fao56_published_daily_example_18(self):
        # FAO-56 ch.4 Example 18, published 3.9 mm/day, rounded inputs.
        # https://www.fao.org/4/X0490E/x0490e08.htm (already consulted).
        values = dict(TempMedia=16.9, TempMin=12.3, TempMax=21.5,
                      HumedadMin=63, HumedadMax=84, VelViento=2.078,
                      Radiacion=22.07, EtPMon=3.9)
        result = calculate(values, date(2001, 7, 6), 50.8, 100, 1, 1,
                           "fao56_sensitivity", self.legacy)
        self.assertAlmostEqual(result["et0"], 3.9, delta=.05)

    def test_recorded_artifacts_have_paired_cohorts_and_source_hashes(self):
        import hashlib
        results = Path(__file__).parent / "results/units-offline-v2"
        manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["output_sha256"].items():
            self.assertEqual(hashlib.sha256((results/name).read_bytes()).hexdigest(), digest)
        for source in manifest["source_files"]:
            # The captured Windows checkout had core.autocrlf=true. Reconstruct
            # either text checkout byte stream in memory, without touching raw.
            # This verifies exact captured bytes modulo Git's LF/CRLF checkout,
            # never accepts a meteorological value or other content change.
            data = (self.repo/source["path"]).read_bytes()
            lf = data.replace(b"\r\n", b"\n")
            candidates = (data, lf, lf.replace(b"\n", b"\r\n"))
            self.assertIn(source["sha256"], {hashlib.sha256(candidate).hexdigest() for candidate in candidates})
            self.assertEqual(source["rows"], source["eligible"] + sum(source["excluded"].values()))
        self.assertEqual(hashlib.sha256((Path(__file__).parent/"units_offline.py").read_bytes()).hexdigest(), manifest["tool_sha256"])
        groups = defaultdict(list)
        with (results/"scores.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                groups[row["corpus"],row["kernel"],row["group"],row["key"]].append(row)
        for group in groups.values():
            self.assertEqual({r["hypothesis"] for r in group}, set(HYPOTHESES))
            self.assertEqual(len({r["n"] for r in group}), 1)

    def test_mask_not_sigma_or_auxiliary_dropna(self):
        row = dict(self.sample, Fecha="2004-04-01", Precipitacion="", TempSuelo1="")
        self.assertIsNone(parse_row(row)[1])
        for change in ({"Radiacion": ""}, {"VelViento": "nan"}, {"HumedadMin": 101}, {"TempMax": 1}):
            self.assertIsNotNone(parse_row(dict(row, **change))[1])
        self.assertIsNone(parse_row(dict(row, EtPMon=0))[1])

    def test_output_overwrite_and_protected_paths_rejected(self):
        with self.assertRaisesRegex(ValueError, "new directory"):
            run(self.repo, self.repo / "legacy")
        with self.assertRaisesRegex(ValueError, "reference/acquisition"):
            run(self.repo, self.repo / "datos_siar_baleares" / "must-not-write")

    def test_catalog_coordinate(self):
        self.assertAlmostEqual(latitude("393344000N"), 39+33/60+44/3600)

    def test_report_matches_reviewed_evidence(self):
        report = render(Path(__file__).parent / "results/units-offline-v2")
        expected = (self.repo/"scientific_pipeline/docs/units_offline_forensic.md").read_text(encoding="utf-8")
        self.assertEqual(report, expected)
        self.assertIn("IB101:2008, n=26", report)

    def test_new_results_do_not_inherit_unit_certification(self):
        with patch.object(Path, "read_text", return_value=json.dumps({"output_sha256": {"scores.csv": "different"}})):
            with self.assertRaisesRegex(ValueError, "review new evidence"):
                render("a-new-experiment")


if __name__ == "__main__":
    unittest.main()
