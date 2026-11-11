import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from siar_science.et0_v2 import load_protocol, publish
from siar_science.et0_v2_ib09 import PENDING_PATH, verify_admission
from siar_science.et0_v2_support import choose_candidate, design_vector, inner_split, outer_plan, support
from siar_science.serialization import digest


ROOT = Path(__file__).resolve().parents[2]


class V2PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, _ = load_protocol(ROOT)
        cls.lock = cls.protocol["ib09_handoff"]

    def test_admission_is_development_only_and_requires_review(self):
        path = ROOT / self.lock["path"]
        valid = verify_admission(ROOT, path, self.lock, official_evidence_contradiction=False)
        self.assertEqual(valid["scope"], "DEVELOPMENT_ONLY")
        self.assertEqual(valid["official_period_verification"], "UNRESOLVED")
        for evidence in (True, None):
            with self.subTest(evidence=evidence), self.assertRaisesRegex(ValueError, "STOP IB09"):
                verify_admission(ROOT, path, self.lock, official_evidence_contradiction=evidence)
        with self.assertRaisesRegex(ValueError, "not effective"):
            verify_admission(ROOT, ROOT / PENDING_PATH, self.lock, official_evidence_contradiction=False)

    def test_handoff_byte_change_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "handoff.json"
            path.write_bytes((ROOT / self.lock["path"]).read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_admission(ROOT, path, self.lock, official_evidence_contradiction=False)

    def test_inner_splits_are_reproducible_and_do_not_look_at_outcomes(self):
        train = [f"2019-02-{day:02d}" for day in range(1, 21)]
        context = {"station": "IB09", "family": "HS_RA_V1", "year": "2015", "seed": 20261111}
        fit_a, valid_a = inner_split(train, "A", **context)
        self.assertEqual(valid_a, train[-3:])
        self.assertEqual(fit_a, train[:-3])
        fit_e, valid_e = inner_split(train, "E", **context)
        self.assertEqual((fit_e, valid_e), inner_split(reversed(train), "E", **context))
        self.assertNotEqual(valid_e, valid_a)
        self.assertEqual(len(valid_e), 3)
        self.assertEqual(set(fit_e) | set(valid_e), set(train))
        self.assertFalse(set(fit_e) & set(valid_e))

    def test_outer_plan_excludes_held_out_year_and_flags_unsupported_calendar(self):
        pairs = {"2016-07-07": 1, "2017-01-01": 1, "2017-01-02": 1,
                 "2018-01-01": 1, "2018-01-02": 1, "2019-01-01": 1, "2019-01-02": 1}
        args = dict(station="IB09", family="HS_RA_V1", protocol=self.protocol)
        fold = next(outer_plan(pairs, **args))
        self.assertEqual(fold["held_out"], ["2016-07-07"])
        self.assertEqual(fold["score"]["annual"], fold["held_out"])
        self.assertEqual(fold["score"]["monthly"], [])
        self.assertTrue(all(not day.startswith("2016") for day in fold["train"]))
        self.assertFalse(support(pairs, **args)["calendars"]["monthly"]["groups"]["02"]["ahc_identifiable"])
        with self.assertRaisesRegex(ValueError, "only DEVELOPMENT"):
            list(outer_plan({**pairs, "2020-01-01": 1}, **args))

    def test_calendar_inputs_and_selection(self):
        self.assertEqual(design_vector([4, 11, 20], "2019-12-31", "annual", self.protocol), [4, 11, 20])
        self.assertEqual(design_vector([4, 11, 20], "2019-12-31", "seasonal", self.protocol), [4, 11, 20, 1, 0, 0, 0])
        unstable = {"candidate_id": "x", "algorithm": "LINEAR", "yearly_rmse": {2018: .5, 2019: 1.5}}
        stable = {"candidate_id": "y", "algorithm": "RIDGE", "yearly_rmse": {2018: 1.0005, 2019: 1.0005}}
        chosen = choose_candidate([unstable, stable], family_type="ml", protocol=self.protocol,
                                  supported_years=[2018, 2019])
        self.assertEqual(chosen["candidate_id"], "y")
        with self.assertRaisesRegex(ValueError, "common annual RMSE"):
            choose_candidate([unstable, {**stable, "yearly_rmse": {2019: 1}}], family_type="ml",
                             protocol=self.protocol, supported_years=[2018, 2019])

    def test_publication_is_deterministic_and_refuses_existing_output(self):
        inventory = dict(status="PREPARATION_ONLY", protocol_sha256="protocol", dataset_v1_sha256="dataset", ib09={})
        view = {"rows": [{"date": "2019-12-31"}]}
        with TemporaryDirectory() as folder:
            a, b = Path(folder) / "a", Path(folder) / "b"
            publish(a, view, inventory)
            publish(b, view, inventory)
            self.assertEqual((a / "manifest.json").read_bytes(), (b / "manifest.json").read_bytes())
            manifest = json.loads((a / "manifest.json").read_bytes())
            self.assertEqual(manifest["test_metrics"], 0)
            for name, item in manifest["outputs"].items():
                self.assertEqual(digest((a / name).read_bytes()), item["sha256"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                publish(a, view, inventory)


if __name__ == "__main__":
    unittest.main()
