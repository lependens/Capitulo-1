"""Synthetic fixtures only. No repository SIAR observations are opened."""
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from siar_science.datasets import COLUMNS, REPORT_HEADERS, UNITS, build_dataset, read_dataset, validate_dataset


def digest(data):
    return hashlib.sha256(data).hexdigest()


def csv_bytes(header, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def synthetic_dataset():
    values = {key: None for key in COLUMNS}
    values.update(Fecha="2020-02-29", Estacion="SYNTH", TempMedia=15, TempMin=10, TempMax=20)
    return {"schema_version": "observational-v1", "dataset_id": "synthetic-only", "audit_source_repo_sha": "a"*40, "source_repo_sha": "a"*40,
            "source_data_hashes": {"synthetic.csv": "b"*64}, "quality_post": {"id": "SYNTH-POST", "sha256": "c"*64},
            "units": dict(UNITS), "stations": {"SYNTH": {"latitude_deg": 39.5, "altitude_m": None}},
            "rows": [{"station_code": "SYNTH", "date": "2020-02-29", "values": values,
                      "observation_status": "OBSERVED", "quality_status": "WARN", "quality_flags": [],
                      "source_data_sha": "b"*64, "provenance": {"source_file": "synthetic.csv", "source_row": 2}}]}


class DatasetValidationTests(unittest.TestCase):
    def test_partial_observation_and_unverified_units_are_retained(self):
        data = synthetic_dataset()
        data["units_provenance_status"] = "PARTIAL"
        data["units_interpretation"] = "RESOLVED_FOR_V1"
        validate_dataset(data)
        self.assertIsNone(data["rows"][0]["values"]["Radiacion"])
        self.assertIsNone(data["stations"]["SYNTH"]["altitude_m"])

    def test_unknown_and_explicit_two_meter_height_accepted(self):
        for height in (None, 2, 2.0, 10):
            data = synthetic_dataset()
            data["stations"]["SYNTH"]["wind_height_m"] = height
            validate_dataset(data)
        data["stations"]["SYNTH"]["wind_height_m"] = -1
        with self.assertRaisesRegex(ValueError, "wind height"):
            validate_dataset(data)

    def test_duplicate_date_invalid_date_identity_and_nonfinite_rejected(self):
        mutations = [
            lambda d: d["rows"].append(copy.deepcopy(d["rows"][0])),
            lambda d: d["rows"][0].update(date="2021-02-29"),
            lambda d: d["rows"][0]["values"].update(Estacion="OTHER"),
            lambda d: d["rows"][0]["values"].update(TempMedia=float("nan")),
            lambda d: d["rows"][0]["values"].update(TempMedia=True),
            lambda d: d["rows"][0]["values"].pop("PePMon"),
            lambda d: d["rows"][0].update(source_data_sha="d"*64),
        ]
        for mutate in mutations:
            data = synthetic_dataset()
            mutate(data)
            with self.assertRaises(ValueError):
                validate_dataset(data)

    def test_missing_reference_does_not_invalidate_dataset(self):
        data = synthetic_dataset()
        data["rows"][0]["values"]["EtPMon"] = None
        validate_dataset(data)

    def test_ledger_does_not_invent_observations(self):
        data = synthetic_dataset()
        data["missing_days"] = [{"station_code": "SYNTH", "date": "2020-03-01", "observation_status": "MISSING"}]
        validate_dataset(data)
        data["missing_days"][0]["date"] = "2020-02-29"
        with self.assertRaisesRegex(ValueError, "contradiction"):
            validate_dataset(data)

    def test_read_is_validated_and_does_not_mutate_file(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("SIAR_SCIENCE_TEST_TMP")) as tmp:
            path = Path(tmp)/"dataset.json"
            raw = json.dumps(synthetic_dataset()).encode()
            path.write_bytes(raw)
            self.assertEqual(read_dataset(path), synthetic_dataset())
            self.assertEqual(path.read_bytes(), raw)


class FutureHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("SIAR_SCIENCE_TEST_TMP"))
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.handoff_path = self.base/"handoff.json"
        self.values = {key: "" for key in COLUMNS}
        self.values.update(Fecha="2020-02-29T00:00:00", Estacion="SYNTH", TempMedia="15", TempMin="10", TempMax="20")
        self.raw = csv_bytes(COLUMNS, [self.values])
        (self.base/"synthetic.csv").write_bytes(self.raw)
        self.catalog_bytes = csv_bytes(["Codigo", "Latitud", "Altitud"], [{"Codigo": "SYNTH", "Latitud": "393000000N", "Altitud": ""}])
        (self.base/"catalog.csv").write_bytes(self.catalog_bytes)
        common = {"audit_id": "SYNTH-POST", "repo_git_sha": "a"*40, "siar_sync_version": "synthetic", "audit_tool_version": "synthetic", "estaciones_catalog_sha256": digest(self.catalog_bytes), "quality_spec_version": "v1"}
        measurements = {"station_code": "SYNTH", "canonical_file": "synthetic.csv", "canonical_sha256": digest(self.raw), "file_size_bytes": str(len(self.raw)), "filesystem_mode": "0644", "filesystem_mode_ok": "true", "row_count": "1", "column_count": "23", "schema_23_columns_ok": "true", "first_observation_date": "2020-02-29", "last_observation_date": "2020-02-29", "expected_start_date": "2020-02-29", "expected_end_date": "2020-03-02", "official_period_days": "3", "days_with_data": "1", "coverage_percent": "33.33", "duplicate_date_count": "0", "invalid_date_count": "0", "rows_outside_official_period_count": "0", "estacion_value_mismatch_count": "0", "chronological_order_ok": "true", "audit_status": "WARN"}
        audit = {key: "" for key in REPORT_HEADERS["siar_station_audit.csv"].split(",")}
        audit.update(common, **measurements)
        audit.update(station_state="ACTIVE", fecha_instalacion="2020-02-29", active_cutoff_date="2020-03-02", missing_day_count="2", invalid_row_count="0", out_of_order_row_count="0")
        manifest = {key: "" for key in REPORT_HEADERS["siar_dataset_manifest.csv"].split(",")}
        manifest.update(common, **measurements)
        gap = {key: "" for key in REPORT_HEADERS["siar_missing_days.csv"].split(",")}
        gap.update(common, station_code="SYNTH", observation_date="2020-03-01", gap_status="PENDING_RETRY", last_query_result="NOT_ATTEMPTED", attempt_count="0")
        self.report_rows = {"siar_station_audit.csv": [audit], "siar_dataset_manifest.csv": [manifest], "siar_missing_days.csv": [gap]}
        self.package = {"phase": "POST", "reports": {}}
        self.handoff = {"gate": "SIAR-QUALITY-CLOSEOUT", "authorized": True, "dataset_id": "synthetic-closeout", "source_repo_sha": "a"*40, "source_data_hashes": {"synthetic.csv": digest(self.raw)}, "units": dict(UNITS), "stations": {"SYNTH": {"latitude_deg": 39.5, "altitude_m": None}}, "quality_post": {"id": "SYNTH-POST", "path": "post.json", "sha256": ""}, "units_provenance_status": "PARTIAL"}
        self.handoff["catalog"] = {"path": "catalog.csv", "sha256": digest(self.catalog_bytes)}
        self.handoff["audit_source_repo_sha"] = "a"*40
        self.save()

    def save(self):
        for name, rows in self.report_rows.items():
            content = csv_bytes(REPORT_HEADERS[name].split(","), rows)
            (self.base/name).write_bytes(content)
            self.package["reports"][name] = {"path": name, "sha256": digest(content)}
        content = json.dumps(self.package).encode()
        (self.base/"post.json").write_bytes(content)
        self.handoff["quality_post"]["sha256"] = digest(content)
        self.handoff_path.write_text(json.dumps(self.handoff), encoding="utf-8")

    def expand_to_twelve_stations(self):
        """Controlled tiny CSVs, not the canonical data or real POST artifacts."""
        codes = [f"IB{i:02d}" for i in range(1, 12)] + ["IB101"]
        templates = {name: copy.deepcopy(rows[0]) for name, rows in self.report_rows.items()}
        self.catalog_bytes = csv_bytes(["Codigo", "Latitud", "Altitud"], [
            {"Codigo": code, "Latitud": "393000000N", "Altitud": ""} for code in codes])
        (self.base/"catalog.csv").write_bytes(self.catalog_bytes)
        self.handoff["catalog"]["sha256"] = digest(self.catalog_bytes)
        self.handoff["stations"] = {}
        self.handoff["source_data_hashes"] = {}
        self.report_rows = {name: [] for name in templates}
        for code in codes:
            raw = csv_bytes(COLUMNS, [dict(self.values, Estacion=code)])
            path = code + ".csv"
            (self.base/path).write_bytes(raw)
            self.handoff["stations"][code] = {"latitude_deg": 39.5, "altitude_m": None}
            self.handoff["source_data_hashes"][path] = digest(raw)
            for name, template in templates.items():
                record = dict(template, station_code=code, estaciones_catalog_sha256=digest(self.catalog_bytes))
                if "canonical_file" in record:
                    record.update(canonical_file=path, canonical_sha256=digest(raw), file_size_bytes=str(len(raw)))
                self.report_rows[name].append(record)
        self.save()

    def test_dual_sha_matching_audit_and_equal_canonical_remain_valid(self):
        self.expand_to_twelve_stations()
        dataset = build_dataset(self.handoff_path)
        self.assertEqual(len(dataset["source_data_hashes"]), 12)
        self.assertEqual(dataset["audit_source_repo_sha"], "a"*40)
        self.assertEqual(dataset["source_repo_sha"], "a"*40)

    def test_dual_sha_distinct_commits_pass_with_twelve_matching_files(self):
        self.expand_to_twelve_stations()
        self.handoff["source_repo_sha"] = "e"*40
        self.save()
        before = {path.name: path.read_bytes() for path in self.base.iterdir()}
        dataset = build_dataset(self.handoff_path)
        self.assertEqual(dataset["audit_source_repo_sha"], "a"*40)
        self.assertEqual(dataset["source_repo_sha"], "e"*40)
        self.assertEqual(len(dataset["rows"]), 12)
        self.assertEqual(dataset["source_data_hashes"], self.handoff["source_data_hashes"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.base.iterdir()})

    def test_dual_sha_each_post_report_must_match_audited_commit(self):
        self.expand_to_twelve_stations()
        self.handoff["source_repo_sha"] = "e"*40
        for name in self.report_rows:
            with self.subTest(report=name):
                self.report_rows[name][0]["repo_git_sha"] = "e"*40
                self.save()
                with self.assertRaisesRegex(ValueError, "audit_source_repo_sha"):
                    build_dataset(self.handoff_path)
                self.report_rows[name][0]["repo_git_sha"] = "a"*40

    def test_dual_sha_checks_every_source_against_post_manifest(self):
        self.expand_to_twelve_stations()
        self.handoff["source_repo_sha"] = "e"*40
        for path, expected_hash in list(self.handoff["source_data_hashes"].items()):
            with self.subTest(source=path):
                self.handoff["source_data_hashes"][path] = "f"*64
                self.save()
                with self.assertRaisesRegex(ValueError, "Source/manifest SHA mismatch"):
                    build_dataset(self.handoff_path)
                self.handoff["source_data_hashes"][path] = expected_hash

    def test_dual_sha_missing_or_invalid_audit_identity_rejected(self):
        for value in (None, "", "a"*39, "a"*41, "g"*40, True):
            with self.subTest(value=value):
                if value is None:
                    self.handoff.pop("audit_source_repo_sha", None)
                else:
                    self.handoff["audit_source_repo_sha"] = value
                self.save()
                with self.assertRaisesRegex(ValueError, "audit_source_repo_sha"):
                    build_dataset(self.handoff_path)
                dataset = synthetic_dataset()
                if value is None:
                    dataset.pop("audit_source_repo_sha")
                else:
                    dataset["audit_source_repo_sha"] = value
                with self.assertRaisesRegex(ValueError, "audit_source_repo_sha"):
                    validate_dataset(dataset)

    def test_build_is_read_only_preserves_nulls_and_unknown_gap(self):
        before = {p.name: p.read_bytes() for p in self.base.iterdir()}
        data = build_dataset(self.handoff_path)
        self.assertEqual(data["rows"][0]["values"]["Fecha"], "2020-02-29T00:00:00")
        self.assertIsNone(data["rows"][0]["values"]["Radiacion"])
        self.assertEqual([r["observation_status"] for r in data["missing_days"]], ["PENDING_RETRY", "MISSING"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.base.iterdir()})

    def test_authorization_alone_or_pre_never_suffices(self):
        self.handoff["authorized"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "Authorized"):
            build_dataset(self.handoff_path)
        self.handoff["authorized"] = True
        self.package["phase"] = "PRE"
        self.save()
        with self.assertRaisesRegex(ValueError, "POST"):
            build_dataset(self.handoff_path)

    def test_tampering_raw_report_and_package_is_detected(self):
        for name in ("synthetic.csv", "siar_station_audit.csv", "post.json", "catalog.csv"):
            with self.subTest(name=name):
                original = (self.base/name).read_bytes()
                (self.base/name).write_bytes(original+b" ")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    build_dataset(self.handoff_path)
                (self.base/name).write_bytes(original)

    def test_catalog_metadata_and_nonfinite_fields_remain_explicit(self):
        self.handoff["stations"]["SYNTH"]["latitude_deg"] = 40
        self.save()
        with self.assertRaisesRegex(ValueError, "catalog"):
            build_dataset(self.handoff_path)
        self.handoff["stations"]["SYNTH"]["latitude_deg"] = 39.5
        self.values["Radiacion"] = "NaN"
        self.raw = csv_bytes(COLUMNS, [self.values])
        (self.base/"synthetic.csv").write_bytes(self.raw)
        self.handoff["source_data_hashes"]["synthetic.csv"] = digest(self.raw)
        for name in ("siar_station_audit.csv", "siar_dataset_manifest.csv"):
            self.report_rows[name][0].update(canonical_sha256=digest(self.raw), file_size_bytes=str(len(self.raw)))
        self.save()
        dataset = build_dataset(self.handoff_path)
        row = dataset["rows"][0]
        self.assertIsNone(row["values"]["Radiacion"])
        self.assertIn("source_nonfinite:Radiacion", row["quality_flags"])
        self.assertTrue(row["eligibility"]["hargreaves"]["eligible"])
        self.assertFalse(row["eligibility"]["pm"]["eligible"])

    def test_all_three_reports_and_exact_headers_required(self):
        del self.package["reports"]["siar_missing_days.csv"]
        content = json.dumps(self.package).encode()
        (self.base/"post.json").write_bytes(content)
        self.handoff["quality_post"]["sha256"] = digest(content)
        self.handoff_path.write_text(json.dumps(self.handoff))
        with self.assertRaisesRegex(ValueError, "three"):
            build_dataset(self.handoff_path)

    def test_recovered_observation_and_positive_absence_evidence(self):
        gap = self.report_rows["siar_missing_days.csv"][0]
        gap.update(gap_status="CONFIRMED_NO_DATA", last_query_result="ZERO_ROWS", siar_records_returned="0", attempt_count="1", evidence_ref="synthetic-evidence")
        self.save()
        self.assertEqual(build_dataset(self.handoff_path)["missing_days"][0]["observation_status"], "CONFIRMED_NO_DATA")
        gap["evidence_ref"] = ""
        self.save()
        with self.assertRaisesRegex(ValueError, "positive evidence"):
            build_dataset(self.handoff_path)
        gap.update(observation_date="2020-02-29", gap_status="RECOVERED", last_query_result="DATA_RETURNED", siar_records_returned="1", recovered_at_utc="2020-03-03T00:00:00Z", evidence_ref="synthetic-evidence")
        self.save()
        self.assertEqual(build_dataset(self.handoff_path)["rows"][0]["observation_status"], "RECOVERED")

    def test_cross_report_conflict_and_outside_gap_rejected(self):
        self.report_rows["siar_station_audit.csv"][0]["row_count"] = "2"
        self.save()
        with self.assertRaisesRegex(ValueError, "Audit/manifest"):
            build_dataset(self.handoff_path)
        self.report_rows["siar_station_audit.csv"][0]["row_count"] = "1"
        self.report_rows["siar_missing_days.csv"][0]["observation_date"] = "2020-02-28"
        self.save()
        with self.assertRaisesRegex(ValueError, "outside official"):
            build_dataset(self.handoff_path)


if __name__ == "__main__":
    unittest.main()
