"""Read-only dataset adapter. No function writes data or authorizes a closeout.

``build_dataset`` consumes a future handoff with gate SIAR-QUALITY-CLOSEOUT,
authorized true, and quality_post {id, sha256, path}. That path identifies a
JSON package {phase: POST, reports: {filename: {path, sha256}}}. Report paths
are relative to the package; source_data_hashes paths to the handoff. All three
Quality Audit v1 reports are mandatory, even when the missing-days report is
empty. A verified hash proves identity, not institutional authorization.
audit_source_repo_sha identifies the audited commit; source_repo_sha identifies
the canonical publication. They may differ: POST identities bind to the former,
while source/catalog bytes must match the preserved POST hashes.
"""
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import math
from pathlib import Path
import re


COLUMNS = tuple("Fecha TempMedia TempMax HorMinTempMax TempMin HorMinTempMin HumedadMedia HumedadMax HorMinHumMax HumedadMin HorMinHumMin VelViento DirViento VelVientoMax HorMinVelMax DirVientoVelMax Radiacion Precipitacion TempSuelo1 TempSuelo2 EtPMon PePMon Estacion".split())
UNITS = {"temperature": "degC", "wind": "m/s", "radiation": "MJ/m2/day", "humidity": "%", "altitude": "m"}
OBSERVATION_STATUSES = {"OBSERVED", "RECOVERED", "MISSING", "CONFIRMED_NO_DATA", "PENDING_RETRY", "OUTSIDE_OFFICIAL_PERIOD"}
QUALITY_STATUSES = {"PASS", "WARN", "FAIL", "INCOMPLETE", "NOT_RUN"}
COMMON = "audit_id,repo_git_sha,siar_sync_version,audit_tool_version,estaciones_catalog_sha256,quality_spec_version"
REPORT_HEADERS = {
    "siar_station_audit.csv": "audit_id,audited_at_utc,repo_git_sha,siar_sync_version,audit_tool_version,estaciones_catalog_sha256,quality_spec_version,station_code,station_name,station_state,fecha_instalacion,fecha_baja,active_cutoff_date,expected_start_date,expected_end_date,official_period_days,canonical_file,canonical_sha256,file_size_bytes,filesystem_mode,filesystem_mode_ok,permission_normalized_at_utc,row_count,column_count,schema_23_columns_ok,first_observation_date,last_observation_date,invalid_date_count,rows_outside_official_period_count,estacion_value_mismatch_count,chronological_order_ok,out_of_order_row_count,duplicate_date_count,invalid_row_count,days_with_data,coverage_percent,missing_day_count,audit_status,notes",
    "siar_missing_days.csv": COMMON + ",station_code,observation_date,gap_status,last_query_result,first_detected_at_utc,last_attempt_at_utc,attempt_count,recovered_at_utc,siar_records_returned,evidence_ref,error_code,notes",
    "siar_dataset_manifest.csv": "audit_id,manifest_created_at_utc,repo_git_sha,siar_sync_version,audit_tool_version,estaciones_catalog_sha256,quality_spec_version,station_code,canonical_file,canonical_sha256,file_size_bytes,filesystem_mode,filesystem_mode_ok,permission_normalized_at_utc,row_count,column_count,schema_23_columns_ok,first_observation_date,last_observation_date,expected_start_date,expected_end_date,official_period_days,days_with_data,coverage_percent,duplicate_date_count,invalid_date_count,rows_outside_official_period_count,estacion_value_mismatch_count,chronological_order_ok,audit_status",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value, name="SHA-256"):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid " + name)


def _repo_sha(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value), "Invalid " + name)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _day(value):
    _require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value), "Expected ISO date")
    return date.fromisoformat(value)


def _source_day(value):
    # SIAR Fecha may be a midnight ISO timestamp. Preserve the original string.
    _require(isinstance(value, str), "Fecha must be a string")
    if len(value) > 10:
        from datetime import datetime
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _day(value[:10])


def validate_dataset(dataset):
    """Validate structure and provenance; model eligibility remains independent.

    Missing altitude and unverified unit provenance are retained for QC. An
    unknown wind height follows the explicitly adopted SIAR 2 m convention;
    positive evidence of a different height must be resolved before v1 use.
    """
    _require(isinstance(dataset, dict), "Dataset must be an object")
    _require(dataset.get("schema_version") == "observational-v1", "Unsupported dataset schema")
    _require(isinstance(dataset.get("dataset_id"), str) and dataset["dataset_id"].strip(), "Missing dataset_id")
    _repo_sha(dataset.get("source_repo_sha"), "source_repo_sha")
    _repo_sha(dataset.get("audit_source_repo_sha"), "audit_source_repo_sha")
    sources = dataset.get("source_data_hashes")
    _require(isinstance(sources, dict) and sources, "Missing source_data_hashes")
    for path, digest in sources.items():
        _require(isinstance(path, str) and path, "Invalid source path")
        _hash(digest)
    post = dataset.get("quality_post")
    _require(isinstance(post, dict) and isinstance(post.get("id"), str) and post["id"], "Missing Quality POST")
    _hash(post.get("sha256"))
    units = dataset.get("units")
    _require(isinstance(units, dict) and all(units.get(k) == v for k, v in UNITS.items()), "Unsupported units")
    stations = dataset.get("stations")
    _require(isinstance(stations, dict) and stations, "Missing stations")
    for code, metadata in stations.items():
        _require(isinstance(code, str) and code and isinstance(metadata, dict), "Invalid station")
        latitude = metadata.get("latitude_deg")
        _require(_finite(latitude) and -90 < latitude < 90, "Invalid station latitude")
        altitude = metadata.get("altitude_m")
        _require(altitude is None or _finite(altitude), "Invalid station altitude")
        height = metadata.get("wind_height_m")
        _require(height is None or (_finite(height) and height > 0), "Invalid wind height metadata")
    rows = dataset.get("rows")
    _require(isinstance(rows, list), "Missing rows")
    seen = set()
    for row in rows:
        _require(isinstance(row, dict), "Invalid row")
        code, day = row.get("station_code"), row.get("date")
        _require(isinstance(code, str) and code in stations, "Unknown row station")
        _day(day)
        key = (code, day)
        _require(key not in seen, "Duplicate station/date")
        seen.add(key)
        _require(row.get("observation_status") in OBSERVATION_STATUSES, "Invalid observation_status")
        _require(row.get("quality_status") in QUALITY_STATUSES, "Invalid quality_status")
        _require(isinstance(row.get("quality_flags"), list) and all(isinstance(flag, str) for flag in row["quality_flags"]), "Invalid quality_flags")
        _require(isinstance(row.get("provenance"), dict) and row["provenance"], "Missing row provenance")
        _hash(row.get("source_data_sha"))
        _require(row["source_data_sha"] in sources.values(), "Unknown source_data_sha")
        source_file = row["provenance"].get("source_file")
        source_row = row["provenance"].get("source_row")
        _require(isinstance(source_file, str) and sources.get(source_file) == row["source_data_sha"], "Row source_file/hash mismatch")
        _require(isinstance(source_row, int) and not isinstance(source_row, bool) and source_row >= 2, "Invalid source_row")
        values = row.get("values")
        _require(isinstance(values, dict) and set(values) == set(COLUMNS), "Expected all 23 original columns")
        _require(_source_day(values["Fecha"]) == _day(day), "Fecha/date mismatch")
        _require(values["Estacion"] == code, "Estacion/station_code mismatch")
        for column in COLUMNS:
            if column not in ("Fecha", "Estacion"):
                _require(values[column] is None or _finite(values[column]), "Invalid numeric value: " + column)
        _require(row["observation_status"] not in {"MISSING", "CONFIRMED_NO_DATA", "PENDING_RETRY"}, "Absent observations belong in missing_days ledger")
    row_statuses = {(r["station_code"], r["date"]): r["observation_status"] for r in rows}
    ledger = dataset.get("missing_days", [])
    _require(isinstance(ledger, list), "Invalid missing_days ledger")
    ledger_seen = set()
    for item in ledger:
        _require(isinstance(item, dict), "Invalid missing day")
        code, day = item.get("station_code"), item.get("date")
        _require(isinstance(code, str) and code in stations, "Unknown ledger station")
        _day(day)
        key = (code, day)
        _require(key not in ledger_seen, "Duplicate ledger station/date")
        ledger_seen.add(key)
        status = item.get("observation_status")
        _require(status in OBSERVATION_STATUSES - {"OBSERVED", "OUTSIDE_OFFICIAL_PERIOD"}, "Invalid ledger status")
        _require((key in seen) == (status == "RECOVERED"), "Ledger/observation contradiction")
        if status == "RECOVERED":
            _require(row_statuses[key] == "RECOVERED", "Recovered status mismatch")


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read JSON: " + str(path)) from exc


def read_dataset(path):
    """Read and validate; callers hash the original file separately for manifests."""
    dataset = _json(path)
    validate_dataset(dataset)
    return dataset


def _verified_bytes(base, entry):
    _require(isinstance(entry, dict) and isinstance(entry.get("path"), str), "Missing artifact path")
    _hash(entry.get("sha256"))
    try:
        content = (base / entry["path"]).read_bytes()
    except OSError as exc:
        raise ValueError("Missing artifact: " + entry["path"]) from exc
    _require(hashlib.sha256(content).hexdigest() == entry["sha256"], "Artifact hash mismatch: " + entry["path"])
    return content


def _csv(content, header):
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")), strict=True)
        _require(reader.fieldnames == list(header), "CSV header mismatch")
        rows = list(reader)
        _require(all(None not in row and all(value is not None for value in row.values()) for row in rows), "Malformed CSV row")
        return rows
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("Invalid CSV") from exc


def _latitude(value):
    value = str(value).strip()
    match = re.fullmatch(r"(\d{2})(\d{2})(\d{2})(\d{3})([NS])", value)
    if match:
        deg, minute, second, millis = map(int, match.groups()[:4])
        _require(minute < 60 and second < 60, "Invalid latitude DMS")
        return (deg + minute/60 + (second + millis/1000)/3600) * (-1 if match[5] == "S" else 1)
    latitude = float(value)
    _require(_finite(latitude) and -90 < latitude < 90, "Invalid catalog latitude")
    return latitude


def build_dataset(handoff_path):
    """Build an in-memory dataset ONLY from a verified, authorized future closeout.

    No network requests, writes, raw mutations, recovery or implicit authorization.
    Report snapshots must be internally consistent; retaining FAIL is deliberate
    because outside-period evidence is not erased to manufacture a PASS.
    """
    handoff_path = Path(handoff_path)
    handoff = _json(handoff_path)
    _require(isinstance(handoff, dict) and handoff.get("gate") == "SIAR-QUALITY-CLOSEOUT" and handoff.get("authorized") is True, "Authorized SIAR-QUALITY-CLOSEOUT handoff required")
    _repo_sha(handoff.get("source_repo_sha"), "source_repo_sha")
    _repo_sha(handoff.get("audit_source_repo_sha"), "audit_source_repo_sha")
    post = handoff.get("quality_post")
    package_bytes = _verified_bytes(handoff_path.parent, post)
    package = json.loads(package_bytes)
    _require(isinstance(package, dict) and package.get("phase") == "POST", "Verified Quality POST phase required")
    reports = package.get("reports")
    _require(isinstance(reports, dict) and set(reports) == set(REPORT_HEADERS), "All three Quality POST reports required")
    package_base = (handoff_path.parent / post["path"]).parent
    parsed = {name: _csv(_verified_bytes(package_base, reports[name]), header.split(",")) for name, header in REPORT_HEADERS.items()}
    audits = parsed["siar_station_audit.csv"]
    manifests = parsed["siar_dataset_manifest.csv"]
    missing = parsed["siar_missing_days.csv"]
    _require(audits and manifests, "Empty Quality station/manifest reports")
    _require(isinstance(post.get("id"), str) and post["id"], "Missing POST id")
    for row in audits + manifests + missing:
        _require(row["audit_id"] == post["id"] and row["repo_git_sha"] == handoff["audit_source_repo_sha"], "Quality audit_source_repo_sha provenance mismatch")
        _require(all(row[k] for k in COMMON.split(",")), "Missing Quality traceability")
        _require(all(row[k] == audits[0][k] for k in COMMON.split(",")), "Inconsistent Quality traceability")
    catalog = handoff.get("catalog")
    catalog_bytes = _verified_bytes(handoff_path.parent, catalog)
    _require(catalog["sha256"] == audits[0]["estaciones_catalog_sha256"], "Catalog hash differs from Quality POST")
    catalog_reader = csv.DictReader(io.StringIO(catalog_bytes.decode("utf-8-sig")))
    _require(catalog_reader.fieldnames is not None and {"Codigo", "Latitud", "Altitud"} <= set(catalog_reader.fieldnames), "Catalog fields missing")
    catalog_rows = list(catalog_reader)
    catalog_by_code = {row["Codigo"]: row for row in catalog_rows}
    _require(len(catalog_rows) == len(catalog_by_code), "Duplicate catalog code")
    audit_by_code = {row["station_code"]: row for row in audits}
    _require(len(audit_by_code) == len(audits), "Duplicate station audit")
    manifest_by_code = {row["station_code"]: row for row in manifests}
    _require(len(manifest_by_code) == len(manifests) and set(audit_by_code) == set(manifest_by_code), "Quality station set mismatch")
    _require(isinstance(handoff.get("stations"), dict) and set(handoff["stations"]) == set(audit_by_code), "Station metadata set mismatch")
    for code, station in handoff["stations"].items():
        _require(code in catalog_by_code, "Station absent from verified catalog")
        metadata = catalog_by_code[code]
        _require(_finite(station.get("latitude_deg")) and abs(station["latitude_deg"] - _latitude(metadata["Latitud"])) < 1e-8, "Latitude/catalog mismatch")
        altitude = float(metadata["Altitud"]) if metadata["Altitud"].strip() else None
        _require(station.get("altitude_m") == altitude, "Altitude/catalog mismatch")
    sources = handoff.get("source_data_hashes")
    _require(isinstance(sources, dict) and set(sources) == {row["canonical_file"] for row in manifests}, "Source file set differs from Quality manifest")
    dataset = {key: handoff.get(key) for key in ("dataset_id", "audit_source_repo_sha", "source_repo_sha", "source_data_hashes", "units", "stations")}
    dataset.update(schema_version="observational-v1", quality_post={"id": post["id"], "sha256": post["sha256"],
                   "path": str((handoff_path.parent/post["path"]).resolve()),
                   "reports": {name: {"path": str((package_base/entry["path"]).resolve()), "sha256": entry["sha256"]} for name, entry in reports.items()}},
                   catalog={"path": str((handoff_path.parent/catalog["path"]).resolve()), "sha256": catalog["sha256"]},
                   assumptions={"units_interpretation": "RESOLVED_FOR_V1", "units_provenance": "UNITS_PROVENANCE_PARTIAL",
                                "wind_height_assumption": "2 m / SIAR convention", "station_period_provenance": "not exhaustively verified"},
                   rows=[], missing_days=[])
    for key in ("units_interpretation", "units_provenance", "units_provenance_status"):
        if key in handoff:
            dataset[key] = handoff[key]
    periods, observed = {}, {}
    for code, manifest in manifest_by_code.items():
        audit = audit_by_code[code]
        for key in set(audit) & set(manifest):
            _require(audit[key] == manifest[key], "Audit/manifest mismatch: " + key)
        _require(manifest["audit_status"] in {"PASS", "WARN", "FAIL"}, "Quality POST is incomplete")
        _require(manifest["column_count"] == "23" and manifest["schema_23_columns_ok"] == "true", "Source schema not certified")
        _require(all(manifest[k] == "0" for k in ("duplicate_date_count", "invalid_date_count", "estacion_value_mismatch_count")), "Unresolved source integrity errors")
        start, end = _day(manifest["expected_start_date"]), _day(manifest["expected_end_date"])
        _require(start <= end and int(manifest["official_period_days"]) == (end-start).days+1, "Invalid official period")
        _require(audit["station_state"] in {"ACTIVE", "CLOSED"}, "Unknown station state")
        _require(_day(audit["fecha_instalacion"]) == start, "Installation/period mismatch")
        _require(_day(audit["active_cutoff_date"] if audit["station_state"] == "ACTIVE" else audit["fecha_baja"]) == end, "Official end mismatch")
        periods[code] = (start, end)
        path = manifest["canonical_file"]
        # Audit -> canonical continuity is byte identity, not Git SHA equality.
        _hash(sources[path], "source_data_hashes SHA-256")
        _hash(manifest["canonical_sha256"], "POST canonical_sha256")
        _require(sources[path] == manifest["canonical_sha256"], "Source/manifest SHA mismatch")
        content = _verified_bytes(handoff_path.parent, {"path": path, "sha256": sources[path]})
        _require(len(content) == int(manifest["file_size_bytes"]), "Source size mismatch")
        raw_rows = _csv(content, COLUMNS)
        _require(len(raw_rows) == int(manifest["row_count"]), "Source row count mismatch")
        days = []
        for index, raw in enumerate(raw_rows, start=2):
            day = _source_day(raw["Fecha"])
            _require(raw["Estacion"] == code, "Source station mismatch")
            _require((code, day.isoformat()) not in observed, "Duplicate source station/date")
            values = {k: raw[k] if k in {"Fecha", "Estacion"} else (None if raw[k].strip() == "" else float(raw[k])) for k in COLUMNS}
            nonfinite = [key for key, value in values.items() if isinstance(value, float) and not math.isfinite(value)]
            for key in nonfinite:
                values[key] = None
            outside = not start <= day <= end
            row = {"station_code": code, "date": day.isoformat(), "values": values, "observation_status": "OUTSIDE_OFFICIAL_PERIOD" if outside else "OBSERVED", "quality_status": manifest["audit_status"], "quality_flags": ["outside_official_period"] if outside else [], "source_data_sha": sources[path], "provenance": {"source_file": path, "source_row": index, "quality_post_id": post["id"]}}
            row["quality_flags"].extend("source_nonfinite:" + key for key in nonfinite)
            dataset["rows"].append(row)
            observed[(code, day.isoformat())] = row
            days.append(day)
        inside = sum(start <= day <= end for day in days)
        _require(inside == int(manifest["days_with_data"]), "Quality observed-day count mismatch")
        _require(len(days)-inside == int(manifest["rows_outside_official_period_count"]), "Quality outside-period count mismatch")
        _require((min(days).isoformat() if days else "") == manifest["first_observation_date"] and (max(days).isoformat() if days else "") == manifest["last_observation_date"], "Quality date range mismatch")
        _require(manifest["chronological_order_ok"] == str(days == sorted(days)).lower(), "Quality chronology mismatch")
        _require(int(audit["missing_day_count"]) == (end-start).days+1-inside, "Quality missing-day count mismatch")
    seen_gaps = set()
    for gap in missing:
        code, day = gap["station_code"], gap["observation_date"]
        _require(code in periods, "Unknown gap station")
        parsed_day = _day(day)
        _require(periods[code][0] <= parsed_day <= periods[code][1], "Gap outside official period")
        key = (code, day)
        _require(key not in seen_gaps, "Duplicate missing-day evidence")
        seen_gaps.add(key)
        status, query = gap["gap_status"], gap["last_query_result"]
        _require(status in {"RECOVERED", "CONFIRMED_NO_DATA", "PENDING_RETRY"}, "Unknown Quality gap status")
        _require(query in {"NOT_ATTEMPTED", "ZERO_ROWS", "DATA_RETURNED", "HTTP_ERROR", "INVALID_RESPONSE", "QUOTA"}, "Unknown query result")
        _require(int(gap["attempt_count"]) >= 0, "Invalid attempt count")
        if status == "RECOVERED":
            _require(key in observed and query == "DATA_RETURNED" and gap["recovered_at_utc"] and gap["evidence_ref"] and int(gap["siar_records_returned"]) > 0, "Recovered gap lacks evidence/observation")
            observed[key]["observation_status"] = "RECOVERED"
        else:
            _require(key not in observed, "Missing-day evidence conflicts with observation")
            if status == "CONFIRMED_NO_DATA":
                _require(query == "ZERO_ROWS" and gap["siar_records_returned"] == "0" and gap["evidence_ref"] and int(gap["attempt_count"]) > 0, "Confirmed absence lacks positive evidence")
            else:
                _require(query in {"NOT_ATTEMPTED", "HTTP_ERROR", "INVALID_RESPONSE", "QUOTA"}, "Pending gap conflicts with query result")
        dataset["missing_days"].append({"station_code": code, "date": day, "observation_status": status, "provenance": dict(gap)})
    for code, (start, end) in periods.items():
        day = start
        while day <= end:
            key = (code, day.isoformat())
            if key not in observed and key not in seen_gaps:
                dataset["missing_days"].append({"station_code": code, "date": day.isoformat(), "observation_status": "MISSING", "provenance": {"quality_post_id": post["id"], "reason": "no_observation_and_no_query_evidence"}})
            day += timedelta(days=1)
    dataset["rows"].sort(key=lambda row: (row["station_code"], row["date"]))
    dataset["missing_days"].sort(key=lambda row: (row["station_code"], row["date"]))
    validate_dataset(dataset)
    from .quality import eligibility, scientific_flags
    for row in dataset["rows"]:
        station = dataset["stations"][row["station_code"]]
        row["scientific_flags"] = scientific_flags(row, station)
        row["eligibility"] = {}
        for method in ("pm", "hargreaves"):
            reasons = eligibility(row, station, method)
            row["eligibility"][method] = {"eligible": not reasons, "reasons": reasons}
    return dataset
