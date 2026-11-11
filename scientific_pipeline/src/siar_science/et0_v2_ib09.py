"""Bind the V2 exception to IB09's preserved source records and Quality ledger."""
from datetime import timedelta
import json

from .datasets import COLUMNS, REPORT_HEADERS, _csv, _source_day
from .quality import finite
from .quality_ib09_v2 import IB09_END, IB09_RAW_SHA256, IB09_START
from .serialization import digest


RAW_PATH = "datos_siar_baleares/IB09_datos_completos.csv"
CATALOG_PATH = "estaciones_baleares.csv"
POST_ID = "SIAR-QA-POST-20261020T193140Z"
POST_DIR = f"data_quality/siar/post/{POST_ID}"
CATALOG_SHA = "98abd16dcd6ea42ad46a30c331df47150f07dffdca140c14e4445890ce68a357"
LEDGER_SHA = "55a78e6affff09e2a1e8cb28e0465b2073ad79a1e573a1efe272a52462ba3969"
POST_SHA = "a2d1293fcc54254769dbc80c7105b08b4b03b6a14392df0a4e5ef19b4ef51d07"
PENDING_ID = "IB09-QUALITY-HANDOFF-V2-20261111"
PENDING_SHA = "0592a57c73691409ef062272577204bf081b0dec111815f4a2197ed100c5b8bc"
SOURCE_SHA = "97aa8cc0062de5ba6d2ec1af7668495a352c3777"
AUDIT_SHA = "18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865"
OUTSIDE = "2005-10-23"
PENDING_PATH = "data_quality/siar/ib09-handoff-v2/handoff.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_bytes(path, sha):
    content = path.read_bytes()
    require(digest(content) == sha, f"SHA256 mismatch: {path}")
    return content


def verify_admission(repo, path, lock, *, official_evidence_contradiction=None):
    """Fail closed on a pending, superseded, altered or out-of-scope decision."""
    content = path.read_bytes()
    handoff = json.loads(content)
    verification = handoff.get("official_period_verification") or {}
    admission = handoff.get("admission") or {}
    require(handoff.get("decision_effective") is True and admission.get("status") == "APPROVED",
            "IB09 handoff is not effective for V2 admission")
    require(official_evidence_contradiction is False,
            "STOP IB09: official evidence contradicts the catalog or has not been reviewed")
    require(verification.get("status") == "UNRESOLVED"
            and all(verification.get(key) is None for key in ("start", "end", "evidence")),
            "STOP IB09: independent period evidence changed; reconcile with Dirección 00")
    require(handoff.get("schema_version") == "2.0.0"
            and handoff.get("handoff_id") == lock["handoff_id"] and digest(content) == lock["sha256"],
            "IB09 authorized handoff identity mismatch")
    require(all(handoff.get(key) == value for key, value in {
        "station_code": "IB09", "quality_v1_status": "FAIL", "decision_authority": "Direccion 00",
        "decision_id": "DIRECTION-00-IB09-DEVELOPMENT-20261111",
        "official_period_verification_status": "UNRESOLVED",
        "v2_scientific_admission_status": "APPROVED",
        "raw_path": RAW_PATH, "raw_sha256": IB09_RAW_SHA256, "raw_rows": 4423,
        "observed_official_rows": 4422, "outside_period_rows": 1,
        "outside_dates": [OUTSIDE], "confirmed_no_data_count": 313,
        "catalog_path": CATALOG_PATH, "catalog_sha256": CATALOG_SHA,
        "ledger_path": f"{POST_DIR}/siar_missing_days.csv", "ledger_sha256": LEDGER_SHA,
        "source_repo_sha": SOURCE_SHA, "audit_source_repo_sha": AUDIT_SHA,
        "catalog_period": {"start": IB09_START.isoformat(), "end": IB09_END.isoformat(), "inclusive": True},
        "quality_post": {"id": POST_ID, "path": f"{POST_DIR}/quality-post-package.json", "sha256": POST_SHA},
        "supersedes": {"handoff_id": PENDING_ID, "path": PENDING_PATH, "sha256": PENDING_SHA},
    }.items()) and admission.get("scope") == "DEVELOPMENT_ONLY"
        and admission.get("basis") == "DIRECTION_00_CATALOG_EXCEPTION",
        "IB09 handoff does not authorize exactly the locked V2 exception")
    verified_bytes(repo / PENDING_PATH, PENDING_SHA)
    decision = admission["decision_ref"]
    verified_bytes(repo / decision["path"], decision["sha256"])
    return {"handoff_id": lock["handoff_id"], "handoff_sha256": lock["sha256"], "supersedes": PENDING_ID,
            "decision_id": handoff["decision_id"], "decision": decision,
            "status": "APPROVED", "scope": "DEVELOPMENT_ONLY", "basis": admission["basis"],
            "official_period_verification": verification["status"],
            "known_official_evidence_contradiction": False,
            "source_repo_sha": SOURCE_SHA, "audit_source_repo_sha": AUDIT_SHA}


def reconcile(repo, dataset):
    """Compare all 23 fields and physical source record identity for every row."""
    raw = _csv(verified_bytes(repo / RAW_PATH, IB09_RAW_SHA256), COLUMNS)
    catalog = _csv(verified_bytes(repo / CATALOG_PATH, CATALOG_SHA),
                   "Estacion Codigo Termino Longitud Latitud Altitud XUTM YUTM Huso Fecha_Instalacion Fecha_Baja".split())
    [entry] = [item for item in catalog if item["Codigo"] == "IB09"]
    require((_source_day(entry["Fecha_Instalacion"]), _source_day(entry["Fecha_Baja"])) ==
            (IB09_START, IB09_END), "IB09 catalog period mismatch")
    ledger_name = "siar_missing_days.csv"
    post = json.loads(verified_bytes(repo / POST_DIR / "quality-post-package.json", POST_SHA))
    require(dataset["quality_post"]["id"] == POST_ID and dataset["quality_post"]["sha256"] == POST_SHA
            and dataset["audit_source_repo_sha"] == AUDIT_SHA
            and dataset["catalog"]["sha256"] == CATALOG_SHA,
            "IB09 Dataset V1 Quality or catalog identity mismatch")
    for name, header in REPORT_HEADERS.items():
        report = post["reports"][name]
        require(report["path"] == name, "Unexpected Quality report path")
        records = _csv(verified_bytes(repo / POST_DIR / name, report["sha256"]), header.split(","))
        station_records = [record for record in records if record["station_code"] == "IB09"]
        if name == ledger_name:
            require(report["sha256"] == LEDGER_SHA, "IB09 ledger identity mismatch")
            ledger = station_records
            continue
        [station_report] = station_records
        require(all(station_report[key] == value for key, value in {
            "audit_status": "FAIL", "canonical_sha256": IB09_RAW_SHA256, "row_count": "4423",
            "days_with_data": "4422", "rows_outside_official_period_count": "1",
            "expected_start_date": IB09_START.isoformat(), "expected_end_date": IB09_END.isoformat(),
        }.items()), "IB09 POST station evidence mismatch")

    existing = {row["date"]: row for row in dataset["rows"] if row["station_code"] == "IB09"}
    require(len(raw) == len(existing) == 4423, "IB09 raw or Dataset V1 count mismatch")
    seen, inside, outside = set(), set(), set()
    for index, record in enumerate(raw, start=2):
        day = _source_day(record["Fecha"])
        key = day.isoformat()
        require(key not in seen and record["Estacion"] == "IB09", "IB09 duplicate or foreign source row")
        seen.add(key)
        row = existing.get(key)
        is_inside = IB09_START <= day <= IB09_END
        (inside if is_inside else outside).add(key)
        expected_status = "OBSERVED" if is_inside else "OUTSIDE_OFFICIAL_PERIOD"
        require(row is not None and row["observation_status"] == expected_status and row["quality_status"] == "FAIL"
                and row["source_data_sha"] == IB09_RAW_SHA256
                and row["provenance"] == {"source_file": RAW_PATH.split("/")[-1], "source_row": index,
                                          "quality_post_id": POST_ID}, f"IB09 provenance mismatch: {key}")
        for column, value in record.items():
            if column not in ("Fecha", "Estacion"):
                value = float(value) if value.strip() else None
                value = value if finite(value) else None
            require(row["values"][column] == value, f"IB09 source value mismatch: {key}/{column}")
    require(len(inside) == 4422 and outside == {OUTSIDE}, "IB09 period reconciliation failed")
    expected_days = {(IB09_START + timedelta(days=n)).isoformat()
                     for n in range((IB09_END - IB09_START).days + 1)}
    dataset_ledger = {row["date"]: row for row in dataset["missing_days"] if row["station_code"] == "IB09"}
    missing = {record["observation_date"] for record in ledger}
    require(len(ledger) == len(missing) == 313 and missing == expected_days - inside
            and dataset_ledger.keys() == missing, "IB09 calendar/ledger reconciliation failed")
    for record in ledger:
        day = record["observation_date"]
        require(record["gap_status"] == "CONFIRMED_NO_DATA" and record["last_query_result"] == "ZERO_ROWS"
                and record["siar_records_returned"] == "0" and record["evidence_ref"]
                and dataset_ledger[day]["observation_status"] == "CONFIRMED_NO_DATA"
                and dataset_ledger[day]["provenance"] == record, f"IB09 missing-day evidence mismatch: {day}")
    evidence = {
        "raw_sha256": IB09_RAW_SHA256, "catalog_sha256": CATALOG_SHA, "quality_post_id": POST_ID,
        "quality_post_sha256": POST_SHA, "ledger_sha256": LEDGER_SHA,
        "source_rows": len(seen), "values_reconciled_per_row": len(COLUMNS),
        "catalog_interior": len(inside), "outside_dates": sorted(outside), "confirmed_no_data": len(missing),
        "catalog_days": len(expected_days), "quality_v1_status": "FAIL",
        "catalog_period": [IB09_START.isoformat(), IB09_END.isoformat()], "inclusive": True,
    }
    return inside, evidence
