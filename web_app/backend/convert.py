"""One-way, offline packaging of approved outputs. Never executes science."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import duckdb

from .catalog import CSV_SHA, DATASET_SHA, LIMITATIONS, METHOD_IDS, MODELS


def sha(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def package(source: Path, output: Path):
    source = source.resolve()
    csv_path, metrics_path, manifest_path = [source / n for n in
        ("classical-baselines.csv", "metrics.json", "manifest.json")]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha(csv_path) != CSV_SHA:
        raise ValueError("CSV SHA does not match the approved four-baseline publication")
    if manifest["dataset_sha256"] != DATASET_SHA:
        raise ValueError("Unexpected Dataset v1 identity")
    if sha(manifest_path) != "78d249f3736ff0aeefbdd4599b4f2e77ded0c76c21bff26f274664e282ebee62":
        raise ValueError("Manifest does not match approved publication")
    if manifest["status"] != "SUCCESS" or sha(metrics_path) != manifest["outputs"]["metrics.json"]["sha256"]:
        raise ValueError("Scientific publication is not PASS or metrics hash mismatches")
    # These are the producer's actual catalog bytes, not a regenerated snapshot.
    catalog_path = Path(manifest["catalog"]["path"])
    if sha(catalog_path) != manifest["catalog"]["sha256"]:
        raise ValueError("Catalog hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("CREATE TABLE observations AS SELECT * FROM read_csv(?, all_varchar=true)", [str(csv_path)])
        n = db.execute("SELECT count(*) FROM observations").fetchone()[0]
        if n != 75496:
            raise ValueError("Unexpected row count")
        db.execute("COPY observations TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(output / "observations.parquet")])
        stats = db.execute("SELECT station_code, min(date), max(date), count(*), min(quality_status) FROM observations GROUP BY station_code ORDER BY station_code").fetchall()
    with catalog_path.open(encoding="utf-8-sig", newline="") as stream:
        catalog = {row["Codigo"]: row for row in csv.DictReader(stream)}
    audit_source = manifest["quality_post"]["reports"]["siar_station_audit.csv"]
    audit_path = Path(audit_source["path"])
    if sha(audit_path) != audit_source["sha256"]:
        raise ValueError("Quality audit hash mismatch")
    with audit_path.open(encoding="utf-8-sig", newline="") as stream:
        audit = {row["station_code"]: row for row in csv.DictReader(stream)}
    stations = []
    for code, start, end, rows, quality in stats:
        item = catalog[code]
        quality_row = audit[code]
        stations.append(dict(code=code, name=item["Estacion"], status="CLOSED" if item["Fecha_Baja"] else "ACTIVE",
            **{"from": start, "to": end}, rows=rows, quality_status=quality,
            official_from=quality_row["expected_start_date"], official_to=quality_row["expected_end_date"],
            expected_days=int(quality_row["official_period_days"]), observed_days=int(quality_row["days_with_data"]),
            missing_days=int(quality_row["missing_day_count"]), coverage_percent=float(quality_row["coverage_percent"])))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for metric in metrics:
        metric["model"] = METHOD_IDS[metric["method"]]
    artifacts = {"observations.parquet": sha(output / "observations.parquet")}
    for name, value in (("metrics.json", metrics), ("stations.json", stations), ("models.json", MODELS)):
        path = output / name
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        artifacts[name] = sha(path)
    info = dict(schema_version="web-v0.1", conversion_version="1.0.0", dataset_id="siar-balears-observational-v1",
        dataset_sha256=DATASET_SHA, source_sha256=CSV_SHA, rows=n,
        quality_post="SIAR-QA-POST-20261020T193140Z", pipeline_sha=manifest["pipeline"]["git_sha"],
        result_run=manifest["run_id"], limitations=LIMITATIONS,
        reference="EtPMon · referencia calculada SIAR", source_metrics_sha256=sha(metrics_path),
        source_manifest_sha256=sha(manifest_path), catalog_sha256=sha(catalog_path), quality_audit_sha256=sha(audit_path),
        audit_source_repo_sha=manifest["audit_source_repo_sha"], source_repo_sha=manifest["source_repo_sha"],
        artifacts=artifacts, derived_sha256=artifacts["observations.parquet"])
    (output / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.output), indent=2))
