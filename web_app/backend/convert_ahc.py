"""Package published AHC view tables for WEB-V0.2 without recalculating science."""
import argparse
import json
import shutil
from pathlib import Path

import duckdb

from .catalog import AHC_MODELS, CSV_SHA, DATASET_SHA, LIMITATIONS
from .convert import sha

VIEW_MANIFEST_SHA = "f6dc102121f348d1fe9e8d45ecc824a3d1565421ca37285f2b75fc0e6b30156a"
TEST_MANIFEST_SHA = "502d7a39b82c8b5032931370ae8f05f4d8e7ecbbb93f6946810d0e47434fc469"
CLOSEOUT_MANIFEST_SHA = "39be9d706f962722862c50ce6b25fad5d0e93a57d954c4b71aa77cfe5942d6cd"
PROTOCOL_SHA = "9ae64c2089cf650fe8b3c9390182075a14c3f3d47f1addd7db026ab0fd740090"

TABLES = {
    "daily_predictions.csv": "ahc_predictions.parquet",
    "metrics.csv": "ahc_metrics.parquet",
    "ahc_coefficients.csv": "ahc_coefficients.parquet",
    "model_metadata.csv": "ahc_model_metadata.parquet",
    "coverage_metrics.csv": "ahc_coverage.parquet",
    "station_metadata.csv": "ahc_station_metadata.parquet",
}


def _safe_copy_base(base: Path, output: Path):
    info = json.loads((base / "info.json").read_text(encoding="utf-8"))
    if info.get("dataset_sha256") != DATASET_SHA or info.get("source_sha256") != CSV_SHA:
        raise ValueError("Unapproved WEB-V0.1 base package")
    if info.get("schema_version") != "web-v0.1":
        raise ValueError("Expected immutable WEB-V0.1 input package")
    for name, digest in info["artifacts"].items():
        if Path(name).name != name or sha(base / name) != digest:
            raise ValueError(f"Base derivative integrity failure: {name}")
        shutil.copy2(base / name, output / name)
    return info


def _validate_columns(db, csv_path: Path, schema: dict, source_name: str):
    escaped = str(csv_path).replace("'", "''")
    actual = [row[0] for row in db.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto('{escaped}', header=true, sample_size=-1)"
    ).fetchall()]
    expected = [column["name"] for column in schema[source_name]["columns"]]
    if actual != expected:
        raise ValueError(f"Schema mismatch for {source_name}")


def package(base: Path, ahc: Path, output: Path):
    base, ahc, output = base.resolve(), ahc.resolve(), output.resolve()
    test, closeout = ahc / "test", ahc / "closeout"
    view_manifest_path, test_manifest_path = test / "view-manifest.json", test / "manifest.json"
    closeout_manifest_path = closeout / "manifest.json"
    if sha(view_manifest_path) != VIEW_MANIFEST_SHA or sha(test_manifest_path) != TEST_MANIFEST_SHA:
        raise ValueError("AHC TEST manifests do not match the approved publication")
    if sha(closeout_manifest_path) != CLOSEOUT_MANIFEST_SHA:
        raise ValueError("AHC closeout manifest does not match the approved publication")
    view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    closeout_manifest = json.loads(closeout_manifest_path.read_text(encoding="utf-8"))
    if any(doc["status"] != "SUCCESS" for doc in (view_manifest, test_manifest, closeout_manifest)):
        raise ValueError("AHC publication is not PASS")
    if any(doc["dataset_sha256"] != DATASET_SHA for doc in (view_manifest, test_manifest, closeout_manifest)):
        raise ValueError("Unexpected AHC Dataset v1 identity")
    if view_manifest["protocol_sha256"] != PROTOCOL_SHA or view_manifest.get("science_recalculation_allowed") is not False:
        raise ValueError("Unexpected AHC visualization contract identity")
    schema_path = test / "view_schemas.json"
    if sha(schema_path) != view_manifest["outputs"]["view_schemas.json"]["sha256"]:
        raise ValueError("AHC view schema hash mismatch")
    schemas = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    output.mkdir(parents=True, exist_ok=False)
    base_info = _safe_copy_base(base, output)
    source_hashes, artifacts = {}, dict(base_info["artifacts"])
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        for source_name, target_name in TABLES.items():
            source_path = test / source_name
            expected = view_manifest["outputs"][source_name]
            if sha(source_path) != expected["sha256"]:
                raise ValueError(f"AHC source hash mismatch: {source_name}")
            _validate_columns(db, source_path, schemas, source_name)
            escaped_source = str(source_path).replace("'", "''")
            escaped_target = str(output / target_name).replace("'", "''")
            db.execute(f"COPY (SELECT * FROM read_csv_auto('{escaped_source}', header=true, sample_size=-1)) TO '{escaped_target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            rows = db.execute(f"SELECT count(*) FROM read_parquet('{escaped_target}')").fetchone()[0]
            if rows != expected["rows"]:
                raise ValueError(f"AHC row count mismatch: {source_name}")
            source_hashes[f"test/{source_name}"] = expected["sha256"]
            artifacts[target_name] = sha(output / target_name)
        climatology_source = closeout / "ahc_climatology_coefficients.csv"
        climatology_expected = closeout_manifest["outputs"]["ahc_climatology_coefficients.csv"]
        if sha(climatology_source) != climatology_expected["sha256"]:
            raise ValueError("AHC climatology source hash mismatch")
        escaped_source = str(climatology_source).replace("'", "''")
        target = output / "ahc_climatology_coefficients.parquet"
        escaped_target = str(target).replace("'", "''")
        db.execute(f"COPY (SELECT * FROM read_csv_auto('{escaped_source}', header=true, sample_size=-1)) TO '{escaped_target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        if db.execute(f"SELECT count(*) FROM read_parquet('{escaped_target}')").fetchone()[0] != climatology_expected["rows"]:
            raise ValueError("AHC climatology row count mismatch")
        source_hashes["closeout/ahc_climatology_coefficients.csv"] = climatology_expected["sha256"]
        artifacts[target.name] = sha(target)
    ahc_models_path = output / "ahc_models.json"
    ahc_models_path.write_text(json.dumps(AHC_MODELS, ensure_ascii=False), encoding="utf-8")
    artifacts[ahc_models_path.name] = sha(ahc_models_path)
    info = dict(base_info)
    info.update(
        schema_version="web-v0.2-ahc",
        conversion_version="2.0.0",
        web_version="0.2.0",
        limitations=LIMITATIONS,
        ahc_status="AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION=PASS",
        ahc_run_id=view_manifest["run_id"],
        ahc_pipeline_sha=view_manifest["pipeline"]["git_sha"],
        ahc_protocol_sha256=PROTOCOL_SHA,
        ahc_view_manifest_sha256=VIEW_MANIFEST_SHA,
        ahc_test_manifest_sha256=TEST_MANIFEST_SHA,
        ahc_closeout_manifest_sha256=CLOSEOUT_MANIFEST_SHA,
        ahc_test_status=view_manifest["test_status"],
        split_semantics={"ALL": "published rows from every split; no combined metrics", "TRAIN": "ajuste", "VALIDATION": "selección", "TEST": "retrospective_locked"},
        data_recalculated=False,
        scientific_artifacts_modified=False,
        ahc_source_hashes=source_hashes,
        artifacts=artifacts,
        derived_sha256=artifacts["ahc_predictions.parquet"],
    )
    (output / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ahc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.base, args.ahc, args.output), indent=2))
