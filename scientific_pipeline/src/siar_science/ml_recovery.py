"""Append-only evaluation recovery for an interrupted, frozen ML campaign.

This module deliberately imports no candidate selection, fitting, or refit
functions. It validates persisted DEVELOPMENT artifacts, then evaluates their
already-refitted models on TEST exactly once for this recovery run.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

from .ml_campaign import (OFFICIAL_AHC_TEST_SHA256, OFFICIAL_DATASET_SHA256,
                          OFFICIAL_PROTOCOL_FILE_SHA256, _context_for_split,
                          _candidates, _resource_usage, _test_predictions, _thread_limits)
from .ml_classical import ALGORITHMS, candidate_id, score_predictions, validate_ml_protocol
from .ml_pipeline import _csv, _model_diagnostics, _runtime, _write
from .pipeline import _identity, _utc, safe_new_output
from .serialization import digest, json_bytes

INTERRUPTED_RUN_ARTIFACT_SHA256 = {
    "selection.json": "a41ee87956368d6c0e1fae2b39177ffe8c42cae7658477172c46725409f829c7",
    "frozen_configurations.json": "6425f876fa5880090e0c97cdec60ffe72e82880bfa296f04f152eed113f65a86",
    "refitted_models.joblib": "b1be36a43ff465cbf8662e6182c3c862037d1bc6bbc8295320013c99e7893946",
    "refit_metadata.json": "2c2cf1a1099ec3d728e18d0f9f99065ddcba3004e51491330fd1e3d9f56f5b6f",
    "candidate_summaries.json": "3b24d6fffaabdebb117c075406be2d518e73cf1be04673cf659c44d82acc7927",
    "rolling_fold_metrics.csv": "3e275c5153e4b6028213d55795d8ab48eae2c1ef05814f1477ea25dc4721ecbe",
    "protocol.json": OFFICIAL_PROTOCOL_FILE_SHA256,
}


def _journal(path, state, **details):
    record = {"state": state, "at_utc": _utc(), **details}
    with Path(path).open("ab") as stream:
        stream.write((json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _load_json(path):
    content = Path(path).read_bytes()
    return json.loads(content), digest(content)


def _file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()



def _verify_ahc_output_dependencies(ahc_manifest_path):
    """Verify every immutable AHC output before any Dataset v1 read."""
    manifest_bytes = Path(ahc_manifest_path).read_bytes()
    manifest_sha = digest(manifest_bytes)
    if manifest_sha != OFFICIAL_AHC_TEST_SHA256:
        raise ValueError("Official AHC TEST manifest hash mismatch")
    ahc_manifest = json.loads(manifest_bytes)
    outputs = ahc_manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("AHC manifest has no declared outputs")
    root = Path(ahc_manifest_path).resolve().parent
    verified = {}
    for name, entry in sorted(outputs.items()):
        if not isinstance(entry, dict):
            raise ValueError("Invalid AHC output entry: " + str(name))
        relative = Path(entry.get("path", ""))
        if not entry.get("path") or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe or missing AHC output path: " + str(name))
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("AHC output escapes manifest directory: " + str(name))
        if not path.is_file():
            raise FileNotFoundError("Missing AHC manifest output: " + str(name) + " (" + str(path) + ")")
        actual_sha = _file_sha256(path)
        if actual_sha != entry.get("sha256"):
            raise ValueError("AHC manifest output hash mismatch: " + str(name))
        verified[name] = {"path": relative.as_posix(), "sha256": actual_sha}
    return {"manifest_sha256": manifest_sha, "output_count": len(verified), "outputs": verified}


def _validate_frozen_source(source_run, dataset_path, ahc_manifest_path, protocol_path):
    source = Path(source_run).resolve()
    stage = source / ".staging" if (source / ".staging").is_dir() else source
    if not source.is_dir():
        raise ValueError("Campaign source directory is required")
    # The source run is evidence and is only read. No writes are made below it.
    selection, selection_sha = _load_json(stage / "selection.json")
    frozen, frozen_sha = _load_json(stage / "frozen_configurations.json")
    refit, refit_sha = _load_json(stage / "refit_metadata.json")
    candidate_summaries, candidate_summaries_sha = _load_json(stage / "candidate_summaries.json")
    protocol, protocol_file_sha = _load_json(protocol_path)
    validate_ml_protocol(protocol)
    protocol_sha = digest(json_bytes(protocol))
    if selection.get("status") != "FROZEN" or selection.get("selection_split") != "DEVELOPMENT_FOLDS_F1_F4":
        raise ValueError("Persisted selection is not frozen from DEVELOPMENT F1-F4")
    if selection.get("test_inspected") is not False:
        raise ValueError("Selection does not attest that TEST was excluded")
    if selection.get("candidate_count") != 185 or selection.get("selected_cell_count") != 35 or len(selection.get("selected", [])) != 35:
        raise ValueError("Frozen selection does not contain the official 185/35 candidate structure")
    if protocol.get("experiment_id") != "ML_V1_CLASSICAL_ROLLING_20261031":
        raise ValueError("Recovery requires the frozen official rolling protocol")
    if protocol_file_sha != OFFICIAL_PROTOCOL_FILE_SHA256:
        raise ValueError("Official protocol file hash mismatch")
    for name, expected_hash in INTERRUPTED_RUN_ARTIFACT_SHA256.items():
        if _file_sha256(stage / name) != expected_hash:
            raise ValueError("Preserved interrupted-run artifact hash mismatch: " + name)
    # Do not read Dataset v1 before the durable TEST_OPENED journal event.
    if not Path(dataset_path).is_file():
        raise FileNotFoundError("Dataset v1 file is missing")
    if OFFICIAL_DATASET_SHA256 != protocol["dataset_sha256"]:
        raise ValueError("Protocol Dataset v1 hash mismatch")
    ahc_sha = digest(Path(ahc_manifest_path).read_bytes())
    if ahc_sha != OFFICIAL_AHC_TEST_SHA256 or ahc_sha != protocol["ahc_test_manifest_sha256"]:
        raise ValueError("Official AHC TEST manifest hash mismatch")
    expected_configs = [{key: item[key] for key in ("model_id", "candidate_id", "algorithm", "feature_set", "features", "parameters")}
                        for item in selection["selected"]]
    candidate_lookup = {candidate_id(algorithm, family, parameters): (algorithm, family, parameters)
                        for algorithm, family, parameters in _candidates(protocol)}
    selected_cells = set()
    for item in expected_configs:
        candidate = candidate_lookup.get(item["candidate_id"])
        if candidate is None or candidate != (item["algorithm"], item["feature_set"], item["parameters"]):
            raise ValueError("Selection contains a candidate outside the frozen protocol grid")
        selected_cells.add((item["algorithm"], item["feature_set"]))
    expected_cells = {(algorithm, family) for algorithm in ALGORITHMS for family in protocol["feature_sets"]}
    if selected_cells != expected_cells:
        raise ValueError("Selection does not contain exactly one candidate per model/feature cell")
    if frozen.get("selected_cell_count") != 35 or frozen.get("protocol_sha256") != protocol_sha:
        raise ValueError("Frozen configuration identity mismatch")
    if frozen.get("selection_sha256") != selection_sha or frozen.get("configurations") != expected_configs:
        raise ValueError("Frozen configurations do not match selection.json")
    expected_candidate_ids = {candidate_id(algorithm, family, parameters)
                              for algorithm, family, parameters in _candidates(protocol)}
    if len(candidate_summaries) != 185 or {item.get("candidate_id") for item in candidate_summaries} != expected_candidate_ids:
        raise ValueError("Persisted DEVELOPMENT candidate summaries do not match the frozen candidate budget")
    if any(summary.get("test_inspected") is not False or
           len(summary.get("fold_metrics", [])) != len(protocol["development_folds"]) or
           {item.get("fold_id") for item in summary.get("fold_metrics", [])} != {fold["fold_id"] for fold in protocol["development_folds"]}
           for summary in candidate_summaries):
        raise ValueError("Candidate summary is missing one or more DEVELOPMENT folds")
    if len(refit) != 35 or {item.get("candidate_id") for item in refit} != {item["candidate_id"] for item in expected_configs}:
        raise ValueError("Refit metadata does not match the frozen 35 configurations")
    for item in refit:
        if item.get("refit_split") != "DEVELOPMENT" or item.get("refit_period") != protocol["development_period"]:
            raise ValueError("Refit metadata is not full DEVELOPMENT 2004-2020")
        selected_config = next(config for config in expected_configs if config["candidate_id"] == item["candidate_id"])
        if any(item.get(field) != selected_config[field] for field in ("model_id", "algorithm", "feature_set", "parameters")):
            raise ValueError("Refit metadata configuration differs from frozen selection")
    bundle_path = stage / "refitted_models.joblib"
    bundle_sha = _file_sha256(bundle_path)
    import joblib
    bundle = joblib.load(bundle_path)
    if (bundle.get("schema_version") != "ml-refitted-model-bundle-v1"
            or bundle.get("fit_split") != "DEVELOPMENT"
            or bundle.get("fit_period") != protocol["development_period"]
            or bundle.get("protocol_sha256") != protocol_sha
            or bundle.get("selection_sha256") != selection_sha):
        raise ValueError("Refitted model bundle metadata mismatch")
    bundle_configs = [{key: item[key] for key in ("model_id", "candidate_id", "algorithm", "feature_set", "features", "parameters")}
                      for item in bundle.get("selected", [])]
    if bundle_configs != expected_configs or set(bundle.get("models", {})) != {item["candidate_id"] for item in expected_configs}:
        raise ValueError("Refitted bundle does not exactly match frozen configurations")
    if any(item["algorithm"] not in ALGORITHMS for item in expected_configs):
        raise ValueError("Unexpected algorithm in frozen selection")
    return source, stage, selection, selection_sha, frozen, frozen_sha, refit, refit_sha, bundle, bundle_sha, protocol, protocol_sha


def _write_recovery_outputs(stage, rows, predicted, scores, selection, manifest):
    common_keys = ("dataset_id", "dataset_sha256", "audit_source_repo_sha", "source_repo_sha", "protocol_sha256", "run_id")
    identity = {key: manifest[key] for key in common_keys}
    model_ids = [item["model_id"] for item in selection["selected"]]
    daily_fields = list(common_keys) + ["station_code", "date", "split", "EtPMon", "PM_FAO56_V1", *model_ids,
        "eligibility_reasons", "quality_status", "observation_status", "quality_flags", "scientific_flags", "provenance", "source_data_sha"]
    daily = []
    for row in predicted:
        record = dict(identity, station_code=row["station_code"], date=row["date"], split="TEST", EtPMon=row["EtPMon"],
                      PM_FAO56_V1=row["target"], **row["predictions"])
        record.update(eligibility_reasons=row["eligibility_reasons"], quality_status=row["quality_status"],
            observation_status=row["observation_status"], quality_flags=row["quality_flags"],
            scientific_flags=row["scientific_flags"], provenance=row["provenance"], source_data_sha=row["source_data_sha"])
        daily.append(record)
    manifest["outputs"]["test_predictions.csv"] = _csv(stage / "test_predictions.csv", daily, daily_fields)
    if not scores:
        raise ValueError("Recovery produced no metric records")
    fields = list(common_keys) + list(scores[0])
    tables = {"test_metrics.csv": None, "station_metrics.csv": {"station"}, "year_metrics.csv": {"year"},
        "month_metrics.csv": {"month"}, "season_metrics.csv": {"season"},
        "station_year_metrics.csv": {"station_year"}, "season_year_metrics.csv": {"season_year"}}
    for filename, groups in tables.items():
        selected = (score for score in scores if groups is None or score["group"] in groups)
        manifest["outputs"][filename] = _csv(stage / filename, (dict(identity, **score) for score in selected), fields)
    common = [dict(identity, **score) for score in scores if score["cohort_id"] in ("common_ml_35", "common_all")]
    manifest["outputs"]["common_cohort_metrics.csv"] = _csv(stage / "common_cohort_metrics.csv", common, list(common[0]) if common else fields)
    coverage = []
    for family in manifest["protocol"]["feature_sets"]:
        members = [row for row in rows if row["split"] == "TEST"]
        eligible = sum(not row["eligibility_reasons"][family] for row in members)
        reasons = {}
        for row in members:
            for reason in row["eligibility_reasons"][family]: reasons[reason] = reasons.get(reason, 0) + 1
        coverage.append(dict(identity, feature_set=family, split="TEST", source_rows=len(members), eligible_n=eligible, exclusion_reasons=reasons))
    manifest["outputs"]["coverage.csv"] = _csv(stage / "coverage.csv", coverage, list(coverage[0]))
    metadata = []
    for item in selection["selected"]:
        model = manifest["_models"][item["candidate_id"]]
        metadata.append(dict(identity, model_id=item["model_id"], candidate_id=item["candidate_id"], algorithm=item["algorithm"],
            feature_set=item["feature_set"], features=item["features"], parameters=item["parameters"], fit_split="DEVELOPMENT",
            fit_period=manifest["protocol"]["development_period"], selected_on="DEVELOPMENT_FOLDS_F1_F4",
            test_used_for_selection=False, fitted_diagnostics=_model_diagnostics(model, fitted_on="DEVELOPMENT_2004_2020")))
    manifest["outputs"]["model_metadata.csv"] = _csv(stage / "model_metadata.csv", metadata, list(metadata[0]))


def execute_ml_test_recovery(source_run, dataset_path, ahc_manifest_path, protocol_path, output_path, *, pipeline_sha=None):
    """Validate frozen source artifacts, journal TEST access, and evaluate only."""
    output = safe_new_output(output_path, directory=True)
    journal_path = output / "recovery_journal.jsonl"
    staging = output / ".staging"
    staging.mkdir()
    started = time.perf_counter()
    usage_before = _resource_usage()
    manifest = {"schema_version": "ml-test-recovery-v1", "stage": "retrospective_recovery_evaluation",
        "status": "FAILED", "run_id": "ml-recovery-" + uuid.uuid4().hex, "started_at_utc": _utc(),
        "prior_attempt_status": "TEST_ACCESS_OCCURRED_TEST_EVALUATION_NOT_PERFORMED", "test_access_claim": "retrospective recovery; prior technical TEST access without predictions or metrics",
        "pipeline": _identity(pipeline_sha), "runtime": _runtime(), "outputs": {}, "test_inspected": False}
    _journal(journal_path, "RECOVERY_STARTED", source_run=str(Path(source_run).resolve()))
    try:
        ahc_dependencies = _verify_ahc_output_dependencies(ahc_manifest_path)
        _journal(journal_path, "PRE_TEST_DEPENDENCIES_VERIFIED",
                 ahc_manifest_sha256=ahc_dependencies["manifest_sha256"],
                 ahc_output_count=ahc_dependencies["output_count"])
        values = _validate_frozen_source(source_run, dataset_path, ahc_manifest_path, protocol_path)
        source, source_stage, selection, selection_sha, frozen, frozen_sha, refit, refit_sha, bundle, bundle_sha, protocol, protocol_sha = values
        # Persist source validation and all immutable inputs before any TEST loader is invoked.
        manifest.update(source_run=str(source), selection_sha256=selection_sha, frozen_configurations_sha256=frozen_sha,
            refit_metadata_sha256=refit_sha, refitted_model_bundle_sha256=bundle_sha,
            source_artifacts={name: _file_sha256(source_stage / name) for name in
                ("selection.json", "frozen_configurations.json", "candidate_summaries.json", "refit_metadata.json", "refitted_models.joblib")},
            protocol=protocol, protocol_sha256=protocol_sha, protocol_file_sha256=digest(Path(protocol_path).read_bytes()),
            expected_dataset_sha256=protocol["dataset_sha256"],
            ahc_test_manifest_sha256=ahc_dependencies["manifest_sha256"], ahc_dependency_outputs=ahc_dependencies["outputs"],
            selected_cell_count=35, fit_split="DEVELOPMENT", fit_period=protocol["development_period"],
            prior_attempt_status="TEST_ACCESS_OCCURRED_TEST_EVALUATION_NOT_PERFORMED",
            prior_test_access_history=[
                {"attempt":"original campaign", "status":"TEST_ACCESS_UNKNOWN"},
                {"attempt":"server recovery", "status":"TEST_NOT_OPENED"},
                {"attempt":"first Kaggle recovery", "status":"TEST_ACCESS_OCCURRED",
                 "predictions":False, "metrics":False,
                 "manifest_sha256":"9bb4c7bc1f605b121e0d8797743094883a54e773b6275258f0a9fec1769e70f1",
                 "journal_sha256":"307c065336f3be736cf19f1121222e4b499642ca98db8bcef87ef6122a1336df"}])
        _journal(journal_path, "PRE_TEST_VERIFIED", selection_sha256=selection_sha, bundle_sha256=bundle_sha,
                 candidate_count=185, selected_cell_count=35)
        _journal(journal_path, "TEST_ACCESS_INTENT_RECORDED", reason="frozen selection, refit and AHC dependencies verified")
        # Durable access marker precedes hashing, parsing, or any other Dataset v1 read.
        _journal(journal_path, "TEST_OPENED", reason="all frozen and AHC dependency checks passed")
        manifest["test_inspected"] = True
        manifest["test_access_count_this_recovery"] = 1
        dataset_sha = _file_sha256(dataset_path)
        if dataset_sha != OFFICIAL_DATASET_SHA256 or dataset_sha != protocol["dataset_sha256"]:
            raise ValueError("Official Dataset v1 hash mismatch")
        manifest["dataset_sha256"] = dataset_sha
        dataset, rows, loaded_protocol, identity = _context_for_split(dataset_path, ahc_manifest_path, protocol_path, "TEST")
        if loaded_protocol != protocol or identity["dataset_sha256"] != manifest["dataset_sha256"] or identity["protocol_sha256"] != protocol_sha:
            raise ValueError("TEST identity differs from pre-verified inputs")
        _journal(journal_path, "TEST_CONTEXT_IDENTIFIED", row_count=len(rows), identity_verified=True)
        manifest.update(identity)
        manifest["test_inspected"] = True
        manifest["development_selection_sha256"] = selection_sha
        manifest["test_access_count_this_recovery"] = 1
        with _thread_limits():
            predicted = _test_predictions(rows, selection["selected"], bundle["models"])
            scores = list(score_predictions(predicted, [item["model_id"] for item in selection["selected"]]))
            manifest["_models"] = bundle["models"]
            _write_recovery_outputs(staging, rows, predicted, scores, selection, manifest)
            del manifest["_models"]
            _journal(journal_path, "TEST_EVALUATION_COMPLETED", prediction_rows=len(predicted), metric_rows=len(scores))
            result = {"status": "PASS", "stage": manifest["stage"], "test_access_claim": manifest["test_access_claim"],
                "test_status": protocol["test_status"], "selected_cell_count": 35, "test_rows": len(rows),
                "test_common_cohort": [score for score in scores if score["reference_id"] == "PM_FAO56_V1" and score["group"] == "global" and score["cohort_id"] == "common_all"]}
            manifest["outputs"]["recovery_results.json"] = {"path": "recovery_results.json", "sha256": _write(staging / "recovery_results.json", result)}
        manifest["status"] = "SUCCESS"
        manifest["finished_at_utc"] = _utc()
        manifest["runtime_seconds"] = time.perf_counter() - started
        usage_after = _resource_usage()
        manifest["resource_usage"] = {"wall_seconds": manifest["runtime_seconds"],
            "user_cpu_seconds": max(0.0, usage_after["user_cpu_seconds"] - usage_before["user_cpu_seconds"])
                if usage_after["user_cpu_seconds"] is not None and usage_before["user_cpu_seconds"] is not None else None,
            "system_cpu_seconds": max(0.0, usage_after["system_cpu_seconds"] - usage_before["system_cpu_seconds"])
                if usage_after["system_cpu_seconds"] is not None and usage_before["system_cpu_seconds"] is not None else None,
            "peak_rss_mb": usage_after["peak_rss_mb"]}
        for path in sorted(staging.iterdir()):
            if path.is_file(): path.replace(output / path.name)
        staging.rmdir()
        (output / "manifest.json").write_bytes(json_bytes(manifest))
        (output / "manifest.sha256").write_text(digest((output / "manifest.json").read_bytes()) + "  manifest.json\n", encoding="ascii")
        _journal(journal_path, "SUCCESS", manifest_sha256=digest((output / "manifest.json").read_bytes()))
        return manifest
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        manifest.pop("_models", None)
        manifest.update(status="FAILED", error=str(exc), finished_at_utc=_utc(), runtime_seconds=time.perf_counter() - started,
                        resource_usage=_resource_usage())
        (output / "manifest.json").write_bytes(json_bytes(manifest))
        _journal(journal_path, "FAILED", error=str(exc), test_inspected=manifest.get("test_inspected", False))
        raise
