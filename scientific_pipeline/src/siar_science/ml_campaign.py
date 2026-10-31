"""Official, append-only ML v1 rolling-origin campaign orchestration.

Development code receives only DEVELOPMENT rows.  Selection and full-
development refits are persisted in a private staging area before the TEST
loader is called.  A failed campaign removes staging artifacts and retains a
FAILED manifest with no scientific outputs.
"""
import csv
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
import uuid
import warnings

from .datasets import read_dataset
from .metrics import metrics
from .ml_classical import (ALGORITHMS, BASELINES, candidate_id, feature_names,
                           make_estimator, prepare_rows, score_predictions,
                           validate_ml_protocol)
from .ml_pipeline import (_csv, _dump_models, _load_official, _model_diagnostics,
                         _peak_memory_mb, _read, _runtime, _write)
from .ml_rolling import aggregate_fold_metrics, evaluate_rolling_candidates
from .pipeline import _identity, _utc, safe_new_output
from .serialization import digest, json_bytes


OFFICIAL_CAMPAIGN = "ML_V1_CLASSICAL_ROLLING_20261031"
OFFICIAL_DATASET_SHA256 = "0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb"
OFFICIAL_PROTOCOL_FILE_SHA256 = "45f1c4f3ed20bb70dc3b1ca06a8e23ec14f95aae6b60f574e44a8a261cc07048"
OFFICIAL_AHC_TEST_SHA256 = "502d7a39b82c8b5032931370ae8f05f4d8e7ecbbb93f6946810d0e47434fc469"


def _context_for_split(dataset_path, ahc_manifest_path, protocol_path, split):
    """Load only the requested period's modeled rows and reference values."""
    if split not in ("DEVELOPMENT", "TEST"):
        raise ValueError("Unknown ML partition")
    protocol, protocol_file_sha = _read(protocol_path)
    validate_ml_protocol(protocol)
    dataset_bytes = Path(dataset_path).read_bytes()
    dataset_sha = digest(dataset_bytes)
    dataset = read_dataset(dataset_path)
    if digest(Path(dataset_path).read_bytes()) != dataset_sha:
        raise ValueError("Dataset changed during read")
    if dataset_sha != protocol["dataset_sha256"] or dataset["dataset_id"] != protocol["dataset_id"]:
        raise ValueError("Protocol dataset mismatch")
    official, ahc, ahc_sha = _load_official(
        dataset, dataset_sha, ahc_manifest_path, protocol["ahc_test_manifest_sha256"],
        period=protocol["development_period"] if split == "DEVELOPMENT" else protocol["test_period"],
    )
    period = protocol["development_period"] if split == "DEVELOPMENT" else protocol["test_period"]
    selected_rows = [row for row in dataset["rows"] if period[0] <= row["date"] <= period[1]]
    if not selected_rows:
        raise ValueError("No dataset rows in " + split)
    # Do not pass the other partition to feature construction or model code.
    partition_dataset = {"stations": dataset["stations"], "rows": selected_rows}
    rows = prepare_rows(partition_dataset, official, protocol)
    split_sha = digest(json_bytes(sorted(
        [row["station_code"], row["date"],
         "DEVELOPMENT" if protocol["development_period"][0] <= row["date"] <= protocol["development_period"][1]
         else "TEST" if protocol["test_period"][0] <= row["date"] <= protocol["test_period"][1]
         else "OUTSIDE_EXPERIMENT"]
        for row in dataset["rows"])))
    identity = {key: dataset[key] for key in (
        "dataset_id", "audit_source_repo_sha", "source_repo_sha", "quality_post",
        "source_data_hashes", "units")}
    identity.update(
        dataset_sha256=dataset_sha, catalog=dataset.get("catalog"), protocol=protocol,
        protocol_sha256=digest(json_bytes(protocol)), protocol_file_sha256=protocol_file_sha,
        ahc_test_manifest_sha256=ahc_sha, ahc_test_manifest_path=str(Path(ahc_manifest_path).resolve()),
        ahc_pipeline=ahc["pipeline"], split_assignment_sha256=split_sha)
    return dataset, rows, protocol, identity


def _candidates(protocol):
    return [(algorithm, family, parameters)
            for family in protocol["feature_sets"]
            for algorithm in ALGORITHMS
            for parameters in protocol["candidate_grids"][algorithm]]


def _candidate_selection_key(summary):
    aggregate = summary["aggregate"]
    return (aggregate["weighted_mean_fold_rmse"], aggregate["weighted_sd_fold_rmse"], summary["candidate_id"])


def _select_cell(summaries, protocol):
    eligible = [item for item in summaries if item["aggregate"]["weighted_mean_fold_rmse"] is not None]
    if not eligible:
        raise ValueError("No eligible candidate in model/feature cell")
    minimum = min(item["aggregate"]["weighted_mean_fold_rmse"] for item in eligible)
    tied = [item for item in eligible if item["aggregate"]["weighted_mean_fold_rmse"]
            <= minimum * (1 + protocol["selection"]["relative_tie_tolerance"])]
    return min(tied, key=_candidate_selection_key)


def _common_validation(rows, protocol, selected):
    """Score selected cell candidates on each fold's common eligible keys."""
    from collections import defaultdict
    common_records = []
    for fold in protocol["development_folds"]:
        validation = [row for row in rows if row["split"] == "DEVELOPMENT"
                      and fold["validation"][0] <= row["date"] <= fold["validation"][1]
                      and all(row["features"][family] is not None for family in protocol["feature_sets"])]
        if not validation:
            raise ValueError("Empty common validation cohort for " + fold["fold_id"])
        for candidate in selected:
            train = [row for row in rows if row["split"] == "DEVELOPMENT"
                     and fold["train"][0] <= row["date"] <= fold["train"][1]
                     and row["features"][candidate["feature_set"]] is not None]
            if not train:
                raise ValueError("Empty fold train for selected candidate")
            model = make_estimator(candidate["algorithm"], candidate["parameters"], protocol)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit([row["features"][candidate["feature_set"]] for row in train],
                          [row["target"] for row in train])
            prediction = [float(value) for value in model.predict(
                [row["features"][candidate["feature_set"]] for row in validation])]
            if any(not math.isfinite(value) for value in prediction):
                raise ValueError("Nonfinite common-cohort validation prediction")
            common_records.append({
                "candidate_id": candidate["candidate_id"], "algorithm": candidate["algorithm"],
                "feature_set": candidate["feature_set"], "fold_id": fold["fold_id"],
                "train_n": len(train), "validation_n": len(validation),
                "metrics": metrics([row["target"] for row in validation], prediction),
                "fit_warnings": [{"category": type(item.message).__name__, "message": str(item.message)} for item in caught],
            })
    by_candidate = defaultdict(list)
    for record in common_records:
        by_candidate[record["candidate_id"]].append(record)
    summaries = {}
    for candidate in selected:
        values = by_candidate[candidate["candidate_id"]]
        summaries[candidate["candidate_id"]] = {
            "candidate_id": candidate["candidate_id"], "algorithm": candidate["algorithm"],
            "feature_set": candidate["feature_set"],
            "parameters": candidate["parameters"],
            "fold_metrics": [{"fold_id": item["fold_id"], **item["metrics"]} for item in values],
            "aggregate": aggregate_fold_metrics(values),
        }
    return common_records, summaries


def _choose_champion(summaries, protocol, candidates=None):
    candidates = list(summaries.values()) if candidates is None else list(candidates)
    if not candidates:
        raise ValueError("No candidate for development champion")
    minimum = min(item["aggregate"]["weighted_mean_fold_rmse"] for item in candidates)
    tolerance = protocol["selection"]["relative_tie_tolerance"]
    tied = [item for item in candidates if item["aggregate"]["weighted_mean_fold_rmse"] <= minimum * (1 + tolerance)]
    ranks = {name: i for i, name in enumerate(protocol["selection"]["complexity_rank"])}
    return min(tied, key=lambda item: (
        item["aggregate"]["weighted_sd_fold_rmse"], ranks[item["algorithm"]],
        len(feature_names(protocol, item["feature_set"])), item["candidate_id"]))


def _frozen_selection(rows, protocol, outcome):
    by_cell = {}
    for summary in outcome["candidate_summaries"]:
        by_cell.setdefault((summary["algorithm"], summary["feature_set"]), []).append(summary)
    selected = []
    for algorithm in ALGORITHMS:
        for family in protocol["feature_sets"]:
            winner = deepcopy(_select_cell(by_cell[algorithm, family], protocol))
            winner["model_id"] = "ML_" + algorithm + "__" + family
            winner["features"] = list(feature_names(protocol, family))
            selected.append(winner)
    common_records, common_summaries = _common_validation(rows, protocol, selected)
    for item in selected:
        item["common_validation_aggregate"] = common_summaries[item["candidate_id"]]["aggregate"]
    champion = _choose_champion(common_summaries, protocol)
    selected_ids = {item["candidate_id"] for item in selected}
    if champion["candidate_id"] not in selected_ids:
        raise ValueError("Campaign champion is not one of the frozen cell selections")
    best_by_algorithm = {
        algorithm: _choose_champion(common_summaries, protocol,
            [item for item in common_summaries.values() if item["algorithm"] == algorithm])
        for algorithm in ALGORITHMS
    }
    best_groups = {
        "linear": _choose_champion(common_summaries, protocol,
            [item for item in common_summaries.values() if item["algorithm"] in ("LINEAR","RIDGE","ELASTIC_NET")]),
        "tree": _choose_champion(common_summaries, protocol,
            [item for item in common_summaries.values() if item["algorithm"] == "TREE"]),
        "ensemble": _choose_champion(common_summaries, protocol,
            [item for item in common_summaries.values() if item["algorithm"] in ("RANDOM_FOREST","EXTRA_TREES")]),
        "boosting": _choose_champion(common_summaries, protocol,
            [item for item in common_summaries.values() if item["algorithm"] == "HIST_GRADIENT_BOOSTING"]),
    }
    return {
        "schema_version": "ml-rolling-selection-v1", "status": "FROZEN",
        "selection_split": "DEVELOPMENT_FOLDS_F1_F4", "test_inspected": False,
        "candidate_count": len(outcome["candidate_summaries"]), "selected_cell_count": len(selected),
        "selected": selected, "common_validation_folds": common_records,
        "common_validation_summaries": list(common_summaries.values()),
        "campaign_champion": champion,
        "best_by_algorithm": best_by_algorithm,
        "best_model_groups": best_groups,
        "best_feature_set": champion["feature_set"],
        "common_validation_cohort_rule": "all 35 selected cells eligible within each validation fold",
    }


def _fit_full_development(rows, protocol, selected):
    models, records = {}, []
    period = protocol["development_period"]
    for candidate in selected:
        family = candidate["feature_set"]
        train = [row for row in rows if row["split"] == "DEVELOPMENT"
                 and period[0] <= row["date"] <= period[1]
                 and row["features"][family] is not None]
        if not train:
            raise ValueError("Empty full-DEVELOPMENT refit cohort for " + family)
        model = make_estimator(candidate["algorithm"], candidate["parameters"], protocol)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit([row["features"][family] for row in train], [row["target"] for row in train])
        models[candidate["candidate_id"]] = model
        records.append({
            "candidate_id": candidate["candidate_id"], "model_id": candidate["model_id"],
            "algorithm": candidate["algorithm"], "feature_set": family,
            "parameters": candidate["parameters"], "refit_period": period,
            "refit_split": "DEVELOPMENT", "refit_n": len(train),
            "refit_key_sha256": digest(json_bytes(sorted([r["station_code"], r["date"]] for r in train))),
            "fit_warnings": [{"category": type(item.message).__name__, "message": str(item.message)} for item in caught],
            "fitted_diagnostics": _model_diagnostics(model, fitted_on="DEVELOPMENT_2004_2020"),
        })
    return models, records


def _permutation_importance(rows, protocol, selected, champion, common_summaries):
    from sklearn.inspection import permutation_importance
    fold = next(item for item in protocol["development_folds"] if item["fold_id"] == "F4")
    choices = []
    common_by_id = {item["candidate_id"]: item for item in common_summaries}
    for algorithm in ALGORITHMS:
        options = [item for item in common_by_id.values() if item["algorithm"] == algorithm]
        choices.append(_choose_champion(common_by_id, protocol, options))
    if champion["candidate_id"] not in {item["candidate_id"] for item in choices}:
        choices.append(champion)
    output = []
    for candidate in choices:
        family = candidate["feature_set"]
        train = [row for row in rows if row["split"] == "DEVELOPMENT"
                 and fold["train"][0] <= row["date"] <= fold["train"][1]
                 and row["features"][family] is not None]
        valid = [row for row in rows if row["split"] == "DEVELOPMENT"
                 and fold["validation"][0] <= row["date"] <= fold["validation"][1]
                 and row["features"][family] is not None]
        valid.sort(key=lambda row: hashlib.sha256(
            (row["station_code"]+"/"+row["date"]).encode("utf-8")).hexdigest())
        valid = valid[:protocol["permutation_importance"]["maximum_rows"]]
        model = make_estimator(candidate["algorithm"], candidate["parameters"], protocol)
        model.fit([row["features"][family] for row in train], [row["target"] for row in train])
        result = permutation_importance(
            model, [row["features"][family] for row in valid], [row["target"] for row in valid],
            scoring="neg_root_mean_squared_error",
            n_repeats=protocol["permutation_importance"]["n_repeats"],
            random_state=protocol["runtime"]["random_seed"], n_jobs=1)
        estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
        intrinsic = getattr(estimator, "feature_importances_", None)
        coefficients = getattr(estimator, "coef_", None)
        for index, name in enumerate(feature_names(protocol, family)):
            output.append({
                "candidate_id": candidate["candidate_id"], "algorithm": candidate["algorithm"],
                "feature_set": family, "feature": name, "partition": "F4_VALIDATION",
                "validation_sample_n": len(valid),
                "permutation_rmse_increase_mean": float(result.importances_mean[index]),
                "permutation_rmse_increase_sd": float(result.importances_std[index]),
                "n_repeats": int(result.importances.shape[1]),
                "intrinsic_tree_importance": float(intrinsic[index]) if intrinsic is not None else None,
                "standardized_linear_coefficient": float(coefficients[index]) if coefficients is not None else None,
                "used_for_selection": False, "causal_interpretation": False,
            })
    return output


def _test_predictions(rows, selected, models):
    output = []
    mappings = {}
    from .ml_classical import predict
    for candidate in selected:
        mappings[candidate["model_id"]] = predict(models[candidate["candidate_id"]], rows, candidate["feature_set"])
    for row in rows:
        key = (row["station_code"], row["date"])
        predictions = dict(row["baselines"])
        predictions.update({model_id: by_key.get(key) for model_id, by_key in mappings.items()})
        flags = list(row["scientific_flags"])
        flags.extend("negative_prediction:" + name for name, value in predictions.items()
                     if value is not None and value < 0)
        output.append({
            "station_code": row["station_code"], "date": row["date"], "split": "TEST",
            "target": row["target"], "EtPMon": row["EtPMon"], "predictions": predictions,
            "eligibility_reasons": row["eligibility_reasons"],
            "quality_status": row["quality_status"], "observation_status": row["observation_status"],
            "quality_flags": row["quality_flags"], "scientific_flags": flags,
            "provenance": row["provenance"], "source_data_sha": row["source_data_sha"],
        })
    return output


def _write_campaign_views(stage_dir, development_rows, test_rows, predicted, scores, selected, refit_records,
                          importance_rows, manifest, selection):
    identity_keys = ("dataset_id", "dataset_sha256", "audit_source_repo_sha", "source_repo_sha",
                     "protocol_sha256", "run_id")
    identity = {key: manifest[key] for key in identity_keys}
    model_ids = [item["model_id"] for item in selected]
    baseline_ids = list(BASELINES)
    daily_fields = list(identity_keys) + ["station_code", "date", "split", "EtPMon"] + baseline_ids + model_ids + [
        "eligibility_reasons", "quality_status", "observation_status", "quality_flags", "scientific_flags",
        "provenance", "source_data_sha"]
    daily_records = []
    for row in predicted:
        item = dict(identity, station_code=row["station_code"], date=row["date"], split="TEST", EtPMon=row["EtPMon"])
        item.update(row["predictions"])
        item.update(PM_FAO56_V1=row["target"], eligibility_reasons=row["eligibility_reasons"],
                    quality_status=row["quality_status"], observation_status=row["observation_status"],
                    quality_flags=row["quality_flags"], scientific_flags=row["scientific_flags"],
                    provenance=row["provenance"], source_data_sha=row["source_data_sha"])
        daily_records.append(item)
    manifest["outputs"]["test_predictions.csv"] = _csv(stage_dir/"test_predictions.csv", daily_records, daily_fields)

    score_fields = list(identity_keys) + list(scores[0])
    group_files = {
        "test_metrics.csv": None, "station_metrics.csv": {"station"}, "year_metrics.csv": {"year"},
        "month_metrics.csv": {"month"}, "season_metrics.csv": {"season"},
        "station_year_metrics.csv": {"station_year"}, "season_year_metrics.csv": {"season_year"},
    }
    for filename, groups in group_files.items():
        manifest["outputs"][filename] = _csv(stage_dir/filename,
            (dict(identity, **score) for score in scores if groups is None or score["group"] in groups), score_fields)
    common = [dict(identity, **score) for score in scores
              if score["cohort_id"] in ("common_ml_35", "common_all")]
    manifest["outputs"]["common_cohort_metrics.csv"] = _csv(
        stage_dir/"common_cohort_metrics.csv", common, list(common[0]) if common else score_fields)

    coverage = []
    for family in manifest["protocol"]["feature_sets"]:
        for partition, period, fold_id in (("DEVELOPMENT_FULL", manifest["protocol"]["development_period"], None),
                                          ("TEST", manifest["protocol"]["test_period"], None)):
            source_rows = [row for row in (development_rows if partition == "DEVELOPMENT_FULL" else test_rows)
                           if period[0] <= row["date"] <= period[1]]
            eligible = [row for row in source_rows if row["features"][family] is not None]
            reasons = {}
            for row in source_rows:
                for reason in row["eligibility_reasons"][family]: reasons[reason] = reasons.get(reason, 0) + 1
            coverage.append({**identity, "partition":partition, "fold_id":fold_id, "feature_set":family,
                "source_rows":len(source_rows), "eligible_n":len(eligible),
                "eligible_key_sha256":digest(json_bytes(sorted([row["station_code"],row["date"]] for row in eligible))),
                "exclusion_reasons":reasons})
        for fold in manifest["protocol"]["development_folds"]:
            for partition, period in (("FOLD_TRAIN",fold["train"]),("FOLD_VALIDATION",fold["validation"])):
                source_rows=[row for row in development_rows if period[0] <= row["date"] <= period[1]]
                eligible=[row for row in source_rows if row["features"][family] is not None]
                reasons={}
                for row in source_rows:
                    for reason in row["eligibility_reasons"][family]: reasons[reason]=reasons.get(reason,0)+1
                coverage.append({**identity,"partition":partition,"fold_id":fold["fold_id"],"feature_set":family,
                    "source_rows":len(source_rows),"eligible_n":len(eligible),
                    "eligible_key_sha256":digest(json_bytes(sorted([row["station_code"],row["date"]] for row in eligible))),
                    "exclusion_reasons":reasons})
    manifest["outputs"]["coverage.csv"] = _csv(stage_dir/"coverage.csv", coverage,
        list(coverage[0]) if coverage else list(identity_keys)+["partition","fold_id","feature_set","source_rows","eligible_n","eligible_key_sha256","exclusion_reasons"])
    metadata = []
    refit_by_id = {row["candidate_id"]: row for row in refit_records}
    for item in selected:
        metadata.append({**identity, "model_id": item["model_id"], "candidate_id": item["candidate_id"],
            "algorithm": item["algorithm"], "feature_set": item["feature_set"], "features": item["features"],
            "parameters": item["parameters"], "selected_on": "DEVELOPMENT_FOLDS_F1_F4",
            "development_metrics": item["aggregate"], "refit": refit_by_id[item["candidate_id"]],
            "target": "PM_FAO56_V1", "target_meaning": "calculated reference; not measured ET"})
    manifest["outputs"]["model_metadata.csv"] = _csv(stage_dir/"model_metadata.csv", metadata, list(metadata[0]))
    manifest["outputs"]["permutation_importance.csv"] = _csv(
        stage_dir/"permutation_importance.csv", importance_rows,
        list(importance_rows[0]) if importance_rows else ["candidate_id","algorithm","feature_set","feature"])
    schema = {}
    for name, entry in manifest["outputs"].items():
        if name.endswith(".csv"):
            with (stage_dir/entry["path"]).open(encoding="utf-8-sig", newline="") as stream:
                schema[name] = {"rows": entry["rows"], "columns": next(csv.reader(stream))}
    manifest["outputs"]["view_schemas.json"] = {"path":"view_schemas.json", "sha256":_write(stage_dir/"view_schemas.json", schema)}
    view = {key: manifest[key] for key in ("dataset_id","dataset_sha256","audit_source_repo_sha","source_repo_sha",
        "quality_post","catalog","source_data_hashes","units","protocol_sha256","split_assignment_sha256",
        "pipeline","runtime","run_id","ahc_test_manifest_sha256")}
    view.update(schema_version="ml-science-view-v1", status="SUCCESS", test_status="retrospective_locked",
                science_recalculation_allowed=False, outputs=deepcopy(manifest["outputs"]))
    manifest["outputs"]["view-manifest.json"] = {"path":"view-manifest.json", "sha256":_write(stage_dir/"view-manifest.json", view)}


def _resource_usage():
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss / (1024 if platform.system() != "Darwin" else 1024 * 1024)
        return {"user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
                "peak_rss_mb": rss}
    except (ImportError, AttributeError):
        return {"user_cpu_seconds": None, "system_cpu_seconds": None, "peak_rss_mb": _peak_memory_mb()}


def execute_ml_rolling_campaign(dataset_path, ahc_manifest_path, protocol_path, output_path,
                                *, pipeline_sha=None, require_official_identity=True):
    """Run rolling selection, full-development refit, then one TEST evaluation."""
    output = safe_new_output(output_path, directory=True)
    staging = output/".staging"
    staging.mkdir()
    started = time.perf_counter()
    usage_before = _resource_usage()
    manifest = {"schema_version":"ml-rolling-campaign-run-v1", "stage":"rolling-campaign",
                "status":"FAILED", "run_id":"ml-rolling-"+uuid.uuid4().hex,
                "started_at_utc":_utc(), "pipeline":None, "runtime":{}, "outputs":{},
                "scientific_selection_allowed":True, "test_inspected":False,
                "phase_order":["DEVELOPMENT_F1_F4", "SELECTION_FROZEN", "REFIT_DEVELOPMENT_2004_2020", "TEST_ONCE"]}
    try:
        manifest["pipeline"] = _identity(pipeline_sha)
        manifest["runtime"] = _runtime()
        dataset, development, protocol, identity = _context_for_split(
            dataset_path, ahc_manifest_path, protocol_path, "DEVELOPMENT")
        if require_official_identity:
            if protocol.get("experiment_id") != OFFICIAL_CAMPAIGN:
                raise ValueError("Protocol is not the frozen official ML v1 campaign")
            if (identity["dataset_sha256"] != OFFICIAL_DATASET_SHA256
                    or identity["protocol_file_sha256"] != OFFICIAL_PROTOCOL_FILE_SHA256
                    or identity["ahc_test_manifest_sha256"] != OFFICIAL_AHC_TEST_SHA256):
                raise ValueError("Official dataset/protocol/AHC identity mismatch")
            if len(_candidates(protocol)) != 185 or len(protocol["feature_sets"])*len(ALGORITHMS) != 35:
                raise ValueError("Frozen official candidate budget must be 185 / 35 cells")
        manifest.update(identity)
        manifest["input_rows"] = len(dataset["rows"])
        manifest["development_rows_loaded"] = len(development)
        manifest["test_rows_loaded_before_freeze"] = 0
        protocol_copy = staging/"protocol.json"
        protocol_copy.write_bytes(Path(protocol_path).read_bytes())
        manifest["outputs"]["protocol.json"] = {"path":"protocol.json", "sha256":digest(protocol_copy.read_bytes()),
                                                   "bytes":protocol_copy.stat().st_size}
        candidates = _candidates(protocol)
        with _thread_limits():
            outcome = evaluate_rolling_candidates(development, protocol, candidates)
            if len(outcome["candidate_summaries"]) != len(candidates):
                raise ValueError("Rolling candidate count mismatch")
            selection = _frozen_selection(development, protocol, outcome)
            if len(selection["selected"]) != 35:
                raise ValueError("Selection did not freeze all 35 model/feature cells")
            # Publish selection/frozen configurations in staging before the TEST context exists.
            manifest["development_completed_at_utc"] = _utc()
            manifest["test_inspected"] = False
            manifest["selection_frozen_at_utc"] = _utc()
            manifest["candidate_budget_frozen"] = True
            manifest["outputs"]["rolling_fold_metrics.csv"] = _csv(
                staging/"rolling_fold_metrics.csv", outcome["fold_records"],
                ["candidate_id","algorithm","feature_set","parameters","fold_id","train_period","validation_period",
                 "train_n","validation_n","train_key_sha256","validation_key_sha256","metrics","fit_warnings","test_inspected"])
            summaries = []
            selected_ids = {item["candidate_id"] for item in selection["selected"]}
            for item in outcome["candidate_summaries"]:
                summaries.append({**item, "selected_for_cell": item["candidate_id"] in selected_ids,
                                  "scientific_selection_allowed":True, "test_inspected":False})
            manifest["outputs"]["candidate_summaries.json"] = {"path":"candidate_summaries.json",
                "sha256":_write(staging/"candidate_summaries.json", summaries)}
            manifest["outputs"]["selection.json"] = {"path":"selection.json",
                "sha256":_write(staging/"selection.json", selection)}
            frozen = {"schema_version":"ml-frozen-configurations-v1", "protocol_sha256":identity["protocol_sha256"],
                "selection_sha256":manifest["outputs"]["selection.json"]["sha256"],
                "selected_cell_count":len(selection["selected"]), "campaign_champion":selection["campaign_champion"],
                "configurations":[{key:item[key] for key in ("model_id","candidate_id","algorithm","feature_set","features","parameters")}
                                  for item in selection["selected"]]}
            manifest["outputs"]["frozen_configurations.json"] = {"path":"frozen_configurations.json",
                "sha256":_write(staging/"frozen_configurations.json", frozen)}
            models, refit_records = _fit_full_development(development, protocol, selection["selected"])
            bundle = {"schema_version":"ml-refitted-model-bundle-v1", "fit_split":"DEVELOPMENT",
                      "fit_period":protocol["development_period"], "protocol_sha256":identity["protocol_sha256"],
                      "selection_sha256":manifest["outputs"]["selection.json"]["sha256"],
                      "selected":selection["selected"], "models":models}
            manifest["outputs"]["refitted_models.joblib"] = _dump_models(staging/"refitted_models.joblib", bundle)
            manifest["outputs"]["refit_metadata.json"] = {"path":"refit_metadata.json",
                "sha256":_write(staging/"refit_metadata.json", refit_records)}
            manifest["refit_completed_at_utc"] = _utc()
            manifest["full_development_refit"] = True
            importance_rows = _permutation_importance(
                development, protocol, selection["selected"], selection["campaign_champion"],
                selection["common_validation_summaries"])
            # The single TEST access occurs only after selection, frozen configs and refit bundle exist.
            manifest["test_opened_at_utc"] = _utc()
            manifest["test_access_count"] = 1
            manifest["test_inspected"] = True
            test_dataset, test_rows, test_protocol, test_identity = _context_for_split(
                dataset_path, ahc_manifest_path, protocol_path, "TEST")
            if (test_identity["dataset_sha256"] != identity["dataset_sha256"]
                    or test_identity["protocol_sha256"] != identity["protocol_sha256"]
                    or test_identity["split_assignment_sha256"] != identity["split_assignment_sha256"]):
                raise ValueError("Input identity changed before TEST")
            predicted = _test_predictions(test_rows, selection["selected"], models)
            scores = list(score_predictions(predicted, [item["model_id"] for item in selection["selected"]]))
            _write_campaign_views(staging, development, test_rows, predicted, scores, selection["selected"],
                                  refit_records, importance_rows, manifest, selection)
            result = {"status":"PASS", "test_status":protocol["test_status"],
                "selection_split":"DEVELOPMENT_FOLDS_F1_F4", "selection_frozen_before_test":True,
                "refit_split":"DEVELOPMENT", "refit_period":protocol["development_period"],
                "test_access_count":1, "candidate_count":len(candidates), "selected_cell_count":len(selection["selected"]),
                "campaign_champion":selection["campaign_champion"],
                "best_by_algorithm":selection["best_by_algorithm"],
                "best_model_groups":selection["best_model_groups"],
                "best_feature_set":selection["best_feature_set"],
                "test_rows":len(test_rows),
                "test_common_cohort": [score for score in scores if score["reference_id"]=="PM_FAO56_V1"
                    and score["group"]=="global" and score["cohort_id"]=="common_all"],
                "test_best_descriptive": min((score for score in scores if score["reference_id"]=="PM_FAO56_V1"
                    and score["group"]=="global" and score["cohort_id"]=="own" and score["method_id"].startswith("ML_")),
                    key=lambda score: score["rmse"] if score["rmse"] is not None else math.inf),
            }
            manifest["outputs"]["test_results.json"] = {"path":"test_results.json",
                "sha256":_write(staging/"test_results.json", result)}
        manifest["status"] = "SUCCESS"
        manifest["scientific_selection_allowed"] = True
        manifest["finished_at_utc"] = _utc()
        manifest["runtime_seconds"] = time.perf_counter() - started
        usage_after = _resource_usage()
        manifest["resource_usage"] = {
            "wall_seconds":manifest["runtime_seconds"],
            "user_cpu_seconds":max(0.0, usage_after["user_cpu_seconds"]-usage_before["user_cpu_seconds"])
                if usage_after["user_cpu_seconds"] is not None and usage_before["user_cpu_seconds"] is not None else None,
            "system_cpu_seconds":max(0.0, usage_after["system_cpu_seconds"]-usage_before["system_cpu_seconds"])
                if usage_after["system_cpu_seconds"] is not None and usage_before["system_cpu_seconds"] is not None else None,
            "peak_rss_mb":usage_after["peak_rss_mb"],
        }
        # Finalize all hashes before atomically moving the complete result set.
        for path in sorted(staging.iterdir()):
            if path.is_file():
                path.replace(output/path.name)
        staging.rmdir()
        manifest["outputs"] = {name:entry for name,entry in manifest["outputs"].items()}
        (output/"manifest.json").write_bytes(json_bytes(manifest))
        manifest_sha = digest((output/"manifest.json").read_bytes())
        (output/"manifest.sha256").write_text(manifest_sha+"  manifest.json\n", encoding="ascii")
        return manifest
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        # The output directory was created empty by this invocation. Remove
        # only its own staged/published files so a failed run cannot resemble
        # a partial scientific result.
        for child in output.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        manifest["status"] = "FAILED"
        manifest["error"] = str(exc)
        manifest["outputs"] = {}
        manifest["finished_at_utc"] = _utc()
        manifest["runtime_seconds"] = time.perf_counter()-started
        manifest["resource_usage"] = _resource_usage()
        (output/"manifest.json").write_bytes(json_bytes(manifest))
        raise


def _thread_limits():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()
