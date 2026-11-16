"""Leakage-safe rolling-origin development helpers for classical ML v1.

The functions in this module never inspect rows labelled TEST.  They are used
both by the small Linux smoke and by the later full classical campaign.
"""
from collections import defaultdict
import math
import time
import warnings

from .metrics import metrics
from .ml_classical import candidate_id, make_estimator
from .serialization import digest, json_bytes


def _inside(day, period):
    return period[0] <= day <= period[1]


def _eligible(rows, family, period):
    return [row for row in rows
            if row["split"] == "DEVELOPMENT"
            and _inside(row["date"], period)
            and row["features"][family] is not None]


def aggregate_fold_metrics(fold_records):
    """Aggregate predeclared fold RMSE with eligible validation n as weights."""
    valid = [record for record in fold_records
             if record["metrics"]["rmse"] is not None and record["metrics"]["n"] > 0]
    if not valid:
        raise ValueError("No finite rolling-origin validation score")
    total_n = sum(record["metrics"]["n"] for record in valid)
    weighted_mean = sum(record["metrics"]["n"] * record["metrics"]["rmse"] for record in valid) / total_n
    weighted_variance = sum(record["metrics"]["n"] * (record["metrics"]["rmse"] - weighted_mean) ** 2
                            for record in valid) / total_n
    pooled_rmse = math.sqrt(sum(record["metrics"]["n"] * record["metrics"]["rmse"] ** 2
                                for record in valid) / total_n)
    rmses = [record["metrics"]["rmse"] for record in valid]
    return {
        "fold_count": len(valid),
        "validation_n_total": total_n,
        "weighted_mean_fold_rmse": weighted_mean,
        "pooled_rmse": pooled_rmse,
        "weighted_sd_fold_rmse": math.sqrt(weighted_variance),
        "min_fold_rmse": min(rmses),
        "max_fold_rmse": max(rmses),
        "aggregation_weights": "eligible validation n in each fold",
    }


def evaluate_rolling_candidates(rows, protocol, candidates):
    """Fit candidates independently in every causal fold and score held-out windows.

    `candidates` is an iterable of (algorithm, feature_set, parameters).  Models
    are deliberately not returned: a later scientific selection must use only
    these development scores and then refit the frozen configuration on the
    full DEVELOPMENT period.
    """
    records = []
    started = time.perf_counter()
    for algorithm, family, parameters in candidates:
        identifier = candidate_id(algorithm, family, parameters)
        candidate_folds = []
        for fold in protocol["development_folds"]:
            train = _eligible(rows, family, fold["train"])
            validation = _eligible(rows, family, fold["validation"])
            if not train or not validation:
                raise ValueError(f"Empty rolling fold {fold['fold_id']} for {family}")
            model = make_estimator(algorithm, parameters, protocol)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit([row["features"][family] for row in train], [row["target"] for row in train])
            predictions = [float(value) for value in model.predict([row["features"][family] for row in validation])]
            if any(not math.isfinite(value) for value in predictions):
                raise ValueError("Nonfinite rolling-origin prediction")
            score = metrics([row["target"] for row in validation], predictions)
            record = {
                "candidate_id": identifier,
                "algorithm": algorithm,
                "feature_set": family,
                "parameters": parameters,
                "fold_id": fold["fold_id"],
                "train_period": fold["train"],
                "validation_period": fold["validation"],
                "train_n": len(train),
                "validation_n": len(validation),
                "train_key_sha256": digest(json_bytes(sorted([row["station_code"], row["date"]] for row in train))),
                "validation_key_sha256": digest(json_bytes(sorted([row["station_code"], row["date"]] for row in validation))),
                "metrics": score,
                "fit_warnings": [{"category": type(item.message).__name__, "message": str(item.message)} for item in caught],
                "test_inspected": False,
            }
            records.append(record)
            candidate_folds.append(record)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["candidate_id"]].append(record)
    summaries = []
    for identifier, values in grouped.items():
        first = values[0]
        summaries.append({
            "candidate_id": identifier,
            "algorithm": first["algorithm"],
            "feature_set": first["feature_set"],
            "parameters": first["parameters"],
            "fold_metrics": [{"fold_id": value["fold_id"], **value["metrics"]} for value in values],
            "aggregate": aggregate_fold_metrics(values),
            "test_inspected": False,
            "scientific_selection_allowed": False,
        })
    summaries.sort(key=lambda item: (item["algorithm"], item["candidate_id"]))
    return {
        "fold_records": records,
        "candidate_summaries": summaries,
        "runtime_seconds": time.perf_counter() - started,
        "test_inspected": False,
    }


def linux_smoke_candidates(protocol):
    linux = protocol["linux_execution"]
    family = linux["smoke_feature_set"]
    if linux["smoke_algorithms"] != ["LINEAR", "RIDGE"]:
        raise ValueError("Smoke must remain LINEAR + RIDGE")
    return [("LINEAR", family, {}), ("RIDGE", family, {"alpha": 1.0})]

