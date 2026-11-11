"""DEVELOPMENT-only V2 fitting for one station, input family and calendar."""
from collections import defaultdict
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from time import perf_counter
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler

from .ann_modern import _epochs, _model, final_epochs
from .et0_v2_support import category, choose_candidate, design_vector, outer_plan
from .metrics import metrics
from .ml_classical import candidate_id, make_estimator
from .serialization import digest, json_bytes


@dataclass
class Cell:
    station: str
    family: str
    calendar: str
    protocol: dict
    rows: dict
    plans: list
    x: np.ndarray
    y: np.ndarray
    raw: np.ndarray
    positions: dict
    plan_hashes: dict

    def indices(self, dates):
        return np.asarray([self.positions[day] for day in dates], dtype=int)

    def groups(self, dates):
        return [category(day, self.calendar) for day in dates]


def make_cell(station_rows, key, protocol, expected=None):
    station, family, calendar = key
    start, end = protocol["development"]
    rows = {}
    for row in station_rows:
        day = row["date"]
        if row["station_code"] != station or not start <= day <= end or day in rows:
            raise ValueError("Invalid or duplicate DEVELOPMENT key")
        if row["target"] is not None and row["inputs"][family] is not None:
            rows[day] = row
    rows = dict(sorted(rows.items()))
    empirical = {day: row["baselines"][family]["estimate"] for day, row in rows.items()}
    if any(not isfinite(value) for value in empirical.values()):
        raise ValueError("Invalid empirical input")
    plans = list(outer_plan(empirical, station=station, family=family, protocol=protocol))
    plan_hashes = {plan["year"]: digest(json_bytes(plan)) for plan in plans}
    if expected is not None:
        if list(plan_hashes.values()) != [record["keys_sha256"] for record in expected["folds"]]:
            raise ValueError(f"Development keys diverge from the approved inventory: {key}")
    return Cell(station, family, calendar, protocol, rows,
                [plan for plan in plans if plan["score"][calendar]],
                np.asarray([design_vector(row["inputs"][family], day, calendar, protocol)
                            for day, row in rows.items()], dtype=np.float64),
                np.asarray([row["target"] for row in rows.values()], dtype=np.float64),
                np.asarray(list(empirical.values()), dtype=np.float64),
                {day: index for index, day in enumerate(rows)}, plan_hashes)


def ahc_coefficients(groups, empirical, targets):
    samples = defaultdict(list)
    for group, x, y in zip(groups, empirical, targets, strict=True):
        samples[group].append((float(x), float(y)))
    result = {}
    for group, pairs in sorted(samples.items()):
        denominator = fsum(x*x for x, _ in pairs)
        result[group] = {"n": len(pairs), "sum_x2": denominator,
                         "coefficient": fsum(x*y for x, y in pairs) / denominator if denominator > 0 else None}
    return result


def score_fold(cell, plan, predictions):
    dates = plan["score"][cell.calendar]
    actual = cell.y[cell.indices(dates)].tolist()
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(actual),) or not np.isfinite(predictions).all():
        raise ValueError("Nonfinite predictions or incorrect DEVELOPMENT cohort")
    return {"year": int(plan["year"]), "keys_sha256": digest(json_bytes(dates)),
            "plan_sha256": cell.plan_hashes[plan["year"]], **metrics(actual, predictions.tolist())}


def fit_ahc(cell, plans):
    folds = []
    for plan in plans:
        train = cell.indices(plan["train"])
        fit = ahc_coefficients(cell.groups(plan["train"]), cell.raw[train], cell.y[train])
        validation = plan["score"][cell.calendar]
        preds = [x * fit[group]["coefficient"] for x, group in
                 zip(cell.raw[cell.indices(validation)], cell.groups(validation), strict=True)]
        folds.append({**score_fold(cell, plan, preds), "coefficients": fit})
    groups = ahc_coefficients(cell.groups(cell.rows), cell.raw, cell.y)
    return {"groups": groups, "folds": folds, "paired": len(cell.rows)}


def candidates(stage, cell, ml_protocol):
    if stage == "ML":
        return [{"candidate_id": candidate_id(algorithm, cell.family, params), "algorithm": algorithm,
                 "parameters": params} for algorithm in cell.protocol["ml"]["complexity_rank"]
                for params in ml_protocol["candidate_grids"][algorithm]]
    if stage not in {"ANN_A", "ANN_E"}:
        raise ValueError("Unknown stage")
    result = []
    for architecture in cell.protocol["ann"]["architectures"]:
        sizes = [cell.x.shape[1], *architecture, 1]
        count = sum((left + 1) * right for left, right in zip(sizes, sizes[1:]))
        result.append({"candidate_id": "ann-" + "-".join(map(str, architecture)),
                       "architecture": architecture, "parameter_count": count})
    return result


def _network(cell, architecture, seed, size):
    config = cell.protocol["ann"]
    return _model(architecture, seed, {**config, "batch_size": min(config["batch_size_max"], size)})


def _fit_seed(cell, plan, rule, architecture, seed):
    inner_fit, inner_valid = map(cell.indices, plan["inner"][rule])
    train = cell.indices(plan["train"])
    validation = cell.indices(plan["score"][cell.calendar])
    scale = StandardScaler().fit(cell.x[inner_fit])
    fit, valid = scale.transform(cell.x[inner_fit]), scale.transform(cell.x[inner_valid])
    model = _network(cell, architecture, seed, len(fit))
    settings = cell.protocol["ann"]["early_stopping"]
    best, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, settings["max_epochs"] + 1):
        model.partial_fit(fit, cell.y[inner_fit])
        loss = sqrt(float(np.mean((model.predict(valid) - cell.y[inner_valid]) ** 2)))
        if not isfinite(loss):
            raise ValueError("Nonfinite ANN inner RMSE")
        if loss < best - settings["min_delta_rmse"]:
            best, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    scale = StandardScaler().fit(cell.x[train])
    model = _epochs(_network(cell, architecture, seed, len(train)),
                    scale.transform(cell.x[train]), cell.y[train], best_epoch)
    prediction = model.predict(scale.transform(cell.x[validation]))
    return prediction, {"seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
                        "inner_rmse": best, **metrics(cell.y[validation].tolist(), prediction.tolist())}


def fit_candidate(stage, cell, plans, config, ml_protocol):
    started = perf_counter()
    folds = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for plan in plans:
            if stage == "ML":
                train = cell.indices(plan["train"])
                valid = cell.indices(plan["score"][cell.calendar])
                estimator = make_estimator(config["algorithm"], config["parameters"], ml_protocol)
                estimator.fit(cell.x[train], cell.y[train])
                folds.append(score_fold(cell, plan, estimator.predict(cell.x[valid])))
                del estimator
            else:
                fits = [_fit_seed(cell, plan, stage[-1], config["architecture"], seed)
                        for seed in cell.protocol["ann"]["seeds"]]
                folds.append({**score_fold(cell, plan, np.mean([pred for pred, _ in fits], axis=0)),
                              "seeds": [scores for _, scores in fits]})
    n = sum(fold["n"] for fold in folds)
    average = fsum(fold["rmse"] for fold in folds) / len(folds)
    return {**config, "folds": folds, "n": n,
            "mean_annual_rmse": average,
            "sd_annual_rmse": sqrt(fsum((fold["rmse"] - average) ** 2 for fold in folds) / len(folds)),
            "pooled_rmse": sqrt(fsum(fold["n"] * fold["rmse"] ** 2 for fold in folds) / n),
            "warnings": sorted({str(record.message) for record in caught}),
            "wall_seconds": perf_counter() - started}


def selections(stage, cell, summaries):
    records = [{**item, "yearly_rmse": {fold["year"]: fold["rmse"] for fold in item["folds"]}}
               for item in summaries]
    years = [int(plan["year"]) for plan in cell.plans]
    by_id = {item["candidate_id"]: item for item in summaries}

    def choose(group):
        return choose_candidate(group, family_type="ml" if stage == "ML" else "ann",
                                protocol=cell.protocol, supported_years=years)

    if stage == "ML":
        chosen = {algorithm: choose([item for item in records if item["algorithm"] == algorithm])
                  for algorithm in cell.protocol["ml"]["complexity_rank"]}
        ids = {item["candidate_id"] for item in chosen.values()}
        principal = choose([item for item in records if item["candidate_id"] in ids])
    else:
        principal = choose(records)
        chosen = {stage: principal}
    return principal, {label: {**by_id[item["candidate_id"]], **item} for label, item in chosen.items()}


def refit(stage, cell, chosen, ml_protocol):
    """Yield one frozen model at a time with its selected configuration."""
    for label, config in chosen.items():
        if stage == "ML":
            model = make_estimator(config["algorithm"], config["parameters"], ml_protocol)
            yield label, model.fit(cell.x, cell.y), {}
        else:
            seeds = cell.protocol["ann"]["seeds"]
            folds = [{"best_epochs": [item["best_epoch"] for item in fold["seeds"]]} for fold in config["folds"]]
            epochs = final_epochs(folds, seeds)
            scale = StandardScaler().fit(cell.x)
            x = scale.transform(cell.x)
            models = [_epochs(_network(cell, config["architecture"], seed, len(x)), x, cell.y, epochs[str(seed)])
                      for seed in seeds]
            yield label, {"scale": scale, "models": models}, {"epochs": epochs, "seeds": seeds}
