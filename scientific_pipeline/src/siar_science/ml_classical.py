"""Pure ML-v1 feature, estimator, selection and metric helpers.

Imports scikit-learn lazily so the non-ML scientific package stays usable
without the optional dependency. No helper reads TEST during model selection.
"""
from collections import defaultdict
from datetime import date
import hashlib
import math
import warnings

from .calibration import partition, season
from .metrics import metrics
from .quality import finite
from .serialization import digest, json_bytes
from .solar import extraterrestrial_radiation

ALGORITHMS = ("LINEAR", "RIDGE", "ELASTIC_NET", "TREE", "RANDOM_FOREST", "EXTRA_TREES", "HIST_GRADIENT_BOOSTING")
BASELINES = ("PM_FAO56_V1", "HS_RA_V1", "HS_RA_AHC_VALIDATED", "HG_RS_V1", "HG_RS_AHC_VALIDATED",
             "HG_RS_AHC_VALIDATED_SELECTED", "VALIANTZAS_V1", "VALIANTZAS_AHC_VALIDATED")


def validate_ml_protocol(protocol):
    if protocol.get("schema_version") != "ml-classical-protocol-v1" or protocol.get("target") != "PM_FAO56_V1":
        raise ValueError("Unsupported ML protocol or target")
    if protocol.get("test_status") != "retrospective_locked" or protocol.get("preprocessing", {}).get("refit_train_plus_validation") is not False:
        raise ValueError("Unsupported TEST/refit policy")
    if set(protocol.get("splits", {})) != {"TRAIN", "VALIDATION", "TEST"}:
        raise ValueError("Require TRAIN/VALIDATION/TEST")
    previous = None
    for name in ("TRAIN", "VALIDATION", "TEST"):
        start, end = protocol["splits"][name]
        date.fromisoformat(start); date.fromisoformat(end)
        if start > end or (previous is not None and start <= previous):
            raise ValueError("Invalid temporal split")
        previous = end
    if set(protocol.get("candidate_grids", {})) != set(ALGORITHMS):
        raise ValueError("Require the seven ordered algorithm families")
    if any(not isinstance(grid, list) or not grid for grid in protocol["candidate_grids"].values()):
        raise ValueError("Empty candidate grid")
    if not protocol.get("primary_stations") or len(set(protocol["primary_stations"])) != len(protocol["primary_stations"]):
        raise ValueError("Invalid primary station population")
    if not protocol.get("feature_sets") or any(not isinstance(v, list) or not v for v in protocol["feature_sets"].values()):
        raise ValueError("Invalid feature families")
    if protocol.get("runtime", {}).get("random_seed") is None or protocol["selection"].get("test_tuning_allowed") is not False:
        raise ValueError("Missing deterministic seed or anti-tuning policy")
    return protocol


def feature_names(protocol, family):
    if family not in protocol["feature_sets"]:
        raise ValueError("Unknown feature family")
    return tuple(protocol["shared_context"] + protocol["feature_sets"][family])


def _features(dataset, row):
    values = row["values"]
    station = dataset["stations"][row["station_code"]]
    tmin, tmax = values["TempMin"], values["TempMax"]
    day = date.fromisoformat(row["date"])
    year_days = date(day.year, 12, 31).timetuple().tm_yday
    ordinal = day.timetuple().tm_yday
    angle = 2 * math.pi * ordinal / year_days
    return {
        "latitude_deg": station.get("latitude_deg"), "altitude_m": station.get("altitude_m"),
        "doy_sin": math.sin(angle), "doy_cos": math.cos(angle),
        "Ra_mj_m2_day": extraterrestrial_radiation(station["latitude_deg"], ordinal),
        "Tcalc": (tmin+tmax)/2 if finite(tmin) and finite(tmax) else None,
        "TD": tmax-tmin if finite(tmin) and finite(tmax) else None,
        "HumedadMedia": values["HumedadMedia"], "HumedadMin": values["HumedadMin"], "HumedadMax": values["HumedadMax"],
        "Radiacion": values["Radiacion"], "VelViento": values["VelViento"]}


def prepare_rows(dataset, official, protocol):
    """Thin immutable projections; no imputation and family-specific eligibility."""
    validate_ml_protocol(protocol)
    primary = set(protocol["primary_stations"])
    result = []
    for row in dataset["rows"]:
        key = (row["station_code"], row["date"])
        if key not in official:
            raise ValueError("AHC daily projection missing dataset key")
        source = official[key]
        values = _features(dataset, row)
        accepted = (row["station_code"] in primary and row["quality_status"] in protocol["eligibility"]["quality_status"]
                    and row["observation_status"] in protocol["eligibility"]["observation_status"]
                    and partition(row["date"], protocol) in ("TRAIN", "VALIDATION", "TEST"))
        target = source.get("PM_FAO56_V1")
        accepted = accepted and finite(target) and target >= 0
        families, reasons = {}, {}
        for family in protocol["feature_sets"]:
            names = feature_names(protocol, family)
            reason = []
            if not accepted:
                reason.append("outside_primary_population_or_source_target_ineligible")
            missing = [name for name in names if not finite(values.get(name))]
            if missing:
                reason.extend("missing_or_nonfinite:"+name for name in missing)
            if finite(values.get("TD")) and values["TD"] < 0:
                reason.append("negative_temperature_range")
            if any(finite(values.get(name)) and not 0 <= values[name] <= 100 for name in ("HumedadMedia", "HumedadMin", "HumedadMax") if name in names):
                reason.append("humidity_outside_0_100")
            if "Radiacion" in names and finite(values.get("Radiacion")) and values["Radiacion"] < 0:
                reason.append("negative_radiation")
            if "VelViento" in names and finite(values.get("VelViento")) and values["VelViento"] < 0:
                reason.append("negative_wind")
            reasons[family] = sorted(set(reason))
            families[family] = None if reason else [values[name] for name in names]
        result.append(dict(station_code=row["station_code"], date=row["date"], split=partition(row["date"], protocol),
            target=target, EtPMon=source.get("EtPMon"), baselines={m: source.get(m) for m in BASELINES},
            feature_values=values, features=families, eligibility_reasons=reasons,
            quality_status=row["quality_status"], observation_status=row["observation_status"],
            quality_flags=row["quality_flags"], scientific_flags=row.get("scientific_flags", []),
            provenance=row["provenance"], source_data_sha=row["source_data_sha"]))
    if len(result) != len(official):
        raise ValueError("AHC daily projection has extra keys")
    return result


def candidate_id(algorithm, family, parameters):
    return "mlc-" + digest(json_bytes(dict(algorithm=algorithm, feature_set=family, parameters=parameters)))


def make_estimator(algorithm, parameters, protocol):
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
    except ImportError as exc:
        raise ValueError("Install the scientific_pipeline[ml] optional dependency") from exc
    fixed = protocol.get("fixed_estimator_parameters", {}).get(algorithm, {})
    kwargs = {**fixed, **parameters}
    if algorithm == "LINEAR": model = LinearRegression(**kwargs)
    elif algorithm == "RIDGE": model = Ridge(**kwargs)
    elif algorithm == "ELASTIC_NET": model = ElasticNet(**kwargs)
    elif algorithm == "TREE": model = DecisionTreeRegressor(**kwargs)
    elif algorithm == "RANDOM_FOREST": model = RandomForestRegressor(**kwargs)
    elif algorithm == "EXTRA_TREES": model = ExtraTreesRegressor(**kwargs)
    elif algorithm == "HIST_GRADIENT_BOOSTING": model = HistGradientBoostingRegressor(**kwargs)
    else: raise ValueError("Unknown algorithm")
    return Pipeline([("scale", StandardScaler()), ("model", model)]) if algorithm in ("LINEAR", "RIDGE", "ELASTIC_NET") else model


def fit_candidates(rows, protocol):
    """Fit every preregistered candidate on TRAIN only."""
    models, records = {}, []
    for family in protocol["feature_sets"]:
        chosen = [r for r in rows if r["split"] == "TRAIN" and r["features"][family] is not None]
        if not chosen:
            raise ValueError("No TRAIN rows for "+family)
        x, y = [r["features"][family] for r in chosen], [r["target"] for r in chosen]
        key_sha = digest(json_bytes(sorted([r["station_code"], r["date"]] for r in chosen)))
        for algorithm in ALGORITHMS:
            for parameters in protocol["candidate_grids"][algorithm]:
                identifier = candidate_id(algorithm, family, parameters)
                model = make_estimator(algorithm, parameters, protocol)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(x, y)
                models[identifier] = model
                records.append(dict(candidate_id=identifier, algorithm=algorithm, feature_set=family,
                    parameters=parameters, fixed_parameters=protocol.get("fixed_estimator_parameters", {}).get(algorithm, {}),
                    features=list(feature_names(protocol, family)), train_n=len(chosen), train_key_sha256=key_sha,
                    fit_warnings=[dict(category=type(item.message).__name__,message=str(item.message)) for item in caught]))
    return models, records


def predict(model, rows, family):
    eligible = [r for r in rows if r["features"][family] is not None]
    values = model.predict([r["features"][family] for r in eligible]) if eligible else []
    if any(not finite(float(value)) for value in values):
        raise ValueError("Nonfinite ML prediction")
    return {(r["station_code"], r["date"]): float(value) for r, value in zip(eligible, values)}


def _choose(scored, protocol, *, cross_cell=False):
    finite_scores = [r for r in scored if r["metrics"]["rmse"] is not None]
    if not finite_scores:
        raise ValueError("No finite VALIDATION score")
    minimum = min(r["metrics"]["rmse"] for r in finite_scores)
    tolerance = protocol["selection"]["relative_tie_tolerance"]
    tied = [r for r in finite_scores if r["metrics"]["rmse"] <= minimum*(1+tolerance)]
    ranks = {name:i for i, name in enumerate(protocol["selection"]["complexity_rank"])}
    if cross_cell:
        tied.sort(key=lambda r:(ranks[r["algorithm"]], len(feature_names(protocol, r["feature_set"])), r["candidate_id"]))
    else:
        tied.sort(key=lambda r:r["candidate_id"])
    return tied[0]


def validate_candidates(rows, models, candidates, protocol):
    """Select on VALIDATION. Returned objects contain no TEST computation."""
    validation = [r for r in rows if r["split"] == "VALIDATION"]
    scores, selected = [], []
    for family in protocol["feature_sets"]:
        for algorithm in ALGORITHMS:
            cell = []
            for candidate in [c for c in candidates if c["feature_set"] == family and c["algorithm"] == algorithm]:
                predictions = predict(models[candidate["candidate_id"]], validation, family)
                eligible = [r for r in validation if (r["station_code"], r["date"]) in predictions]
                score = metrics([r["target"] for r in eligible], [predictions[r["station_code"],r["date"]] for r in eligible])
                record = dict(**candidate, metrics=score,
                    validation_key_sha256=digest(json_bytes(sorted([r["station_code"],r["date"]] for r in eligible))))
                scores.append(record); cell.append(record)
            selected.append(_choose(cell, protocol))
    prediction_maps = {r["candidate_id"]: predict(models[r["candidate_id"]], validation, r["feature_set"]) for r in selected}
    common = set.intersection(*(set(values) for values in prediction_maps.values()))
    common_rows = [r for r in validation if (r["station_code"],r["date"]) in common]
    common_scores = []
    for record in selected:
        values = prediction_maps[record["candidate_id"]]
        score = metrics([r["target"] for r in common_rows], [values[r["station_code"],r["date"]] for r in common_rows])
        common_record = dict(record)
        common_record.update(metrics=score,
            validation_key_sha256=digest(json_bytes(sorted([r["station_code"],r["date"]] for r in common_rows))))
        common_scores.append(common_record)
    champion = _choose(common_scores, protocol, cross_cell=True)
    return dict(status="PASS", selection_split="VALIDATION", test_inspected=False, refit_train_plus_validation=False,
        selected=selected, all_candidate_scores=scores, common_validation_scores=common_scores,
        campaign_champion=champion, common_validation_n=len(common_rows),
        common_validation_key_sha256=digest(json_bytes(sorted([r["station_code"],r["date"]] for r in common_rows))))


def importance(rows, models, selection, protocol):
    """Permutation importance on target-independent VALIDATION sample only."""
    from sklearn.inspection import permutation_importance
    chosen = []
    for algorithm in ALGORITHMS:
        cells = [r for r in selection["common_validation_scores"] if r["algorithm"] == algorithm]
        chosen.append(_choose(cells, protocol, cross_cell=True))
    if selection["campaign_champion"]["candidate_id"] not in {r["candidate_id"] for r in chosen}:
        chosen.append(selection["campaign_champion"])
    output = []
    for record in chosen:
        family, identifier = record["feature_set"], record["candidate_id"]
        eligible = [r for r in rows if r["split"] == "VALIDATION" and r["features"][family] is not None]
        eligible.sort(key=lambda r:hashlib.sha256((r["station_code"]+"/"+r["date"]).encode()).hexdigest())
        eligible = eligible[:protocol["permutation_importance"]["maximum_rows"]]
        result = permutation_importance(models[identifier], [r["features"][family] for r in eligible], [r["target"] for r in eligible],
            scoring="neg_root_mean_squared_error", n_repeats=protocol["permutation_importance"]["n_repeats"],
            random_state=protocol["runtime"]["random_seed"], n_jobs=1)
        names = feature_names(protocol, family)
        estimator = models[identifier].named_steps["model"] if hasattr(models[identifier], "named_steps") else models[identifier]
        intrinsic = getattr(estimator, "feature_importances_", None)
        coefficients = getattr(estimator, "coef_", None)
        for index, name in enumerate(names):
            output.append(dict(candidate_id=identifier, algorithm=record["algorithm"], feature_set=family, feature=name,
                validation_sample_n=len(eligible), permutation_rmse_increase_mean=float(result.importances_mean[index]),
                permutation_rmse_increase_sd=float(result.importances_std[index]), n_repeats=result.importances.shape[1],
                intrinsic_tree_importance=float(intrinsic[index]) if intrinsic is not None else None,
                standardized_linear_coefficient=float(coefficients[index]) if coefficients is not None else None,
                used_for_selection=False, causal_interpretation=False))
    return output


def group_labels(row):
    day = date.fromisoformat(row["date"]); station = row["station_code"]
    sy = f"{day.year + (day.month == 12)}-{season(row['date'])}"
    return (("global", "all", None), ("station", station, station), ("year", str(day.year), None),
            ("month", f"{day.month:02}", None), ("season", season(row["date"]), None),
            ("station_year", f"{station}:{day.year}", station), ("season_year", sy, None))


def score_predictions(rows, ml_ids):
    groups = defaultdict(list)
    for row in rows:
        if row["split"] in ("TRAIN","VALIDATION","TEST"):
            for group, key, station in group_labels(row): groups[row["split"],group,key,station].append(row)
    all_ids = list(BASELINES) + list(ml_ids)
    for (split, group, key, station), members in sorted(groups.items()):
        for reference in ("PM_FAO56_V1", "EtPMon"):
            target = lambda r: r["target"] if reference == "PM_FAO56_V1" else r["EtPMon"]
            valid = [r for r in members if finite(target(r)) and target(r) >= 0]
            cohorts = [("own", (m,)) for m in all_ids]
            cohorts += [("common_ml_35", tuple(ml_ids)), ("common_all", tuple(all_ids))]
            for cohort, methods in cohorts:
                selected = [r for r in valid if all(finite(r["predictions"].get(m)) for m in methods)]
                keys = sorted([r["station_code"],r["date"]] for r in selected)
                for method in methods:
                    score = metrics([target(r) for r in selected], [r["predictions"][method] for r in selected])
                    yield dict(split=split, reference_id=reference, group=group, key=key, station_code=station,
                        cohort_id=cohort, cohort_sha256=digest(json_bytes(keys)), method_id=method,
                        start_date=min((r["date"] for r in selected),default=None), end_date=max((r["date"] for r in selected),default=None),
                        r2_status="DEFINED" if score["r2"] is not None else "EMPTY_OR_SINGLE_OR_CONSTANT_REFERENCE", **score)
