"""Pure staged AHC experiment. Decisions see TRAIN/VALIDATION, never TEST.

No data acquisition, raw mutation, regional fallback, random split or ANN.
"""
from collections import defaultdict
from copy import deepcopy
import math

from .calibration import (EMPIRICAL, block_stability, coefficient_for, fit_scale,
                          fit_station, group_key, partition, support, valid_pair, validate_protocol)
from .metrics import metrics
from .serialization import digest, json_bytes
from .quality import finite


def _id(record, protocol):
    return "ahc-" + digest(json_bytes(dict(coefficient=record, protocol=protocol)))


def train_analysis(rows, stations, protocol):
    """Pure fit/stability/reference sensitivity; only TRAIN values are used."""
    validate_protocol(protocol)
    train = [r for r in rows if partition(r["date"], protocol) == "TRAIN"]
    coefficients, stability, sensitivity, material = [], [], [], []
    issues = []
    for station in sorted(stations):
        for method, (field, _) in EMPIRICAL.items():
            annual = fit_station(train, station, method, protocol)[0]
            coefficients.append(annual)
            check = block_stability(train, station, method, protocol, protocol["training_stability_blocks"])
            stability.append(check)
            if check["status"].startswith("STOP"):
                issues.append(dict(reason=check["status"], station_code=station, base_method_id=method))
            chosen = [r for r in train if r["station_code"] == station and valid_pair(r, field)]
            estimands = {name: fit_scale([(r[field], r["ET0_FAO56"]) for r in chosen], name)
                         for name in ("OLS0", "MEAN_RATIO", "RATIO_SUMS")}
            same = [r for r in chosen if valid_pair(r, field, "EtPMon")]
            pm = fit_scale([(r[field], r["ET0_FAO56"]) for r in same])
            siar = fit_scale([(r[field], r["EtPMon"]) for r in same])
            paired_support = support(same, protocol["support"])
            a, b = pm["coefficient"], siar["coefficient"]
            relative = abs(a-b)/abs(a) if a not in (None, 0) and b is not None else None
            adjusted_difference = (math.fsum(abs(a-b)*r[field] for r in same)/len(same)
                                   if a is not None and b is not None and same else None)
            gate = protocol["stability_gate"]
            disagreement = paired_support["robust"] and (
                (relative is not None and relative > gate["reference_stop_relative_difference_above"]) or
                (adjusted_difference is not None and adjusted_difference > gate["reference_stop_mean_adjusted_difference_mm_day_above"]) or
                (a == 0 and b not in (None, 0)))
            if disagreement:
                issues.append(dict(reason="STOP_REFERENCE_DISAGREEMENT", station_code=station, base_method_id=method))
            sensitivity.append(dict(station_code=station, base_method_id=method, train_estimands=estimands,
                same_pairs_reference_comparison=dict(pm=pm, siar=siar, support=paired_support,
                    relative_coefficient_difference=relative, mean_adjusted_difference_mm_day=adjusted_difference,
                    status="STOP" if disagreement else "PASS" if paired_support["robust"] else "INSUFFICIENT_SUPPORT")))
            seasonal = fit_station(train, station, method, protocol, granularity="seasonal")
            values = [c["coefficient"] for c in seasonal if c["status"] == "ROBUST"]
            spread = (max(values)-min(values))/annual["coefficient"] if len(values) == 4 and annual["coefficient"] not in (None, 0) else None
            material.append(dict(station_code=station, base_method_id=method, seasonal_range_over_annual=spread,
                material=spread is not None and spread >= gate["seasonal_material_range_over_annual"]))
    enabled = {m: any(r["material"] for r in material if r["base_method_id"] == m) for m in EMPIRICAL}
    for station in sorted(stations):
        for method in EMPIRICAL:
            if enabled[method]:
                for granularity in ("seasonal", "monthly"):
                    coefficients.extend(fit_station(train, station, method, protocol, granularity=granularity))
    for c in coefficients:
        c["coefficient_id"] = _id(c, protocol)
    return dict(status="STOP" if issues else "PASS", stop_reasons=issues, fit_split="TRAIN", reference="PM_FAO56_V1",
                coefficients=coefficients, training_stability=stability, train_sensitivity=sensitivity,
                complexity_enabled=enabled, seasonal_materiality=material,
                train_partition_key_sha256=digest(json_bytes(sorted([r["station_code"], r["date"]] for r in train))),
                description="All coefficient fits and gates use TRAIN only; no TEST or VALIDATION values inspected")


def _prediction(row, method, coefficients, granularity):
    field = EMPIRICAL[method][0]
    record = coefficient_for(coefficients, row["station_code"], method, row["date"], granularity)
    value = row.get(field)
    accepted = row.get("quality_status") in ("PASS", "WARN") and row.get("observation_status") in ("OBSERVED", "RECOVERED")
    if not accepted or not finite(value) or record is None or record["coefficient"] is None or not record["deployable"]:
        return None, record
    result = value * record["coefficient"]
    if not finite(result):
        raise ValueError("Nonfinite AHC prediction")
    return result, record


def validate_selection(rows, training, protocol):
    """Choose among frozen TRAIN fits using VALIDATION only; never refit."""
    validate_protocol(protocol)
    if training["status"] != "PASS" or training["fit_split"] != "TRAIN":
        raise ValueError("Validation requires successful frozen TRAIN coefficients")
    validation = [r for r in rows if partition(r["date"], protocol) == "VALIDATION"]
    coefficients = training["coefficients"]
    selections = []
    for method, (field, _) in EMPIRICAL.items():
        chosen = [r for r in validation if valid_pair(r, field) and _prediction(r, method, coefficients, "annual")[0] is not None]
        observed = [r["ET0_FAO56"] for r in chosen]
        annual = [_prediction(r, method, coefficients, "annual")[0] for r in chosen]
        annual_score = metrics(observed, annual)
        candidate_scores = {}
        accepted = {}
        for granularity in ("seasonal", "monthly"):
            if not training["complexity_enabled"][method]:
                candidate_scores[granularity] = dict(status="NOT_ENABLED_BY_TRAIN", n=0)
                continue
            predictions = [_prediction(r, method, coefficients, granularity)[0] for r in chosen]
            complete = all(v is not None for v in predictions)
            score = metrics(observed, predictions)
            by_station = []
            station_veto = False
            for station in sorted({r["station_code"] for r in chosen}):
                idx = [i for i, r in enumerate(chosen) if r["station_code"] == station]
                old = metrics([observed[i] for i in idx], [annual[i] for i in idx])
                new = metrics([observed[i] for i in idx], [predictions[i] for i in idx])
                ratio = new["rmse"] / old["rmse"] if new["rmse"] is not None and old["rmse"] not in (None, 0) else None
                veto = old["n"] >= protocol["selection"]["station_minimum_validation_n"] and (
                    new["n"] != old["n"] or (ratio is not None and ratio > 1+protocol["selection"]["maximum_station_rmse_degradation"]) or
                    (old["rmse"] == 0 and new["rmse"] not in (None, 0)))
                station_veto |= veto
                by_station.append(dict(station_code=station, annual=old, candidate=new, veto=veto))
            improvement = 1-score["rmse"]/annual_score["rmse"] if score["rmse"] is not None and annual_score["rmse"] not in (None, 0) else None
            acceptable = complete and not station_veto and improvement is not None and improvement >= protocol["selection"]["relative_validation_rmse_improvement"]
            candidate_scores[granularity] = dict(status="ACCEPTABLE" if acceptable else "RETAIN_ANNUAL", metrics=score,
                complete_annual_cohort=complete, relative_rmse_improvement=improvement, stations=by_station)
            accepted[granularity] = acceptable
        selected = "seasonal" if accepted.get("seasonal") else "annual"
        if accepted.get("monthly"):
            if selected == "annual":
                selected = "monthly"
            else:
                seasonal_rmse = candidate_scores["seasonal"]["metrics"]["rmse"]
                monthly_rmse = candidate_scores["monthly"]["metrics"]["rmse"]
                if seasonal_rmse > 0 and 1-monthly_rmse/seasonal_rmse >= protocol["selection"]["relative_validation_rmse_improvement"]:
                    selected = "monthly"
        selections.append(dict(base_method_id=method, selected_granularity=selected, annual_metrics=annual_score,
            candidates=candidate_scores, validation_key_sha256=digest(json_bytes(sorted([r["station_code"], r["date"]] for r in chosen)))))
    return dict(status="PASS", selection_split="VALIDATION", coefficients_refitted=False,
                frozen_training_coefficients_sha256=digest(json_bytes(coefficients)), selections=selections,
                test_inspected=False)


def apply_validated(rows, coefficients, protocol, selections):
    """Return thin per-day predictions; prediction does not require PM target."""
    if any(c["kind"] != "AHC_VALIDATED" or c["fit_split"] != "TRAIN" for c in coefficients):
        raise ValueError("Validated output requires exclusively TRAIN-fitted coefficients")
    selected = {s["base_method_id"]: s["selected_granularity"] for s in selections}
    result = []
    for row in rows:
        predictions = {"PM_FAO56_V1": row.get("ET0_FAO56")}
        predictions.update({m: row.get(field) for m, (field, _) in EMPIRICAL.items()})
        identifiers, eligibility = {}, deepcopy(row["eligibility"])
        flags = list(row.get("scientific_flags", []))
        for method, (field, adjusted) in EMPIRICAL.items():
            granularities = [("annual", adjusted)]
            if selected.get(method, "annual") != "annual":
                granularities.append((selected[method], adjusted + "_SELECTED"))
            for granularity, model in granularities:
                value, coefficient = _prediction(row, method, coefficients, granularity)
                predictions[model] = value
                identifiers[model] = coefficient.get("coefficient_id") if coefficient is not None else None
                eligibility[model] = dict(eligible=value is not None, reasons=[] if value is not None else
                    ["missing_base_prediction" if not finite(row.get(field)) else "unavailable_train_coefficient_or_source_status"],
                    granularity=granularity)
                if value is not None and value < 0:
                    flags.append("negative_prediction:" + model)
        result.append(dict(station_code=row["station_code"], date=row["date"], split=partition(row["date"], protocol),
            EtPMon=row["EtPMon"], predictions=predictions, coefficient_ids=identifiers, eligibility=eligibility,
            observation_status=row["observation_status"], quality_status=row["quality_status"],
            quality_flags=row["quality_flags"], scientific_flags=flags, provenance=row["provenance"], source_data_sha=row["source_data_sha"]))
    return result


def closeout_analysis(rows, stations, protocol, *, test_completed=False):
    """Post-TEST description only; no selection or validated-coefficient mutation."""
    if test_completed is not True:
        raise ValueError("Climatology/post-TEST diagnostics require a completed TEST")
    stability = [block_stability(rows, s, m, protocol, protocol["post_test_descriptive_blocks"])
                 for s in sorted(stations) for m in EMPIRICAL]
    issues = [dict(reason=r["status"], station_code=r["station_code"], base_method_id=r["base_method_id"])
              for r in stability if r["status"].startswith("STOP")]
    climatology = []
    if not issues:
        for station in sorted(stations):
            for method in EMPIRICAL:
                records = fit_station(rows, station, method, protocol, split="ALL", kind="AHC_CLIMATOLOGY", descriptive=True)
                for record in records:
                    record["coefficient_id"] = _id(record, protocol)
                climatology.extend(records)
    return dict(status="STOP" if issues else "PASS", stop_reasons=issues,
                post_test_stability=stability, climatology=climatology,
                out_of_sample_claim=False, used_for_validated_selection=False)
