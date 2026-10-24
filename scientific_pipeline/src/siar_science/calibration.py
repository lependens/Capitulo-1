"""Pure AHC estimands, chronological partitions, support and stability.

Multipliers emulate versioned daily PM, not measured evapotranspiration.
No fitting function can read a future partition implicitly.
"""
from collections import Counter
from datetime import date
import math
import statistics

from .quality import finite
from .serialization import digest, json_bytes

EMPIRICAL = {
    "HS_RA_V1": ("ET0_Hargreaves", "HS_RA_AHC_VALIDATED"),
    "HG_RS_V1": ("ET0_Hargreaves_Rs", "HG_RS_AHC_VALIDATED"),
    "VALIANTZAS_V1": ("ET0_Valiantzas", "VALIANTZAS_AHC_VALIDATED"),
}


def validate_protocol(protocol):
    """Validate all cut boundaries before looking at a single response value."""
    if (protocol.get("schema_version") != "ahc-protocol-v1" or protocol.get("estimator") != "OLS0"
            or protocol.get("reference") != "PM_FAO56_V1"):
        raise ValueError("Unsupported AHC protocol/estimand/reference")
    if set(protocol.get("splits", {})) != {"TRAIN", "VALIDATION", "TEST"}:
        raise ValueError("Require exactly three temporal partitions")
    previous_end = None
    for name in ("TRAIN", "VALIDATION", "TEST"):
        start, end = protocol["splits"][name]
        if (date.fromisoformat(start).isoformat() != start or date.fromisoformat(end).isoformat() != end or
                start > end or (previous_end is not None and previous_end >= start)):
            raise ValueError("Reversed or overlapping chronology")
        previous_end = end
    for start, end in protocol["training_stability_blocks"]:
        if start > end or start < int(protocol["splits"]["TRAIN"][0][:4]) or end > int(protocol["splits"]["TRAIN"][1][:4]):
            raise ValueError("Training stability blocks must stay in TRAIN")
    for key, lower, upper in (("training_stability_blocks", int(protocol["splits"]["TRAIN"][0][:4]), int(protocol["splits"]["TRAIN"][1][:4])),
                              ("post_test_descriptive_blocks", int(protocol["splits"]["TRAIN"][0][:4]), int(protocol["splits"]["TEST"][1][:4]))):
        previous = None
        for start, end in protocol[key]:
            if (type(start) is not int or type(end) is not int or start > end or start < lower or end > upper
                    or (previous is not None and start <= previous)):
                raise ValueError("Invalid or overlapping stability blocks")
            previous = end
    if any(type(v) is not int or v <= 0 for v in protocol["support"].values()):
        raise ValueError("Support thresholds must be positive integers")
    gate, selection = protocol["stability_gate"], protocol["selection"]
    if type(gate["minimum_robust_blocks"]) is not int or gate["minimum_robust_blocks"] < 2:
        raise ValueError("Stability requires at least two robust blocks")
    for key in ("stop_cv_above", "reference_stop_relative_difference_above", "reference_stop_mean_adjusted_difference_mm_day_above", "seasonal_material_range_over_annual"):
        if not finite(gate[key]) or gate[key] <= 0:
            raise ValueError("Invalid stability threshold")
    if not finite(gate["stop_max_min_ratio_above"]) or gate["stop_max_min_ratio_above"] <= 1:
        raise ValueError("Invalid stability ratio")
    for key in ("relative_validation_rmse_improvement", "maximum_station_rmse_degradation"):
        if not finite(selection[key]) or not 0 <= selection[key] < 1:
            raise ValueError("Invalid validation threshold")
    if type(selection["station_minimum_validation_n"]) is not int or selection["station_minimum_validation_n"] <= 0 or selection.get("prefer_simpler_on_tie") is not True:
        raise ValueError("Invalid station support or tie policy")
    if (protocol["selection"].get("fit_only_train") is not True or
            protocol["selection"].get("refit_train_plus_validation") is not False or
            protocol["selection"].get("require_full_validation_coverage") is not True):
        raise ValueError("Unsupported fit/validation policy")
    return protocol


def season(day):
    month = date.fromisoformat(day).month
    return "DJF" if month in (12, 1, 2) else "MAM" if month < 6 else "JJA" if month < 9 else "SON"


def partition(day, protocol):
    date.fromisoformat(day)
    matches = [name for name, (start, end) in protocol["splits"].items() if start <= day <= end]
    if len(matches) > 1:
        raise ValueError("Overlapping temporal partitions")
    return matches[0] if matches else "OUTSIDE_EXPERIMENT"


def valid_pair(row, field, reference="ET0_FAO56"):
    return (row.get("quality_status") in ("PASS", "WARN") and
            row.get("observation_status") in ("OBSERVED", "RECOVERED") and
            finite(row.get(field)) and row[field] >= 0 and
            finite(row.get(reference)) and row[reference] >= 0)


def fit_scale(pairs, estimator="OLS0"):
    """Return a dimensionless scale and denominator/zero diagnostics.

    pairs=(model,reference); both must be finite/nonnegative. OLS includes
    zeros and needs nonzero sum(x²); mean-ratio excludes x=0 only here.
    No epsilon, capping, intercept, slope clipping or replacement of missing.
    """
    pairs = list(pairs)
    if estimator not in ("OLS0", "MEAN_RATIO", "RATIO_SUMS"):
        raise ValueError("Unknown AHC estimator")
    if any(not finite(x) or not finite(y) or x < 0 or y < 0 for x, y in pairs):
        raise ValueError("Calibration requires finite nonnegative pairs")
    used = [(x, y) for x, y in pairs if x > 0] if estimator == "MEAN_RATIO" else pairs
    numerator = math.fsum(y / x for x, y in used) if estimator == "MEAN_RATIO" else math.fsum(
        x*y if estimator == "OLS0" else y for x, y in used)
    denominator = len(used) if estimator == "MEAN_RATIO" else math.fsum(
        x*x if estimator == "OLS0" else x for x, y in used)
    if not finite(numerator) or not finite(denominator):
        raise ValueError("Unrepresentable calibration sums")
    if estimator == "OLS0" and denominator == 0 and any(x > 0 for x, _ in pairs):
        raise ValueError("Squared predictor denominator underflow")
    coefficient = numerator / denominator if denominator > 0 else None
    if coefficient is not None and not finite(coefficient):
        raise ValueError("Nonfinite calibration coefficient")
    return dict(estimator=estimator, coefficient=coefficient, n_input=len(pairs), n_used=len(used),
                zero_predictors=sum(x == 0 for x, _ in pairs),
                zero_references=sum(y == 0 for _, y in pairs),
                smallest_positive_predictor=min((x for x, _ in pairs if x > 0), default=None),
                numerator=numerator, denominator=denominator,
                status="FITTED" if coefficient is not None else "NO_POSITIVE_DENOMINATOR")


def support(rows, spec, granularity="annual"):
    """Describe support without filtering fitted pairs or inventing observations."""
    counts = Counter(r["date"][:4] for r in rows)
    qualified = []
    for year, n in counts.items():
        seasons = Counter(season(r["date"]) for r in rows if r["date"].startswith(year))
        if n >= spec["qualified_year_n"] and all(seasons[s] >= spec["qualified_year_season_n"] for s in ("DJF", "MAM", "JJA", "SON")):
            qualified.append(year)
    if granularity == "annual":
        enough = len(rows) >= spec["minimum_fit_n"] and len(qualified) >= spec["minimum_fit_years"]
    elif granularity in ("seasonal", "monthly"):
        threshold = spec["minimum_season_fit_n"] if granularity == "seasonal" else spec["minimum_month_fit_n"]
        year_threshold = spec["minimum_season_year_n"] if granularity == "seasonal" else spec["minimum_month_year_n"]
        qualified = [year for year, n in counts.items() if n >= year_threshold]
        enough = len(rows) >= threshold and len(qualified) >= spec["minimum_group_years"]
    else:
        raise ValueError("Unknown coefficient granularity")
    return dict(n=len(rows), years=len(counts), qualified_years=sorted(qualified),
                season_counts=dict(Counter(season(r["date"]) for r in rows)),
                robust=enough, status="ROBUST_SUPPORT" if enough else "INSUFFICIENT_SUPPORT")


def group_key(day, granularity):
    return "all" if granularity == "annual" else season(day) if granularity == "seasonal" else day[5:7] if granularity == "monthly" else None


def fit_station(rows, station, method, protocol, *, split="TRAIN", granularity="annual", reference="ET0_FAO56", kind="AHC_VALIDATED", descriptive=False):
    """Fit only declared dates. Validated coefficients can only use TRAIN."""
    validate_protocol(protocol)
    if kind == "AHC_VALIDATED" and (split != "TRAIN" or descriptive):
        raise ValueError("AHC_VALIDATED must be fitted exclusively on TRAIN")
    if kind not in ("AHC_VALIDATED", "AHC_CLIMATOLOGY"):
        raise ValueError("Unknown AHC coefficient kind")
    if kind == "AHC_CLIMATOLOGY" and split != "ALL":
        raise ValueError("Climatology requires the explicitly bounded full history")
    if method not in EMPIRICAL or reference not in ("ET0_FAO56", "EtPMon"):
        raise ValueError("Unknown calibration method/reference")
    field = EMPIRICAL[method][0]
    chosen = [r for r in rows if r["station_code"] == station and valid_pair(r, field, reference)
              and (partition(r["date"], protocol) in ("TRAIN", "VALIDATION", "TEST") if split == "ALL" else partition(r["date"], protocol) == split)]
    keys = ["all"] if granularity == "annual" else ["DJF", "MAM", "JJA", "SON"] if granularity == "seasonal" else [f"{m:02}" for m in range(1, 13)] if granularity == "monthly" else []
    if not keys:
        raise ValueError("Unknown coefficient granularity")
    result = []
    for key in keys:
        subset = [r for r in chosen if group_key(r["date"], granularity) == key]
        coverage = support(subset, protocol["support"], granularity)
        estimate = fit_scale([(r[field], r[reference]) for r in subset])
        coefficient = estimate["coefficient"] if coverage["robust"] or descriptive else None
        result.append(dict(station_code=station, base_method_id=method, kind=kind, reference_id=reference,
            granularity=granularity, group_key=key, fit_split=split, coefficient=coefficient,
            status="ROBUST" if coverage["robust"] and coefficient is not None else "SHORT_SERIES_DESCRIPTIVE" if descriptive and coefficient is not None else "UNAVAILABLE",
            deployable=coverage["robust"] and coefficient is not None, support=coverage, estimate=estimate,
            fit_key_sha256=digest(json_bytes(sorted([r["station_code"], r["date"]] for r in subset))),
            calibration_start=min((r["date"] for r in subset), default=None),
            calibration_end=max((r["date"] for r in subset), default=None)))
    return result


def coefficient_for(coefficients, station, method, day, granularity="annual"):
    matches = [c for c in coefficients if c["station_code"] == station and c["base_method_id"] == method
               and c["granularity"] == granularity and c["group_key"] == group_key(day, granularity)]
    if len(matches) > 1:
        raise ValueError("Duplicate/ambiguous coefficient identity")
    if matches and (matches[0]["kind"] != "AHC_VALIDATED" or matches[0]["reference_id"] != "ET0_FAO56"
                    or matches[0]["fit_split"] != "TRAIN"):
        raise ValueError("Prediction requires primary-reference TRAIN validated coefficients")
    return matches[0] if matches else None


def block_stability(rows, station, method, protocol, blocks):
    """Block coefficients, sample CV and Theil–Sen trend; no uncertainty claim."""
    field = EMPIRICAL[method][0]
    summaries = []
    for start, end in blocks:
        subset = [r for r in rows if r["station_code"] == station and start <= int(r["date"][:4]) <= end and valid_pair(r, field)]
        fit = fit_scale([(r[field], r["ET0_FAO56"]) for r in subset])
        summaries.append(dict(start_year=start, end_year=end, midpoint_year=(start+end)/2,
                              support=support(subset, protocol["support"]), **fit))
    robust = [r for r in summaries if r["support"]["robust"] and r["coefficient"] is not None]
    values = [r["coefficient"] for r in robust]
    mean = statistics.mean(values) if values else None
    sd = statistics.stdev(values) if len(values) > 1 else None
    slopes = [(b["coefficient"]-a["coefficient"])/(b["midpoint_year"]-a["midpoint_year"])
              for i, a in enumerate(robust) for b in robust[i+1:]]
    cv = sd / mean if sd is not None and mean > 0 else None
    ratio = max(values)/min(values) if values and min(values)>0 else None
    gates = protocol["stability_gate"]
    enough = len(robust) >= gates["minimum_robust_blocks"]
    stop = enough and ((cv is not None and cv > gates["stop_cv_above"]) or
                       (ratio is not None and ratio > gates["stop_max_min_ratio_above"]))
    return dict(station_code=station, base_method_id=method, blocks=summaries, robust_blocks=len(robust),
                mean=mean, sample_sd=sd, min=min(values, default=None), max=max(values, default=None),
                range=max(values)-min(values) if values else None, cv=cv, max_min_ratio=ratio,
                theil_sen_slope_per_year=statistics.median(slopes) if slopes else None,
                status="STOP_TEMPORAL_INSTABILITY" if stop else "INSUFFICIENT_BLOCK_SUPPORT" if not enough else "PASS")
