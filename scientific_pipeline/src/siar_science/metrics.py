"""Pairwise finite evaluation. Bias = prediction minus SIAR, R² = 1-SSE/SST."""
import math
from collections import defaultdict
from datetime import date

CLASSICAL_FIELDS = ("ET0_FAO56", "ET0_Hargreaves", "ET0_Valiantzas")
CLASSICAL_COHORTS = {
    "PM_HS": CLASSICAL_FIELDS[:2],
    "PM_VAL": (CLASSICAL_FIELDS[0], CLASSICAL_FIELDS[2]),
    "HS_VAL": CLASSICAL_FIELDS[1:],
    "PM_HS_VAL": CLASSICAL_FIELDS,
}


def reference_value(row):
    """Reference eligibility is distinct from prediction eligibility; zero valid."""
    value = row["values"].get("EtPMon")
    return value if (isinstance(value, (float, int)) and not isinstance(value, bool)
                     and math.isfinite(value) and value >= 0
                     and row["quality_status"] in ("PASS", "WARN")
                     and row["observation_status"] in ("OBSERVED", "RECOVERED")) else None


def metrics(observed, predicted):
    observed, predicted = list(observed), list(predicted)
    if len(observed) != len(predicted):
        raise ValueError("Unequal vector lengths")
    pairs = [(o, p) for o, p in zip(observed, predicted)
             if isinstance(o, (int, float)) and isinstance(p, (int, float))
             and not isinstance(o, bool) and not isinstance(p, bool)
             and math.isfinite(o) and math.isfinite(p)]
    n = len(pairs)
    if not n:
        return dict(n=0, mae=None, rmse=None, bias=None, r2=None)
    errors = [p-o for o, p in pairs]
    sse = sum(e*e for e in errors)
    mean = sum(o for o, _ in pairs) / n
    sst = sum((o-mean)**2 for o, _ in pairs)
    return dict(n=n, mae=sum(abs(e) for e in errors)/n,
                rmse=math.sqrt(sse/n), bias=sum(errors)/n,
                r2=1-sse/sst if n >= 2 and sst > 0 else None)


def evaluate(rows, prediction_fields=("ET0_FAO56", "ET0_Hargreaves"), *, classical=False):
    """Each model's finite pairs; global is pooled, never mean of station scores.

    Seasons are northern meteorological DJF/MAM/JJA/SON. December belongs
    to the following season-year. No claim of independent ground truth for SIAR.
    """
    groups = defaultdict(list)
    groups["global", "all"] = []
    for row in rows:
        d = date.fromisoformat(row["date"])
        season = ("DJF" if d.month in (12, 1, 2) else
                  "MAM" if d.month < 6 else "JJA" if d.month < 9 else "SON")
        for kind, key in (("global", "all"), ("station", row["station_code"]),
                          ("year", str(d.year)), ("month", f"{d.month:02d}"),
                          ("station_year", f"{row['station_code']}:{d.year}"),
                          ("season", season),
                          ("season_year", f"{d.year + (d.month == 12)}-{season}")):
            groups[kind, key].append(row)
    result = []
    for (kind, key), group in sorted(groups.items()):
        paired = [r for r in group if reference_value(r) is not None
                  and all(isinstance(r.get(m), (float, int)) and not isinstance(r.get(m), bool)
                          and math.isfinite(r[m]) for m in ("ET0_FAO56", "ET0_Hargreaves"))]
        cohorts = [("own", group, prediction_fields)]
        if classical:
            for name, fields in CLASSICAL_COHORTS.items():
                selected = [r for r in group if reference_value(r) is not None and
                            all(isinstance(r.get(f), (int, float)) and not isinstance(r.get(f), bool)
                                and math.isfinite(r[f]) for f in fields)]
                cohorts.append((name, selected, fields))
        else:
            cohorts.append(("paired", paired, prediction_fields))
        for cohort, selected, fields in cohorts:
            for method in fields:
                score = metrics([reference_value(r) for r in selected], [r.get(method) for r in selected])
                result.append(dict(group=kind, key=key, cohort=cohort, method=method, **score))
    return result
