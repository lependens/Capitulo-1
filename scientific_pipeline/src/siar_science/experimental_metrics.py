"""Precomputed temporal experiment metrics; frontends must not recompute science."""
from collections import defaultdict
from datetime import date

from .calibration import EMPIRICAL, season
from .metrics import metrics
from .quality import finite
from .serialization import digest, json_bytes

ANNUAL_MODELS = ("PM_FAO56_V1",) + tuple(EMPIRICAL) + tuple(v[1] for v in EMPIRICAL.values())


def reference(row, name):
    value = row["predictions"].get(name) if name == "PM_FAO56_V1" else row.get("EtPMon")
    accepted = row["quality_status"] in ("PASS", "WARN") and row["observation_status"] in ("OBSERVED", "RECOVERED")
    return value if accepted and finite(value) and value >= 0 else None


def labels(row):
    d = date.fromisoformat(row["date"])
    station = row["station_code"]
    sy = f"{d.year + (d.month == 12)}-{season(row['date'])}"
    ym = f"{d.year}-{d.month:02}"
    return (("global", "all", None), ("station", station, station),
            ("year", str(d.year), None), ("station_year", f"{station}:{d.year}", station),
            ("month", f"{d.month:02}", None), ("season", season(row["date"]), None),
            ("year_month", ym, None), ("station_year_month", f"{station}:{ym}", station),
            ("season_year", sy, None), ("station_season_year", f"{station}:{sy}", station))


def experiment_scores(rows):
    """Yield own, original/annual pairs, common seven and selected comparisons.

    All comparison members share exactly the same keys and reference. Global
    scores pool pairs; station means and per-period R² are never averaged.
    """
    groups = defaultdict(list)
    model_ids = sorted({name for row in rows for name in row["predictions"]})
    selected_models = [m for m in model_ids if m.endswith("_SELECTED")]
    for row in rows:
        if row["split"] in ("TRAIN", "VALIDATION", "TEST"):
            for kind, key, station in labels(row):
                groups[row["split"], kind, key, station].append(row)
    for (split, kind, key, station), group in sorted(groups.items()):
        for target in ("PM_FAO56_V1", "EtPMon"):
            valid = [r for r in group if reference(r, target) is not None]
            cohorts = [("own:"+name, (name,)) for name in model_ids]
            cohorts += [("paired:"+method, (method, adjusted)) for method, (_, adjusted) in EMPIRICAL.items()]
            cohorts += [("common_7", ANNUAL_MODELS)]
            for selected in selected_models:
                base = next(m for m, (_, adjusted) in EMPIRICAL.items() if selected == adjusted+"_SELECTED")
                cohorts.append(("selected:"+base, (base, EMPIRICAL[base][1], selected)))
            if selected_models:
                chosen = ("PM_FAO56_V1",) + tuple(adjusted+"_SELECTED" if adjusted+"_SELECTED" in selected_models else adjusted
                                                  for _, adjusted in EMPIRICAL.values())
                cohorts.append(("common_selected", chosen))
            for cohort, fields in cohorts:
                chosen = [r for r in valid if all(finite(r["predictions"].get(f)) for f in fields)]
                keys = sorted([r["station_code"], r["date"]] for r in chosen)
                cohort_sha = digest(json_bytes(keys))
                observed = [reference(r, target) for r in chosen]
                for method in fields:
                    score = metrics(observed, [r["predictions"][method] for r in chosen])
                    yield dict(split=split, reference_id=target, group=kind, key=key, station_code=station,
                        start_date=min((r["date"] for r in chosen), default=None),
                        end_date=max((r["date"] for r in chosen), default=None),
                        cohort_id="own" if cohort.startswith("own:") else cohort, cohort_sha256=cohort_sha,
                        method_id=method, r2_status="DEFINED" if score["r2"] is not None else "EMPTY_OR_SINGLE_OR_CONSTANT_REFERENCE", **score)
