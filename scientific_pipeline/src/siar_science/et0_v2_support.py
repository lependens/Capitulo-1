"""Target-independent partitions and support checks for the V2 contract."""
from collections import Counter, defaultdict
from hashlib import sha256
from math import fsum, sqrt

from .serialization import digest, json_bytes


def category(day, calendar):
    if calendar == "annual":
        return "annual"
    month = day[5:7]
    if calendar == "monthly":
        return month
    if calendar == "seasonal":
        return ("DJF", "MAM", "JJA", "SON")[(int(month) % 12) // 3]
    raise ValueError("Unknown V2 calendar")


def design_vector(base, day, calendar, protocol):
    """Exactly the family inputs, with the same calendar information as AHC."""
    group = category(day, calendar)
    return list(base) + ([] if calendar == "annual" else
                         [int(group == label) for label in protocol["calendars"][calendar]])


def inner_split(train, rule, *, station, family, year, seed):
    train = sorted(train)
    if len(train) < 2 or len(set(train)) != len(train):
        raise ValueError("Inner validation requires at least two unique keys")
    n_valid = (3 * len(train) + 19) // 20  # ceil(15%)
    if rule == "A":
        valid = train[-n_valid:]
    elif rule == "E":
        def rank(day):
            return sha256(f"{seed}:{station}:{family}:{year}:{day}".encode("ascii")).digest(), day

        valid = sorted(sorted(train, key=rank)[:n_valid])
    else:
        raise ValueError("Unknown ANN rule")
    return sorted(set(train).difference(valid)), valid


def outer_plan(pairs, *, station, family, protocol):
    """Return exact date keys by fold. Values are used only for AHC support."""
    start, end = protocol["development"]
    if any(not start <= day <= end for day in pairs):
        raise ValueError("Outer planner accepts only DEVELOPMENT dates")
    by_year = defaultdict(list)
    for day in sorted(pairs):
        by_year[day[:4]].append(day)
    for year, held_out in sorted(by_year.items()):
        train = sorted(set(pairs).difference(held_out))
        if len(train) < 2:
            continue
        inner = {
            rule: inner_split(train, rule, station=station, family=family, year=year,
                              seed=protocol["ann"]["inner_split_seed_E"])
            for rule in protocol["ann"]["run_order"]
        }
        score = {}
        for calendar in protocol["calendars"]:
            # An unseen category or undefined AHC cannot enter the common scoring cohort.
            supported = {category(day, calendar) for day in train if pairs[day] != 0}
            for fit, _ in inner.values():
                supported.intersection_update(category(day, calendar) for day in fit)
            score[calendar] = [day for day in held_out if category(day, calendar) in supported]
        yield {"year": year, "train": train, "held_out": held_out, "inner": inner, "score": score}


def support(pairs, *, station, family, protocol):
    """Compact, hash-bound split support. No model fitting or scoring."""
    folds = []
    for plan in outer_plan(pairs, station=station, family=family, protocol=protocol):
        folds.append({
            "year": int(plan["year"]), "train": len(plan["train"]), "held_out": len(plan["held_out"]),
            "score": {k: len(v) for k, v in plan["score"].items()},
            "inner": {k: {"fit": len(fit), "validation": len(valid)}
                      for k, (fit, valid) in plan["inner"].items()},
            "keys_sha256": digest(json_bytes(plan)),
        })
    calendars = {}
    for calendar, labels in protocol["calendars"].items():
        n = Counter(category(day, calendar) for day in pairs)
        years = defaultdict(set)
        identifiable = set()
        for day, value in pairs.items():
            group = category(day, calendar)
            years[group].add(int(day[:4]))
            if value != 0:
                identifiable.add(group)
        supported_years = [fold["year"] for fold in folds if fold["score"][calendar] > 0]
        calendars[calendar] = {
            "outer_years": supported_years,
            "selection_supported": len(supported_years) >= protocol["outer_validation"]["minimum_supported_folds"],
            "groups": {label: {"paired": n[label], "years": sorted(years[label]),
                               "ahc_identifiable": label in identifiable} for label in labels},
        }
    return {"calendars": calendars, "folds": folds}


def choose_candidate(candidates, *, family_type, protocol, supported_years):
    """Pick using one comparable DEVELOPMENT RMSE per supported year.

    Each record has candidate_id, yearly_rmse and either algorithm (ML) or
    parameter_count and architecture (ANN). The runner may call this on a
    single algorithm or on all ML candidates under the same keys.
    """
    from math import isfinite

    if family_type not in {"ml", "ann"}:
        raise ValueError("Unknown V2 selection family")
    years = set(supported_years)
    if len(years) < protocol["outer_validation"]["minimum_supported_folds"] or any(
        not str(year).isdigit() or not int(protocol["development"][0][:4]) <= int(year)
        <= int(protocol["development"][1][:4]) for year in years
    ):
        raise ValueError("Selection requires common supported DEVELOPMENT years")
    scores = []
    ids = set()
    for candidate in candidates:
        by_year = candidate["yearly_rmse"]
        candidate_id = candidate["candidate_id"]
        if candidate_id in ids or set(by_year) != years or any(
            not isfinite(value) or value < 0 for value in by_year.values()
        ):
            raise ValueError("Duplicate candidate or missing/invalid common annual RMSE")
        ids.add(candidate_id)
        mean = fsum(by_year.values()) / len(years)
        sd = sqrt(fsum((value - mean) ** 2 for value in by_year.values()) / len(years))
        scores.append((mean, sd, candidate))
    if not scores:
        raise ValueError("No candidates to compare")
    minimum = min(item[0] for item in scores)
    if family_type == "ml":
        tolerance = minimum * protocol["ml"]["tie_relative_tolerance"]
        rank = protocol["ml"]["complexity_rank"]

        def tie(item):
            return item[1], rank.index(item[2]["algorithm"]), item[2]["candidate_id"]
    else:
        tolerance = protocol["ann"]["tie_absolute_tolerance"]

        def tie(item):
            return item[1], item[2]["parameter_count"], item[2]["architecture"]
    chosen = min((item for item in scores if item[0] - minimum <= tolerance), key=tie)
    return {"candidate_id": chosen[2]["candidate_id"], "mean_rmse": chosen[0], "sd_rmse": chosen[1]}
