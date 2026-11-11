"""Prepare an auditable V2 DEVELOPMENT view. This module never fits a model."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .datasets import validate_dataset
from .et0_v2_ib09 import reconcile, require, verified_bytes, verify_admission
from .et0_v2_support import support
from .pipeline import METHODS, calculate
from .quality import eligibility, finite
from .serialization import digest, json_bytes


METHOD_KEYS = {name: (alias, field) for alias, (name, field) in METHODS.items()}
PM = "PM_FAO56_V1"
PROTOCOL = "scientific_pipeline/experiments/et0-v2-preparation.json"


def load_protocol(repo):
    content = (repo / PROTOCOL).read_bytes()
    protocol = json.loads(content)
    ml = protocol["ml"]
    prior = json.loads(verified_bytes(repo / ml["protocol_path"], ml["protocol_sha256"]))
    require(sum(map(len, prior["candidate_grids"].values())) == ml["candidate_count_per_station_family_calendar"]
            and prior["selection"]["complexity_rank"] == ml["complexity_rank"],
            "V2 ML candidate registry mismatch")
    return protocol, digest(content)


def _inputs(row, name):
    values = row["values"]
    tmin, tmax = values["TempMin"], values["TempMax"]
    if name == "HG_RS_V1":
        return [(tmin + tmax) / 2, values["Radiacion"]]
    ra = row["model_diagnostics"][name]["ra_mj_m2_day"]
    return [tmin, tmax, ra] if name == "HS_RA_V1" else [tmin, tmax, values["HumedadMedia"], ra]


def _development_row(row, admitted):
    pm = row[METHOD_KEYS[PM][1]]
    results, inputs = {}, {}
    for name, (alias, field) in METHOD_KEYS.items():
        item = row["eligibility"][alias]
        results[name] = {"estimate": row[field], "status": item["status"], "reasons": item["reasons"]}
        if name != PM:
            inputs[name] = _inputs(row, name) if item["status"] == "CALCULATED" else None
    return {
        "station_code": row["station_code"], "date": row["date"],
        "observation_status": row["observation_status"],
        "quality_v1_status": "FAIL" if admitted else row["quality_status"],
        "quality_flags": row["quality_flags"], "scientific_flags": row["scientific_flags"],
        "v2_admission": "IB09_DEVELOPMENT_CATALOG_EXCEPTION" if admitted else "QUALITY_V1",
        "source_data_sha": row["source_data_sha"], "provenance": row["provenance"],
        "target": pm if finite(pm) and pm >= 0 else None,
        "baselines": results, "inputs": inputs,
    }


def _count(tally, status, own, has_pm):
    tally["raw_observations"] += 1
    tally["outside_period"] += status == "OUTSIDE_OFFICIAL_PERIOD"
    tally["observations_in_period"] += status in {"OBSERVED", "RECOVERED"}
    tally["recovered"] += status == "RECOVERED"
    for name, available in own.items():
        tally[f"own:{name}"] += available
        tally[f"paired_pm:{name}"] += available and has_pm
    tally["common_four"] += has_pm and all(own.values())


def _summary(counter):
    summary = {key: counter[key] for key in
               ("raw_observations", "observations_in_period", "outside_period", "recovered",
                "confirmed_no_data", "common_four")}
    summary["methods"] = {name: {"own": counter[f"own:{name}"], "paired_pm": counter[f"paired_pm:{name}"]}
                          for name in METHOD_KEYS}
    return summary


def prepare(repo, dataset_path, handoff_path=None, *, official_evidence_contradiction=None):
    """Return DEVELOPMENT rows plus a support-only inventory for both periods."""
    protocol, protocol_sha = load_protocol(repo)
    dataset_sha = protocol["dataset_v1_sha256"]
    dataset = json.loads(verified_bytes(dataset_path, dataset_sha))
    validate_dataset(dataset)
    admitted, evidence = reconcile(repo, dataset)
    if handoff_path is None:
        admitted.clear()
        evidence["admission"] = {"status": "WITHHELD_PENDING_EFFECTIVE_HANDOFF"}
    else:
        evidence["admission"] = verify_admission(repo, handoff_path, protocol["ib09_handoff"],
                                               official_evidence_contradiction=official_evidence_contradiction)
    dev_start, dev_end = protocol["development"]
    test_start, test_end = protocol["retrospective_test"]
    require(all(dev_start <= day <= dev_end for day in admitted), "IB09 exception outside DEVELOPMENT")
    require(not any(row["station_code"] == "IB09" and test_start <= row["date"] <= test_end
                    for row in dataset["rows"]), "STOP IB09: unexpected retrospective TEST observation")

    def is_admitted(row):
        return row["station_code"] == "IB09" and row["date"] in admitted

    # A temporary calculation-only view satisfies the Quality gate on the
    # verified keys. Method-specific checks run unchanged. V1 remains FAIL.
    development = [({**row, "quality_status": "WARN"} if is_admitted(row) else row)
                   for row in dataset["rows"] if dev_start <= row["date"] <= dev_end]
    calculated = calculate({**dataset, "rows": development, "missing_days": []}, tuple(METHODS))
    rows = [_development_row(row, is_admitted(row)) for row in calculated["rows"]]
    totals = defaultdict(Counter)
    yearly = defaultdict(Counter)
    pairs = defaultdict(dict)
    for row in rows:
        code, day = row["station_code"], row["date"]
        own = {name: result["status"] == "CALCULATED" for name, result in row["baselines"].items()}
        for tally in (totals[code, "DEVELOPMENT"], yearly[code, "DEVELOPMENT", day[:4]]):
            _count(tally, row["observation_status"], own, row["target"] is not None)
        if row["target"] is not None:
            for name in protocol["families"]:
                if own[name]:
                    pairs[code, name][day] = row["baselines"][name]["estimate"]

    # Keep the TEST outcome sealed. Count input/QC readiness only, without
    # calculating PM, predictions or errors and without exporting daily TEST keys.
    for row in dataset["rows"]:
        code, day = row["station_code"], row["date"]
        if not test_start <= day <= test_end:
            continue
        station = dataset["stations"][code]
        own = {name: not eligibility(row, station, alias) for name, (alias, _) in METHOD_KEYS.items()}
        for tally in (totals[code, "RETROSPECTIVE_TEST_QC_ONLY"],
                      yearly[code, "RETROSPECTIVE_TEST_QC_ONLY", day[:4]]):
            _count(tally, row["observation_status"], own, own[PM])
    for item in dataset["missing_days"]:
        if item["observation_status"] != "CONFIRMED_NO_DATA":
            continue
        code, day = item["station_code"], item["date"]
        if dev_start <= day <= dev_end:
            partition = "DEVELOPMENT"
        elif test_start <= day <= test_end:
            partition = "RETROSPECTIVE_TEST_QC_ONLY"
        else:
            continue
        totals[code, partition]["confirmed_no_data"] += 1
        yearly[code, partition, day[:4]]["confirmed_no_data"] += 1

    historical = set(protocol["historical_stations"])
    stations = {}
    for code, metadata in sorted(dataset["stations"].items()):
        stations[code] = {
            "series": "HISTORICAL_SEPARATE" if code in historical else "PRIMARY",
            "metadata": metadata,
            "development": _summary(totals[code, "DEVELOPMENT"]),
            "retrospective_test_qc_only": _summary(totals[code, "RETROSPECTIVE_TEST_QC_ONLY"]),
            "by_year": {year: _summary(tally) for (station, _, year), tally in sorted(yearly.items())
                        if station == code},
            "support": {family: support(pairs[code, family], station=code, family=family, protocol=protocol)
                        for family in protocol["families"]},
        }
    supported = [item for station in stations.values() for family in station["support"].values()
                 for item in family["calendars"].values() if item["selection_supported"]]
    folds = sum(len(item["outer_years"]) for item in supported)
    inventory = {
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha, "dataset_v1_sha256": dataset_sha,
        "status": "PREPARATION_ONLY" if handoff_path is not None else "PARTIAL_IB09_WITHHELD",
        "development": protocol["development"], "retrospective_test": protocol["retrospective_test"],
        "test_inventory": "input/QC counts only; no calculated reference, estimates, outcomes or scores",
        "ib09": evidence, "stations": stations,
        "fit_budget_if_launched": {
            "station_family_calendar_cells": len(supported), "supported_outer_folds": folds,
            "ml_candidate_fits": folds * protocol["ml"]["candidate_count_per_station_family_calendar"],
            "ml_refits": len(supported) * len(protocol["ml"]["complexity_rank"]),
            "ann_fits_per_rule": 2 * folds * len(protocol["ann"]["architectures"]) * len(protocol["ann"]["seeds"])
                                 + len(supported) * len(protocol["ann"]["seeds"]),
        },
    }
    gaps = [item for item in dataset["missing_days"] if dev_start <= item["date"] <= dev_end]
    view = {"schema": "et0-v2-development-preparation-v1", "dataset_id": "siar-balears-et0-v2-development",
            "dataset_v1_id": dataset["dataset_id"], "dataset_v1_sha256": dataset_sha,
            "protocol_sha256": protocol_sha, "station_metadata": dataset["stations"],
            "units": dataset["units"], "ib09": evidence, "rows": rows, "ledger": gaps}
    return view, inventory


def publish(output, view, inventory):
    require(not output.exists(), f"Output already exists: {output}")
    payloads = {
        "development.json.gz": gzip.compress(json_bytes(view), mtime=0),
        "support-inventory.json": json_bytes(inventory),
    }
    manifest = {key: inventory[key] for key in ("status", "protocol_sha256", "dataset_v1_sha256", "ib09")}
    manifest.update({
        "outputs": {name: {"sha256": digest(content), "bytes": len(content)} for name, content in payloads.items()},
        "fitted_models": 0, "test_predictions": 0, "test_metrics": 0,
    })
    payloads["manifest.json"] = json_bytes(manifest)
    with TemporaryDirectory(dir=output.parent) as staging:
        staged = Path(staging) / "run"
        staged.mkdir()
        for name, content in payloads.items():
            (staged / name).write_bytes(content)
        staged.rename(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--ib09-handoff", type=Path, help="Withhold IB09 when absent")
    parser.add_argument("--official-evidence-reviewed-no-contradiction", action="store_true",
                        help="Required with the effective IB09 handoff after reviewing available official evidence")
    parser.add_argument("--output", required=True, type=Path, help="New directory with an existing parent")
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), f"Output already exists: {args.output}")
        view, inventory = prepare(args.repo, args.dataset, args.ib09_handoff,
                                  official_evidence_contradiction=False
                                  if args.official_evidence_reviewed_no_contradiction else None)
        publish(args.output, view, inventory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"V2 preparation stopped: {error}\n")
    print(f"{inventory['status']}: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
