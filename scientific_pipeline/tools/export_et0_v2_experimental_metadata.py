"""Export ET0 V2 DEVELOPMENT metadata from the verified frozen campaign."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

FREEZE = "97acad7ce7cc39c65d82f6efdc36290ed0325caaef2dedb69d9ffbefe1addfbd"
HISTORY = "0be2ce0f8e9534251743a3aea091ba1c530d4aa52305694cc43b8814a712b70d"
FAMILIES = {
    "HS_RA_V1": ["Tmin", "Tmax", "Ra"],
    "HG_RS_V1": ["Tmean_extremes", "Rs"],
    "VALIANTZAS_V1": ["Tmin", "Tmax", "RHmean", "Ra"],
}
CALENDARS = {"annual": [], "seasonal": ["DJF", "MAM", "JJA", "SON"],
             "monthly": [f"{n:02d}" for n in range(1, 13)]}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return json.loads(path.read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_table(directory, name, rows):
    if not rows:
        raise ValueError(f"Empty table: {name}")
    fields = sorted({field for row in rows for field in row})
    path = directory / name
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return {"sha256": sha(path), "rows": len(rows), "bytes": path.stat().st_size, "columns": fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    freeze = read(run / "freeze.json", FREEZE)
    capsule = read(args.capsule / "manifest.json", freeze["identity"]["capsule_sha256"])
    support = read(args.capsule / "support-inventory.json",
                   capsule["outputs"]["support-inventory.json"]["sha256"])
    development_path = args.capsule / "development.json.gz"
    if sha(development_path) != capsule["outputs"]["development.json.gz"]["sha256"]:
        raise ValueError("Unexpected DEVELOPMENT view")
    with gzip.open(development_path, "rt", encoding="utf-8") as stream:
        development = json.load(stream)["rows"]
    eligible_dates = {}
    for row in development:
        for family in FAMILIES:
            if row["target"] is not None and row["inputs"][family] is not None:
                eligible_dates.setdefault((row["station_code"], family), []).append(row["date"])
    if sha(args.history) != HISTORY:
        raise ValueError("Unexpected full-history file")
    tables = {k: [] for k in ("experimental-cells", "ahc-frozen-coefficients", "ahc-fold-coefficients",
                               "ahc-history-in-sample", "ml-development-candidates", "ml-final-models",
                               "ml-fold-metrics", "ann-development-candidates", "ann-selected-architectures",
                               "ann-fold-seed-metrics")}
    for stage, expected_index in freeze["stages"].items():
        index = read(run / stage / "index.json", expected_index)
        for cell, expected_cell in sorted(index["cells"].items()):
            folder = run / stage / cell
            info = read(folder / "frozen.json", expected_cell)
            station, family, calendar = info["key"]
            if "__".join(info["key"]) != cell or info["stage"] != stage:
                raise ValueError(f"Invalid cell {stage}/{cell}")
            key = {"station_code": station, "family": family, "calendar": calendar,
                   "cell_id": cell, "stage": stage, "frozen_sha256": expected_cell}
            if stage == "AHC":
                ahc = info["ahc"]
                for group, data in ahc["groups"].items():
                    tables["ahc-frozen-coefficients"].append({**key, "group": group, **data,
                        "fit_start": "2004-01-01", "fit_end": "2019-12-31",
                        "fit_period_semantics": "development_window", "support": "IDENTIFIABLE" if data["coefficient"] is not None else "UNSUPPORTED"})
                for fold in ahc["folds"]:
                    for group, data in fold["coefficients"].items():
                        tables["ahc-fold-coefficients"].append({**key, "held_out_year": fold["year"],
                            "plan_sha256": fold["plan_sha256"], "group": group, **data})
                continue
            selected = {v["candidate_id"] for v in info["selected"].values()}
            principal = info["principal"]["candidate_id"]
            for label, model in info["models"].items():
                model_path = folder / label
                if model_path.stat().st_size != model["bytes"] or sha(model_path) != model["sha256"]:
                    raise ValueError(f"Model identity mismatch: {model_path}")
                item = info["selected"][label.removesuffix(".joblib")]
                row = {**key, "model_label": label, "model_sha256": model["sha256"], "model_bytes": model["bytes"],
                       "candidate_id": item["candidate_id"], "is_principal": item["candidate_id"] == principal,
                       "selected_configuration": {k: item.get(k) for k in ("algorithm", "architecture", "parameters", "parameter_count")},
                       "epochs": model.get("epochs"), "seeds": model.get("seeds")}
                tables["ml-final-models" if stage == "ML" else "ann-selected-architectures"].append(row)
            for name, expected in sorted(info["checkpoints"].items()):
                checkpoint = read(folder / "candidates" / name, expected)
                summary = checkpoint["summary"]
                summary_bytes = (json.dumps(summary, ensure_ascii=False, sort_keys=True,
                                            indent=2, allow_nan=False) + "\n").encode("utf-8")
                if hashlib.sha256(summary_bytes).hexdigest() != checkpoint["summary_sha256"]:
                    raise ValueError(f"Candidate summary mismatch: {name}")
                if name != summary["candidate_id"] + ".json" or [f["plan_sha256"] for f in summary["folds"]] != checkpoint["plans"]:
                    raise ValueError(f"Candidate identity mismatch: {name}")
                spec = {**key, "checkpoint_sha256": expected, "candidate_id": summary["candidate_id"],
                        "configuration": checkpoint["config"], "inputs": FAMILIES[family] + CALENDARS[calendar],
                        "fold_count": len(summary["folds"]), "selected_by_algorithm": summary["candidate_id"] in selected,
                        "is_principal": summary["candidate_id"] == principal,
                        **{k: summary.get(k) for k in ("n", "mean_annual_rmse", "sd_annual_rmse", "pooled_rmse", "warnings")}}
                tables["ml-development-candidates" if stage == "ML" else "ann-development-candidates"].append(spec)
                for fold in summary["folds"]:
                    basic = {**key, "candidate_id": summary["candidate_id"], "checkpoint_sha256": expected,
                             **{k: fold.get(k) for k in ("year", "n", "mae", "rmse", "bias", "r2", "plan_sha256", "keys_sha256")}}
                    if stage == "ML":
                        tables["ml-fold-metrics"].append(basic)
                    else:
                        for seed in fold["seeds"]:
                            tables["ann-fold-seed-metrics"].append({**basic, "ensemble_rmse": basic["rmse"],
                                "ensemble_mae": basic["mae"], "ensemble_bias": basic["bias"], "ensemble_r2": basic["r2"],
                                **{f"seed_{k}": v for k, v in seed.items()}})
    for station, sinfo in sorted(support["stations"].items()):
        for family in FAMILIES:
            for calendar in CALENDARS:
                cell = "__".join((station, family, calendar))
                state = sinfo["support"][family]["calendars"][calendar]
                dates = eligible_dates.get((station, family), [])
                tables["experimental-cells"].append({"station_code": station, "family": family, "calendar": calendar,
                    "cell_id": cell, "ahc_available": True, "ml_available": state["selection_supported"],
                    "ann_a_available": state["selection_supported"], "ann_e_available": state["selection_supported"],
                    "category_support": state["groups"], "outer_years": state["outer_years"],
                    "reason": "" if state["selection_supported"] else "INSUFFICIENT_SUPPORTED_OUTER_FOLDS",
                    "development_period": support["development"], "effective_first_date": min(dates) if dates else None,
                    "effective_last_date": max(dates) if dates else None, "eligible_paired_days": len(dates),
                    "station_metadata": sinfo["metadata"]})
    with args.history.open(encoding="utf-8", newline="") as f:
        tables["ahc-history-in-sample"] = list(csv.DictReader(f))
    expected = {"experimental-cells": 108, "ahc-frozen-coefficients": 600, "ahc-fold-coefficients": 8382,
                "ahc-history-in-sample": 36, "ml-development-candidates": 3885, "ml-final-models": 735,
                "ml-fold-metrics": 55056, "ann-development-candidates": 1050,
                "ann-selected-architectures": 210, "ann-fold-seed-metrics": 44640}
    for name, count in expected.items():
        if len(tables[name]) != count:
            raise ValueError(f"{name}: expected {count}, got {len(tables[name])}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=False)
    outputs = {name + ".csv": write_table(args.output, name + ".csv", rows) for name, rows in tables.items()}
    manifest = {"schema": "et0-v2-experimental-metadata-v1", "freeze_sha256": FREEZE,
                "dataset_sha256": freeze["identity"]["dataset_v1_sha256"],
                "capsule_sha256": freeze["identity"]["capsule_sha256"], "history_sha256": HISTORY,
                "pipeline_sha": freeze["identity"]["pipeline_sha"], "stage_index_sha256": freeze["stages"],
                "publication_status": "NOT_PUBLISHED", "outputs": outputs}
    (args.output / "catalog.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"catalog_sha256": sha(args.output / "catalog.json"), "counts": expected}))


if __name__ == "__main__":
    main()
