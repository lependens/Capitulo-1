"""Score V2 against calculated PM after verifying the complete frozen campaign."""
import argparse
from collections import defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import gzip
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .datasets import REPORT_HEADERS, _csv, validate_dataset
from .et0_v2 import _development_row, load_protocol
from .et0_v2_campaign import cell_path, freeze, read_json, sha_file, write_json
from .et0_v2_ib09 import POST_DIR, POST_SHA, require, verified_bytes
from .et0_v2_models import ahc_coefficients
from .et0_v2_support import category, design_vector
from .metrics import metrics
from .pipeline import METHODS, calculate
from .serialization import digest, json_bytes


CONTRAST = ("FORMULA", "AHC", "ML", "ANN_A", "ANN_E")


def load_retrospective(dataset_path, protocol):
    data = json.loads(verified_bytes(dataset_path, protocol["dataset_v1_sha256"]))
    validate_dataset(data)
    start, end = protocol["retrospective_test"]
    source = [row for row in data["rows"] if start <= row["date"] <= end]
    require(not set(row["station_code"] for row in source).intersection(protocol["historical_stations"]),
            "Unexpected retrospective observation for a historical station")
    return [_development_row(row, False) for row in
            calculate({**data, "rows": source, "missing_days": []}, tuple(METHODS))["rows"]]


def _station_periods(repo, protocol):
    name = "siar_station_audit.csv"
    post = json.loads(verified_bytes(repo / POST_DIR / "quality-post-package.json", POST_SHA))
    audit = _csv(verified_bytes(repo / POST_DIR / name, post["reports"][name]["sha256"]),
                 REPORT_HEADERS[name].split(","))
    start, end = protocol["retrospective_test"]
    return {item["station_code"]: (max(start, item["expected_start_date"]), min(end, item["expected_end_date"]))
            for item in audit}


def _frozen_predictions(run, key, rows, protocol):
    station, family, calendar = key
    possible = [row for row in rows if row["inputs"][family] is not None]
    by_day = {row["date"]: {"FORMULA": row["baselines"][family]["estimate"]} for row in possible}
    ahc = read_json(cell_path(run, "AHC", key) / "frozen.json")["ahc"]["groups"]
    for day, predictions in by_day.items():
        scale = ahc.get(category(day, calendar), {}).get("coefficient")
        if scale is not None:
            predictions["AHC"] = predictions["FORMULA"] * scale

    for stage in ("ML", "ANN_A", "ANN_E"):
        path = cell_path(run, stage, key)
        if not (path / "frozen.json").is_file():
            continue
        meta = read_json(path / "frozen.json")
        # Groups with no paired training rows cannot be scored or used for prediction.
        supported = [row for row in possible if ahc.get(category(row["date"], calendar), {}).get("n", 0) > 0]
        if not supported:
            continue
        x = np.asarray([design_vector(row["inputs"][family], row["date"], calendar, protocol)
                        for row in supported], dtype=float)
        for label, config in meta["selected"].items():
            model = joblib.load(path / f"{label}.joblib")
            if stage == "ML":
                pred = model.predict(x)
            else:
                transformed = model["scale"].transform(x)
                pred = np.mean([member.predict(transformed) for member in model["models"]], axis=0)
            require(pred.shape == (len(x),) and np.isfinite(pred).all(), "Invalid frozen retrospective prediction")
            for row, value in zip(supported, pred, strict=True):
                predictions = by_day[row["date"]]
                if stage != "ML" or config["candidate_id"] == meta["principal"]["candidate_id"]:
                    predictions[stage] = float(value)
                if stage == "ML":
                    predictions[f"ML_{label}"] = float(value)
            del model
    return by_day


def _groups(station, day):
    year, month = day[:4], day[5:7]
    return (("global", "all"), ("station", station), ("year", year),
            ("station_year", f"{station}:{year}"), ("month", month), ("season", category(day, "seasonal")))


def _calendar_counts(periods):
    totals = defaultdict(int)
    for station, (start, end) in periods.items():
        totals["station", station] = 0
        if start <= end:
            current, last = date.fromisoformat(start), date.fromisoformat(end)
            while current <= last:
                for group in _groups(station, current.isoformat()):
                    totals[group] += 1
                current += timedelta(days=1)
    return totals


def retrospective_metrics(rows, methods, calendar_days):
    grouped = defaultdict(list)
    for row in rows:
        for group in _groups(row["station_code"], row["date"]):
            grouped[group].append(row)
    for (kind, group), denominator in sorted(calendar_days.items()):
        records = grouped[kind, group]
        target_rows = [r for r in records if r["target"] is not None]
        common = [r for r in target_rows if all(r.get(method) is not None for method in CONTRAST)]
        cohort_hash = digest(json_bytes([[r["station_code"], r["date"]] for r in common]))
        for method in methods:
            emitted = sum(r.get(method) is not None for r in records)
            paired = sum(r.get(method) is not None for r in target_rows)
            scored = [r for r in common if r.get(method) is not None]
            yield {
                "group": kind, "key": group, "method": method, "calendar_days": denominator,
                "pm_days": len(target_rows), "predicted_days": emitted, "pm_pairs_own": paired,
                "coverage_calendar": emitted / denominator if denominator else None,
                "coverage_pm_own": paired / len(target_rows) if target_rows else None,
                "common_keys_sha256": cohort_hash if len(scored) == len(common) else
                digest(json_bytes([[r["station_code"], r["date"]] for r in scored])),
                **metrics([r["target"] for r in scored], [r[method] for r in scored]),
            }


def _history(capsule, test_rows, protocol):
    with gzip.open(capsule / "development.json.gz", "rt", encoding="utf-8") as stream:
        dev_rows = json.load(stream)["rows"]
    for partition, source in (("DEVELOPMENT", dev_rows), ("RETROSPECTIVE", test_rows)):
        for row in source:
            row["history_partition"] = partition
    all_rows = defaultdict(list)
    for row in [*dev_rows, *test_rows]:
        all_rows[row["station_code"]].append(row)
    for station, rows in sorted(all_rows.items()):
        for family in protocol["families"]:
            eligible = [r for r in rows if r["target"] is not None and r["inputs"][family] is not None]
            groups = ["annual"] * len(eligible)
            x = [row["baselines"][family]["estimate"] for row in eligible]
            y = [row["target"] for row in eligible]
            fit = ahc_coefficients(groups, x, y).get("annual", {"n": 0, "coefficient": None, "sum_x2": 0})
            a = fit["coefficient"]
            yield {"station_code": station, "family": family, "scope": "FULL_HISTORY_IN_SAMPLE",
                   "development_pairs": sum(r["history_partition"] == "DEVELOPMENT" for r in eligible),
                   "retrospective_pairs": sum(r["history_partition"] == "RETROSPECTIVE" for r in eligible),
                   "years": ",".join(sorted({row["date"][:4] for row in eligible})),
                   "coefficient": a, **metrics(y, [a * v if a is not None else None for v in x])}


def write_csv(path, records):
    records = iter(records)
    first = next(records)
    if path.suffix == ".gz":
        with path.open("wb") as compressed:
            with io.TextIOWrapper(gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0),
                                  encoding="utf-8", newline="") as stream:
                _write_csv(stream, first, records)
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        _write_csv(stream, first, records)


def _write_csv(stream, first, records):
    writer = csv.DictWriter(stream, fieldnames=list(first), lineterminator="\n")
    writer.writeheader()
    writer.writerow(first)
    writer.writerows(records)


def _report(metrics_rows, sha):
    title = [
        "# ET₀ V2 · evaluación retrospectiva 2020–2025", "",
        f"Congelación DEVELOPMENT SHA256 `{sha}`.", "",
        "Referencia diaria calculada con PM_FAO56_V1. RMSE, MAE y sesgo en mm/día. "
        "El error usa claves comunes dentro de cada familia y calendario. La cobertura propia usa los días de catálogo.",
        "", "| Familia | Calendario | Método | Pares comunes | RMSE | MAE | Sesgo | Cobertura propia |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        if row["group"] == "global" and row["method"] in CONTRAST and row["n"]:
            title.append(f"| {row['family']} | {row['calendar']} | {row['method']} | {row['n']} | "
                         f"{row['rmse']:.4f} | {row['mae']:.4f} | {row['bias']:+.4f} | "
                         f"{row['coverage_calendar']:.2%} |")
    title.extend(["", "IB07, IB09 e IB101 son series históricas separadas con cero observaciones retrospectivas.",
                  "El AHC de histórico completo y su error dentro de muestra figuran en una tabla separada.", "",
                  "[Predicciones](predictions.csv.gz) · [Métricas por grupo](retrospective-metrics.csv) · "
                  "[AHC de histórico completo](ahc-full-history-in-sample.csv) · [Manifiesto](manifest.json)", ""])
    return "\n".join(title)


def evaluate(repo, capsule, run, dataset_path, *, expected_freeze_sha, output):
    require(not output.exists(), "Choose a new retrospective output directory")
    require((run / "freeze.json").is_file(), "Freeze all DEVELOPMENT stages before TEST access")
    require(freeze(repo, capsule, run) == expected_freeze_sha, "Incorrect freeze identity")
    require(dataset_path.is_file() and output.parent.is_dir(), "Check dataset and output parent paths")
    protocol, _ = load_protocol(repo)
    opened = datetime.now(timezone.utc).isoformat()
    access = run / "retrospective-access.json"
    entry = {"event": "RETROSPECTIVE_ACCESS_INTENT", "utc": opened, "freeze_sha256": expected_freeze_sha,
             "dataset_sha256": protocol["dataset_v1_sha256"]}
    with access.open("xb") as stream:
        stream.write(json_bytes(entry))
    test_rows = load_retrospective(dataset_path, protocol)
    opened_event = {"event": "RETROSPECTIVE_OPENED", "freeze_sha256": expected_freeze_sha,
                    "utc": datetime.now(timezone.utc).isoformat(), "rows": len(test_rows)}
    write_json(run / "retrospective-opened.json", opened_event)
    station_periods = _station_periods(repo, protocol)
    calendar_days = _calendar_counts(station_periods)
    stations = defaultdict(list)
    for row in test_rows:
        stations[row["station_code"]].append(row)
    methods = (*CONTRAST, *(f"ML_{name}" for name in protocol["ml"]["complexity_rank"]))
    predictions, all_metrics = [], []
    with threadpool_limits(limits=1):
        for family in protocol["families"]:
            for calendar in protocol["calendars"]:
                evaluated = []
                for station in sorted(stations):
                    rows = stations[station]
                    predicted = _frozen_predictions(run, (station, family, calendar), rows, protocol)
                    for row in rows:
                        day = row["date"]
                        values = predicted.get(day, {})
                        evaluated.append({"family": family, "calendar": calendar, "station_code": station,
                                          "date": day, "target": row["target"],
                                          "source_quality": row["quality_v1_status"],
                                          "observation_status": row["observation_status"],
                                          "formula_status": row["baselines"][family]["status"],
                                          "formula_reasons": "|".join(row["baselines"][family]["reasons"]),
                                          "pm_reasons": "|".join(row["baselines"]["PM_FAO56_V1"]["reasons"]),
                                          **{method: values.get(method) for method in methods}})
                predictions.extend(evaluated)
                all_metrics.extend({"family": family, "calendar": calendar, **record}
                                   for record in retrospective_metrics(evaluated, methods, calendar_days))
    with TemporaryDirectory(dir=output.parent) as staging:
        staged = Path(staging) / "results"
        staged.mkdir()
        write_csv(staged / "predictions.csv.gz", predictions)
        write_csv(staged / "retrospective-metrics.csv", all_metrics)
        write_csv(staged / "ahc-full-history-in-sample.csv", _history(capsule, test_rows, protocol))
        (staged / "report.md").write_text(_report(all_metrics, expected_freeze_sha), encoding="utf-8", newline="\n")
        manifest = {"status": "PASS", "period": protocol["retrospective_test"],
                    "target": "calculated PM_FAO56_V1", "identity": read_json(run / "identity.json"),
                    "test_access": [entry, opened_event], "retrospective_rows": len(test_rows),
                    "historical_station_observations": dict.fromkeys(protocol["historical_stations"], 0),
                    "outputs": {p.name: {"sha256": sha_file(p), "bytes": p.stat().st_size}
                                for p in sorted(staged.iterdir())}}
        write_json(staged / "manifest.json", manifest)
        staged.rename(output)
    return sha_file(output / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "capsule", "run", "dataset", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    print(evaluate(args.repo, args.capsule, args.run, args.dataset,
                   expected_freeze_sha=args.freeze_sha256, output=args.output))


if __name__ == "__main__":
    main()
