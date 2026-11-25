"""Read-only, hash-pinned adapter for the complementary DEVELOPMENT handoff."""
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from .convert import sha
from .et0_v2 import checked
from .et0_v2_preview import Calendar, Formula, FORMULAS

CATALOG_SHA = "3207fe042abb004b7cd69122339845adc8d55cd65b5cfd4d676d9e5512e6eea9"
COUNTS = {
    "experimental-cells": 108,
    "ahc-frozen-coefficients": 600,
    "ahc-fold-coefficients": 8382,
    "ahc-history-in-sample": 36,
    "ml-development-candidates": 3885,
    "ml-final-models": 735,
    "ml-fold-metrics": 55056,
    "ann-development-candidates": 1050,
    "ann-selected-architectures": 210,
    "ann-fold-seed-metrics": 44640,
}
KEYS = {
    "experimental-cells": ("cell_id",),
    "ahc-frozen-coefficients": ("cell_id", "group"),
    "ahc-fold-coefficients": ("cell_id", "held_out_year", "group"),
    "ahc-history-in-sample": ("station_code", "family"),
    "ml-development-candidates": ("cell_id", "candidate_id"),
    "ml-final-models": ("cell_id", "model_label"),
    "ml-fold-metrics": ("cell_id", "candidate_id", "year"),
    "ann-development-candidates": ("cell_id", "stage", "candidate_id"),
    "ann-selected-architectures": ("cell_id", "stage"),
    "ann-fold-seed-metrics": ("cell_id", "stage", "candidate_id", "year", "seed_seed"),
}
STAGES = {"AHC", "ML", "ANN_A", "ANN_E"}
CALENDARS = {"annual", "seasonal", "monthly"}
JSON_FIELDS = {"configuration", "selected_configuration", "inputs", "warnings", "epochs", "seeds",
               "category_support", "outer_years", "development_period", "station_metadata"}
BOOLEAN_FIELDS = {"ahc_available", "ml_available", "ann_a_available", "ann_e_available",
                  "selected_by_algorithm", "is_principal"}
INTEGER_FIELDS = {"n", "development_pairs", "retrospective_pairs", "eligible_paired_days", "fold_count",
                  "held_out_year", "year", "seed_seed", "seed_n", "seed_best_epoch", "seed_epochs_run",
                  "model_bytes"}
FLOAT_FIELDS = {"coefficient", "sum_x2", "mae", "rmse", "bias", "r2", "mean_annual_rmse",
                "sd_annual_rmse", "pooled_rmse", "ensemble_rmse", "ensemble_mae", "ensemble_bias",
                "ensemble_r2", "seed_rmse", "seed_mae", "seed_bias", "seed_r2", "seed_inner_rmse"}
Stage = Literal["AHC", "ML", "ANN_A", "ANN_E"]


def records(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def decode(row):
    decoded = {}
    for key, value in row.items():
        if value == "":
            decoded[key] = None
        elif key in JSON_FIELDS:
            decoded[key] = json.loads(value)
        elif key in BOOLEAN_FIELDS:
            decoded[key] = value == "True"
        elif key in INTEGER_FIELDS:
            decoded[key] = int(value)
        elif key in FLOAT_FIELDS:
            decoded[key] = float(value)
        else:
            decoded[key] = value
    return decoded


def router(directory, retrospective, stations):
    root = Path(directory).resolve(strict=True)
    catalog_path = root / "catalog.json"
    digest = sha(catalog_path)
    if digest != CATALOG_SHA:
        raise ValueError("ET0 DEVELOPMENT catalog SHA mismatch")
    manifest = json.loads(catalog_path.read_bytes())
    if (manifest["schema"] != "et0-v2-experimental-metadata-v1"
            or manifest["publication_status"] != "NOT_PUBLISHED"
            or any(manifest[key] != retrospective[key] for key in ("dataset_sha256", "freeze_sha256"))
            or manifest["outputs"].keys() != {name + ".csv" for name in COUNTS}):
        raise ValueError("ET0 DEVELOPMENT catalog is incompatible with the signed retrospective")
    paths = {}
    cached = {}
    station_codes = set(stations)
    expected_cells = {(s, f, c) for s in station_codes for f in FORMULAS for c in CALENDARS}
    available = {}
    for name, count in COUNTS.items():
        filename = name + ".csv"
        entry = manifest["outputs"][filename]
        path = paths[name] = checked(root, filename, entry)
        keys = set()
        keep = name in {"experimental-cells", "ml-final-models", "ann-selected-architectures"}
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != entry["columns"]:
                raise ValueError(f"Unexpected columns: {filename}")
            for row in reader:
                key = tuple(row[field] for field in KEYS[name])
                if key in keys or not all(key):
                    raise ValueError(f"Invalid or duplicate key: {filename}")
                keys.add(key)
                station, family = row["station_code"], row["family"]
                if station not in station_codes or family not in FORMULAS:
                    raise ValueError(f"Unknown station or family: {filename}")
                if name == "ahc-history-in-sample":
                    if row["scope"] != "FULL_HISTORY_IN_SAMPLE":
                        raise ValueError("Unexpected AHC historical scope")
                else:
                    cell = (station, family, row["calendar"])
                    if cell not in expected_cells or row["cell_id"] != "__".join(cell):
                        raise ValueError(f"Invalid cell: {filename}")
                    if name == "experimental-cells":
                        available[cell] = {stage for stage in STAGES if row[stage.lower() + "_available"] == "True"}
                    elif row["stage"] not in available[cell]:
                        raise ValueError(f"Invalid stage support: {filename}")
                if keep:
                    cached.setdefault(name, []).append(decode(row))
        if len(keys) != count or count != entry["rows"]:
            raise ValueError(f"Unexpected row count: {filename}")
    if available.keys() != expected_cells:
        raise ValueError("Incomplete experimental cell inventory")

    source = {k: manifest[k] for k in ("schema", "freeze_sha256", "dataset_sha256", "publication_status")}
    source.update(catalog_sha256=digest, integrity="VERIFIED", tables=manifest["outputs"])
    api = APIRouter(prefix="/development", tags=["ET₀ V2 DEVELOPMENT"])

    @api.get("/catalog")
    def catalog():
        return dict(source=source, cells=cached["experimental-cells"])

    @api.get("/frequency")
    def frequency(stage: Literal["ML", "ANN_A", "ANN_E"], dimension: Literal["global", "station_code", "family", "calendar"] = "global",
                  station: str | None = None, family: Formula | None = None, calendar: Calendar | None = None):
        if station and station not in station_codes:
            raise HTTPException(422, "Estación desconocida")
        name = "ml-final-models" if stage == "ML" else "ann-selected-architectures"
        bins = Counter()
        for row in cached[name]:
            if (row["stage"] != stage or not row["is_principal"]
                    or station and station != row["station_code"]
                    or family and family != row["family"]
                    or calendar and calendar != row["calendar"]):
                continue
            config = row["selected_configuration"]
            choice = config["algorithm"] if stage == "ML" else "–".join(map(str, config["architecture"]))
            bins[row.get(dimension, "Todas las celdas"), choice] += 1
        return dict(scope="DEVELOPMENT", rows=[dict(group=group, choice=choice, count=count)
                    for (group, choice), count in sorted(bins.items())], total=sum(bins.values()))

    @api.get("/table/{table}")
    def table_rows(table: str, station: str, family: Formula, calendar: Calendar | None = None,
                   stage: Stage | None = None, candidate: str | None = Query(None, max_length=100),
                   year: int | None = Query(None, ge=2004, le=2019),
                   offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=150)):
        if table not in paths:
            raise HTTPException(404, "Tabla experimental desconocida")
        if station not in station_codes:
            raise HTTPException(422, "Estación desconocida")
        if table in {"ml-fold-metrics", "ann-fold-seed-metrics"} and not (calendar and candidate):
            raise HTTPException(422, "Seleccione una configuración y candidato para ver los folds")
        is_history = table == "ahc-history-in-sample"
        rows = []
        total = 0
        for row in (cached[table] if table in cached else records(paths[table])):
            if (row["station_code"] != station or row["family"] != family
                    or calendar and not is_history and row.get("calendar") != calendar
                    or stage and row.get("stage") != stage
                    or candidate and row.get("candidate_id") != candidate
                    or year and str(row.get("year", row.get("held_out_year"))) != str(year)):
                continue
            if offset <= total < offset + limit:
                rows.append(row if table in cached else decode(row))
            total += 1
        return dict(scope="FULL_HISTORY_IN_SAMPLE" if is_history else "DEVELOPMENT",
                    table=table + ".csv", rows=rows, total=total, limit=limit, offset=offset)

    return api
