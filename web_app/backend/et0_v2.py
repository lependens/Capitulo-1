"""Read-only adapter for the catalog and the retrospective tables delivered by 05."""
import csv
from datetime import date
import json
from pathlib import Path
import re
from typing import Annotated, Literal

import duckdb
from fastapi import APIRouter, HTTPException, Query

from .convert import sha
from .et0_v2_preview import Calendar, FORMULAS, Formula, Group, METHODS

STATION_CATALOG_SHA = "98abd16dcd6ea42ad46a30c331df47150f07dffdca140c14e4445890ce68a357"
METHOD_LABELS = {
    **METHODS, "ML_LINEAR": "Regresión lineal", "ML_RIDGE": "Ridge",
    "ML_ELASTIC_NET": "Elastic Net", "ML_TREE": "Árbol de decisión",
    "ML_RANDOM_FOREST": "Random Forest", "ML_EXTRA_TREES": "Extra Trees",
    "ML_HIST_GRADIENT_BOOSTING": "Hist Gradient Boosting",
}
METRICS = "retrospective-metrics.csv"
HISTORY = "ahc-full-history-in-sample.csv"
INTEGERS = {"calendar_days", "pm_days", "predicted_days", "pm_pairs_own",
            "n", "development_pairs", "retrospective_pairs"}
NUMBERS = {"mae", "rmse", "bias", "r2", "coverage_calendar", "coverage_pm_own", "coefficient"}
PARTITION = re.compile(r"parquet/station_id=(IB\d+)/year=(\d{4})/predictions\.parquet")
PREDICTION_COLUMNS = (
    "station_id date method_id method_family resolution prediction reference_value reference_id unit "
    "cohort_id eligibility_status quality_status observation_status formula_status formula_reasons "
    "pm_reasons source_run_id freeze_sha256 dataset_sha256 schema_version"
).split()


def checked(root, relative, entry):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
        raise ValueError(f"ET0 V2 integrity failure: {relative}")
    return path


def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [
            {key: (int(value) if value else None) if key in INTEGERS else
                  (float(value) if value else None) if key in NUMBERS else value
             for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def coordinate(value):
    match = re.fullmatch(r"(\d{2})(\d{2})(\d{2})(\d{3})([NSEW])", value)
    if not match:
        raise ValueError("Unknown SIAR coordinate encoding")
    degrees, minutes, seconds, thousandths, hemisphere = match.groups()
    if int(minutes) >= 60 or int(seconds) >= 60:
        raise ValueError("Invalid SIAR coordinate")
    result = int(degrees) + int(minutes) / 60 + (int(seconds) + int(thousandths) / 1000) / 3600
    return -result if hemisphere in ("S", "W") else result


def station_locations(path):
    if not path:
        return {}
    if sha(path) != STATION_CATALOG_SHA:
        raise ValueError("Unexpected SIAR station catalog SHA")
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return {r["Codigo"]: dict(latitude=coordinate(r["Latitud"]), longitude=coordinate(r["Longitud"]),
                                  altitude=int(r["Altitud"]), municipality=r["Termino"])
                for r in csv.DictReader(stream)}


def router(directory, catalog_sha, stations, location_path=None, mode="approved"):
    root = Path(directory).resolve(strict=True)
    catalog_path = root / "catalog.json"
    if not catalog_sha or sha(catalog_path) != catalog_sha:
        raise ValueError("ET0 V2 catalog SHA mismatch")
    contract = json.loads(catalog_path.read_text(encoding="utf-8"))
    statuses = {"review": {"DRAFT", "VERIFIED", "PUBLISHED"}, "approved": {"VERIFIED", "PUBLISHED"}}
    if mode not in statuses or contract["status"] not in statuses[mode]:
        raise ValueError("ET0 V2 publication status is incompatible with the reader mode")
    if contract["schema_version"] != "et0-v2-web-draft-1" or contract["reference_id"] != "PM_FAO56_V1":
        raise ValueError("Unsupported ET0 V2 schema or reference")
    methods = {m: METHOD_LABELS[m] for m in contract["methods"]}
    inventory = {entry["path"]: entry for entry in contract["files"]}
    if len(inventory) != len(contract["files"]):
        raise ValueError("Duplicated ET0 V2 catalog paths")
    partitions = {}
    for relative, entry in inventory.items():
        if match := PARTITION.fullmatch(relative):
            if match[1] not in contract["stations"]:
                raise ValueError("Unexpected prediction station")
            partitions[match[1], int(match[2])] = str(checked(root, relative, entry))
    first_day, last_day = map(date.fromisoformat, contract["period"])
    expected = {(station, year) for station in contract["stations"]
                for year in range(first_day.year, last_day.year + 1)}
    if partitions.keys() != expected:
        raise ValueError("Incomplete ET0 V2 station-year partition inventory")
    metric_rows = load_rows(checked(root, METRICS, inventory[METRICS]))
    history_rows = load_rows(checked(root, HISTORY, inventory[HISTORY]))
    with duckdb.connect(config={"threads": 1}) as db:
        columns = db.execute("DESCRIBE SELECT * FROM read_parquet(?, hive_partitioning=false)",
                             [list(partitions.values())]).fetchall()
        if [column[0] for column in columns] != PREDICTION_COLUMNS:
            raise ValueError("Unexpected ET0 V2 prediction columns")
    groups = {}
    for row in metric_rows:
        groups.setdefault(row["group"], set()).add(row["key"])
    group_keys = {group: sorted(keys, key=lambda key: key if not key.startswith("IB") else
                                 f"{int(key[2:].split(':')[0]):03d}{key.partition(':')[2]}")
                  for group, keys in groups.items()}
    locations = station_locations(location_path)
    station_names = {row["code"]: row["name"] for row in stations}
    station_ids = [*contract["stations"], *contract["historical_stations_without_retrospective_data"]]
    source = {
        "mode": mode, "status": contract["status"], "publication_status": contract.get("publication_status", "UNKNOWN"), "limitations": contract.get("limitations", []), "version": contract["result_version"],
        "schema_version": contract["schema_version"], "catalog_sha256": catalog_sha,
        "reference": contract["reference_id"], "reference_kind": "calculated",
        "retrospective_period": contract["period"],
        **{key: contract[key] for key in ("dataset_sha256", "freeze_sha256", "source_run_id", "original_manifest_sha256")},
    }
    api = APIRouter(prefix="/api/et0-v2", tags=["ET₀ V2"])

    @api.get("/catalog")
    def catalog():
        return dict(**source, formulas=FORMULAS, methods=methods,
                    stations={code: station_names.get(code, code) for code in station_ids},
                    station_details=[dict(row, **locations.get(row["code"], {}),
                                          retrospective=row["code"] in contract["stations"])
                                     for row in stations if row["code"] in station_ids],
                    station_catalog_sha256=STATION_CATALOG_SHA if locations else None,
                    calendars=contract["resolutions"], group_keys=group_keys)

    def validate_methods(selection):
        if selection and (len(selection) > len(methods) or not set(selection) <= methods.keys()):
            raise HTTPException(422, "Selección de métodos inválida")

    def scores(rows, selection):
        return [dict(row, cohort_id=row["common_keys_sha256"])
                for row in rows if not selection or row["method"] in selection]

    @api.get("/metrics")
    def metrics(formula: Formula, calendar: Calendar, group: Group, key: str,
                method: Annotated[list[str] | None, Query()] = None):
        validate_methods(method)
        if key not in group_keys.get(group, []):
            raise HTTPException(422, "Grupo y clave incompatibles")
        rows = [row for row in metric_rows if (row["family"], row["calendar"], row["group"], row["key"])
                == (formula, calendar, group, key)]
        return dict(source=source, scope="RETROSPECTIVE", rows=scores(rows, method))

    @api.get("/evolution")
    def evolution(formula: Formula, calendar: Calendar, station: str | None = None,
                  method: Annotated[list[str] | None, Query()] = None):
        validate_methods(method)
        if station and station not in station_ids:
            raise HTTPException(422, "Estación desconocida")
        group = "station_year" if station else "year"
        rows = [dict(row, year=row["key"].split(":")[-1]) for row in metric_rows
                if (row["family"], row["calendar"], row["group"]) == (formula, calendar, group)
                and (not station or row["key"].startswith(station + ":"))]
        return dict(source=source, scope="RETROSPECTIVE", rows=scores(rows, method))

    @api.get("/series")
    def series(formula: Formula, calendar: Calendar, station: str, start: date, end: date,
               cohort: Literal["own", "common_core"] = "own",
               method: Annotated[list[str] | None, Query()] = None):
        validate_methods(method)
        if station not in station_ids or not first_day <= start <= end <= last_day or (end - start).days > 366:
            raise HTTPException(422, "Seleccione una estación y hasta 367 días del periodo retrospectivo")
        selection = list(dict.fromkeys(method or list(METHODS)))
        if cohort == "common_core" and not set(selection) <= METHODS.keys():
            raise HTTPException(422, "La cohorte común de la serie cubre solo los cinco métodos principales")
        files = [partitions[station, year] for year in range(start.year, end.year + 1)
                 if (station, year) in partitions]
        points = {}
        if files:
            with duckdb.connect(config={"threads": 1, "memory_limit": "128MB",
                                        "max_temp_directory_size": "0B"}) as db:
                records = db.execute("""
                    SELECT date, method_id, prediction, reference_value
                    FROM read_parquet(?, hive_partitioning=false)
                    WHERE station_id=? AND method_family=? AND resolution=? AND date BETWEEN ? AND ?
                      AND list_contains(?, method_id) AND (? = 'own' OR cohort_id = 'common_core')
                    ORDER BY date, method_id
                """, [files, station, formula, calendar, start.isoformat(), end.isoformat(), selection, cohort]).fetchall()
            for day, name, prediction, reference in records:
                point = points.setdefault(day, dict(date=day, reference=reference, **dict.fromkeys(selection)))
                point[name] = prediction
        return dict(source=source, scope="RETROSPECTIVE", cohort=cohort, points=list(points.values()))

    @api.get("/in-sample")
    def in_sample(formula: Formula, station: str):
        if station not in station_ids:
            raise HTTPException(422, "Estación desconocida")
        rows = [dict(row, method="AHC") for row in history_rows
                if row["family"] == formula and row["station_code"] == station]
        return dict(source=source, scope="FULL_HISTORY_IN_SAMPLE", calendar="annual", rows=rows)

    return api
