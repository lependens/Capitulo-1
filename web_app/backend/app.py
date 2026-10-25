"""Read-only API over a verified derivative of a closed scientific publication."""
from datetime import date
import json
import math
import os
from pathlib import Path
from typing import Literal

import duckdb
from fastapi import FastAPI, HTTPException, Query

from .convert import sha
from .catalog import CSV_SHA, DATASET_SHA, SUFFIXES


def create_app(data_dir=None):
    folder = Path(data_dir or os.environ.get("WEB_DATA_DIR", "/data"))
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    if info["dataset_sha256"] != DATASET_SHA or info["source_sha256"] != CSV_SHA:
        raise ValueError("Unapproved scientific source")
    for name, digest in info["artifacts"].items():
        if Path(name).name != name or sha(folder / name) != digest:
            raise ValueError(f"Derived artifact integrity failure: {name}")
    load = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    stations, models, published = load("stations.json"), load("models.json"), load("metrics.json")
    station_codes = {s["code"] for s in stations}
    model_ids = {m["id"] for m in models}
    predictions = model_ids - {"EtPMon"}
    api = FastAPI(title="SIAR ET₀ Lab", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    db = duckdb.connect(config={"threads": 1, "memory_limit": "256MB", "max_temp_directory_size": "0B"})
    parquet_path = str(folder / "observations.parquet").replace("'", "''")
    db.execute(f"CREATE VIEW observations AS SELECT * FROM read_parquet('{parquet_path}')")
    api.state.db = db

    def check_station(station):
        if station and station not in station_codes:
            raise HTTPException(422, "Unknown station")

    def check_model(model, allow_reference=False):
        if model not in (model_ids if allow_reference else predictions):
            raise HTTPException(422, "Unknown model")

    def where(station, start, end):
        check_station(station)
        if start and end and start > end:
            raise HTTPException(422, "from must be on or before to")
        clauses, values = [], []
        for expression, value in (("station_code = ?", station), ("date >= ?", start), ("date <= ?", end)):
            if value is not None:
                clauses.append(expression)
                values.append(str(value))
        return " AND ".join(clauses) or "true", values

    def query(sql, values=()):
        cursor = db.execute(sql, values)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    @api.get("/api/health")
    async def health():
        return dict(status="ok", dataset_id=info["dataset_id"], rows=info["rows"], integrity="verified at startup", version="0.1.0")

    @api.get("/api/stations")
    async def get_stations():
        return dict(stations=stations)

    @api.get("/api/models")
    async def get_models():
        return dict(models=models)

    @api.get("/api/dataset/info")
    async def get_info():
        return info

    @api.get("/api/timeseries")
    async def timeseries(station: str, start: date | None = Query(None, alias="from"),
                   end: date | None = Query(None, alias="to"), models: str = "EtPMon,PM_FAO56_V1",
                   max_points: int = Query(800, ge=50, le=2000)):
        ids = list(dict.fromkeys(models.split(",")))
        for model in ids:
            check_model(model, True)
        condition, values = where(station, start, end)
        count = query(f"SELECT count(*) n FROM observations WHERE {condition}", values)[0]["n"]
        width = max(1, math.ceil(count / max_points))
        columns = [f'avg(try_cast("{("EtPMon" if m == "EtPMon" else "ET0_" + m)}" AS DOUBLE)) AS "{m}"' for m in ids]
        # Means are visual summaries only; independently preserve null model values.
        sql = f'''WITH selected AS (SELECT *, row_number() OVER (ORDER BY date) - 1 AS rn
            FROM observations WHERE {condition}) SELECT min(date) AS date,
            max(date) AS date_end, count(*) AS observations, {', '.join(columns)}
            FROM selected GROUP BY floor(rn / {width}) ORDER BY date'''
        points = query(sql, values)
        return dict(points=points, source_rows=count, returned_points=len(points),
                    aggregation="daily" if width == 1 else f"visual mean · {width} observed rows per bucket",
                    affects_metrics=False)

    @api.get("/api/metrics")
    async def metrics(station: str | None = None, model: str | None = None,
                group: Literal["global", "year", "month", "season"] = "global",
                cohort: Literal["common", "own"] = "common"):
        check_station(station)
        if model:
            check_model(model)
        response = dict(scope="published scientific metrics", period_filter_applied=False,
                        station=station, cohort=cohort, group=group, metrics=[])
        if station and group in ("month", "season"):
            return dict(**response, unavailable_reason="El productor no publicó métricas estación-mes ni estación-temporada. Seleccione todas las estaciones.")
        actual_group = {"global": "station", "year": "station_year"}.get(group, group) if station else group
        actual_cohort = "PM_HS_VAL_HGRS" if cohort == "common" else "own"
        response["metrics"] = [dict(m, published_cohort=m["cohort"], cohort=cohort) for m in published
            if m["group"] == actual_group and m["cohort"] == actual_cohort and
            (not model or m["model"] == model) and
            (not station or m["key"] == station or m["key"].startswith(station + ":"))]
        return response

    @api.get("/api/scatter")
    async def scatter(station: str, model: str, start: date | None = Query(None, alias="from"),
                end: date | None = Query(None, alias="to"), cohort: Literal["common", "own"] = "common",
                max_points: int = Query(1500, ge=50, le=3000)):
        check_model(model)
        condition, values = where(station, start, end)
        required = predictions if cohort == "common" else [model]
        condition += " AND " + " AND ".join(f'evaluated_{SUFFIXES[m]} = \'true\'' for m in sorted(required))
        count = query(f"SELECT count(*) n FROM observations WHERE {condition}", values)[0]["n"]
        stride = max(1, math.ceil(count / max_points))
        points = query(f'''WITH selected AS (SELECT *, row_number() OVER(ORDER BY date) - 1 rn
            FROM observations WHERE {condition}) SELECT date, cast(EtPMon AS DOUBLE) AS reference,
            cast("ET0_{model}" AS DOUBLE) AS predicted FROM selected WHERE rn % {stride} = 0 ORDER BY date''', values)
        return dict(points=points, total_rows=count, returned_points=len(points), cohort=cohort,
                    sampling="all" if stride == 1 else f"deterministic every {stride} evaluated rows", affects_metrics=False)

    @api.get("/api/coverage")
    async def coverage(station: str | None = None):
        check_station(station)
        condition, values = where(station, None, None)
        columns = []
        for model, suffix in SUFFIXES.items():
            columns += [f"count(*) FILTER(WHERE eligible_{suffix} = 'true') AS {suffix}_predicted",
                        f"count(*) FILTER(WHERE evaluated_{suffix} = 'true') AS {suffix}_evaluated"]
        counts = query(f"SELECT station_code, {', '.join(columns)} FROM observations WHERE {condition} GROUP BY station_code", values)
        by_code = {r["station_code"]: r for r in counts}
        results = []
        for s in stations:
            if station and s["code"] != station:
                continue
            result = dict(s)
            result["models"] = [dict(model=m, predicted=by_code[s["code"]][f"{suffix}_predicted"],
                                     evaluated=by_code[s["code"]][f"{suffix}_evaluated"]) for m, suffix in SUFFIXES.items()]
            results.append(result)
        return dict(stations=results, meaning="Counts of preserved observations and producer eligibility; not completeness of expected calendar days")

    @api.get("/api/observations")
    async def observations(station: str, start: date | None = Query(None, alias="from"),
                     end: date | None = Query(None, alias="to"), offset: int = Query(0, ge=0),
                     limit: int = Query(50, ge=1, le=200)):
        condition, values = where(station, start, end)
        return dict(rows=query(f"SELECT * FROM observations WHERE {condition} ORDER BY date LIMIT ? OFFSET ?", [*values, limit, offset]),
                    offset=offset, limit=limit)

    return api


def app():
    return create_app()
