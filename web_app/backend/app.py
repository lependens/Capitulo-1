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
from .catalog import AHC_MODELS, AHC_MODEL_IDS, AHC_PAIRS, AHC_SERIES_IDS, CSV_SHA, DATASET_SHA, SUFFIXES


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
    ahc_enabled = info.get("schema_version") == "web-v0.2-ahc"
    ahc_models = load("ahc_models.json") if ahc_enabled else []
    station_codes = {s["code"] for s in stations}
    model_ids = {m["id"] for m in models}
    predictions = model_ids - {"EtPMon"}
    web_version = "0.2.0" if ahc_enabled else "0.1.0"
    api = FastAPI(title="SIAR ET₀ Lab", version=web_version, docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    db = duckdb.connect(config={"threads": 1, "memory_limit": "256MB", "max_temp_directory_size": "0B"})
    parquet_path = str(folder / "observations.parquet").replace("'", "''")
    db.execute(f"CREATE VIEW observations AS SELECT * FROM read_parquet('{parquet_path}')")
    if ahc_enabled:
        for view, filename in {
            "ahc_predictions": "ahc_predictions.parquet",
            "ahc_metrics": "ahc_metrics.parquet",
            "ahc_coefficients": "ahc_coefficients.parquet",
            "ahc_climatology": "ahc_climatology_coefficients.parquet",
            "ahc_model_metadata": "ahc_model_metadata.parquet",
            "ahc_coverage": "ahc_coverage.parquet",
            "ahc_station_metadata": "ahc_station_metadata.parquet",
        }.items():
            path = str(folder / filename).replace("'", "''")
            db.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{path}')")
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

    def require_ahc():
        if not ahc_enabled:
            raise HTTPException(404, "AHC package is not installed")

    def check_split(split):
        if split not in {"ALL", "TRAIN", "VALIDATION", "TEST"}:
            raise HTTPException(422, "Unknown split")

    def check_ahc_model(model):
        if model not in AHC_SERIES_IDS:
            raise HTTPException(422, "Unknown AHC model")

    def ahc_where(station, start, end, split):
        check_station(station)
        check_split(split)
        if start and end and start > end:
            raise HTTPException(422, "from must be on or before to")
        clauses, values = [], []
        for expression, value in (("station_code = ?", station), ("date >= ?", start), ("date <= ?", end)):
            if value is not None:
                clauses.append(expression)
                values.append(str(value))
        if split != "ALL":
            clauses.append("split = ?")
            values.append(split)
        return " AND ".join(clauses) or "true", values

    def decode_json_fields(rows, fields):
        for row in rows:
            for field in fields:
                if isinstance(row.get(field), str):
                    row[field] = json.loads(row[field])
        return rows

    @api.get("/api/health")
    async def health():
        return dict(status="ok", dataset_id=info["dataset_id"], rows=info["rows"], integrity="verified at startup",
                    version=web_version, release="WEB-V0.2-AHC" if ahc_enabled else "WEB-V0.1",
                    ahc="PASS" if ahc_enabled else "not installed", data_recalculated=False,
                    scientific_artifacts_modified=False)

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

    @api.get("/api/ahc/models")
    async def get_ahc_models():
        require_ahc()
        published_metadata = query("SELECT * FROM ahc_model_metadata ORDER BY method_id")
        decode_json_fields(published_metadata, ("input_parameters", "input_requirements", "selection"))
        by_id = {row["method_id"]: row for row in published_metadata}
        return dict(models=[dict(model, published=by_id.get(model["id"])) for model in ahc_models],
                    split_semantics=info["split_semantics"], reference="PM_FAO56_V1")

    @api.get("/api/ahc/timeseries")
    async def ahc_timeseries(station: str, split: str = "TEST",
                  start: date | None = Query(None, alias="from"), end: date | None = Query(None, alias="to"),
                  models: str = "PM_FAO56_V1,HG_RS_V1,HG_RS_AHC_VALIDATED,HG_RS_AHC_VALIDATED_SELECTED",
                  max_points: int = Query(800, ge=50, le=2000)):
        require_ahc()
        ids = list(dict.fromkeys(models.split(",")))
        if not ids or any(not model for model in ids):
            raise HTTPException(422, "At least one AHC model is required")
        for model in ids:
            check_ahc_model(model)
        condition, values = ahc_where(station, start, end, split)
        count = query(f"SELECT count(*) n FROM ahc_predictions WHERE {condition}", values)[0]["n"]
        width = max(1, math.ceil(count / max_points))
        columns = [f'avg(try_cast("{model}" AS DOUBLE)) AS "{model}"' for model in ids]
        points = query(f'''WITH selected AS (SELECT *, row_number() OVER (ORDER BY date) - 1 rn
            FROM ahc_predictions WHERE {condition}) SELECT min(date) date, max(date) date_end,
            count(*) observations, min(split) split, {', '.join(columns)}
            FROM selected GROUP BY floor(rn / {width}) ORDER BY date''', values)
        return dict(points=points, source_rows=count, returned_points=len(points), split=split,
                    aggregation="daily" if width == 1 else f"visual mean · {width} published rows per bucket",
                    affects_metrics=False, data_recalculated=False)

    @api.get("/api/ahc/comparison")
    async def ahc_comparison(base_method: str, split: str = "TEST", station: str | None = None):
        require_ahc()
        check_station(station)
        check_split(split)
        if base_method not in AHC_PAIRS:
            raise HTTPException(422, "Unknown AHC base method")
        cohort_id, methods = AHC_PAIRS[base_method]
        clauses, values = ['reference_id = ?', '"group" = ?', 'cohort_id = ?'], ["PM_FAO56_V1", "station" if station else "global", cohort_id]
        if station:
            clauses.append("station_code = ?")
            values.append(station)
        if split != "ALL":
            clauses.append("split = ?")
            values.append(split)
        placeholders = ",".join("?" for _ in methods)
        values.extend(methods)
        rows = query(f'''SELECT split, reference_id, "group", key, station_code, cohort_id, cohort_sha256,
            method_id, model_instance_id, start_date, end_date, n, mae, rmse, bias, r2, r2_status
            FROM ahc_metrics WHERE {' AND '.join(clauses)} AND method_id IN ({placeholders})
            ORDER BY CASE split WHEN 'TRAIN' THEN 1 WHEN 'VALIDATION' THEN 2 ELSE 3 END, method_id''', values)
        return dict(base_method=base_method, methods=methods, split=split, station=station, metrics=rows,
                    scope="published AHC metrics; paired cohort", period_filter_applied=False,
                    test_status="retrospective_locked", data_recalculated=False)

    @api.get("/api/ahc/coefficients")
    async def coefficients(station: str | None = None, base_method: str | None = None,
                           granularity: Literal["ALL", "annual", "seasonal", "monthly"] = "ALL",
                           kind: Literal["AHC_VALIDATED", "AHC_CLIMATOLOGY"] = "AHC_VALIDATED"):
        require_ahc()
        check_station(station)
        if base_method and base_method not in AHC_PAIRS:
            raise HTTPException(422, "Unknown AHC base method")
        clauses, values = [], []
        for expression, value in (("station_code = ?", station), ("base_method_id = ?", base_method)):
            if value:
                clauses.append(expression)
                values.append(value)
        if granularity != "ALL":
            clauses.append("granularity = ?")
            values.append(granularity)
        view = "ahc_coefficients" if kind == "AHC_VALIDATED" else "ahc_climatology"
        rows = query(f'''SELECT coefficient_id, station_code, base_method_id, kind, reference_id,
            granularity, group_key, coefficient, status, deployable, fit_split, calibration_start,
            calibration_end, fit_key_sha256, support, estimate FROM {view}
            WHERE {' AND '.join(clauses) if clauses else 'true'}
            ORDER BY station_code, base_method_id, granularity, group_key''', values)
        decode_json_fields(rows, ("support", "estimate"))
        return dict(kind=kind, coefficients=rows, data_recalculated=False,
                    interpretation="operational description only; no TEST performance" if kind == "AHC_CLIMATOLOGY" else "TRAIN-fitted frozen coefficients")

    @api.get("/api/ahc/seasonal")
    async def ahc_seasonal(station: str | None = None, split: str = "TEST"):
        require_ahc()
        check_station(station)
        check_split(split)
        group = "station_season_year" if station else "season_year"
        clauses, values = ['reference_id = ?', '"group" = ?', 'cohort_id = ?'], ["PM_FAO56_V1", group, "selected:HG_RS_V1"]
        if station:
            clauses.append("station_code = ?")
            values.append(station)
        if split != "ALL":
            clauses.append("split = ?")
            values.append(split)
        rows = query(f'''SELECT split, key, station_code, cohort_id, method_id, start_date, end_date,
            n, mae, rmse, bias, r2, r2_status FROM ahc_metrics
            WHERE {' AND '.join(clauses)} ORDER BY start_date, method_id''', values)
        return dict(split=split, station=station, metrics=rows, scope="published seasonal metrics",
                    test_status="retrospective_locked", data_recalculated=False)

    @api.get("/api/ahc/coverage")
    async def ahc_coverage(station: str | None = None, split: str = "ALL"):
        require_ahc()
        check_station(station)
        check_split(split)
        clauses, values = [], []
        if station:
            clauses.append("station_code = ?")
            values.append(station)
        if split != "ALL":
            clauses.append("split = ?")
            values.append(split)
        rows = query(f'''SELECT method_id, model_instance_id, station_code, split, source_rows,
            eligible_n, predicted_n, evaluated_PM_n, evaluated_SIAR_n, overlapping_reason_counts
            FROM ahc_coverage WHERE {' AND '.join(clauses) if clauses else 'true'}
            ORDER BY station_code, split, method_id''', values)
        decode_json_fields(rows, ("overlapping_reason_counts",))
        return dict(split=split, rows=rows, data_recalculated=False)

    return api


def app():
    return create_app()
