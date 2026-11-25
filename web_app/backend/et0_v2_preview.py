"""Synthetic preview. No filesystem access and no scientific calculations."""
from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/et0-v2", tags=["ET₀ V2 preview"])
Formula = Literal["HS_RA_V1", "HG_RS_V1", "VALIANTZAS_V1"]
Calendar = Literal["annual", "seasonal", "monthly"]
Group = Literal["global", "station", "year", "station_year", "month", "season"]
Method = Literal["FORMULA", "AHC", "ML", "ANN_A", "ANN_E"]

FORMULAS = {
    "HS_RA_V1": {"label": "Hargreaves–Samani Ra", "inputs": ["Tmin", "Tmax", "Ra"]},
    "HG_RS_V1": {"label": "Hargreaves Rs", "inputs": ["Media de extremos", "Rs"]},
    "VALIANTZAS_V1": {"label": "Valiantzas", "inputs": ["Tmin", "Tmax", "HR media", "Ra"]},
}
METHODS = {
    "FORMULA": "Fórmula original", "AHC": "AHC", "ML": "ML clásico",
    "ANN_A": "ANN A", "ANN_E": "ANN E",
}
STATIONS = {"DEMO-01": "Estación de ejemplo", "DEMO-H": "Serie histórica de ejemplo"}
GROUP_KEYS = {
    "global": ["all"], "station": list(STATIONS), "year": ["2020", "2021"],
    "station_year": ["DEMO-01:2020", "DEMO-01:2021"],
    "month": ["07"], "season": ["JJA"],
}
SOURCE = {
    "mode": "synthetic", "version": "preview-1", "reference": "PM_FAO56_V1",
    "reference_kind": "calculated", "retrospective_period": ["2020-01-01", "2025-12-31"],
}

# Illustrative fixtures, intentionally independent of the chart points.
FIXTURE_SCORES = {
    "FORMULA": (.51, .64, .23, .78, .96),
    "AHC": (.34, .43, -.04, .89, .96),
    "ML": (.23, .32, .02, .94, .91),
    "ANN_A": (.22, .30, -.01, .95, .90),
    "ANN_E": (.24, .33, .03, .93, .90),
}
YEAR_2021_SCORES = {
    "FORMULA": (.56, .71, .28, .76, .92),
    "AHC": (.35, .45, -.06, .87, .92),
    "ML": (.23, .31, .01, .94, .88),
    "ANN_A": (.27, .35, -.02, .92, .87),
}
DAILY = [
    ("2020-07-01", 4.5, 5.1, 4.7, 4.4, 4.5, 4.6),
    ("2020-07-02", 4.9, 5.5, 5.1, 4.8, 4.9, 5.0),
    ("2020-07-03", 4.6, 5.0, 4.7, 4.7, 4.5, 4.7),
    ("2020-07-04", 5.5, 6.0, 5.8, 5.4, 5.6, 5.5),
    ("2020-07-05", 5.2, 5.8, 5.4, 5.3, 5.2, 5.3),
    ("2020-07-06", 5.7, 6.2, 5.8, None, 5.6, 5.5),
    ("2020-07-07", 4.8, 5.0, 4.7, 4.9, 4.8, 4.9),
    ("2020-07-08", 4.3, 4.8, 4.5, 4.2, 4.3, 4.2),
]


@router.get("/catalog")
def catalog():
    return {
        **SOURCE, "formulas": FORMULAS, "methods": METHODS, "stations": STATIONS,
        "calendars": ["annual", "seasonal", "monthly"], "group_keys": GROUP_KEYS,
    }


@router.get("/metrics")
def metrics(formula: Formula, calendar: Calendar, group: Group, key: str,
            method: Annotated[list[Method] | None, Query()] = None):
    if key not in GROUP_KEYS[group]:
        raise HTTPException(422, "Grupo y clave incompatibles")
    scores = []
    is_2021 = key in ("2021", "DEMO-01:2021")
    fixtures = YEAR_2021_SCORES if is_2021 else FIXTURE_SCORES
    for name, (mae, rmse, bias, r2, coverage) in fixtures.items():
        if method and name not in method:
            continue
        if key == "DEMO-H":
            continue
        scores.append(dict(method=name, mae=mae, rmse=rmse, bias=bias, r2=r2,
                           coverage_calendar=coverage, n=166 if is_2021 else 180,
                           cohort_id=f"demo:{formula}:{calendar}:{group}:{key}"))
    return dict(source=SOURCE, scope="RETROSPECTIVE", formula=formula,
                calendar=calendar, group=group, key=key, rows=scores)


@router.get("/series")
def series(formula: Formula, calendar: Calendar, station: str, start: date, end: date):
    if station not in STATIONS or start > end or (end - start).days > 366:
        raise HTTPException(422, "Seleccione una estación y un intervalo de hasta 367 días")
    points = []
    if (station, formula, calendar) == ("DEMO-01", "HG_RS_V1", "annual"):
        points = [
            dict(date=day, reference=target, **dict(zip(METHODS, values)))
            for day, target, *values in DAILY if start.isoformat() <= day <= end.isoformat()
        ]
    return dict(source=SOURCE, scope="RETROSPECTIVE", station=station, formula=formula,
                calendar=calendar, start=start, end=end, points=points)


@router.get("/in-sample")
def in_sample(formula: Formula, station: str):
    if station not in STATIONS:
        raise HTTPException(422, "Estación desconocida")
    rows = []
    if station == "DEMO-01" and formula == "HG_RS_V1":
        rows.append(dict(method="AHC", coefficient=.91, n=512, mae=.29, rmse=.39,
                         bias=.02, r2=.90, development_pairs=332, retrospective_pairs=180))
    return dict(source=SOURCE, scope="FULL_HISTORY_IN_SAMPLE", calendar="annual",
                formula=formula, station=station, rows=rows)
