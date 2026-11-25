"""Descriptive summaries of the existing verified daily station observations."""
from datetime import date
import math
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query

VARIABLES = (
    "TempMedia", "TempMin", "TempMax", "HumedadMedia", "VelViento",
    "Radiacion", "Precipitacion", "ET0_PM_FAO56_V1",
)


def router(query, where, columns):
    api = APIRouter(prefix="/api/et0-v2/weather", tags=["Observaciones SIAR"])
    available = [name for name in VARIABLES if name in columns]

    def field(name):
        return f'try_cast("{name}" AS DOUBLE)'

    def aggregates(selection):
        return ", ".join(
            f'avg({field(name)}) AS "{name}", count({field(name)}) AS "{name}_n"'
            for name in selection
        )

    @api.get("/stations")
    async def stations():
        rows = query(f"""SELECT station_code, min(date) first_date, max(date) last_date,
            count(*) observations, {aggregates(available)}
            FROM observations GROUP BY station_code ORDER BY station_code""")
        return dict(variables=available, rows=rows, source="observations.parquet")

    @api.get("/series")
    async def series(station: str, variable: Annotated[list[str], Query(min_length=1, max_length=8)],
                     start: date | None = None, end: date | None = None,
                     view: Literal["daily", "year", "month", "scatter"] = "daily"):
        selection = list(dict.fromkeys(variable))
        if not set(selection) <= set(available) or view == "scatter" and len(selection) != 2:
            raise HTTPException(422, "La vista de dispersión requiere dos variables distintas")
        condition, values = where(station, start, end)
        overall = query(f"SELECT count(*) observations, {aggregates(selection)} FROM observations WHERE {condition}",
                        values)[0]
        weight = 1
        if view == "scatter":
            paired = " AND ".join(f"{field(name)} IS NOT NULL" for name in selection)
            total = query(f"SELECT count(*) n FROM observations WHERE {condition} AND {paired}", values)[0]["n"]
            weight = max(1, math.ceil(total / 1200))
            fields = ", ".join(f'{field(name)} AS "{name}"' for name in selection)
            points = query(f"""WITH paired AS (SELECT date, {fields},
                row_number() OVER (ORDER BY date) - 1 idx FROM observations WHERE {condition} AND {paired})
                SELECT * EXCLUDE (idx) FROM paired WHERE idx % {weight} = 0 ORDER BY date""", values)
        elif view == "daily":
            weight = max(1, math.ceil(overall["observations"] / 1000))
            points = query(f"""WITH ordered AS (SELECT *, row_number() OVER (ORDER BY date) - 1 idx
                    FROM observations WHERE {condition})
                SELECT min(date) period, max(date) period_end, count(*) observations, {aggregates(selection)}
                FROM ordered GROUP BY idx // {weight} ORDER BY period""", values)
        else:
            part = "year" if view == "year" else "month"
            points = query(f"""SELECT date_part('{part}', cast(date AS DATE)) period,
                    count(*) observations, {aggregates(selection)}
                FROM observations WHERE {condition} GROUP BY period ORDER BY period""", values)
        return dict(view=view, points=points, overall=overall, stride=weight,
                    source="observations.parquet", scope="DESCRIPTIVE_OBSERVATIONS", affects_metrics=False)
    return api
