"""Checks for descriptive weather and the real complementary science handoff."""
import csv
import os
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException
import pytest

from test_api import ASGIClient
from web_app.backend.et0_v2_development import CATALOG_SHA, COUNTS, router
from web_app.backend.station_weather import router as weather_router

IDENTITY = {
    "dataset_sha256": "0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb",
    "freeze_sha256": "97acad7ce7cc39c65d82f6efdc36290ed0325caaef2dedb69d9ffbefe1addfbd",
}


@pytest.fixture(scope="module")
def development():
    path = os.getenv("TEST_ET0_V2_DEVELOPMENT_DIR")
    if not path:
        pytest.skip("Provide TEST_ET0_V2_DEVELOPMENT_DIR for the real scientific handoff")
    catalog = Path(__file__).parents[2] / "estaciones_baleares.csv"
    with catalog.open() as f:
        stations = [r["Codigo"] for r in csv.DictReader(f)]
    app = FastAPI()
    app.include_router(router(path, IDENTITY, stations))
    return ASGIClient(app)


def test_handoff_catalog_support_and_frequency(development):
    metadata = development.get("/development/catalog").json()
    source = metadata["source"]
    assert source["catalog_sha256"] == CATALOG_SHA
    assert {k: source[k] for k in IDENTITY} == IDENTITY
    assert source["publication_status"] == "NOT_PUBLISHED"
    assert {name.removesuffix(".csv"): entry["rows"] for name, entry in source["tables"].items()} == COUNTS
    assert len(metadata["cells"]) == 108
    excluded = [row for row in metadata["cells"] if not row["ml_available"]]
    assert len(excluded) == 3
    assert all(r["station_code"] == "IB101" and r["calendar"] == "monthly"
               and r["ahc_available"] and not r["ann_a_available"] and not r["ann_e_available"] for r in excluded)
    for stage in ("ML", "ANN_A", "ANN_E"):
        for dimension in ("global", "station_code", "family", "calendar"):
            frequency = development.get("/development/frequency", {"stage": stage, "dimension": dimension}).json()
            assert sum(r["count"] for r in frequency["rows"]) == frequency["total"] == 105


def test_handoff_frozen_candidates_folds_and_history(development):
    key = dict(station="IB01", family="HG_RS_V1", calendar="annual")

    def get(name, **params):
        return development.get("/development/table/" + name, params={**key, **params}).json()

    assert get("ahc-frozen-coefficients")["rows"][0]["coefficient"] == 0.9189636639528016
    assert get("ahc-fold-coefficients", year=2019)["rows"][0]["held_out_year"] == 2019
    assert get("ahc-history-in-sample")["scope"] == "FULL_HISTORY_IN_SAMPLE"
    ml_candidates = get("ml-development-candidates")
    assert ml_candidates["total"] == 37
    assert get("ml-development-candidates", limit=3, offset=1)["rows"] == ml_candidates["rows"][1:4]
    finals = get("ml-final-models")["rows"]
    assert len(finals) == 7 and sum(r["is_principal"] for r in finals) == 1
    assert {r["candidate_id"] for r in finals} == {r["candidate_id"] for r in ml_candidates["rows"] if r["selected_by_algorithm"]}
    candidate = finals[0]["candidate_id"]
    folds = get("ml-fold-metrics", candidate=candidate)
    assert folds["scope"] == "DEVELOPMENT" and folds["total"] > 0
    assert get("ann-development-candidates", stage="ANN_A")["total"] == 5
    ann = get("ann-selected-architectures", stage="ANN_A")["rows"][0]
    assert ann["candidate_id"] == "ann-64" and ann["epochs"]["20261101"] == 199
    rows = get("ann-fold-seed-metrics", stage="ANN_A", candidate=ann["candidate_id"])["rows"]
    assert len(rows) == 3 * len({r["year"] for r in rows})
    for year in {r["year"] for r in rows}:
        fold = [r for r in rows if r["year"] == year]
        assert len({r["ensemble_rmse"] for r in fold}) == 1
        assert len({r["seed_seed"] for r in fold}) == 3


def test_weather_long_intervals_totals_and_pairing():
    with duckdb.connect() as db:
        db.execute("""CREATE TABLE observations AS
            SELECT 'IB01' station_code, cast(cast(day AS DATE) AS VARCHAR) date, 10 + dayofmonth(day) TempMedia,
                   CASE WHEN dayofmonth(day) % 2 = 0 THEN 40 END HumedadMedia
            FROM generate_series(DATE '2009-01-01', DATE '2020-12-31', INTERVAL '1 day') t(day)""")

        def query(sql, params=()):
            result = db.execute(sql, params)
            return [dict(zip([d[0] for d in result.description], row)) for row in result.fetchall()]

        def where(station, start, end):
            if station != "IB01":
                raise HTTPException(422, "Unknown station")
            return "station_code = ? AND date BETWEEN ? AND ?", [station, str(start), str(end)]

        api = FastAPI()
        api.include_router(weather_router(query, where, {"TempMedia", "HumedadMedia"}))
        client = ASGIClient(api)
        params = dict(station="IB01", variable=["TempMedia", "HumedadMedia"], start="2009-01-01", end="2020-12-31")
        daily = client.get("/api/et0-v2/weather/series", params).json()
        assert len(daily["points"]) <= 1000 and daily["stride"] > 1
        assert daily["points"][0]["period"] == "2009-01-01"
        assert daily["points"][-1]["period_end"] == "2020-12-31"
        for view, size in [("year", 12), ("month", 12)]:
            response = client.get("/api/et0-v2/weather/series", dict(params, view=view)).json()
            assert len(response["points"]) == size
            assert sum(row["TempMedia_n"] for row in response["points"]) == daily["overall"]["TempMedia_n"]
            assert sum(row["HumedadMedia_n"] for row in response["points"]) < daily["overall"]["TempMedia_n"]
        scatter = client.get("/api/et0-v2/weather/series", dict(params, view="scatter")).json()
        assert all(row["HumedadMedia"] == 40 for row in scatter["points"])
        assert client.get("/api/et0-v2/weather/series", dict(params, variable=["unknown"])).status_code == 422
