"""Synthetic derivatives test serving semantics, never scientific validity."""
from copy import deepcopy
from datetime import date, timedelta
import asyncio
import json
import os
from pathlib import Path

import duckdb
import httpx
import pytest

from web_app.backend.app import create_app
from web_app.backend.catalog import CSV_SHA, DATASET_SHA, MODELS, SUFFIXES
from web_app.backend.convert import sha


@pytest.fixture
def derivative(tmp_path):
    columns = ["station_code", "date", "EtPMon", "quality_status"]
    columns += ["ET0_" + model for model in SUFFIXES]
    columns += [prefix + suffix for suffix in SUFFIXES.values() for prefix in ("eligible_", "evaluated_")]
    records = []
    for index in range(125):
        station = "IB01" if index < 120 else "IB09"
        row = dict.fromkeys(columns)
        row.update(station_code=station, date=str(date(2020, 1, 1) + timedelta(days=index)),
                   EtPMon=str(index % 9), quality_status="WARN" if station == "IB01" else "FAIL")
        for model, suffix in SUFFIXES.items():
            row["ET0_" + model] = str(index % 9 + 0.25) if station == "IB01" else None
            row["eligible_" + suffix] = row["evaluated_" + suffix] = "true" if station == "IB01" else "false"
        # Missing HS does not discard a usable PM value; zero reference is preserved.
        if index == 1:
            row["ET0_HS_RA_V1"] = None
            row["eligible_HS"] = row["evaluated_HS"] = "false"
        if index == 0:
            row["ET0_PM_FAO56_V1"] = "0"
        records.append(row)
    with duckdb.connect() as db:
        db.execute("CREATE TABLE observations (" + ",".join('"' + column + '" VARCHAR' for column in columns) + ")")
        db.executemany("INSERT INTO observations VALUES (" + ",".join("?" for _ in columns) + ")",
                       [[row[column] for column in columns] for row in records])
        db.execute("COPY observations TO ? (FORMAT PARQUET)", [str(tmp_path / "observations.parquet")])
    stations = [dict(code="IB01", name="Synthetic active", status="ACTIVE", rows=120, quality_status="WARN",
                     **{"from": "2020-01-01", "to": "2020-04-29"}),
                dict(code="IB09", name="Synthetic excluded", status="CLOSED", rows=5, quality_status="FAIL",
                     **{"from": "2020-04-30", "to": "2020-05-04"})]
    metrics = []
    for cohort in ("own", "PM_HS_VAL_HGRS"):
        for group, key in (("global", "all"), ("station", "IB01"), ("station_year", "IB01:2020"),
                           ("month", "01"), ("season", "DJF"), ("year", "2020")):
            metrics.append(dict(model="PM_FAO56_V1", method="ET0_FAO56", cohort=cohort, group=group, key=key,
                                n=120 if cohort == "own" else 119, mae=0.123456789123, rmse=0.234567891234,
                                bias=-0.012345678912, r2=-0.567891234567,
                                audit_source_repo_sha="a" * 40, source_repo_sha="b" * 40))
    for name, value in (("stations.json", stations), ("models.json", MODELS), ("metrics.json", metrics)):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    info = dict(dataset_id="synthetic-test-only", dataset_sha256=DATASET_SHA, source_sha256=CSV_SHA,
                rows=len(records), artifacts={name: sha(tmp_path / name) for name in
                    ("observations.parquet", "stations.json", "models.json", "metrics.json")})
    (tmp_path / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(derivative):
    return ASGIClient(create_app(derivative))


class ASGIClient:
    """Small sync facade that exercises ASGI without TestClient worker threads."""
    def __init__(self, app):
        self.app = app

    def get(self, path, params=None):
        async def request():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get(path, params=params)
        return asyncio.run(request())


def test_metadata_and_health_identify_input(client):
    assert client.get("/api/health").json()["rows"] == 125
    assert len(client.get("/api/models").json()["models"]) == 5
    stations = client.get("/api/stations").json()["stations"]
    assert [row["status"] for row in stations] == ["ACTIVE", "CLOSED"]
    assert client.get("/api/dataset/info").json()["source_sha256"] == CSV_SHA


def test_series_filters_inclusively_preserving_zero_and_independent_nulls(client):
    response = client.get("/api/timeseries", params={"station": "IB01", "from": "2020-01-01", "to": "2020-01-02",
                                                   "models": "EtPMon,PM_FAO56_V1,HS_RA_V1"})
    assert response.status_code == 200
    data = response.json()
    assert data["source_rows"] == data["returned_points"] == 2
    assert data["points"][0]["EtPMon"] == 0
    assert data["points"][0]["PM_FAO56_V1"] == 0
    assert data["points"][1]["HS_RA_V1"] is None
    assert data["points"][1]["PM_FAO56_V1"] == 1.25
    assert data["affects_metrics"] is False


@pytest.mark.parametrize("path,params", [
    ("/api/timeseries", {"station": "IB01' OR true --"}),
    ("/api/timeseries", {"station": "IB01", "models": 'EtPMon\" FROM observations;--'}),
    ("/api/timeseries", {"station": "IB01", "from": "2020-02-01", "to": "2020-01-01"}),
    ("/api/timeseries", {"station": "IB01", "from": "not-a-date"}),
    ("/api/timeseries", {"station": "IB01", "max_points": 2001}),
    ("/api/scatter", {"station": "IB01", "model": "EtPMon"}),
    ("/api/scatter", {"station": "IB01", "model": "PM_FAO56_V1", "max_points": 3001}),
    ("/api/metrics", {"group": "arbitrary"}),
    ("/api/metrics", {"cohort": "pair"}),
    ("/api/metrics", {"model": "unapproved"}),
    ("/api/coverage", {"station": "unapproved"}),
    ("/api/observations", {"station": "IB01", "limit": 201}),
    ("/api/observations", {"station": "IB01", "offset": -1}),
])
def test_rejects_invalid_selection_and_injection(client, path, params):
    assert client.get(path, params=params).status_code == 422


def test_visual_payload_is_bounded_and_does_not_change_metrics(client):
    before = client.get("/api/metrics").json()
    data = client.get("/api/timeseries", params={"station": "IB01", "max_points": 50}).json()
    assert data["source_rows"] == 120
    assert data["returned_points"] <= 50
    assert sum(row["observations"] for row in data["points"]) == 120
    assert "visual mean" in data["aggregation"]
    assert client.get("/api/metrics").json() == before


def test_scatter_cohorts_follow_producer_flags_include_zero_and_are_bounded(client):
    own = client.get("/api/scatter", params={"station": "IB01", "model": "PM_FAO56_V1", "cohort": "own"}).json()
    common = client.get("/api/scatter", params={"station": "IB01", "model": "PM_FAO56_V1", "cohort": "common"}).json()
    assert own["total_rows"] == 120
    assert common["total_rows"] == 119
    assert own["points"][0] == dict(date="2020-01-01", reference=0.0, predicted=0.0)
    assert all(row["date"] != "2020-01-02" for row in common["points"])
    bounded = client.get("/api/scatter", params={"station": "IB01", "model": "PM_FAO56_V1", "max_points": 50}).json()
    assert bounded["returned_points"] <= 50
    assert bounded["total_rows"] == 119
    assert "deterministic" in bounded["sampling"]


def test_scatter_date_filter_and_empty_excluded_station(client):
    selected = client.get("/api/scatter", params={"station": "IB01", "model": "PM_FAO56_V1",
                                                 "cohort": "own", "from": "2020-01-02", "to": "2020-01-03"}).json()
    assert selected["total_rows"] == 2
    empty = client.get("/api/scatter", params={"station": "IB09", "model": "PM_FAO56_V1"}).json()
    assert empty["total_rows"] == 0
    assert empty["points"] == []


def test_published_metrics_are_served_verbatim_with_explicit_scope(client, derivative):
    originals = json.loads((derivative / "metrics.json").read_text())
    for cohort in ("own", "common"):
        data = client.get("/api/metrics", params={"cohort": cohort, "model": "PM_FAO56_V1"}).json()
        assert data["scope"] == "published scientific metrics"
        assert data["period_filter_applied"] is False
        metric = data["metrics"][0]
        source = next(m for m in originals if m["group"] == "global" and
                      m["cohort"] == ("own" if cohort == "own" else "PM_HS_VAL_HGRS"))
        assert {key: metric[key] for key in ("n", "mae", "rmse", "bias", "r2")} == {
            key: source[key] for key in ("n", "mae", "rmse", "bias", "r2")}
        assert metric["published_cohort"] == source["cohort"]


def test_metrics_station_year_and_unavailable_month_are_explicit(client):
    station = client.get("/api/metrics", params={"station": "IB01"}).json()
    assert station["metrics"][0]["key"] == "IB01"
    annual = client.get("/api/metrics", params={"station": "IB01", "group": "year"}).json()
    assert annual["metrics"][0]["key"] == "IB01:2020"
    for group in ("month", "season"):
        unavailable = client.get("/api/metrics", params={"station": "IB01", "group": group}).json()
        assert unavailable["metrics"] == []
        assert unavailable["unavailable_reason"]
        assert len(client.get("/api/metrics", params={"group": group}).json()["metrics"]) == 1


def test_coverage_counts_and_observation_pagination(client):
    coverage = client.get("/api/coverage", params={"station": "IB01"}).json()["stations"]
    assert len(coverage) == 1
    by_model = {row["model"]: row for row in coverage[0]["models"]}
    assert by_model["PM_FAO56_V1"]["evaluated"] == 120
    assert by_model["HS_RA_V1"]["evaluated"] == 119
    excluded = client.get("/api/coverage", params={"station": "IB09"}).json()["stations"][0]
    assert all(row["predicted"] == 0 for row in excluded["models"])
    page = client.get("/api/observations", params={"station": "IB01", "limit": 2, "offset": 1}).json()
    assert len(page["rows"]) == 2
    assert page["rows"][0]["date"] == "2020-01-02"
    assert page["rows"][0]["ET0_HS_RA_V1"] is None


@pytest.mark.parametrize("name", ["observations.parquet", "models.json", "stations.json", "metrics.json"])
def test_startup_rejects_modified_artifact(derivative, name):
    with (derivative / name).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        create_app(derivative)


@pytest.mark.parametrize("field", ["dataset_sha256", "source_sha256"])
def test_startup_rejects_wrong_source_identity(derivative, field):
    path = derivative / "info.json"
    info = json.loads(path.read_text())
    info[field] = "0" * 64
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Unapproved"):
        create_app(derivative)


def test_startup_rejects_artifact_path_traversal(derivative):
    path = derivative / "info.json"
    info = json.loads(path.read_text())
    info["artifacts"]["../observations.parquet"] = "0" * 64
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="integrity"):
        create_app(derivative)


@pytest.mark.skipif(not os.environ.get("WEB_OFFICIAL_DATA_DIR"), reason="Official derivative package not configured")
def test_official_publication_rows_and_common_metrics():
    """Opt-in integration: only reads the already packaged official publication."""
    client = ASGIClient(create_app(Path(os.environ["WEB_OFFICIAL_DATA_DIR"])))
    assert client.get("/api/health").json()["rows"] == 75496
    assert len(client.get("/api/stations").json()["stations"]) == 12
    expected = {
            "PM_FAO56_V1": (0.011540, 0.028298, -0.001565, 0.999735),
            "HS_RA_V1": (0.712843, 0.874359, 0.461627, 0.746559),
            "VALIANTZAS_V1": (0.455952, 0.608984, 0.168919, 0.877055),
            "HG_RS_V1": (0.471171, 0.565637, 0.275804, 0.893935),
    }
    metrics = client.get("/api/metrics", params={"cohort": "common"}).json()["metrics"]
    assert len(metrics) == 4
    for row in metrics:
        assert row["n"] == 70860
        assert tuple(round(row[key], 6) for key in ("mae", "rmse", "bias", "r2")) == expected[row["model"]]
    coverage = client.get("/api/coverage").json()["stations"]
    assert sum(row["rows"] for row in coverage) == 75496
    ib09 = next(row for row in coverage if row["code"] == "IB09")
    assert ib09["quality_status"] == "FAIL"
    assert all(row["evaluated"] == 0 for row in ib09["models"])
