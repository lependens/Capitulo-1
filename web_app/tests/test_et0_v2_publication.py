"""Opt-in contract checks against the actual, pinned DRAFT publication."""
import csv
import os
from pathlib import Path

import pytest

from test_api import ASGIClient, derivative
from web_app.backend.app import create_app
from web_app.backend.et0_v2 import router

CATALOG_SHA = "f497074a39d25a38d637329a87a652c470875502a2c9f796cf2eca6706148d81"


@pytest.fixture
def v2(derivative, monkeypatch):
    path = os.environ.get("TEST_ET0_V2_DIR")
    if not path:
        pytest.skip("Provide TEST_ET0_V2_DIR to verify the real publication")
    monkeypatch.setenv("WEB_ET0_V2_DIR", path)
    monkeypatch.setenv("WEB_ET0_V2_CATALOG_SHA256", CATALOG_SHA)
    monkeypatch.setenv("WEB_ET0_V2_MODE", "review")
    monkeypatch.setenv("WEB_ET0_V2_STATION_CSV", str(Path(__file__).parents[2] / "estaciones_baleares.csv"))
    yield ASGIClient(create_app(derivative)), Path(path)


def test_real_metrics_history_and_provenance(v2):
    client, root = v2
    catalog = client.get("/api/et0-v2/catalog").json()
    assert catalog["status"] == "DRAFT" and catalog["catalog_sha256"] == CATALOG_SHA
    assert len(catalog["stations"]) == len(catalog["methods"]) == 12
    with (root / "retrospective-metrics.csv").open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    assert len(source) == 9612
    for grouping, key in [("global", "all"), ("station", "IB02"), ("station_year", "IB11:2025")]:
        response = client.get("/api/et0-v2/metrics", params=dict(
            formula="HG_RS_V1", calendar="annual", group=grouping, key=key)).json()
        rows = [r for r in source if (r["family"], r["calendar"], r["group"], r["key"])
                == ("HG_RS_V1", "annual", grouping, key)]
        assert len(response["rows"]) == len(rows) == 12
        for actual, original in zip(response["rows"], rows):
            for column in ("mae", "rmse", "bias", "r2", "coverage_calendar", "coverage_pm_own"):
                assert actual[column] == (float(original[column]) if original[column] else None)
            assert actual["n"] == int(original["n"])
            assert actual["cohort_id"] == original["common_keys_sha256"]
    historic = client.get("/api/et0-v2/in-sample", params=dict(formula="HG_RS_V1", station="IB09")).json()
    assert historic["scope"] == "FULL_HISTORY_IN_SAMPLE"
    assert historic["rows"][0]["retrospective_pairs"] == 0
    with pytest.raises(ValueError, match="status"):
        router(root, CATALOG_SHA, [], mode="approved")
    with pytest.raises(ValueError, match="catalog SHA"):
        router(root, "0" * 64, [])


def test_real_series_retains_publication_values(v2):
    client, _ = v2
    params = dict(formula="HG_RS_V1", calendar="annual", station="IB01", start="2020-01-01",
                  end="2020-01-01", method=["ML_ELASTIC_NET", "ANN_A", "ANN_E"])
    response = client.get("/api/et0-v2/series", params=params)
    assert response.status_code == 200
    assert len(response.json()["points"]) == 1
    point = response.json()["points"][0]
    assert point["ML_ELASTIC_NET"] == 0.48400743995047346
    assert point["ANN_A"] == 0.854429475184935
    assert point["ANN_E"] == 0.8546901355262677
    assert client.get("/api/et0-v2/series", params=dict(params, cohort="common_core")).status_code == 422
    assert client.get("/api/et0-v2/series", params=dict(params, end="2022-01-01")).status_code == 422
    assert client.get("/api/et0-v2/series", params=dict(params, station="IB09")).json()["points"] == []


def test_comparison_matches_published_metrics(v2):
    client, _ = v2
    selection = dict(formula="HG_RS_V1", group="station", key=["IB01", "IB04"],
                     calendar=["annual", "monthly"], method=["AHC", "ANN_A"])
    response = client.get("/api/et0-v2/comparison", params=selection)
    assert response.status_code == 200
    rows = response.json()["rows"]
    expected = []
    for key in selection["key"]:
        for calendar in selection["calendar"]:
            result = client.get("/api/et0-v2/metrics", params=dict(
                selection, key=key, calendar=calendar))
            expected.extend(result.json()["rows"])
    identity = lambda row: (row["key"], row["calendar"], row["method"])
    assert len(rows) == 8
    assert sorted(rows, key=identity) == sorted(expected, key=identity)
    assert client.get("/api/et0-v2/comparison", params=dict(
        selection, key=["wrong"])).status_code == 422


def test_comparison_combines_published_families_and_years(v2):
    client, _ = v2
    families = ["HG_RS_V1", "HS_RA_V1", "VALIANTZAS_V1"]
    years = ["IB01:2020", "IB01:2025"]
    selection = dict(formula=families, group="station_year", key=years,
                     calendar=["annual"], method=["AHC"])
    response = client.get("/api/et0-v2/comparison", params=selection)
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert {(r["family"], r["key"]) for r in rows} == {(f, y) for f in families for y in years}
    for row in rows:
        original = client.get("/api/et0-v2/metrics", params=dict(
            selection, formula=row["family"], key=row["key"], calendar="annual")).json()["rows"][0]
        assert row == original
    assert {r["cohort_id"] for r in rows if r["key"] == years[0]}.isdisjoint(
        r["cohort_id"] for r in rows if r["key"] == years[1])
