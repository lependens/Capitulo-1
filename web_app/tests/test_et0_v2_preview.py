from web_app.backend.app import create_app
from test_api import ASGIClient
from test_api import derivative


def test_preview_requires_explicit_activation(derivative, monkeypatch):
    monkeypatch.delenv("WEB_ET0_V2_PREVIEW", raising=False)
    app = create_app(derivative)
    assert ASGIClient(app).get("/api/et0-v2/catalog").status_code == 404
    monkeypatch.setenv("WEB_ET0_V2_PREVIEW", "1")
    client = ASGIClient(create_app(derivative))
    assert client.get("/api/et0-v2/catalog").json()["mode"] == "synthetic"


def test_preview_scopes_and_filtering(derivative, monkeypatch):
    monkeypatch.setenv("WEB_ET0_V2_PREVIEW", "1")
    client = ASGIClient(create_app(derivative))
    base = "/api/et0-v2"
    context = dict(formula="HG_RS_V1", calendar="annual")
    params = dict(**context, group="year", key="2020", method=["ANN_A", "ANN_E"])
    earlier = client.get(base + "/metrics", params=params).json()
    later = client.get(base + "/metrics", params={**params, "key": "2021"}).json()
    assert [row["method"] for row in earlier["rows"]] == ["ANN_A", "ANN_E"]
    assert [row["method"] for row in later["rows"]] == ["ANN_A"]
    assert earlier["rows"][0]["cohort_id"] != later["rows"][0]["cohort_id"]
    assert client.get(base + "/metrics", params={**params, "key": "DEMO-01"}).status_code == 422
    series = client.get(base + "/series", params=dict(**context, station="DEMO-01",
                        start="2020-07-06", end="2020-07-06")).json()
    assert len(series["points"]) == 1 and series["points"][0]["ML"] is None
    history = client.get(base + "/in-sample", params=dict(formula="HG_RS_V1", station="DEMO-01")).json()
    assert history["scope"] == "FULL_HISTORY_IN_SAMPLE"
    assert "cohort_id" not in history["rows"][0]
