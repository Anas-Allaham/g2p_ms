"""FastAPI contract: auth, envelopes, request ids, media types, OpenAPI."""

from conftest import AUTH

# The client fixture sets a default Authorization header; send an empty one to
# exercise the unauthenticated path.
NO_AUTH = {"Authorization": ""}


def test_health_live_is_public_and_enveloped(client):
    r = client.get("/health/live", headers=NO_AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["status"] == "alive"
    assert body["meta"]["api_version"] == "v1"
    assert body["meta"]["request_id"]


def test_health_ready_reports_checks(client):
    r = client.get("/health/ready", headers=NO_AUTH)
    assert r.status_code == 200  # model mocked present + bank seeded in the fixture
    checks = r.json()["data"]["checks"]
    assert checks["exercise_bank_populated"] is True
    assert checks["database_ok"] is True


def test_api_requires_bearer_token(client):
    assert client.post("/api/v1/g2p", json={"text": "hi"}, headers=NO_AUTH).status_code == 401
    bad = client.post("/api/v1/g2p", json={"text": "hi"}, headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401
    assert bad.json()["error"]["code"] == "unauthorized"


def test_success_envelope_shape_and_request_id_header(client):
    r = client.post("/api/v1/g2p", json={"text": "school"})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"data", "meta"}
    assert body["data"]["arpabet"]
    assert "ipa" not in body["data"]
    assert r.headers["X-Request-ID"] == body["meta"]["request_id"]


def test_incoming_request_id_is_echoed(client):
    r = client.post("/api/v1/g2p", json={"text": "hi"}, headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.json()["meta"]["request_id"] == "abc123"


def test_validation_error_envelope(client):
    r = client.post("/api/v1/g2p", json={}, headers=AUTH)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_unsupported_media_type_on_analysis(client):
    r = client.post(
        "/api/v1/pronunciation/analyses",
        data={"text": "school"},
        files={"audio": ("x.txt", b"not audio", "text/plain")},
    )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"


def test_openapi_is_generated(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    for p in (
        "/api/v1/g2p",
        "/api/v1/subjects/{subject_id}",
        "/api/v1/pronunciation/analyses",
        "/api/v1/subjects/{subject_id}/analyses",
        "/api/v1/exercises/generate",
        "/api/v1/subjects/{subject_id}/exercises/next",
        "/api/v1/capabilities",
        "/health/ready",
    ):
        assert p in paths

    analysis_schema = schema["components"]["schemas"]["AnalysisData"]
    assert "pronunciation_errors" in analysis_schema["properties"]
    assert "PronunciationError" in schema["components"]["schemas"]
    assert "ReferenceSpan" in schema["components"]["schemas"]


def test_capabilities_exposes_trust_and_limits(client):
    data = client.get("/api/v1/capabilities").json()["data"]
    assert "scoring" in data and "engine" in data["scoring"]
    assert data["g2p"]["alphabet"] == "arpabet"
    assert data["limits"]["language"] == "en"
    assert data["scientific_limitations"]["provisional"] is True
