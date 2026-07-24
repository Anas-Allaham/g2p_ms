"""Exercise generation + adaptive next-selection endpoints."""


def test_stateless_generate_from_metrics(client):
    r = client.post("/api/v1/exercises/generate", json={"metrics": {}})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["exercise"] is not None
    assert data["assessment"]["assessment_source"] == "stateless_raw_metrics"


def test_next_cold_start_is_diagnostic(client):
    client.put("/api/v1/subjects/ex-cold")
    r = client.post(
        "/api/v1/subjects/ex-cold/exercises/next",
        headers={"Idempotency-Key": "cold-1"},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["mode"] == "diagnostic"
    assert data["diagnostic"]["in_diagnostic"] is True
    assert data["sentence_id"]


def test_next_requires_idempotency_key(client):
    client.put("/api/v1/subjects/ex-nokey")
    r = client.post("/api/v1/subjects/ex-nokey/exercises/next")
    assert r.status_code == 422


def test_next_idempotent_retry_replays_same_assignment(client):
    client.put("/api/v1/subjects/ex-idem")
    a = client.post("/api/v1/subjects/ex-idem/exercises/next", headers={"Idempotency-Key": "k"})
    b = client.post("/api/v1/subjects/ex-idem/exercises/next", headers={"Idempotency-Key": "k"})
    assert a.json()["data"]["sentence_id"] == b.json()["data"]["sentence_id"]
