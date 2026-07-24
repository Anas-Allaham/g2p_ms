"""Subject lifecycle, read models, and validation."""


def test_put_subject_is_idempotent(client):
    r1 = client.put("/api/v1/subjects/subj-1")
    assert r1.status_code == 201
    assert r1.json()["data"]["created"] is True
    r2 = client.put("/api/v1/subjects/subj-1")
    assert r2.status_code == 200
    assert r2.json()["data"]["created"] is False


def test_bad_subject_id_is_rejected(client):
    assert client.put("/api/v1/subjects/has space").status_code == 422
    assert client.put("/api/v1/subjects/" + "x" * 200).status_code == 422


def test_reads_on_unknown_subject_are_empty_not_404(client):
    assert client.get("/api/v1/subjects/never-seen/assessment").json()["data"]["assessment"] is None
    assert client.get("/api/v1/subjects/never-seen/gaps").json()["data"]["phonemes"] == []
    hist = client.get("/api/v1/subjects/never-seen/attempts").json()["data"]
    assert hist["attempts"] == [] and hist["has_more"] is False


def test_delete_returns_flag(client):
    client.put("/api/v1/subjects/subj-del")
    assert client.delete("/api/v1/subjects/subj-del").json()["data"]["deleted"] is True
    assert client.delete("/api/v1/subjects/subj-del").json()["data"]["deleted"] is False


def test_attempts_pagination(client, sample_wav):
    subject = "subj-page"
    client.put(f"/api/v1/subjects/{subject}")
    # Record several attempts (real audio path, mocked transcription).
    for i in range(5):
        r = client.post(
            f"/api/v1/subjects/{subject}/analyses",
            data={"text": "school"},
            files={"audio": (f"r{i}.wav", sample_wav, "audio/wav")},
            headers={"Idempotency-Key": f"page-{i}"},
        )
        assert r.status_code == 200

    first = client.get(f"/api/v1/subjects/{subject}/attempts?limit=2").json()["data"]
    assert len(first["attempts"]) == 2
    assert first["has_more"] is True and first["next_cursor"]

    second = client.get(
        f"/api/v1/subjects/{subject}/attempts?limit=2&cursor={first['next_cursor']}"
    ).json()["data"]
    assert len(second["attempts"]) == 2
    # Pages are disjoint and strictly descending by id.
    ids1 = [a["id"] for a in first["attempts"]]
    ids2 = [a["id"] for a in second["attempts"]]
    assert set(ids1).isdisjoint(ids2)
    assert min(ids1) > max(ids2)
