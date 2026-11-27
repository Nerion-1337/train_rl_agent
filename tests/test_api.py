"""Tests de l'API Eagle-1 (lancement : pytest -q)."""
import pytest
from fastapi.testclient import TestClient

from eagle1 import core
from eagle1.api import app

pytestmark = pytest.mark.skipif(not core.BEST_MODEL.exists(), reason="pas de modèle entraîné")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # base de test isolée : ne pollue pas results/eagle1.db
    core.DB_PATH = tmp_path_factory.mktemp("db") / "test.db"
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_play_returns_valid_action(client):
    r = client.post("/play", json={"state": [0.0, 1.4, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]})
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in range(4)
    assert set(body["scores"]) == set(core.ACTIONS)


def test_play_rejects_bad_state(client):
    assert client.post("/play", json={"state": [0.0, 1.0]}).status_code == 422


def test_episode_is_played_and_saved(client):
    r = client.post("/episode", json={"seed": 42, "render": True})
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] in {"atterrissage", "crash", "temps écoulé"}
    assert len(body["telemetry"]) == body["n_steps"]
    assert body["gif_base64"]
    assert any(e["id"] == body["id"] for e in client.get("/episodes").json())


def test_model_reaches_target(client):
    """Cahier des charges : récompense moyenne ≥ 200 (ici sur 20 parties pour rester rapide)."""
    r = client.post("/evaluate", json={"n_episodes": 20, "seed_start": 123})
    assert r.json()["mean_reward"] >= 200
