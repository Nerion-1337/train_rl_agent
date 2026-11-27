"""API du pilote automatique Eagle-1 (FastAPI). Toute la logique RL est ici, côté backend.

Lancement :  uvicorn eagle1.api:app --host 0.0.0.0 --port 8000   (documentation interactive : /docs)
"""
import base64
import io
import json
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from eagle1.core import (ACTIONS, BEST_MODEL, OBS_NAMES, RESULTS, load_experiments, load_model, load_table,
                         play_episode, save_episode)

STATE = {}


@asynccontextmanager
async def lifespan(_app):
    if BEST_MODEL.exists():
        STATE["model"], STATE["meta"] = load_model()
    yield


app = FastAPI(title="Eagle-1 — API du pilote automatique", version="1.0", lifespan=lifespan,
              description="Décisions du pilote automatique (LunarLander-v3), parties complètes et statistiques.")


def _model():
    if "model" not in STATE:
        raise HTTPException(503, "Aucun modèle entraîné : lancez `python -m eagle1.train` puis `--promote`.")
    return STATE["model"]


def _records(df):
    """DataFrame -> liste de dicts JSON (les valeurs manquantes deviennent null)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def _gif_base64(frames, every=2, width=360, fps=50):
    imgs = [Image.fromarray(f) for f in frames[::every]]
    size = (width, round(imgs[0].height * width / imgs[0].width))
    buf = io.BytesIO()
    imgs = [im.resize(size) for im in imgs]
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=round(1000 * every / fps), loop=0)
    return base64.b64encode(buf.getvalue()).decode()


# --- Schémas ---------------------------------------------------------------------------------------------

class PlayRequest(BaseModel):
    state: list[float] = Field(..., min_length=8, max_length=8, description=f"Observation : {', '.join(OBS_NAMES)}",
                               examples=[[0.0, 1.4, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]])


class PlayResponse(BaseModel):
    action: int
    action_name: str
    scores: dict[str, float] = Field(description="Probabilités (PPO) ou Q-values (DQN) de chaque action")


class EpisodeRequest(BaseModel):
    seed: int | None = Field(None, description="Graine du terrain et de la position de départ (aléatoire si vide)")
    wind: bool = Field(False, description="Ajoute vent et turbulences (test de robustesse)")
    render: bool = Field(True, description="Renvoie l'animation GIF de la partie")
    save: bool = Field(True, description="Enregistre la partie pour le tableau de bord")


class EvaluateRequest(BaseModel):
    n_episodes: int = Field(20, ge=1, le=200)
    wind: bool = False
    seed_start: int | None = None


# --- Endpoints -------------------------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/model")
def model_info():
    _model()
    return STATE["meta"]


@app.post("/play", response_model=PlayResponse)
def play(req: PlayRequest):
    """Décision du pilote pour un état donné."""
    model = _model()
    obs = np.array(req.state, dtype=np.float32)
    action = int(model.predict(obs, deterministic=True)[0])
    obs_t = torch.as_tensor(obs).unsqueeze(0)
    with torch.no_grad():
        if STATE["meta"]["algo"] == "PPO":
            scores = model.policy.get_distribution(obs_t).distribution.probs[0]
        else:
            scores = model.q_net(obs_t)[0]
    return PlayResponse(action=action, action_name=ACTIONS[action],
                        scores={name: round(float(s), 4) for name, s in zip(ACTIONS, scores)})


@app.post("/episode")
def episode(req: EpisodeRequest):
    """Joue une partie complète côté serveur : résumé, télémétrie pas à pas et animation."""
    seed = req.seed if req.seed is not None else int(np.random.randint(0, 2**31 - 1))
    result = play_episode(_model(), seed=seed, wind=req.wind, render=req.render)
    episode_id = save_episode(result, source="api") if req.save else None
    return {
        "id": episode_id,
        **{k: result[k] for k in ("seed", "wind", "total_reward", "n_steps", "outcome", "success", "fuel",
                                  "main_engine", "side_engine")},
        "telemetry": result["steps"].assign(action_name=lambda d: d["action"].map(dict(enumerate(ACTIONS))))
                                    .round(4).to_dict(orient="records"),
        "gif_base64": _gif_base64(result["frames"]) if req.render else None,
    }


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    """Joue N parties sans rendu, les enregistre et renvoie les statistiques."""
    rng = np.random.default_rng(req.seed_start)
    start = req.seed_start if req.seed_start is not None else int(rng.integers(0, 2**31 - 1 - req.n_episodes))
    rewards, outcomes = [], []
    for i in range(req.n_episodes):
        result = play_episode(_model(), seed=start + i, wind=req.wind)
        save_episode(result, source="evaluation")
        rewards.append(result["total_reward"])
        outcomes.append(result["outcome"])
    rewards = np.array(rewards)
    return {"n_episodes": req.n_episodes, "wind": req.wind, "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()), "success_rate": float((rewards >= 200).mean()),
            "outcomes": {o: outcomes.count(o) for o in set(outcomes)}}


@app.get("/episodes")
def episodes():
    """Historique des parties jouées."""
    return _records(load_table("episodes"))


@app.get("/decisions")
def decisions(limit_episodes: int = 200):
    """Décisions pas à pas des dernières parties (pour l'analyse par circonstance)."""
    df = load_table("decisions")
    if df.empty:
        return []
    keep = sorted(df["episode_id"].unique())[-limit_episodes:]
    return _records(df[df["episode_id"].isin(keep)].round(4))


@app.get("/experiments")
def experiments():
    """Résultats des expériences d'entraînement (baselines et optimisation des hyperparamètres)."""
    df = load_experiments()
    return [] if df.empty else _records(df.drop(columns=["episode_rewards"]))


@app.get("/experiments/curves")
def curves():
    """Courbes d'apprentissage : récompense moyenne d'évaluation en fonction des pas d'entraînement."""
    out = []
    for f in sorted((RESULTS / "curves").glob("*/evaluations.npz")):
        data = np.load(f)
        out += [{"experiment": f.parent.name, "timesteps": int(t), "mean_reward": float(r.mean())}
                for t, r in zip(data["timesteps"], data["results"])]
    return out


@app.get("/experiments/{name}/rewards")
def experiment_rewards(name: str):
    path = RESULTS / "experiments" / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Expérience inconnue : {name}")
    return json.loads(path.read_text(encoding="utf-8"))["episode_rewards"]
