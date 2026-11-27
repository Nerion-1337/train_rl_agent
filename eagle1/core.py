"""Briques communes du pilote automatique Eagle-1 : environnement, modèle, épisodes instrumentés, stockage."""
import json
import sqlite3
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

ENV_ID = "LunarLander-v3"
ACTIONS = ["Rien", "Moteur gauche", "Moteur principal", "Moteur droit"]
OBS_NAMES = ["x", "y", "vx", "vy", "angle", "vitesse_angulaire", "jambe_gauche", "jambe_droite"]
ALGOS = {"PPO": PPO, "DQN": DQN}

ROOT = Path(__file__).resolve().parent.parent
MODELS, RESULTS, MEDIA, LOGS = (ROOT / d for d in ("models", "results", "media", "logs"))
BEST_MODEL = MODELS / "eagle1_best.zip"
DB_PATH = RESULTS / "eagle1.db"


def make_env(render_mode=None, wind=False):
    """LunarLander-v3 (actions discrètes). `wind=True` ajoute vent et turbulences : test de généralisation."""
    return gym.make(ENV_ID, render_mode=render_mode, enable_wind=wind)


def load_model(path=BEST_MODEL):
    """Charge un modèle SB3 ; l'algorithme est lu dans le fichier .json associé."""
    meta = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    return ALGOS[meta["algo"]].load(path, device="cpu"), meta


def fuel_cost(actions):
    """Carburant consommé, dans l'unité de pénalité de l'environnement (0.3 / pas moteur principal, 0.03 latéral)."""
    actions = np.asarray(actions)
    return 0.3 * np.sum(actions == 2) + 0.03 * np.sum((actions == 1) | (actions == 3))


def play_episode(model, seed=None, wind=False, render=False):
    """Joue un épisode avec la politique déterministe du modèle et renvoie toutes les mesures.

    Issue : « atterrissage » (module posé et immobile, +100), « crash » (-100 : écrasé ou sorti de l'écran),
    « temps écoulé » (1000 pas sans se poser).
    """
    env = make_env("rgb_array" if render else None, wind)
    obs, _ = env.reset(seed=seed)
    rows, frames = [], [env.render()] if render else []
    terminated = truncated = False
    reward = 0.0
    while not (terminated or truncated):
        action = int(model.predict(obs, deterministic=True)[0])
        new_obs, reward, terminated, truncated, _ = env.step(action)
        rows.append([*obs.tolist(), action, float(reward)])
        obs = new_obs
        if render:
            frames.append(env.render())
    env.close()

    steps = pd.DataFrame(rows, columns=[*OBS_NAMES, "action", "reward"])
    outcome = "temps écoulé" if truncated else ("atterrissage" if reward == 100 else "crash")
    total = float(steps["reward"].sum())
    return {
        "seed": seed, "wind": wind, "total_reward": total, "n_steps": len(steps), "outcome": outcome,
        "success": total >= 200, "fuel": float(fuel_cost(steps["action"])),
        "main_engine": int((steps["action"] == 2).sum()),
        "side_engine": int(steps["action"].isin([1, 3]).sum()),
        "steps": steps, "frames": frames,
    }


# --- Stockage des parties jouées (SQLite) : alimente le tableau de bord ---------------------------------

def save_episode(result, source):
    """Enregistre une partie (résumé + décisions pas à pas). Renvoie son identifiant."""
    RESULTS.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        summary = {k: result[k] for k in ("seed", "wind", "total_reward", "n_steps", "outcome", "success",
                                          "fuel", "main_engine", "side_engine")}
        summary.update(source=source, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        con.execute("CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY AUTOINCREMENT, seed INTEGER, "
                    "wind INTEGER, total_reward REAL, n_steps INTEGER, outcome TEXT, success INTEGER, fuel REAL, "
                    "main_engine INTEGER, side_engine INTEGER, source TEXT, timestamp TEXT)")
        cur = con.execute(f"INSERT INTO episodes ({', '.join(summary)}) VALUES ({', '.join('?' * len(summary))})",
                          list(summary.values()))
        episode_id = cur.lastrowid
        steps = result["steps"].assign(episode_id=episode_id, step=range(len(result["steps"])))
        steps.to_sql("decisions", con, if_exists="append", index=False)
    return episode_id


def load_table(name):
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as con:
        try:
            return pd.read_sql(f"SELECT * FROM {name}", con)
        except pd.errors.DatabaseError:
            return pd.DataFrame()


def load_experiments():
    """Résultats de toutes les expériences d'entraînement (un .json par expérience)."""
    files = sorted((RESULTS / "experiments").glob("*.json"))
    return pd.DataFrame([json.loads(f.read_text(encoding="utf-8")) for f in files])
