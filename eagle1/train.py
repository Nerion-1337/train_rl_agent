"""Entraînement et évaluation des expériences (baselines, optimisation des hyperparamètres, modèle final).

Chaque expérience ne change **qu'un seul** hyperparamètre par rapport à sa référence, sauf les configurations
« final » qui combinent les changements gagnants. Exemples (dans le container) :

    python -m eagle1.train --list
    python -m eagle1.train --only dqn_baseline,ppo_baseline
    python -m eagle1.train --round 1 --workers 12
    python -m eagle1.train --promote ppo_final_s1     # devient models/eagle1_best.zip
"""
import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool

import numpy as np

from eagle1.core import ALGOS, BEST_MODEL, ENV_ID, LOGS, MODELS, RESULTS

N_EVAL_EPISODES = 100
EVAL_SEED = 10_000          # graines jamais vues à l'entraînement : mesure de généralisation


@dataclass
class Experiment:
    name: str
    algo: str
    changed: str                     # hyperparamètre modifié par rapport à la référence
    params: dict = field(default_factory=dict)
    timesteps: int = 1_000_000
    n_envs: int = 8
    seed: int = 0
    round: int = 1
    wind: bool = False               # entraînement avec vent et turbulences (robustesse)


PPO_REF = {}                         # PPO par défaut de SB3 (lr 3e-4, gamma 0.99, n_steps 2048, ...)
PPO_FINAL = dict(n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01)
DQN_TUNED = dict(learning_rate=6.3e-4, batch_size=128, buffer_size=50_000, learning_starts=0, gamma=0.99,
                 target_update_interval=250, train_freq=4, gradient_steps=-1, exploration_fraction=0.12,
                 exploration_final_eps=0.1, policy_kwargs=dict(net_arch=[256, 256]))

EXPERIMENTS = [
    # Étape 1 — baselines : paramètres par défaut de SB3
    Experiment("dqn_baseline", "DQN", "défaut SB3", {}, 300_000, 1),
    Experiment("ppo_baseline", "PPO", "défaut SB3", PPO_REF),
    # Étape 2 — DQN : un hyperparamètre à la fois
    Experiment("dqn_buffer_50k", "DQN", "buffer_size=50 000", dict(buffer_size=50_000), 300_000, 1),
    Experiment("dqn_lr_6e-4", "DQN", "learning_rate=6.3e-4", dict(learning_rate=6.3e-4), 300_000, 1),
    Experiment("dqn_target_250", "DQN", "target_update_interval=250", dict(target_update_interval=250), 300_000, 1),
    Experiment("dqn_tuned", "DQN", "combinaison RL Zoo", DQN_TUNED, 300_000, 1),
    # Étape 2 — PPO : un hyperparamètre à la fois
    Experiment("ppo_gamma_0.999", "PPO", "gamma=0.999", dict(gamma=0.999)),
    Experiment("ppo_nsteps_1024", "PPO", "n_steps=1024", dict(n_steps=1024)),
    Experiment("ppo_lr_1e-3", "PPO", "learning_rate=1e-3", dict(learning_rate=1e-3)),
    Experiment("ppo_ent_0.01", "PPO", "ent_coef=0.01", dict(ent_coef=0.01)),
    Experiment("ppo_gae_0.98", "PPO", "gae_lambda=0.98", dict(gae_lambda=0.98)),
    Experiment("ppo_epochs_4", "PPO", "n_epochs=4", dict(n_epochs=4)),
    # Étape 2 bis — configuration finale (combinaison des gains), 3 graines pour mesurer la stabilité
    *[Experiment(f"ppo_final_s{s}", "PPO", "combinaison finale", PPO_FINAL, 2_000_000, 16, s, round=2)
      for s in range(3)],
    # Étape 3 — robustesse : même configuration, entraînée avec vent et turbulences
    Experiment("ppo_final_wind", "PPO", "entraînement avec vent", PPO_FINAL, 2_000_000, 16, 0, round=3, wind=True),
]
BY_NAME = {e.name: e for e in EXPERIMENTS}


def run(exp: Experiment):
    """Entraîne, évalue sur 100 épisodes inédits et sauvegarde modèle + courbes + résultats."""
    import torch
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    torch.set_num_threads(1)                         # plusieurs expériences tournent en parallèle
    out_model = MODELS / "experiments" / f"{exp.name}.zip"
    curves = RESULTS / "curves" / exp.name
    for d in (out_model.parent, curves, RESULTS / "experiments"):
        d.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(ENV_ID, n_envs=exp.n_envs, seed=exp.seed, env_kwargs=dict(enable_wind=exp.wind))
    model = ALGOS[exp.algo]("MlpPolicy", env, seed=exp.seed, verbose=0, device="cpu",
                            tensorboard_log=str(LOGS / "mission"), **exp.params)
    # Évaluation périodique (10 épisodes tous les ~25k pas) -> courbe d'apprentissage evaluations.npz
    eval_cb = EvalCallback(make_vec_env(ENV_ID, n_envs=1, seed=EVAL_SEED + 500), n_eval_episodes=10,
                           eval_freq=max(25_000 // exp.n_envs, 1), log_path=str(curves), verbose=0)
    start = time.time()
    model.learn(exp.timesteps, callback=eval_cb, tb_log_name=exp.name)
    train_minutes = (time.time() - start) / 60
    model.save(out_model)

    rewards, lengths = evaluate_policy(model, make_vec_env(ENV_ID, n_envs=1, seed=EVAL_SEED),
                                       n_eval_episodes=N_EVAL_EPISODES, return_episode_rewards=True)
    rewards = np.array(rewards)
    # robustesse : 100 épisodes avec vent et turbulences (condition absente de l'entraînement standard)
    wind_rewards, _ = evaluate_policy(model, make_vec_env(ENV_ID, n_envs=1, seed=EVAL_SEED + 1000,
                                                          env_kwargs=dict(enable_wind=True)),
                                      n_eval_episodes=N_EVAL_EPISODES, return_episode_rewards=True)
    result = {
        **{k: v for k, v in asdict(exp).items() if k != "params"},
        "params": json.dumps(exp.params),
        "mean_reward": float(rewards.mean()), "std_reward": float(rewards.std()),
        "min_reward": float(rewards.min()), "success_rate": float((rewards >= 200).mean()),
        "mean_length": float(np.mean(lengths)), "train_minutes": round(train_minutes, 1),
        "wind_mean_reward": float(np.mean(wind_rewards)), "wind_success_rate": float(np.mean(np.array(wind_rewards) >= 200)),
        "episode_rewards": rewards.round(2).tolist(),
    }
    (RESULTS / "experiments" / f"{exp.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{exp.name:18s} {result['mean_reward']:8.1f} ± {result['std_reward']:6.1f}  "
          f"succès {result['success_rate']:.0%}  ({train_minutes:.1f} min)", flush=True)
    return result


def wind_eval(name):
    """Ajoute l'évaluation « vent et turbulences » au résultat d'une expérience déjà entraînée."""
    import torch
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    torch.set_num_threads(1)
    path = RESULTS / "experiments" / f"{name}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    model = ALGOS[result["algo"]].load(MODELS / "experiments" / f"{name}.zip", device="cpu")
    rewards, _ = evaluate_policy(model, make_vec_env(ENV_ID, n_envs=1, seed=EVAL_SEED + 1000,
                                                     env_kwargs=dict(enable_wind=True)),
                                 n_eval_episodes=N_EVAL_EPISODES, return_episode_rewards=True)
    result.update(wind_mean_reward=float(np.mean(rewards)), wind_success_rate=float(np.mean(np.array(rewards) >= 200)))
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{name:18s} vent : {result['wind_mean_reward']:7.1f}  succès {result['wind_success_rate']:.0%}", flush=True)


def promote(name):
    """Copie le modèle d'une expérience comme modèle de production (models/eagle1_best.zip + .json)."""
    result = json.loads((RESULTS / "experiments" / f"{name}.json").read_text(encoding="utf-8"))
    shutil.copy(MODELS / "experiments" / f"{name}.zip", BEST_MODEL)
    meta = {k: result[k] for k in ("name", "algo", "changed", "params", "timesteps", "n_envs", "seed",
                                   "mean_reward", "std_reward", "success_rate")}
    BEST_MODEL.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Modèle de production : {name} ({result['mean_reward']:.1f} ± {result['std_reward']:.1f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="lister les expériences")
    parser.add_argument("--only", help="noms séparés par des virgules")
    parser.add_argument("--round", type=int, help="lancer toutes les expériences d'une étape (1, 2 ou 3)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--promote", help="expérience à promouvoir en modèle de production")
    parser.add_argument("--wind-eval", action="store_true", help="ajouter l'évaluation avec vent aux expériences existantes")
    args = parser.parse_args()

    if args.list:
        for e in EXPERIMENTS:
            print(f"[{e.round}] {e.name:18s} {e.algo}  {e.changed:28s} {e.timesteps:>9,} pas  {e.n_envs} env(s)")
    elif args.promote:
        promote(args.promote)
    elif args.wind_eval:
        names = args.only.split(",") if args.only else [f.stem for f in (RESULTS / "experiments").glob("*.json")]
        with Pool(min(args.workers, len(names))) as pool:
            pool.map(wind_eval, names)
    else:
        todo = [BY_NAME[n] for n in args.only.split(",")] if args.only else \
               [e for e in EXPERIMENTS if args.round in (None, e.round)]
        # les plus longues d'abord pour équilibrer les workers
        todo.sort(key=lambda e: -e.timesteps / e.n_envs)
        with Pool(min(args.workers, len(todo))) as pool:
            pool.map(run, todo)


if __name__ == "__main__":
    main()
