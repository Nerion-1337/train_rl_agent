<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/uv-gestion%20des%20d%C3%A9pendances-DE5FE9?logo=uv&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Jupyter-Lab-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Gymnasium-Farama-0081A5" alt="Gymnasium">
  <img src="https://img.shields.io/badge/Stable--Baselines3-DQN%20%7C%20PPO-6A1B9A" alt="Stable-Baselines3">
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Matplotlib-11557C" alt="Matplotlib">
  <img src="https://img.shields.io/badge/TensorBoard-FF6F00?logo=tensorflow&logoColor=white" alt="TensorBoard">
</p>

# Eagle-1 — Pilote automatique par apprentissage par renforcement

Projet d'école (AstroDynamics, fictif) : apprendre les bases de l'**apprentissage par renforcement** à travers
trois exercices guidés, puis développer le pilote automatique du module lunaire **Eagle-1** sur
`LunarLander-v3` (objectif : récompense moyenne **≥ 200** sur 100 épisodes), avec API, GUI et tableau de bord.

> **État d'avancement** : ✅ exercices 1 à 3 · ⏳ mission LunarLander (notebook, vidéo, API, GUI, dashboard)

Tout s'exécute dans un **container Docker** ; les dépendances et le venv sont gérés par **uv** à l'intérieur du container.

---

## Ce que fait le projet

| Notebook | Environnement | Ce qu'on apprend | Résultat |
|---|---|---|---|
| [exercice_1_cartpole_bases.ipynb](notebooks/exercice_1_cartpole_bases.ipynb) | CartPole-v1 | Espaces d'observation (`Box`) / d'action (`Discrete`), boucle `reset → step`, agent aléatoire sur 10 épisodes | Agent aléatoire : **26,5 ± 22,5** sur 10 épisodes (max 500) |
| [exercice_2_qlearning_frozenlake.ipynb](notebooks/exercice_2_qlearning_frozenlake.ipynb) | FrozenLake-v1 (glissant) | Q-learning tabulaire : Q-table 16×4, epsilon-greedy, équation de Bellman, évaluation sans exploration | **66 %** de réussite sur 100 épisodes d'évaluation |
| [exercice_3_dqn_cartpole.ipynb](notebooks/exercice_3_dqn_cartpole.ipynb) | CartPole-v1 | DQN manuel PyTorch (réseau, replay buffer, target network) puis DQN Stable-Baselines3 (défaut vs ajusté), sauvegarde `.zip` | DQN manuel ≈ 285 · SB3 défaut (25k pas) 22,5 · SB3 ajusté (50k pas) **500 ± 0** → `models/dqn_cartpole.zip` |
| *mission (à venir)* | LunarLander-v3 | Baseline, optimisation d'hyperparamètres, vidéo, API FastAPI, GUI, dashboard | — |

---

## Organisation du dépôt

```
train_rl_agent/
├── .github/workflows/ci.yml  # CI GitHub Actions : lock uv + exécution des notebooks dans Docker
├── contexte/                 # Énoncés : exercices, mission, brief PDF, livrables
├── notebooks/                # Exercices 1-3 (+ notebook de la mission finale)
│   ├── exercice_1_cartpole_bases.ipynb
│   ├── exercice_2_qlearning_frozenlake.ipynb
│   └── exercice_3_dqn_cartpole.ipynb
├── models/                   # Modèles entraînés (.zip SB3) — ex. dqn_cartpole.zip
├── logs/                     # Logs TensorBoard (générés, ignorés par git)
├── build/graph_src/          # Code des notebooks exporté en .py + graphe graphify (généré)
├── pyproject.toml            # Dépendances (uv) — torch en version CPU
├── uv.lock                   # Versions figées, reproductibles
├── Dockerfile                # python:3.11-slim + uv, venv dans /opt/venv
├── docker-compose.yml        # Services jupyter (8888) et tensorboard (6006)
└── .env                      # Token Jupyter et ports (valeurs fictives)
```

---

## Architecture d'exécution

```mermaid
flowchart LR
    subgraph Hote["Machine hôte (Windows / macOS / Linux)"]
        user(["Navigateur"])
        repo[("Dossier du projet<br/>notebooks · models · logs")]
        env[".env<br/>JUPYTER_TOKEN · ports"]
    end

    subgraph Compose["docker compose"]
        subgraph J["Service jupyter — image eagle1-autopilot"]
            uv["uv run<br/>venv /opt/venv"] --> lab["JupyterLab :8888"]
            lab --> kernel["Kernel Python 3.11<br/>Gymnasium · SB3 · PyTorch CPU"]
        end
        subgraph T["Service tensorboard — même image"]
            tb["TensorBoard :6006"]
        end
    end

    user -- "http://localhost:8888" --> lab
    user -- "http://localhost:6006" --> tb
    env -. env_file .-> J
    repo <-- "volume .:/app" --> J
    repo -- "volume .:/app (lecture logs/)" --> T
    kernel -- "écrit .zip" --> repo
    kernel -- "écrit events TensorBoard" --> repo
```

Le venv est placé **hors** du dossier monté (`/opt/venv`) : le code de l'hôte est visible en direct dans le
container sans écraser les dépendances installées à la construction de l'image.

---

## Le cycle de l'apprentissage par renforcement

```mermaid
sequenceDiagram
    participant A as Agent
    participant E as Environnement (Gymnasium)
    A->>E: reset(seed)
    E-->>A: observation s₀, info
    loop jusqu'à terminated ou truncated
        A->>A: choisir aₜ (aléatoire / ε-greedy / argmax Q)
        A->>E: step(aₜ)
        E-->>A: sₜ₊₁, rₜ₊₁, terminated, truncated, info
        A->>A: apprendre (Bellman / descente de gradient)
    end
```

---

## Diagramme UML — exercice 3 (DQN)

```mermaid
classDiagram
    direction LR
    class Module["torch.nn.Module"]
    class DQN {
        +Linear layer1 (n_obs → 128)
        +Linear layer2 (128 → 128)
        +Linear layer3 (128 → n_actions)
        +forward(x) Tensor
    }
    class ReplayBuffer {
        -deque memory
        +push(state, action, reward, next_state, done)
        +sample(batch_size) list
        +__len__() int
    }
    class BoucleEntrainement {
        <<fonctions du notebook>>
        +select_action(state, epsilon) int
        +optimize_model() float
    }
    class Env["gymnasium.Env (CartPole-v1)"] {
        +reset(seed)
        +step(action)
    }
    class SB3_DQN["stable_baselines3.DQN"] {
        +learn(total_timesteps)
        +predict(obs)
        +save(path)
        +load(path)$
    }

    Module <|-- DQN
    BoucleEntrainement --> "2" DQN : policy_net / target_net
    BoucleEntrainement --> ReplayBuffer : push / sample
    BoucleEntrainement --> Env : interagit
    SB3_DQN --> Env : interagit
    SB3_DQN ..> DQN : même principe, encapsulé
```

Fonctionnement de `optimize_model()` :

```mermaid
flowchart LR
    B[(ReplayBuffer)] -- "sample(64)" --> T["Tenseurs<br/>s, a, r, s', done"]
    T --> P["policy_net(s).gather(1, a)<br/>= Q(s,a) prédit"]
    T --> C["r + γ · target_net(s').max(1)[0] · (1-done)<br/>= Q cible (no_grad)"]
    P --> L["MSE loss"]
    C --> L
    L --> O["zero_grad → backward → clip → step"]
    O -. "tous les 10 épisodes : load_state_dict" .-> TN["target_net ← policy_net"]
```

---

## Exécution

**Prérequis** : Docker Desktop (ou Docker Engine + Compose v2). Rien d'autre à installer sur l'hôte.

1. Construire l'image (installe le venv uv à partir de `uv.lock`) :

```powershell
docker compose build
```

2. Lancer JupyterLab et TensorBoard :

```powershell
docker compose up
```

- JupyterLab : <http://localhost:8888/lab?token=eagle1> (token défini dans `.env`)
- TensorBoard : <http://localhost:6006>

3. Ouvrir les notebooks de `notebooks/` et exécuter les cellules dans l'ordre.

**Exécuter tous les exercices sans interface** (les sorties sont enregistrées dans les notebooks) :

```powershell
docker compose run --rm --no-deps jupyter uv run --no-sync jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/exercice_1_cartpole_bases.ipynb notebooks/exercice_2_qlearning_frozenlake.ipynb notebooks/exercice_3_dqn_cartpole.ipynb
```

**Ajouter une dépendance** (le lock est régénéré, puis l'image reconstruite) :

```powershell
docker compose run --rm --no-deps jupyter uv add <paquet>
```

```powershell
docker compose build
```

Arrêter : `Ctrl+C` puis `docker compose down`.

---

## Intégration continue (GitHub Actions)

Le workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) s'exécute à chaque push sur `main`,
à chaque pull request, ou à la main (*Run workflow*) :

```mermaid
flowchart LR
    push(["push / PR"]) --> lock["Job lock<br/>uv lock --check"]
    lock --> build["Job notebooks<br/>build image Docker (cache GHA)"]
    build --> run["docker compose run …<br/>nbconvert --execute notebooks/*.ipynb"]
    run --> art[("Artefact<br/>notebooks exécutés + models/")]
```

- **lock** : échoue si `uv.lock` n'est plus en phase avec `pyproject.toml`.
- **notebooks** : construit la même image que `docker compose`, exécute tous les notebooks dans le container
  (échec si une cellule lève une exception) et publie les notebooks exécutés et les modèles en artefact.

---

## Graphe de connaissances du code (graphify)

[graphify](https://github.com/safishamsi/graphify) analyse le code (AST tree-sitter, en local) et produit un
graphe navigable des fonctions, classes et dépendances. Installation (hors container, outil de dev) :

```powershell
uv tool install graphifyy
```

graphify ne lit pas les `.ipynb` : on exporte d'abord le code des notebooks en `.py`, puis on construit le graphe :

```powershell
docker compose run --rm --no-deps jupyter uv run --no-sync jupyter nbconvert --to script 'notebooks/*.ipynb' --output-dir build/graph_src
```

```powershell
graphify update build/graph_src
```

Sorties dans `build/graph_src/graphify-out/` : `graph.html` (visualisation interactive), `graph.json`, `GRAPH_REPORT.md`.
Les nœuds centraux actuels sont `ReplayBuffer` et `DQN` (exercice 3) ; le graphe s'enrichira avec le code
de la mission (API, GUI, dashboard), qui sera écrit en `.py`.
