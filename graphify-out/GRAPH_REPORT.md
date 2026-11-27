# Graph Report - train_rl_agent  (2026-11-27)

## Corpus Check
- 37 files · ~15,622 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 46 file(s) not represented in the graph (top: .zip 17, .npz 16, (none) 4)
- WARNING: 46 of 101 files seen (46%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 250 nodes · 363 edges · 17 communities (11 shown, 6 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2fdccfbb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- api.py
- dashboard
- core.py
- with
- dashboard.py
- build_report.py
- video.py
- rl_viz.py
- test_api.py
- Eagle-1 — Pilote automatique par apprentissage par renforcement
- api
- eagle1-autopilot

## God Nodes (most connected - your core abstractions)
1. `Eagle-1 — Pilote automatique par apprentissage par renforcement` - 12 edges
2. `get()` - 11 edges
3. `play_episode()` - 10 edges
4. `with` - 9 edges
5. `jupyter` - 8 edges
6. `api` - 8 edges
7. `gui` - 8 edges
8. `dashboard` - 8 edges
9. `play()` - 8 edges
10. `episode()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Livrables et cahier des charges` --references--> `test_model_reaches_target()`  [INFERRED]
  README.md → tests/test_api.py
- `lifespan()` --calls--> `load_model()`  [EXTRACTED]
  eagle1/api.py → eagle1/core.py
- `health()` --references--> `get()`  [EXTRACTED]
  eagle1/api.py → eagle1/dashboard.py
- `episode()` --calls--> `play_episode()`  [EXTRACTED]
  eagle1/api.py → eagle1/core.py
- `episode()` --calls--> `save_episode()`  [EXTRACTED]
  eagle1/api.py → eagle1/core.py

## Import Cycles
- None detected.

## Communities (17 total, 6 thin omitted)

### Community 0 - "api.py"
Cohesion: 0.10
Nodes (18): curves(), decisions(), episode(), EpisodeRequest, episodes(), evaluate(), EvaluateRequest, experiment_rewards() (+10 more)

### Community 1 - "dashboard"
Cohesion: 0.06
Nodes (35): condition, command, depends_on, environment, image, ports, restart, volumes (+27 more)

### Community 2 - "core.py"
Cohesion: 0.10
Nodes (12): curve(), fuel_cost(), load_experiments(), load_table(), make_env(), play_episode(), save_episode(), Experiment (+4 more)

### Community 3 - "with"
Cohesion: 0.09
Nodes (24): jobs, lock, notebooks, name, runs-on, steps, name, needs (+16 more)

### Community 4 - "dashboard.py"
Cohesion: 0.11
Nodes (10): circumstances(), concurrency, cancel-in-progress, group, name, on, pull_request, push (+2 more)

### Community 5 - "build_report.py"
Cohesion: 0.12
Nodes (3): bullets(), figure(), P()

### Community 6 - "video.py"
Cohesion: 0.17
Nodes (6): lifespan(), load_model(), main(), main(), make_gif(), pick_seeds()

### Community 7 - "rl_viz.py"
Cohesion: 0.16
Nodes (5): _finish(), rollout(), save_gif(), surface_3d(), trajectories_3d()

### Community 8 - "test_api.py"
Cohesion: 0.13
Nodes (3): Livrables et cahier des charges, client(), test_model_reaches_target()

### Community 9 - "Eagle-1 — Pilote automatique par apprentissage par renforcement"
Cohesion: 0.14
Nodes (14): API, Architecture, Diagramme UML du package `eagle1`, Eagle-1 — Pilote automatique par apprentissage par renforcement, Entraînement et optimisation, Exercices, Exécution, Graphe de connaissances du code (graphify) (+6 more)

### Community 10 - "api"
Cohesion: 0.17
Nodes (12): command, depends_on, healthcheck, image, ports, restart, volumes, interval (+4 more)

## Knowledge Gaps
- **1 isolated node(s):** `eagle1-autopilot`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 142 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `jobs` connect `with` to `dashboard.py`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **What connects `eagle1-autopilot` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `api.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._
- **Why does `Eagle-1 — Pilote automatique par apprentissage par renforcement` connect `Eagle-1 — Pilote automatique par apprentissage par renforcement` to `test_api.py`, `dashboard.py`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Should `dashboard` be split into smaller, more focused modules?**
  _Cohesion score 0.05873015873015873 - nodes in this community are weakly interconnected._
- **Should `core.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10338680926916222 - nodes in this community are weakly interconnected._
- **Should `with` be split into smaller, more focused modules?**
  _Cohesion score 0.09057971014492754 - nodes in this community are weakly interconnected._