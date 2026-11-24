"""Aperçus visuels des expériences : animations GIF d'épisodes et graphiques 3D interactifs (Plotly).

Les fichiers sont écrits dans ../media (GIF + pages HTML ouvrables dans un navigateur).
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import Image as IPyImage
from PIL import Image

# JupyterLab affiche le mimetype Plotly ; le HTML (plotly.js via CDN) sert aux autres lecteurs (nbviewer, VS Code)
pio.renderers.default = "plotly_mimetype+notebook_connected"

MEDIA = Path(__file__).resolve().parent.parent / "media"


def rollout(env, policy, seed=0, max_steps=1000, render=False):
    """Joue un épisode avec `policy(obs) -> action`.

    Renvoie (récompense totale, tableau des observations, images si render=True).
    `env` doit avoir été créé avec render_mode="rgb_array" pour render=True.
    """
    obs, _ = env.reset(seed=seed)
    states, frames, total = [obs], [env.render()] if render else [], 0.0
    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        states.append(obs)
        if render:
            frames.append(env.render())
        if terminated or truncated:
            break
    return total, np.array(states), frames


def save_gif(frames, name, fps=30, every=1, width=360):
    """Enregistre les images en GIF animé dans media/ et le renvoie pour affichage dans le notebook."""
    MEDIA.mkdir(exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames[::every]]
    height = round(imgs[0].height * width / imgs[0].width)
    imgs = [im.resize((width, height)) for im in imgs]
    path = MEDIA / name
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 * every / fps), loop=0)
    return IPyImage(filename=str(path))


def _finish(fig, name, title, labels):
    fig.update_layout(
        title=title, height=560, margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2]),
    )
    MEDIA.mkdir(exist_ok=True)
    fig.write_html(MEDIA / f"{name}.html", include_plotlyjs="cdn")
    return fig


def trajectories_3d(runs, name, title, dims=(0, 2, 3), labels=("x", "y", "z")):
    """Trajectoires dans l'espace d'état. `runs` = [(nom, tableau des observations), ...]."""
    fig = go.Figure()
    for label, states in runs:
        x, y, z = (states[:, d] for d in dims)
        fig.add_scatter3d(x=x, y=y, z=z, mode="lines", name=label, line=dict(width=4))
        fig.add_scatter3d(x=x[-1:], y=y[-1:], z=z[-1:], mode="markers", showlegend=False,
                          marker=dict(size=4, symbol="x"), hoverinfo="skip")
    return _finish(fig, name, title, labels)


def surface_3d(z, name, title, x=None, y=None, labels=("x", "y", "z"), color=None,
               colorscale="Viridis", colorbar_title=None):
    """Surface 3D z = f(x, y) ; `color` colore la surface selon une autre grandeur (ex. action choisie)."""
    fig = go.Figure(go.Surface(z=z, x=x, y=y, surfacecolor=color, colorscale=colorscale,
                               colorbar=dict(title=colorbar_title or labels[2])))
    return _finish(fig, name, title, labels)
