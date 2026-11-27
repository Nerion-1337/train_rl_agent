"""Génère docs/rapport_eagle1.pdf : démarche, résultats, interprétation et notions essentielles du RL.

Lancement (dans le container) :  python docs/build_report.py
Les chiffres sont relus dans results/ : le rapport reste cohérent après un réentraînement.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
RESULTS, MEDIA, DOCS = ROOT / "results", ROOT / "media", ROOT / "docs"
FIG = DOCS / "figures"
OUT = DOCS / "rapport_eagle1.pdf"

# --- Police : DejaVu (livrée avec matplotlib) pour les accents, γ, θ, ε, ≥ ----------------------------------
for name, file in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                   ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"), ("DejaVuMono", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / file)))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                              boldItalic="DejaVu-Bold")

BLUE, ORANGE, GRAY = "#4C72B0", "#DD8452", "#6b7280"
ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="DejaVu-Bold", fontSize=24, leading=30),
    "subtitle": ParagraphStyle("st", fontName="DejaVu", fontSize=12, leading=16, alignment=TA_CENTER,
                               textColor=colors.HexColor(GRAY)),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=16, leading=20, spaceBefore=6, spaceAfter=8,
                         textColor=colors.HexColor("#1f3a68")),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12, leading=15, spaceBefore=8, spaceAfter=4,
                         textColor=colors.HexColor("#1f3a68")),
    "p": ParagraphStyle("p", fontName="DejaVu", fontSize=9.5, leading=13.5, spaceAfter=5),
    "small": ParagraphStyle("s", fontName="DejaVu", fontSize=8, leading=10.5, textColor=colors.HexColor(GRAY)),
    "box": ParagraphStyle("b", fontName="DejaVu", fontSize=9.5, leading=13.5, backColor=colors.HexColor("#eef3fb"),
                          borderColor=colors.HexColor("#c9d6ee"), borderWidth=0.6, borderPadding=7,
                          spaceBefore=6, spaceAfter=10),
    "cell": ParagraphStyle("c", fontName="DejaVu", fontSize=8, leading=10),
    "cellb": ParagraphStyle("cb", fontName="DejaVu-Bold", fontSize=8, leading=10, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="DejaVuMono", fontSize=8, leading=10.5,
                           backColor=colors.HexColor("#f4f4f5"), borderPadding=5, spaceBefore=4, spaceAfter=8),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def P(text, style="p"):
    return Paragraph(text, S[style])


def bullets(items):
    return [P(f"•&nbsp;&nbsp;{it}") for it in items]


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), S["cellb" if header and i == 0 else "cell"]) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d4d4d8")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a68")))
    t.setStyle(TableStyle(style))
    return t


def figure(path, width_cm, caption=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, P(caption, "small")]) if caption else img


# --- Données ---------------------------------------------------------------------------------------------
exps = {json.loads(f.read_text(encoding="utf-8"))["name"]: json.loads(f.read_text(encoding="utf-8"))
        for f in sorted((RESULTS / "experiments").glob("*.json"))}
prod = json.loads((ROOT / "models" / "eagle1_best.json").read_text(encoding="utf-8"))
final_eval = json.loads((RESULTS / "final_evaluation.json").read_text(encoding="utf-8"))
nominal, windy = np.array(final_eval["nominal_rewards"]), np.array(final_eval["wind_rewards"])
wind_model = exps.get("ppo_final_wind")


def curve(name):
    d = np.load(RESULTS / "curves" / name / "evaluations.npz")
    return d["timesteps"], d["results"].mean(axis=1)


def e(name, key="mean_reward", fmt="{:.1f}"):
    return fmt.format(exps[name][key])


# --- Figures ---------------------------------------------------------------------------------------------
FIG.mkdir(parents=True, exist_ok=True)

round1 = sorted([x for x in exps.values() if x["round"] == 1], key=lambda x: x["mean_reward"])
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.barh([x["name"] for x in round1], [x["mean_reward"] for x in round1], xerr=[x["std_reward"] for x in round1],
        color=[BLUE if x["algo"] == "DQN" else ORANGE for x in round1], capsize=2, alpha=0.9)
ax.axvline(200, ls="--", c="gray", lw=1)
ax.set_xlabel("récompense moyenne sur 100 épisodes (trait = écart-type)")
ax.set_title("Un hyperparamètre à la fois — bleu : DQN, orange : PPO")
fig.tight_layout(); fig.savefig(FIG / "experiences.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, names, title in ((axes[0], ["dqn_baseline", "dqn_buffer_50k", "dqn_tuned"], "DQN"),
                         (axes[1], ["ppo_baseline", "ppo_lr_1e-3", "ppo_final_s0", "ppo_final_s1", "ppo_final_s2"], "PPO")):
    for n in names:
        if n in exps:
            ax.plot(*curve(n), label=n, lw=1.3)
    ax.axhline(200, ls="--", c="gray", lw=1); ax.set_title(title); ax.set_xlabel("pas d'entraînement")
    ax.set_ylim(-500, 330); ax.legend(fontsize=7)
axes[0].set_ylabel("récompense (évaluation 10 épisodes)")
fig.tight_layout(); fig.savefig(FIG / "courbes.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].hist(nominal, bins=20, color=ORANGE, alpha=0.85, label="nominal")
axes[0].hist(np.clip(windy, -350, None), bins=20, color=BLUE, alpha=0.55, label="vent (< −350 regroupés)")
axes[0].axvline(200, ls="--", c="gray"); axes[0].set_xlabel("récompense"); axes[0].set_ylabel("épisodes")
axes[0].legend(); axes[0].set_title("Distribution (100 épisodes chacune)")
axes[1].plot(nominal, "o", ms=3, c=ORANGE); axes[1].plot(np.cumsum(nominal) / np.arange(1, 101), c="k", lw=1.2,
                                                         label="moyenne cumulée")
axes[1].axhline(200, ls="--", c="gray"); axes[1].set_xlabel("épisode"); axes[1].set_title("Stabilité, conditions nominales")
axes[1].legend()
fig.tight_layout(); fig.savefig(FIG / "evaluation_finale.png"); plt.close(fig)

LABELLED = {"dqn_baseline", "dqn_tuned", "ppo_baseline", "ppo_lr_1e-3", "ppo_ent_0.01", "ppo_final_s0",
            "ppo_final_wind", "ppo_epochs_4"}
fig, ax = plt.subplots(figsize=(6.5, 4.6))
for x in exps.values():
    if "wind_mean_reward" in x:
        c = BLUE if x["algo"] == "DQN" else ORANGE
        ax.scatter(x["mean_reward"], x["wind_mean_reward"], c=c, s=30)
        if x["name"] in LABELLED:
            ax.annotate(x["name"], (x["mean_reward"], x["wind_mean_reward"]), fontsize=6.5, xytext=(4, -3),
                        textcoords="offset points")
ax.axvline(200, ls="--", c="gray", lw=1); ax.axhline(200, ls="--", c="gray", lw=1)
ax.set_xlabel("récompense — conditions nominales"); ax.set_ylabel("récompense — avec vent")
ax.set_title("Performance vs robustesse")
fig.tight_layout(); fig.savefig(FIG / "robustesse.png"); plt.close(fig)

# images de l'environnement (première image des GIF / vidéo)
from PIL import Image as PILImage  # noqa: E402

for gif, png in (("ex1_cartpole_aleatoire.gif", "cartpole.png"), ("ex2_frozenlake_agent.gif", "frozenlake.png")):
    PILImage.open(MEDIA / gif).convert("RGB").save(FIG / png)
try:
    from moviepy import VideoFileClip
    PILImage.fromarray(VideoFileClip(str(MEDIA / "eagle1_atterrissage.mp4")).get_frame(4.5)).save(FIG / "lunar.png")
except Exception:  # vidéo absente : pas d'illustration
    pass

# --- Contenu ---------------------------------------------------------------------------------------------
story = []
A = story.append

# Page de titre
A(Spacer(1, 4 * cm))
A(P("Eagle-1", "title"))
A(P("Pilote automatique d'atterrissage lunaire par apprentissage par renforcement", "subtitle"))
A(Spacer(1, 0.6 * cm))
A(P("Démarche, résultats, interprétation et notions essentielles", "subtitle"))
A(Spacer(1, 1.2 * cm))
if (FIG / "lunar.png").exists():
    A(figure(FIG / "lunar.png", 11))
A(Spacer(1, 1.2 * cm))
A(table([["Objectif du cahier des charges", "Résultat obtenu"],
         ["Récompense moyenne ≥ 200 sur 100 épisodes", f"<b>{nominal.mean():.1f} ± {nominal.std():.1f}</b> "
                                                       f"({(nominal >= 200).mean():.0%} d'épisodes ≥ 200)"],
         ["Algorithme", f"{prod['algo']} (Stable-Baselines3), modèle {prod['name']}"],
         ["Livrables", "notebooks, vidéo 25 s, API FastAPI, GUI et tableau de bord Streamlit, tests, CI"]],
        [7.5, 8.5]))
A(PageBreak())

# 1. Vue d'ensemble
A(P("1. Vue d'ensemble du projet", "h1"))
A(P("Le projet suit une progression en quatre temps : trois exercices pour construire les fondations de "
    "l'apprentissage par renforcement (RL), puis une mission complète sur un problème plus difficile."))
A(table([["Étape", "Environnement", "Idée clé", "Résultat"],
         ["Exercice 1", "CartPole-v1", "La boucle observation → action → récompense", "Agent aléatoire : 22,7 ± 10,5 / 500"],
         ["Exercice 2", "FrozenLake-v1", "Apprendre une Q-table avec l'équation de Bellman", "66 % de réussite (≈ optimum 74 %)"],
         ["Exercice 3", "CartPole-v1", "Remplacer la table par un réseau : le DQN", "SB3 ajusté : 500 ± 0 / 500"],
         ["Mission", "LunarLander-v3", "Choisir, optimiser, évaluer, déployer",
          f"{nominal.mean():.1f} ± {nominal.std():.1f} (objectif ≥ 200)"]], [2.2, 2.8, 6.2, 4.8]))
A(Spacer(1, 6))
A(P("Tout s'exécute dans un container Docker (image Python 3.11 + uv). Le même container sert les notebooks "
    "(JupyterLab), TensorBoard, l'API, le GUI et le tableau de bord ; un workflow GitHub Actions rejoue les tests "
    "et tous les notebooks à chaque modification."))

# 2. Notions essentielles
A(P("2. Les notions essentielles à maîtriser", "h1"))
A(P("Le cadre : agent, environnement, récompense", "h2"))
A(P("En RL, un <b>agent</b> apprend par essais et erreurs à agir dans un <b>environnement</b>. À chaque pas t, "
    "il observe un <b>état</b> s<sub>t</sub>, choisit une <b>action</b> a<sub>t</sub>, puis reçoit une "
    "<b>récompense</b> r<sub>t+1</sub> et un nouvel état. Personne ne lui montre la bonne réponse : il ne "
    "dispose que de ce signal numérique, souvent rare et différé. C'est la différence fondamentale avec "
    "l'apprentissage supervisé."))
A(table([["Notion", "Définition", "Où on l'a vue"],
         ["Épisode", "Une partie, du reset() à la fin (terminated : fin naturelle ; truncated : limite de temps)", "Ex. 1"],
         ["Espace d'états / d'actions", "Box = continu, Discrete = nombre fini de choix. Détermine les algorithmes possibles", "Ex. 1, mission"],
         ["Politique π(a|s)", "La stratégie de l'agent : quelle action prendre dans quel état", "Partout"],
         ["Retour G", "Somme des récompenses futures actualisées : r<sub>1</sub> + γ r<sub>2</sub> + γ<super>2</super> r<sub>3</sub> + …", "Ex. 2, 3"],
         ["Facteur γ (gamma)", "Entre 0 et 1 : importance du futur. γ proche de 1 = agent prévoyant", "Ex. 2, mission"],
         ["Fonction de valeur Q(s,a)", "Retour espéré en jouant a dans s puis en suivant la politique", "Ex. 2, 3"],
         ["Équation de Bellman", "Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)] : la valeur d'aujourd'hui "
                                 "s'appuie sur l'estimation de demain (bootstrapping)", "Ex. 2, 3"],
         ["Exploration / exploitation", "ε-greedy : action aléatoire avec probabilité ε, sinon la meilleure connue ; "
                                        "ε décroît au fil de l'apprentissage", "Ex. 2, 3"],
         ], [3.6, 9.4, 3]))
A(P("Passer à l'échelle : approximation par réseau de neurones", "h2"))
A(P("Une Q-table n'est possible que pour quelques états discrets (16 pour FrozenLake). Dès que l'état est "
    "continu (CartPole, LunarLander), on approxime Q par un réseau : c'est le <b>DQN</b>. Deux mécanismes le "
    "rendent stable :"))
story += bullets([
    "<b>Replay buffer</b> : les transitions sont stockées puis tirées au hasard, ce qui casse la corrélation "
    "entre pas successifs et réutilise chaque expérience plusieurs fois.",
    "<b>Target network</b> : une copie figée du réseau calcule la cible r + γ max Q(s', a'). Sans elle, le réseau "
    "poursuit une cible qui bouge à chaque mise à jour et diverge facilement.",
])
A(P("Deux familles d'algorithmes", "h2"))
A(table([["", "Basés valeur (DQN)", "Policy gradient (PPO)"],
         ["Ce qu'on apprend", "Q(s,a), la politique = argmax Q", "Directement π(a|s) (acteur) + V(s) (critique)"],
         ["Données", "Off-policy : réutilise d'anciennes transitions", "On-policy : données fraîches à chaque mise à jour"],
         ["Actions", "Discrètes uniquement", "Discrètes et continues"],
         ["Stabilité", "Sensible aux réglages", "Mises à jour bornées (clipping) → robuste"],
         ["Hyperparamètres clés", "learning_rate, buffer_size, target_update_interval, exploration",
          "learning_rate, gamma, n_steps, gae_lambda, ent_coef, n_epochs"]], [3.4, 6.3, 6.3]))
A(Spacer(1, 6))
A(P("Les notions d'ingénieur", "h2"))
story += bullets([
    "<b>Reward shaping</b> : la récompense définit le problème. LunarLander récompense la proximité de la cible, "
    "la lenteur, la stabilité, et pénalise le carburant : l'agent optimise exactement cela, pas « l'idée » qu'on en a.",
    "<b>Variance et graines</b> : deux entraînements identiques avec des graines différentes donnent des résultats "
    "différents. Une conclusion fiable demande plusieurs graines ou de grands écarts.",
    "<b>Évaluer à part</b> : on mesure la politique déterministe (sans exploration), sur des épisodes jamais vus, "
    "et sur assez d'épisodes (100) pour que la moyenne soit fiable.",
    "<b>Généralisation</b> : un bon score sur les conditions d'entraînement ne garantit rien hors de ces conditions "
    "(ici : le vent).",
    "<b>Un hyperparamètre à la fois</b> : seule façon d'attribuer un effet à une cause.",
])
A(PageBreak())

# 3. Exercices
A(P("3. Les exercices : construire les fondations", "h1"))
A(P("Exercice 1 — la boucle d'interaction (CartPole)", "h2"))
A(Table([[figure(FIG / "cartpole.png", 5.5),
          P("On crée l'environnement, on inspecte ses espaces (observation <b>Box</b> de 4 réels : position, vitesse, "
            "angle, vitesse angulaire ; action <b>Discrete(2)</b> : pousser à gauche ou à droite) et on fait jouer un "
            "agent aléatoire 10 épisodes.<br/><br/><b>Résultat</b> : 22,7 ± 10,5 pas sur 500 possibles. "
            "<b>Interprétation</b> : c'est le point zéro, sans apprentissage. Toute méthode doit faire mieux. "
            "Les trajectoires 3D montrent l'angle qui diverge en quelques pas.")]],
        colWidths=[6 * cm, 10 * cm], style=[("VALIGN", (0, 0), (-1, -1), "TOP")]))
A(P("Exercice 2 — Q-learning tabulaire (FrozenLake)", "h2"))
A(Table([[figure(FIG / "frozenlake.png", 4.5),
          P("Une Q-table 16 × 4 remplie de zéros apprend par l'équation de Bellman, avec une exploration ε-greedy "
            "décroissante sur 20 000 épisodes.<br/><br/><b>Résultat</b> : 66 % de réussite sur 100 épisodes "
            "d'évaluation. <b>Interprétation</b> : le sol est glissant (l'action voulue n'est exécutée qu'une fois "
            "sur trois), donc même la politique optimale échoue environ une fois sur quatre (≈ 74 %). La politique "
            "apprise est « prudente » : elle longe les murs pour ne jamais risquer de glisser dans un trou. "
            "<b>Limite</b> : impossible au-delà de quelques états discrets.")]],
        colWidths=[5 * cm, 11 * cm], style=[("VALIGN", (0, 0), (-1, -1), "TOP")]))
A(P("Exercice 3 — DQN manuel puis Stable-Baselines3 (CartPole)", "h2"))
A(P("On implémente un DQN avec PyTorch : réseau 4 → 128 → 128 → 2, replay buffer (deque), target network copié "
    "tous les 10 épisodes, perte MSE entre Q(s,a) prédit (<font name='DejaVuMono'>gather</font>) et la cible "
    "calculée par le target network (<font name='DejaVuMono'>max(1)[0]</font>). Puis on refait la même chose "
    "avec Stable-Baselines3."))
A(table([["Variante", "Résultat", "Lecture"],
         ["DQN manuel (400 épisodes)", "≈ 257 / 500", "Apprend, mais courbe instable : typique du DQN « vanilla »"],
         ["SB3, paramètres par défaut (25 000 pas)", "22,5 / 500", "Défauts pensés pour Atari : target mis à jour tous les 10 000 pas, trop lent ici"],
         ["SB3, réglages RL Zoo (50 000 pas)", "<b>500 ± 0</b> / 500", "Score parfait : les hyperparamètres comptent autant que l'algorithme"]],
        [5, 2.8, 8.2]))
A(P("<b>Leçon principale des exercices</b> : on passe d'un agent qui subit (ex. 1) à un agent qui apprend une "
    "table (ex. 2), puis à un agent qui généralise grâce à un réseau (ex. 3) ; et l'on découvre que le réglage "
    "des hyperparamètres est décisif. C'est exactement le fil de la mission.", "box"))
A(PageBreak())

# 4. Mission
A(P("4. La mission Eagle-1 (LunarLander-v3)", "h1"))
A(P("4.1 Comprendre le terrain de jeu", "h2"))
A(P("L'état comprend 8 variables (position x, y ; vitesses ; angle et vitesse angulaire ; contact de chaque "
    "jambe) et 4 actions discrètes (rien, moteur gauche, moteur principal, moteur droit). La récompense est "
    "façonnée : se rapprocher, ralentir et rester droit rapportent, chaque jambe posée +10, le moteur principal "
    "coûte 0,3 par pas et un moteur latéral 0,03 — <b>le carburant est donc directement pénalisé</b>. Fin "
    "d'épisode : +100 posé et immobile, −100 crash. Un épisode ≥ 200 est un atterrissage réussi. L'agent "
    "aléatoire obtient environ −210 : il faut gagner environ 410 points."))
A(P("4.2 Choix de l'algorithme", "h2"))
A(P("Les actions sont discrètes : le brief recommande DQN, et cite PPO comme alternative plus optimisée. Plutôt "
    "que de trancher a priori, nous avons entraîné <b>les deux</b> avec leurs paramètres par défaut, puis optimisé "
    "chacun. PPO bénéficie en plus de 8 à 16 environnements simulés en parallèle, ce qui accélère "
    "l'entraînement en temps réel."))
A(P("4.3 Baselines et optimisation des hyperparamètres", "h2"))
A(P(f"Avec les paramètres par défaut, <b>DQN échoue</b> ({e('dqn_baseline')}) : ses épisodes durent "
    f"{e('dqn_baseline', 'mean_length', '{:.0f}')} pas en moyenne, soit presque la limite de 1 000 — l'agent a "
    "appris à <b>planer</b> pour éviter le −100 du crash, sans oser se poser. C'est un « optimum local » "
    f"classique. <b>PPO par défaut</b> atteint déjà {e('ppo_baseline')} ± {e('ppo_baseline', 'std_reward')} : "
    "il passe le seuil, mais avec un écart-type élevé (certains épisodes ratés)."))
A(figure(FIG / "experiences.png", 15.5, "Chaque barre : 100 épisodes d'évaluation sur des graines jamais vues. "
                                        "Ligne pointillée : seuil de 200."))
A(table([["Hyperparamètre modifié (PPO)", "Résultat", "Interprétation"],
         ["learning_rate 3e-4 → 1e-3", f"{e('ppo_lr_1e-3')} ± {e('ppo_lr_1e-3', 'std_reward')}",
          "Apprend plus vite dans le budget donné ; meilleur changement isolé"],
         ["gamma 0,99 → 0,999", f"{e('ppo_gamma_0.999')} ± {e('ppo_gamma_0.999', 'std_reward')}",
          "Valoriser le +100 final, lointain, aide à finir l'atterrissage"],
         ["ent_coef 0 → 0,01", f"{e('ppo_ent_0.01')} ± {e('ppo_ent_0.01', 'std_reward')}",
          "Bonus d'entropie : exploration maintenue plus longtemps"],
         ["gae_lambda 0,95 → 0,98", f"{e('ppo_gae_0.98')} ± {e('ppo_gae_0.98', 'std_reward')}",
          "Estimation de l'avantage moins biaisée, sur un horizon plus long"],
         ["n_steps 2048 → 1024", f"{e('ppo_nsteps_1024')} ± {e('ppo_nsteps_1024', 'std_reward')}",
          "Lots de données deux fois plus petits (avec 8 envs) → gradients plus bruités"],
         ["n_epochs 10 → 4", f"{e('ppo_epochs_4')} ± {e('ppo_epochs_4', 'std_reward')}",
          "Moins de passes par lot → apprentissage plus lent à budget égal"]], [4.6, 2.8, 8.6]))
A(Spacer(1, 6))
A(P(f"Côté DQN, réduire le <b>buffer_size</b> à 50 000 aide ({e('dqn_buffer_50k')}) : on oublie plus vite les "
    "vieilles transitions issues d'une politique médiocre. Mais aucun changement isolé ne suffit ; la "
    f"<b>combinaison RL Zoo</b> (réseau 256 × 256, target mis à jour tous les 250 pas, lr 6,3e-4…) atteint "
    f"{e('dqn_tuned')} ± {e('dqn_tuned', 'std_reward')}. DQN <i>peut</i> donc réussir, mais il est bien plus "
    "sensible aux réglages que PPO, ce qui confirme le choix de PPO."))
A(figure(FIG / "courbes.png", 16, "Courbes d'apprentissage (évaluation de 10 épisodes tous les 25 000 pas)."))
A(P("4.4 Configuration finale et choix du modèle", "h2"))
finals = [exps[f"ppo_final_s{s}"] for s in range(3) if f"ppo_final_s{s}" in exps]
A(P("La configuration finale combine gamma 0,999, gae_lambda 0,98, ent_coef 0,01, n_steps 1024 × 16 "
    "environnements (soit des lots de 16 384 transitions, comme la baseline), n_epochs 4, sur 2 millions de pas. "
    "Elle est entraînée avec <b>3 graines</b> pour vérifier la reproductibilité :"))
A(table([["Graine", "Moyenne ± écart-type", "Pire épisode", "Réussite"]] +
        [[f["name"], f"{f['mean_reward']:.1f} ± {f['std_reward']:.1f}", f"{f['min_reward']:.1f}",
          f"{f['success_rate']:.0%}"] for f in finals], [5, 4, 3.5, 3.5]))
A(Spacer(1, 6))
A(P(f"Les trois graines donnent le même résultat à quelques points près : la performance n'est pas un coup de "
    f"chance. Nous avons retenu <b>{prod['name']}</b>, dont le <b>pire épisode est le meilleur des trois</b> — "
    "pour un atterrisseur, le pire cas compte plus que la moyenne. Remarque honnête : n_steps 1024 et n_epochs 4 "
    "dégradaient le score <i>isolément</i> ; ils ne nuisent plus ici car le nombre d'environnements est doublé "
    "(lots identiques) et le budget d'entraînement aussi."))
A(PageBreak())

A(P("4.5 Évaluation finale et généralisation", "h2"))
A(P(f"Évaluation indépendante dans le notebook, 100 épisodes sur des graines jamais vues : "
    f"<b>{nominal.mean():.1f} ± {nominal.std():.1f}</b>, pire épisode {nominal.min():.1f}, "
    f"{(nominal >= 200).mean():.0%} des épisodes au-dessus de 200. Le critère « moyenne ≥ 200, stable, écart-type "
    "faible » est atteint avec une marge confortable."))
A(figure(FIG / "evaluation_finale.png", 16))
wind_text = ""
if wind_model:
    wind_text = (f" Pour tenter d'y remédier, nous avons entraîné la même configuration <b>avec du vent</b> "
                 f"(ppo_final_wind) : {wind_model['mean_reward']:.1f} en conditions nominales et "
                 f"{wind_model['wind_mean_reward']:.1f} avec vent ({wind_model['wind_success_rate']:.0%} de réussite). "
                 "Avec le même budget, la tâche bruitée est plus difficile : le modèle perd en nominal sans gagner "
                 "nettement avec vent. Cette randomisation de l'environnement est la bonne direction, mais demande "
                 "plus d'entraînement : c'est la principale piste d'amélioration.")
A(P(f"<b>Généralisation</b> : nouveaux terrains et positions de départ, aucun problème. Mais avec <b>vent et "
    f"turbulences</b> — une physique jamais vue — le modèle tombe à {windy.mean():.1f} ({(windy >= 200).mean():.0%} "
    "de réussite). Fait instructif : la configuration finale, la meilleure en conditions nominales, est moins "
    "robuste que plusieurs variantes entraînées moins longtemps (ent_coef 0,01, gae_lambda 0,98, lr 1e-3) — un "
    "entraînement prolongé spécialise la politique sur les conditions vues : c'est une forme de surapprentissage "
    f"à l'environnement.{wind_text}"))
A(figure(FIG / "robustesse.png", 12, "Chaque point est un modèle : en haut à droite, performant et robuste."))
A(P("4.6 Comportement appris", "h2"))
A(P("L'analyse des décisions (tableau de bord, vue « Décisions de l'IA ») montre une stratégie lisible : moteur "
    "principal dans 77 % des pas en chute rapide, moteur latéral opposé à l'inclinaison pour se redresser, et "
    "« rien » dans 89 % des pas une fois quasi immobile, ce qui économise le carburant. Les atterrissages durent "
    "environ 5 secondes (≈ 250 pas) et consomment environ 30 unités de pénalité carburant."))

# 5. Architecture
A(PageBreak())
A(P("5. Architecture du prototype", "h1"))
A(table([["Composant", "Rôle", "Accès"],
         ["eagle1/core.py, train.py", "Environnement, épisodes instrumentés, expériences, évaluation", "CLI dans le container"],
         ["API FastAPI (api.py)", "Toute la logique RL : /play (état → action), /episode, /evaluate, statistiques", "localhost:8000/docs"],
         ["GUI Streamlit (gui.py)", "Lance une partie via l'API, affiche animation, télémétrie, actions", "localhost:8501"],
         ["Tableau de bord (dashboard.py)", "Performances, décisions par circonstance, expériences", "localhost:8502"],
         ["video.py", "Vidéo 25 s avec gymnasium.wrappers.RecordVideo", "media/eagle1_atterrissage.mp4"],
         ["JupyterLab, TensorBoard", "Notebooks et courbes d'entraînement", "localhost:8888, :6006"],
         ["Tests + GitHub Actions", "Tests de l'API (dont le critère ≥ 200), exécution de tous les notebooks", "à chaque push"]],
        [4.3, 8, 3.7]))
A(Spacer(1, 6))
A(P("Le GUI et le tableau de bord ne contiennent aucune logique RL : ils appellent l'API, conformément au cahier "
    "des charges. Les parties jouées sont stockées dans une base SQLite qui alimente le tableau de bord. "
    "Tout se lance avec <font name='DejaVuMono'>docker compose up -d</font> ; les entraînements avec "
    "<font name='DejaVuMono'>python -m eagle1.train</font>, ce rapport avec "
    "<font name='DejaVuMono'>python docs/build_report.py</font>."))

# 6. Lire les résultats
A(P("6. Comment interpréter les résultats", "h1"))
A(table([["Indicateur", "Ce qu'il dit", "Piège à éviter"],
         ["Récompense moyenne", "Performance globale", "Une moyenne seule cache les échecs : toujours la lire avec l'écart-type"],
         ["Écart-type", "Régularité du pilote", "Élevé = comportement imprévisible, même si la moyenne passe 200"],
         ["Pire épisode / taux ≥ 200", "Sécurité, pire cas", "Indicateur le plus parlant pour un système critique"],
         ["Courbe d'apprentissage", "Vitesse et stabilité de l'apprentissage", "Bruitée par nature ; regarder la tendance, pas un point"],
         ["Durée d'épisode", "Style de vol", "1 000 pas = l'agent plane sans se poser (DQN par défaut)"],
         ["Score avec vent", "Robustesse hors distribution", "Un modèle peut exceller en nominal et échouer hors conditions vues"],
         ["Plusieurs graines", "Fiabilité de la conclusion", "Un seul entraînement peut être un coup de chance"]],
        [3.6, 4.6, 7.8]))
A(P("<b>Les résultats les plus pertinents</b> : (1) le modèle final atteint ~280 de moyenne avec un écart-type "
    "de ~20 et 100 % de réussite sur des graines inédites ; (2) trois graines donnent le même résultat ; "
    "(3) DQN par défaut échoue en « planant » alors que DQN réglé réussit : les hyperparamètres sont décisifs ; "
    "(4) la robustesse au vent n'est pas acquise par défaut : c'est la principale limite et la piste "
    "d'amélioration.", "box"))

# 7. Limites et pistes
A(KeepTogether([P("7. Limites et pistes d'amélioration", "h1")] + bullets([
    "<b>Robustesse</b> : entraîner avec vent sur un budget plus long (l'essai à 2 M de pas n'a pas suffi) ou "
    "partir d'une configuration plus exploratoire (ent_coef), voire randomiser la gravité.",
    "<b>Recherche d'hyperparamètres automatisée</b> (Optuna) plutôt que manuelle, avec plusieurs graines par essai.",
    "<b>Carburant</b> : ajuster la récompense pour pénaliser davantage la consommation si c'est la priorité métier.",
    "<b>Actions continues</b> : LunarLanderContinuous-v3 avec PPO ou SAC, pour une poussée dosée plus économe.",
])))

# 8. Glossaire
A(P("8. Glossaire", "h1"))
A(table([["Terme", "Définition courte"],
         ["Avantage A(s,a)", "Combien une action fait mieux que la moyenne dans cet état (utilisé par PPO)"],
         ["Bootstrapping", "Estimer une valeur à partir d'une autre estimation (cible de Bellman)"],
         ["Clipping (PPO)", "Limite l'écart entre ancienne et nouvelle politique à chaque mise à jour"],
         ["Entropie (ent_coef)", "Mesure du hasard de la politique ; un bonus d'entropie entretient l'exploration"],
         ["GAE (gae_lambda)", "Estimation de l'avantage qui arbitre entre biais (λ petit) et variance (λ proche de 1)"],
         ["n_steps / n_envs", "Pas collectés par environnement avant chaque mise à jour ; lot = n_steps × n_envs"],
         ["Off-policy / on-policy", "Apprendre de données anciennes (DQN) / seulement de la politique actuelle (PPO)"],
         ["Politique déterministe", "Toujours l'action la plus probable : utilisée pour évaluer et déployer"],
         ["Randomisation de domaine", "Varier l'environnement à l'entraînement (vent…) pour gagner en robustesse"],
         ["Timestep", "Une interaction agent-environnement ; le budget d'entraînement se compte en pas"],
         ["VecEnv", "Plusieurs copies de l'environnement simulées en parallèle"]], [4.5, 11.5]))


def on_page(canvas, doc):
    if doc.page > 1:
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(2 * cm, 1.2 * cm, "Eagle-1 — pilote automatique par apprentissage par renforcement")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
        canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title="Eagle-1 — rapport", author="Projet Eagle-1").build(
    story, onFirstPage=on_page, onLaterPages=on_page)
print("écrit :", OUT)
