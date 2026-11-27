"""Tableau de bord Eagle-1 : suivi interactif des performances du pilote automatique.

Frontend uniquement : les données viennent de l'API (parties jouées, décisions, expériences d'entraînement).
Lancement :  streamlit run eagle1/dashboard.py
"""
import os

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
ACTION_ORDER = ["Rien", "Moteur gauche", "Moteur principal", "Moteur droit"]
TARGET = 200

st.set_page_config(page_title="Eagle-1 · Tableau de bord", page_icon=":material/monitoring:", layout="wide")


@st.cache_data(ttl=30, show_spinner=False)
def get(path):
    resp = requests.get(f"{API_URL}{path}", timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def circumstances(dec):
    """Classe chaque décision selon la situation du module (phase de vol, inclinaison, position, vitesse)."""
    dec = dec.copy()
    dec["phase"] = np.select(
        [(dec.jambe_gauche > 0) | (dec.jambe_droite > 0), dec.y < 0.3, dec.y < 0.8],
        ["au sol", "approche finale", "descente"], "haute altitude")
    dec["inclinaison"] = np.select([dec.angle > 0.1, dec.angle < -0.1],
                                   ["penché à gauche", "penché à droite"], "droit")
    dec["position"] = np.select([dec.x < -0.2, dec.x > 0.2], ["à gauche", "à droite"], "au-dessus de la cible")
    dec["vitesse verticale"] = np.select([dec.vy < -0.5, dec.vy < -0.1], ["chute rapide", "descente douce"], "stationnaire")
    dec["action_name"] = pd.Categorical(dec.action.map(dict(enumerate(ACTION_ORDER))), ACTION_ORDER)
    return dec


# --- En-tête et contrôles --------------------------------------------------------------------------------
st.title("Eagle-1 — tableau de bord du pilote automatique")

with st.sidebar:
    st.header("Simulation")
    n = st.slider("Nombre de parties à jouer", 5, 100, 20, step=5)
    wind = st.toggle("Vent et turbulences")
    if st.button("Lancer les parties", type="primary", icon=":material/play_arrow:", width="stretch"):
        with st.spinner(f"{n} parties en cours côté API…"):
            r = requests.post(f"{API_URL}/evaluate", json={"n_episodes": n, "wind": wind}, timeout=600)
        if r.ok:
            st.cache_data.clear()
            st.success(f"Moyenne {r.json()['mean_reward']:.1f} sur {n} parties.")
        else:
            st.error(r.text)
    st.divider()
    st.caption(f"Source des données : {API_URL}")

try:
    episodes = get("/episodes")
    experiments = get("/experiments")
except requests.RequestException as exc:
    st.error(f"API injoignable ({API_URL}) : {exc}", title="Erreur")
    st.stop()

view = st.segmented_control("Vue", ["Performances", "Décisions de l'IA", "Entraînement"], default="Performances",
                            required=True, label_visibility="collapsed")

# --- Vue 1 : performances sur les parties jouées ---------------------------------------------------------
if view == "Performances":
    if episodes.empty:
        st.info("Aucune partie enregistrée : lancez des parties depuis la barre latérale.", icon=":material/info:")
        st.stop()
    with st.sidebar:
        st.header("Filtres")
        conditions = st.pills("Conditions", ["sans vent", "avec vent"], selection_mode="multi",
                              default=["sans vent", "avec vent"])
    ep = episodes[episodes.wind.map({0: "sans vent", 1: "avec vent", False: "sans vent", True: "avec vent"})
                  .isin(conditions)].reset_index(drop=True)
    if ep.empty:
        st.warning("Aucune partie pour ces filtres.")
        st.stop()
    ep["partie"] = ep.index + 1
    ep["moyenne glissante (20)"] = ep.total_reward.rolling(20, min_periods=1).mean()

    with st.container(horizontal=True):
        st.metric("Parties", len(ep), border=True)
        st.metric("Récompense moyenne", f"{ep.total_reward.mean():.1f}", border=True)
        st.metric("Écart-type", f"{ep.total_reward.std(ddof=0):.1f}", border=True)
        st.metric("Taux de réussite (≥ 200)", f"{ep.success.mean():.0%}", border=True)
        st.metric("Carburant moyen", f"{ep.fuel.mean():.1f}", border=True,
                  help="0.3 par pas de moteur principal, 0.03 par pas de moteur latéral")

    c1, c2 = st.columns([2, 1])
    with c1, st.container(border=True):
        st.subheader("Récompense par partie")
        base = alt.Chart(ep).encode(x=alt.X("partie:Q", title="partie"))
        chart = (base.mark_circle(size=40, opacity=0.6).encode(
                    y=alt.Y("total_reward:Q", title="récompense"), color=alt.Color("outcome:N", title="issue"),
                    tooltip=["partie", "seed", "total_reward", "outcome", "n_steps", "fuel"])
                 + base.mark_line(color="#4C72B0").encode(y="moyenne glissante (20):Q")
                 + alt.Chart(pd.DataFrame({"y": [TARGET]})).mark_rule(strokeDash=[6, 4], color="gray").encode(y="y:Q"))
        st.altair_chart(chart, alt="Récompense de chaque partie, moyenne glissante et seuil de 200")
    with c2, st.container(border=True):
        st.subheader("Distribution")
        hist = alt.Chart(ep).mark_bar().encode(
            x=alt.X("total_reward:Q", bin=alt.Bin(step=25), title="récompense"), y=alt.Y("count():Q", title="parties"))
        st.altair_chart(hist + alt.Chart(pd.DataFrame({"x": [TARGET]})).mark_rule(color="gray").encode(x="x:Q"),
                        alt="Histogramme des récompenses")
        st.bar_chart(ep.outcome.value_counts(), horizontal=True, x_label="parties", alt="Issues des parties")

    with st.container(border=True):
        st.subheader("Carburant et durée")
        st.altair_chart(alt.Chart(ep).mark_circle(size=50).encode(
            x=alt.X("n_steps:Q", title="durée (pas)"), y=alt.Y("fuel:Q", title="carburant"),
            color=alt.Color("total_reward:Q", title="récompense", scale=alt.Scale(scheme="viridis")),
            tooltip=["partie", "total_reward", "fuel", "n_steps"]),
            alt="Carburant consommé en fonction de la durée, coloré par la récompense")

    with st.expander("Données brutes"):
        st.dataframe(ep.drop(columns=["moyenne glissante (20)"]), hide_index=True, alt="Parties jouées")

# --- Vue 2 : décisions prises selon les circonstances ----------------------------------------------------
elif view == "Décisions de l'IA":
    decisions = get("/decisions")
    if decisions.empty:
        st.info("Aucune décision enregistrée : lancez des parties.", icon=":material/info:")
        st.stop()
    dec = circumstances(decisions)
    st.caption(f"{len(dec):,} décisions issues de {dec.episode_id.nunique()} parties.".replace(",", " "))

    with st.container(horizontal=True):
        for action in ACTION_ORDER:
            st.metric(action, f"{(dec.action_name == action).mean():.0%}", border=True)

    criterion = st.segmented_control("Circonstance", ["phase", "inclinaison", "position", "vitesse verticale"],
                                     default="phase", required=True)
    share = (dec.groupby([criterion, "action_name"], observed=False).size()
             .groupby(level=0).transform(lambda s: s / s.sum()).rename("part").reset_index())
    with st.container(border=True):
        st.subheader(f"Action choisie selon : {criterion}")
        st.altair_chart(alt.Chart(share).mark_bar().encode(
            y=alt.Y(f"{criterion}:N", title=None), x=alt.X("part:Q", axis=alt.Axis(format="%"), title="part des décisions"),
            color=alt.Color("action_name:N", sort=ACTION_ORDER, title="action"),
            order=alt.Order("action_name:N"), tooltip=[criterion, "action_name", alt.Tooltip("part:Q", format=".0%")]),
            alt=f"Répartition des actions selon {criterion}")
    st.caption("Seuils — phase : au sol = une jambe touche, approche finale y < 0.3, descente y < 0.8 ; "
               "inclinaison : |angle| > 0.1 rad ; position : |x| > 0.2 (bords de la zone cible) ; "
               "vitesse : chute rapide vy < -0.5, descente douce vy < -0.1.")
    st.info("Comportement appris : moteur principal quand le module chute vite, moteur latéral opposé à "
            "l'inclinaison pour se redresser, et « Rien » une fois stabilisé ou posé, ce qui économise le carburant.",
            icon=":material/lightbulb:")

    with st.container(border=True):
        st.subheader("Carte des décisions (position du module)")
        sample = dec.sample(min(len(dec), 5000), random_state=0)
        st.altair_chart(alt.Chart(sample).mark_circle(size=12, opacity=0.5).encode(
            x=alt.X("x:Q", title="position horizontale x"), y=alt.Y("y:Q", title="altitude y"),
            color=alt.Color("action_name:N", sort=ACTION_ORDER, title="action")),
            alt="Action choisie en fonction de la position du module")

# --- Vue 3 : entraînement et optimisation des hyperparamètres --------------------------------------------
else:
    if experiments.empty:
        st.info("Aucune expérience : lancez `python -m eagle1.train`.", icon=":material/info:")
        st.stop()
    exp = experiments.sort_values("mean_reward", ascending=False)
    prod = requests.get(f"{API_URL}/model", timeout=30).json()
    with st.container(horizontal=True):
        st.metric("Expériences", len(exp), border=True)
        st.metric("Modèle en production", prod["name"], border=True,
                  help="Choisi parmi les 3 graines finales pour sa meilleure pire partie (critère de sécurité)")
        st.metric("Production : récompense (100 épisodes)", f"{prod['mean_reward']:.1f} ± {prod['std_reward']:.1f}",
                  border=True)

    with st.container(border=True):
        st.subheader("Récompense moyenne sur 100 épisodes d'évaluation")
        exp["bas"], exp["haut"] = exp.mean_reward - exp.std_reward, exp.mean_reward + exp.std_reward
        base = alt.Chart(exp).encode(y=alt.Y("name:N", sort="-x", title=None))
        st.altair_chart(
            base.mark_bar().encode(x=alt.X("mean_reward:Q", title="récompense moyenne"), color=alt.Color("algo:N"),
                                   tooltip=["name", "changed", "mean_reward", "std_reward", "success_rate"])
            + base.mark_rule().encode(x="bas:Q", x2="haut:Q")
            + alt.Chart(pd.DataFrame({"x": [TARGET]})).mark_rule(strokeDash=[6, 4], color="gray").encode(x="x:Q"),
            alt="Comparaison des expériences, barres d'erreur = écart-type")

    curves = get("/experiments/curves")
    if not curves.empty:
        with st.container(border=True):
            st.subheader("Courbes d'apprentissage")
            chosen = st.pills("Expériences", sorted(curves.experiment.unique()), selection_mode="multi",
                              default=[n for n in ("dqn_baseline", "ppo_baseline", prod["name"]) if n in set(curves.experiment)])
            line = alt.Chart(curves[curves.experiment.isin(chosen)]).mark_line().encode(
                x=alt.X("timesteps:Q", title="pas d'entraînement", scale=alt.Scale(domainMin=0)),
                y=alt.Y("mean_reward:Q", title="récompense moyenne (10 épisodes)",
                        scale=alt.Scale(domain=[-600, 350], clamp=True)),
                color=alt.Color("experiment:N", title="expérience"), tooltip=["experiment", "timesteps", "mean_reward"])
            st.altair_chart(line + alt.Chart(pd.DataFrame({"y": [TARGET]})).mark_rule(strokeDash=[6, 4], color="gray")
                            .encode(y="y:Q"), alt="Récompense d'évaluation au fil de l'entraînement")
            st.caption("Axe vertical borné à [-600, 350] : les premières évaluations peuvent descendre plus bas.")

    st.dataframe(exp[["name", "algo", "changed", "timesteps", "mean_reward", "std_reward", "success_rate",
                      *[c for c in ("wind_mean_reward", "wind_success_rate") if c in exp], "train_minutes"]], hide_index=True, alt="Tableau des expériences",
                 column_config={"success_rate": st.column_config.ProgressColumn("taux de réussite", format="percent"),
                                "mean_reward": st.column_config.NumberColumn("moyenne", format="%.1f"),
                                "std_reward": st.column_config.NumberColumn("écart-type", format="%.1f"),
                                "wind_mean_reward": st.column_config.NumberColumn("moyenne avec vent", format="%.1f"),
                                "wind_success_rate": st.column_config.ProgressColumn("réussite avec vent", format="percent")})
