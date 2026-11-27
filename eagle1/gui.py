"""GUI Eagle-1 : visualiser une partie jouée par le pilote automatique.

Frontend uniquement : la partie est jouée par l'API (POST /episode), ce fichier ne fait qu'afficher.
Lancement :  streamlit run eagle1/gui.py
"""
import base64
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
OUTCOME_ICON = {"atterrissage": ":material/flight_land:", "crash": ":material/crisis_alert:",
                "temps écoulé": ":material/timer_off:"}

st.set_page_config(page_title="Eagle-1 · Visualisation", page_icon=":material/rocket_launch:", layout="wide")
st.title("Eagle-1 — visualisation d'un atterrissage")
st.caption("Le pilote automatique (agent RL) joue une partie complète côté API ; cette page l'affiche.")

with st.container(horizontal=True, vertical_alignment="bottom"):
    seed = st.number_input("Graine du terrain (vide = aléatoire)", min_value=0, value=None, step=1,
                           placeholder="aléatoire", width=260)
    wind = st.toggle("Vent et turbulences", help="Conditions jamais vues à l'entraînement")
    launch = st.button("Lancer une partie", type="primary", icon=":material/play_arrow:")

if launch:
    with st.spinner("Partie en cours côté API…"):
        try:
            resp = requests.post(f"{API_URL}/episode", json={"seed": seed, "wind": wind}, timeout=120)
            resp.raise_for_status()
            st.session_state.episode = resp.json()
        except requests.RequestException as exc:
            st.error(f"API injoignable ou en erreur ({API_URL}) : {exc}", title="Erreur")

ep = st.session_state.get("episode")
if ep is None:
    st.info("Cliquez sur **Lancer une partie** pour voir le pilote automatique à l'œuvre.", icon=":material/info:")
    st.stop()

left, right = st.columns([1, 1.3], gap="large")
with left:
    st.image(base64.b64decode(ep["gif_base64"]), width="stretch",
             alt=f"Animation de la partie {ep['seed']} : {ep['outcome']}")
    st.caption(f"Partie n° {ep['id']} · graine {ep['seed']} · {'avec' if ep['wind'] else 'sans'} vent")

with right:
    with st.container(horizontal=True):
        st.metric("Récompense", f"{ep['total_reward']:.1f}", border=True,
                  help="≥ 200 = atterrissage réussi selon le cahier des charges")
        st.metric("Issue", ep["outcome"], border=True, icon=OUTCOME_ICON[ep["outcome"]])
        st.metric("Durée", f"{ep['n_steps']} pas", border=True, help="50 pas = 1 seconde")
        st.metric("Carburant", f"{ep['fuel']:.1f}", border=True,
                  help=f"Moteur principal : {ep['main_engine']} pas · latéraux : {ep['side_engine']} pas")
    if ep["success"]:
        st.success("Atterrissage réussi (récompense ≥ 200).", icon=":material/check_circle:")
    else:
        st.warning("Objectif non atteint sur cette partie (récompense < 200).", icon=":material/warning:")

    tel = pd.DataFrame(ep["telemetry"]).reset_index(names="pas")
    st.subheader("Télémétrie")
    st.line_chart(tel, x="pas", y=["y", "vy", "angle"], x_label="pas", y_label="valeur",
                  alt="Altitude, vitesse verticale et angle au cours de la partie")
    actions = alt.Chart(tel).mark_tick(thickness=2, size=18).encode(
        x=alt.X("pas:Q", title="pas"), y=alt.Y("action_name:N", title=None),
        color=alt.Color("action_name:N", legend=None))
    st.altair_chart(actions, alt="Action choisie par le pilote à chaque pas")
