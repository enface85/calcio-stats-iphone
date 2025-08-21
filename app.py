import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
from dateutil import parser as dtparser
from scipy.stats import poisson

# ------------------ CONFIG ------------------
API_KEY = st.secrets.get("b73fb46b6f9c4240a2ec8b2d8facf553", os.getenv("b73fb46b6f9c4240a2ec8b2d8facf553", ""))
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

st.set_page_config(page_title="Calcio Stats & Prob", page_icon="⚽", layout="wide")

# ------------------ UTILS ------------------
@st.cache_data(show_spinner=False)
def search_team(name_query: str):
    """Search teams by name across available competitions (simple heuristic)."""
    # football-data.org doesn't have a global search; we try a few popular comps
    comp_ids = ["PL","SA","PD","BL1","FL1","CL","ELC","PPL","DED","BSA","EC","WC"]  # Premier, Serie A, LaLiga, etc.
    matches = []
    for cid in comp_ids:
        url = f"{BASE_URL}/competitions/{cid}/teams"
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            continue
        for t in r.json().get("teams", []):
            if name_query.lower() in (t.get("name","") + " " + t.get("shortName","") + " " + t.get("tla","")).lower():
                matches.append({"id": t["id"], "name": t["name"], "shortName": t.get("shortName"), "tla": t.get("tla")})
    # Deduplicate by id preserving order
    seen=set(); dedup=[]
    for m in matches:
        if m["id"] not in seen:
            dedup.append(m); seen.add(m["id"])
    return dedup

@st.cache_data(show_spinner=False)
def fetch_recent_matches(team_id: int, limit: int = 20):
    url = f"{BASE_URL}/teams/{team_id}/matches?status=FINISHED&limit={limit}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Errore API ({r.status_code}): {r.text[:150]}")
    games = r.json().get("matches", [])
    rows = []
    for m in games:
        home = m["homeTeam"]["name"]; away = m["awayTeam"]["name"]
        sh = (m["score"]["fullTime"]["home"] if m["score"]["fullTime"]["home"] is not None else 0)
        sa = (m["score"]["fullTime"]["away"] if m["score"]["fullTime"]["away"] is not None else 0)
        date = dtparser.parse(m["utcDate"]).date().isoformat()
        rows.append({"date": date, "home": home, "away": away, "gh": sh, "ga": sa})
    df = pd.DataFrame(rows)
    return df

def compute_form(df: pd.DataFrame, team_name: str):
    if df.empty:
        return pd.DataFrame(columns=["Esito"])
    def result(row):
        if row["home"] == team_name:
            if row["gh"] > row["ga"]: return "V"
            if row["gh"] == row["ga"]: return "N"
            return "P"
        else:
            if row["ga"] > row["gh"]: return "V"
            if row["gh"] == row["ga"]: return "N"
            return "P"
    df = df.copy()
    df["Esito"] = df.apply(result, axis=1)
    return df

def poisson_outcome_probs(lambda_home: float, lambda_away: float, max_goals: int = 8):
    """Compute 1X2 probs by summing independent Poisson home/away goal distributions."""
    gh = np.arange(0, max_goals+1)
    ga = np.arange(0, max_goals+1)
    p_h = poisson.pmf(gh, lambda_home)
    p_a = poisson.pmf(ga, lambda_away)
    matrix = np.outer(p_h, p_a)  # P(gh,ga)
    p_home = np.tril(matrix, -1).sum()  # gh > ga
    p_draw = np.trace(matrix)           # gh == ga
    p_away = np.triu(matrix, 1).sum()   # gh < ga
    # Normalize if truncated
    s = p_home + p_draw + p_away
    return p_home/s, p_draw/s, p_away/s

def estimate_team_lambdas(df_team: pd.DataFrame, team_name: str):
    """Simple average goals for/against splitting home/away contexts."""
    if df_team.empty:
        return 1.0, 1.0  # fallback
    # Goals scored by team in its last matches (context-aware)
    scored = []
    conceded = []
    for _, r in df_team.iterrows():
        if r["home"] == team_name:
            scored.append(r["gh"]); conceded.append(r["ga"])
        else:
            scored.append(r["ga"]); conceded.append(r["gh"])
    lam_for = max(0.1, float(np.mean(scored)))
    lam_against = max(0.1, float(np.mean(conceded)))
    return lam_for, lam_against

def combined_match_lambdas(df1, name1, df2, name2, adjust_home_adv=True):
    lam1_for, lam1_against = estimate_team_lambdas(df1, name1)
    lam2_for, lam2_against = estimate_team_lambdas(df2, name2)
    # Combine using defensive strengths (very simple heuristic)
    lambda_home = (lam1_for + lam2_against)/2.0
    lambda_away = (lam2_for + lam1_against)/2.0
    if adjust_home_adv:
        lambda_home *= 1.10  # +10% home advantage heuristic
    return max(0.1, lambda_home), max(0.1, lambda_away)

# ------------------ UI ------------------
st.title("⚽ Calcio Stats & Probability (Poisson)")
st.caption("Solo a scopo informativo. Nessuna garanzia di vincita.")

with st.sidebar:
    st.header("Impostazioni")
    n_matches = st.slider("Numero di partite recenti da considerare", 5, 40, 20, 1)
    st.write("Inserisci la tua API key in `.streamlit/secrets.toml`.")
    st.write("Stima con modello Poisson basato su medie gol recenti.")

col1, col2 = st.columns(2)
with col1:
    team1_query = st.text_input("Squadra Casa (es. Inter)", "")
with col2:
    team2_query = st.text_input("Squadra Ospite (es. Milan)", "")

if "searched" not in st.session_state:
    st.session_state["searched"] = {"t1": [], "t2": []}

def choose_team(label, query, key):
    if not query:
        return None
    results = search_team(query)
    if not results:
        st.warning(f"Nessuna squadra trovata per: {query}")
        return None
    names = [f'{r["name"]} ({r.get("tla","")})' for r in results]
    idx = st.selectbox(f"Scegli {label}", range(len(results)), format_func=lambda i: names[i])
    return results[idx]

st.divider()
colA, colB, colC = st.columns(3)
team1 = choose_team("Squadra Casa", team1_query, "t1")
team2 = choose_team("Squadra Ospite", team2_query, "t2")

st.divider()
btn_stats = st.button("📊 Calcola statistiche")
btn_probs = st.button("🔮 Stima probabilità 1X2")

if btn_stats or btn_probs:
    if not API_KEY:
        st.error("Configura la API key in `.streamlit/secrets.toml` (FOOTBALL_DATA_API_KEY).")
    elif not team1 or not team2:
        st.warning("Seleziona entrambe le squadre.")
    else:
        try:
            df1 = fetch_recent_matches(team1["id"], n_matches)
            df2 = fetch_recent_matches(team2["id"], n_matches)
        except Exception as e:
            st.error(f"Errore nel recupero dati: {e}")
            st.stop()

        if btn_stats:
            st.subheader("Andamento e medie gol")
            f1 = compute_form(df1, team1["name"])
            f2 = compute_form(df2, team2["name"])
            colL, colR = st.columns(2)
            with colL:
                st.markdown(f"**{team1['name']}** - ultime {len(f1)} partite")
                if not f1.empty:
                    st.write("Forma (più recente in cima):", " ".join(f1["Esito"].tolist()[:10]))
                    st.write("Media gol fatti:", round(estimate_team_lambdas(df1, team1["name"])[0], 2))
                    st.write("Media gol subiti:", round(estimate_team_lambdas(df1, team1["name"])[1], 2))
                st.dataframe(df1.sort_values("date", ascending=False))
            with colR:
                st.markdown(f"**{team2['name']}** - ultime {len(f2)} partite")
                if not f2.empty:
                    st.write("Forma (più recente in cima):", " ".join(f2["Esito"].tolist()[:10]))
                    st.write("Media gol fatti:", round(estimate_team_lambdas(df2, team2["name"])[0], 2))
                    st.write("Media gol subiti:", round(estimate_team_lambdas(df2, team2["name"])[1], 2))
                st.dataframe(df2.sort_values("date", ascending=False))

        if btn_probs:
            st.subheader("Probabilità 1X2 (modello Poisson, semplice)")
            lam_h, lam_a = combined_match_lambdas(df1, team1["name"], df2, team2["name"])
            p1, px, p2 = poisson_outcome_probs(lam_h, lam_a, max_goals=8)
            st.metric("λ Casa (gol attesi)", round(lam_h, 2))
            st.metric("λ Ospite (gol attesi)", round(lam_a, 2))
            colp1, colpx, colp2 = st.columns(3)
            colp1.metric("Prob. 1 (Casa)", f"{p1*100:.1f}%")
            colpx.metric("Prob. X (Pareggio)", f"{px*100:.1f}%")
            colp2.metric("Prob. 2 (Ospite)", f"{p2*100:.1f}%")
            st.caption("Stima ottenuta da distribuzioni di Poisson indipendenti troncate a 8 gol.")
