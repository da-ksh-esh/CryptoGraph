"""CryptoGraph dashboard: the whole project as one story, built around the six notebooks, for a presentation.

    cryptograph-predict                 # writes runs/<run_id>/ (needed for What if? and the coin network today)
    cryptograph-train --config configs/experiment/predict_events_1h_v3.yaml   # the crisis replays
    streamlit run dashboard/app.py

Written for listeners without a finance or deep-learning background: short sentences, everyday words, every technical
term explained once. "Speaker notes" in the top bar shows a ready-to-say line under each chart. Environment:
CRYPTOGRAPH_RUNS_DIR (default <project>/runs), CRYPTOGRAPH_ROOT (project root for data/ and reports/). Every number is
read from the report files and runs (serving/story.py) or recomputed from the processed data with the notebooks'
own functions (serving/story_network.py); "What if?" fits a small VAR on recent hourly returns (serving/what_if.py).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptograph.data.config import PROJECT_ROOT          # torch before pandas (Windows DLL order, see README)
from cryptograph.serving import dashboard_data as dd
from cryptograph.serving import story as sy
from cryptograph.serving import story_network as sn
from cryptograph.serving import what_if as wi

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import plotly.express as px                               # noqa: E402
import plotly.graph_objects as go                         # noqa: E402
import streamlit as st                                    # noqa: E402

ROOT = Path(os.environ.get("CRYPTOGRAPH_ROOT", PROJECT_ROOT))
RUNS = Path(os.environ.get("CRYPTOGRAPH_RUNS_DIR", ROOT / "runs"))

st.set_page_config(page_title="CryptoGraph | crypto shock contagion", page_icon="🕸️", layout="wide")

RED, TEAL, GREY, AMBER, GREEN, VIOLET = "#e5484d", "#2ec4b6", "#8b929c", "#f5a524", "#3ccf7a", "#8e7dff"
FAMILY_COLOR = {"temporal graph": TEAL, "machine learning": AMBER, "network baseline": VIOLET}
SPLIT_COLOR = {"train": GREY, "validation": AMBER, "test": TEAL}
SECTOR_COLORS = [TEAL, AMBER, VIOLET, RED, GREEN, "#4ea8de", "#ff8a3d", "#c77dff", GREY]

st.markdown(f"""<style>
.stApp {{background: linear-gradient(160deg, #07080a 0%, #111317 45%, #1c1f25 100%); background-attachment: fixed;}}
[data-testid="stHeader"] {{background: rgba(9, 10, 12, 0.94);}}
.stMainBlockContainer {{padding-top: 3.75rem; max-width: 1240px;}}
[data-testid="stLayoutWrapper"]:has(> .st-key-navbar) {{position: sticky; top: 3.75rem; z-index: 99;}}
.st-key-navbar {{padding: 0.6rem 1rem; border-radius: 0.75rem; background: rgba(12, 13, 16, 0.92);
    backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.06);}}
.st-key-navbar p {{margin: 0;}}
[data-testid="stVerticalBlockBorderWrapper"] {{border-color: rgba(255, 255, 255, 0.08) !important;
    background: rgba(255, 255, 255, 0.02);}}
h1 {{font-weight: 650; letter-spacing: -0.02em;}}
h2, h3, h4 {{font-weight: 550;}}
[data-testid="stMetricValue"] {{font-weight: 400;}}
[data-testid="stMetricLabel"] p {{color: #9a9fa8;}}
.stTabs [data-baseweb="tab-list"] {{gap: 1.3rem; border-bottom: 1px solid rgba(255, 255, 255, 0.06);}}
.stTabs [data-baseweb="tab"] p {{font-size: 1rem;}}
footer {{visibility: hidden;}}
.hero {{padding: 2rem 0 1rem;}}
.hero .kicker {{color: {TEAL}; letter-spacing: .14em; text-transform: uppercase; font-size: .8rem; font-weight: 600;}}
.hero h1 {{font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.08; margin: .4rem 0 .8rem;}}
.hero p {{color: #c3c8cf; font-size: 1.15rem; max-width: 64ch; line-height: 1.55;}}
.kpis {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .9rem; margin: 1rem 0 1.6rem;}}
.kpi {{border: 1px solid rgba(255,255,255,.08); border-radius: .9rem; padding: 1rem 1.1rem; background: rgba(255,255,255,.025);}}
.kpi b {{display: block; font-size: 2rem; font-weight: 600; letter-spacing: -.02em;}}
.kpi span {{color: #a9b0b9; font-size: .93rem;}}
.flow {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .7rem; margin: .6rem 0 1.4rem;}}
.step {{border-radius: .8rem; padding: .9rem 1rem; background: rgba(46,196,182,.07); border: 1px solid rgba(46,196,182,.25);}}
.step .n {{color: {TEAL}; font-weight: 700; font-size: .78rem; letter-spacing: .1em;}}
.step b {{display: block; margin: .2rem 0 .3rem; font-size: 1.05rem;}}
.step span {{color: #b4bac3; font-size: .9rem; line-height: 1.45;}}
.step code {{font-size: .78rem; color: #9a9fa8; background: none; padding: 0;}}
.cards {{display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .9rem; margin: .6rem 0 1.4rem;}}
.card {{border: 1px solid rgba(255,255,255,.08); border-radius: .9rem; padding: 1rem 1.1rem; background: rgba(255,255,255,.025);}}
.card .big {{font-size: 1.7rem; font-weight: 600; color: #fafafa;}}
.card b {{display: block; margin: .25rem 0 .35rem; font-size: 1.05rem;}}
.card span {{color: #b4bac3; font-size: .93rem; line-height: 1.5;}}
.nb {{display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    color: {AMBER}; border: 1px solid rgba(245,165,36,.4); border-radius: 1rem; padding: .1rem .6rem; margin-top: 1.6rem;}}
.plain {{color: #c3c8cf; font-size: 1.05rem; line-height: 1.6; max-width: 80ch;}}
.say {{border: 1px dashed rgba(245,165,36,.55); border-radius: .7rem; padding: .7rem 1rem; margin: .4rem 0 1.2rem;
    background: rgba(245,165,36,.06); color: #f1e3c8; font-size: .98rem; line-height: 1.55;}}
.say b {{color: {AMBER};}}
.takeaway {{border-left: 3px solid {TEAL}; padding: .7rem 1rem; margin: 1.4rem 0; background: rgba(46,196,182,.07);
    border-radius: 0 .6rem .6rem 0; color: #dfe4ea; font-size: 1.02rem;}}
.takeaway b {{color: {TEAL};}}
.term {{color: #a9b0b9; font-size: .9rem;}}
@media (max-width: 900px) {{.kpis, .cards, .flow {{grid-template-columns: repeat(2, minmax(0, 1fr));}}}}
</style>""", unsafe_allow_html=True)


# --- cached readers ------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _prediction(path: str) -> dd.Prediction:
    return dd.load_prediction(path)


@st.cache_data(show_spinner=False)
def _graph(summary: dict) -> tuple:
    return dd.load_graph_at(summary, ROOT)


@st.cache_data(show_spinner=False)
def _rankings(version: str):
    return dd.load_rankings(version, ROOT)


@st.cache_data(show_spinner=False)
def _sectors() -> dict:
    return dd.load_sectors(ROOT)


@st.cache_data(show_spinner="Loading hourly prices...")
def _returns(summary: dict):
    return wi.load_returns(summary, ROOT)


@st.cache_data(show_spinner=False)
def _shock_model(summary: dict, window: int, horizon: int) -> wi.ShockModel:
    r, times, coins, thr = _returns(summary)
    return wi.fit_shock_model(r, times, coins, int(summary["as_of_bar"]), window, horizon, threshold=thr)


@st.cache_data(show_spinner=False)
def _events_index(path: str) -> dict:
    return dd.load_events_index(path)


@st.cache_data(show_spinner=False)
def _event(path: str, event_id: int) -> dict:
    return dd.load_event(path, event_id)


@st.cache_data(show_spinner=False)
def _story(name: str):
    return getattr(sy, name)(ROOT)


@st.cache_resource(show_spinner="Loading the hourly data of all coins (first time only)...")
def _panel(version: str) -> sn.PanelView | None:
    return sn.panel_view(ROOT, version)


@st.cache_data(show_spinner=False)
def _net(name: str, version: str):
    """One network computation (serving/story_network.py) on the hourly panel; None when the data is missing."""
    v = _panel(version)
    if v is None:
        return None
    if name == "corr":
        return pd.DataFrame(sn.window_corr(v), index=v.coins, columns=v.coins)
    if name == "sector_contagion":
        return sn.sector_contagion(v, _sectors()) if v.sample else None
    return getattr(sn, name)(v)


@st.cache_data(show_spinner=False)
def _market_mode(version: str, start: str):
    """The market factor on the test window from the date all 30 coins trade (as in notebook 02)."""
    ret = sy.deseasonalised_returns(ROOT, version)
    if ret is None:
        return None
    r, times = ret
    return sy.market_mode(r, np.asarray(times >= pd.Timestamp(start)))


@st.cache_data(show_spinner=False)
def _params(experiment: str):
    run = sy.find_run(RUNS, experiment)
    return (sy.parameter_breakdown(run), sy.head_inputs(run), sy.run_settings(run)) if run else (None, None, {})


@st.cache_data(show_spinner=False)
def _tgae():
    return sy.tgae_summary(RUNS)


STATE_LABEL = dict(zip(dd.NODE_STATES, ("big move now", "moved earlier", "calm so far", "not trading yet")))
STATE_COLOR = dict(zip(dd.NODE_STATES, ("#d62728", "#ff9f43", "#9aa5b1", "#3a3f47")))
SHORT_VERDICT = {"Almost every coin made a big move": "almost every coin moved: can't judge",
                 "The model did clearly better than picking coins at random.": "clearly better than random",
                 "The model did about as well as picking coins at random.": "about as random",
                 "The model did worse than picking coins at random here.": "worse than random"}
EVENT_PLAIN = {"src_vol_z": "the shocking coin's recent jumpiness", "src_return_scaled": "the shocking coin's move",
               "lag1_corr": "how the two coins moved one hour apart", "contemp_corr": "how the two coins moved together",
               "src_shock": "shock flag"}
MARKET_WINDOW_START = "2024-04-02"                        # all 30 coins listed (ENA), as in notebook 02
RANKINGS_VERSION = "v2"                                   # notebook 02 wrote the rankings; v3 keeps the same prices
EMBEDDING_DIM = 64                                        # configs/model/tgn.yaml: embedding_dim
MOVE_SCALE = [[0.0, "#d62728"], [0.5, "#3a3f47"], [1.0, "#2ca02c"]]          # price down red, up green


def draw_chart(fig, height: int | None = None) -> None:
    """Plot on the page's gradient: transparent backgrounds, faint grid lines."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(size=13))
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    st.plotly_chart(fig, width="stretch")


def section(notebook: str, title: str, plain: str) -> None:
    """A section header: which notebook it comes from, a title that states the finding, and the idea in plain words."""
    st.markdown(f"<span class='nb'>{notebook}</span>", unsafe_allow_html=True)
    st.markdown(f"#### {title}")
    st.markdown(f"<div class='plain'>{plain}</div>", unsafe_allow_html=True)


def say(text: str) -> None:
    """A speaker note, shown only when 'Speaker notes' is on."""
    if NOTES:
        st.markdown(f"<div class='say'><b>🎤 Say:</b> {text}</div>", unsafe_allow_html=True)


def takeaway(text: str) -> None:
    st.markdown(f"<div class='takeaway'><b>In one line:</b> {text}</div>", unsafe_allow_html=True)


def network_figure(coins: list[str], edges: pd.DataFrame, sector_of: dict, height: int = 560) -> tuple:
    """Coins on a fixed circle (grouped by sector) with the lines of the correlation snapshot; returns (fig, pos)."""
    pos = dd.circle_layout(coins, sector_of)
    ex, ey = [], []
    for e in edges.itertuples():
        if e.a in pos and e.b in pos:
            ex += [pos[e.a][0], pos[e.b][0], None]
            ey += [pos[e.a][1], pos[e.b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="#666"), hoverinfo="none",
                               showlegend=False))
    fig.update_layout(height=height, xaxis=dict(visible=False, range=[-1.35, 1.35]),
                      yaxis=dict(visible=False, range=[-1.25, 1.25]), margin=dict(t=50, b=10))
    return fig, pos


def sector_colors(sectors) -> dict:
    return {s: SECTOR_COLORS[i % len(SECTOR_COLORS)] for i, s in enumerate(sorted(set(sectors)))}


def fmt_int(x) -> str:
    return f"{int(x):,}" if x is not None else "—"


# --- navbar ------------------------------------------------------------------------------------------------------------
runs = dd.prediction_runs(RUNS)
P = _prediction(str(runs[0])) if runs else None              # the latest prediction run (What if?, coin network)
S = P.summary if P is not None else {}
version = S.get("data_version", "v3")
k_hours = list(S.get("k_hours", sy.K_HOURS))
default_h = 1 if 1 in k_hours else k_hours[0]
with st.container(key="navbar"):
    brand, look, notes = st.columns([1.3, 4, 1], vertical_alignment="center")
    brand.markdown("### CryptoGraph", help="Crypto shock contagion: which coins fall next, and how soon?")
    h = look.segmented_control("Look ahead", k_hours, default=default_h, format_func=lambda k: f"{k} h",
                               key="horizon", width="stretch",
                               help="How far ahead we ask: will the coin shock within this time? Used by the results "
                                    "and the crisis replays.")
    NOTES = notes.toggle("🎤 Notes", key="notes", help="Speaker notes: a ready-to-say line under each part.")
h = default_h if h is None else h                         # clicking the selected window again clears it
H = dd.horizon_label(h)

facts = sy.dataset_facts(ROOT, version)
head = _story("headline")
lb = _story("leaderboard")
bt = _story("bootstrap_v3_vs_v2")
thr = facts.get("shock_threshold") or 2.29
v3, m15 = head.get("v3"), head.get("m15")
pers = P.persistence if P is not None else pd.DataFrame()
mm = _market_mode(version, MARKET_WINDOW_START)

(tab_problem, tab_built, tab_moves, tab_net, tab_results, tab_why, tab_crises) = st.tabs(
    ["1 · The problem", "2 · How we built it", "3 · How coins move", "4 · The network", "5 · Model & results",
     "6 · Why, how long, how safe", "7 · Crises & what if"])

# --- 1. the problem -------------------------------------------------------------------------------------------------
with tab_problem:
    st.markdown("""<div class='hero'><div class='kicker'>Deep learning on networks · crypto markets</div>
<h1>When one coin crashes,<br>which coins fall next, and how soon?</h1>
<p>Crypto coins are linked. When one coin suddenly crashes, others often follow, sometimes within the hour.
We built a deep learning model that watches these crashes as they happen and warns which coins are likely to be
hit next.</p></div>""", unsafe_allow_html=True)

    years = (pd.Timestamp(facts["end"]) - pd.Timestamp(facts["start"])).days / 365.25 if facts.get("start") else None
    kpi = [(f"{facts.get('coins', '—')} coins", f"hourly prices from Binance, {years:.1f} years"
            if years else "hourly prices from Binance"),
           (f"{facts.get('hours', 0):,}", "hours of data for every coin"),
           (f"{facts['events'] / 1e6:.1f} million" if facts.get("events") else "—", "shock events the model learned from"),
           (f"{v3['lift']:.1f}×" if v3 else "—", "better than guessing at 1 hour (final model, on unseen years)")]
    st.markdown("<div class='kpis'>" + "".join(f"<div class='kpi'><b>{a}</b><span>{b}</span></div>" for a, b in kpi)
                + "</div>", unsafe_allow_html=True)
    say("Our project asks one question: when one crypto coin suddenly crashes, which other coins will follow, and how "
        "soon? We built a deep learning model that answers it, and we tested it on years it had never seen.")

    st.markdown("#### The problem in three words: shock, contagion, warning")
    st.markdown("<div class='cards'>" + "".join(f"<div class='card'><div class='big'>{a}</div><b>{b}</b><span>{c}</span>"
                                                "</div>" for a, b, c in [
        ("⚡", "A shock", f"A coin moves much more than usual in one hour: more than <b>{thr:.2f} times</b> its normal "
                         "hourly swing, up or down. Like a sudden storm on a normal day. It happens in about 4% of "
                         "hours."),
        ("🦠", "Contagion", "One coin's shock is followed by shocks in other coins, like a cold spreading through a "
                           "classroom. Our professor's idea: study this spread as a network."),
        ("🚨", "An early warning", "Traders and exchanges lose the most in the first hours of a crash. Knowing which "
                                   "coins are likely to be hit next gives them time to act."),
    ]) + "</div>", unsafe_allow_html=True)

    st.markdown("#### Our objective")
    st.markdown("<div class='plain'>The moment a coin shocks, give <b>every other coin a risk score</b>: how likely is "
                "it to have its own shock within <b>1, 3, 6, 12 or 24 hours</b>? Then check, honestly, whether those "
                "scores beat guessing.</div>", unsafe_allow_html=True)
    say("Our objective is simple: the moment one coin shocks, give every other coin a risk score for the next 1, 3, 6, "
        "12 and 24 hours, and prove the scores beat guessing.")

    st.markdown("#### What is new in our work")
    st.markdown("<div class='cards'>" + "".join(f"<div class='card'><b>{a}</b><span>{b}</span></div>" for a, b in [
        ("The market as a living network", "Most studies look at 30 separate price charts. We treat the coins as one "
                                           "network whose links change every hour."),
        ("Contagion as a prediction", "Research usually <i>measures</i> how connected coins are. We turned contagion "
                                      "into a <i>prediction</i> that can be checked against what really happened."),
        ("A model that reads events in order", "A temporal graph network (TGN) reads the shocks one by one, in time "
                                               "order, and can never look into the future."),
        ("Honest testing", "The last 2.7 years were locked away and used only once, at the end. Every result is compared "
                           "with simple models and with random guessing."),
        ("Explanations", "For each warning we show why the model raised it, in plain words."),
        ("Stress tests", "We attacked our own model with noise, missing data and deliberate manipulation."),
    ]) + "</div>", unsafe_allow_html=True)

    st.markdown("#### The questions we asked, and what we found")
    cj = _story("contagion_jump")
    cj1 = cj[(cj["split"] == "test") & (cj["k_hours"] == 1)] if not cj.empty else cj
    found = [
        ("Is contagion real?", f"Yes. Right after a shock, other coins are {cj1['lift'].iloc[0]:.1f}× more likely to "
                               "shock in the next hour." if len(cj1) else "Yes.", "3 · Notebook 01"),
        ("Does Bitcoin move first?", "No. Coins move together in the same hour, not one after another.",
         "3 · Notebook 01"),
        ("How tightly linked are coins?", f"Very. One market-wide 'tide' explains {mm['share']:.0%} of all hourly "
                                          "movement." if mm else "Very tightly.", "4 · Notebook 02"),
        ("What happens in a crisis?", "The network pulls even tighter: in every crisis, coins moved more alike.",
         "4 · Notebook 02"),
        ("Can we predict who is next?", f"Yes: {v3['lift']:.1f}× better than guessing at 1 hour, on years the model "
                                        "never saw." if v3 else "Yes.", "5 · Notebook 03"),
        ("Why does it raise a warning?", "Mostly because the receiving coin is already busy and jumpy.",
         "6 · Notebook 04"),
        ("How long does a shock last?", f"Only hours: {pers['median_hours_to_normal'].min():.0f} to "
                                        f"{pers['median_hours_to_normal'].max():.0f} hours, typically."
         if not pers.empty else "Only hours.", "6 · Notebook 05"),
        ("Can the model be fooled?", "Noise and missing data barely hurt it. A deliberate attack does.",
         "6 · Notebook 06"),
    ]
    st.dataframe(pd.DataFrame(found, columns=["question", "answer", "see tab · notebook"]), hide_index=True,
                 width="stretch")
    takeaway("contagion between coins is real and fast; our model warns which coins are next about twice as well as "
             "guessing, and every claim is tested honestly. The tabs above follow the six notebooks in order.")

# --- 2. how we built it ----------------------------------------------------------------------------------------------
TGN_SVG = """<svg viewBox="-34 0 794 470" width="100%" style="max-width:820px" xmlns="http://www.w3.org/2000/svg"
 font-family="Inter, Segoe UI, sans-serif">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#c9ced6"/></marker></defs>
<rect x="215" y="12" width="330" height="52" rx="10" fill="#1d2129" stroke="#e5484d" stroke-width="1.4"/>
<rect x="190" y="300" width="380" height="58" rx="10" fill="#132b2a" stroke="#2ec4b6" stroke-width="1.4"/>
<rect x="215" y="400" width="330" height="50" rx="10" fill="#1d2129" stroke="#3ccf7a" stroke-width="1.4"/>
<g opacity="MEMORY_OPACITY">
<g fill="none" stroke-width="1.4">
<rect x="20" y="104" width="240" height="62" rx="10" fill="#1d2129" stroke="#8e7dff"/>
<rect x="500" y="104" width="240" height="62" rx="10" fill="#1d2129" stroke="#8e7dff"/>
<rect x="190" y="200" width="380" height="58" rx="10" fill="#132b2a" stroke="#2ec4b6"/>
</g>
<g fill="#eef1f4" text-anchor="middle">
<text x="140" y="129" font-size="15" font-weight="600">2. A memory for each coin</text>
<text x="140" y="149" font-size="12" fill="#a9b0b9">a 64-number diary, updated by a GRU</text>
<text x="620" y="129" font-size="15" font-weight="600">3. Its recent shocks</text>
<text x="620" y="149" font-size="12" fill="#a9b0b9">the last 10 events + when they happened</text>
<text x="380" y="224" font-size="15" font-weight="600">4. Attention</text>
<text x="380" y="244" font-size="12" fill="#a9b0b9">decides which recent events matter most</text>
</g>
<g stroke="#c9ced6" stroke-width="1.4" fill="none">
<path d="M300 64 L170 102" marker-end="url(#a)"/><path d="M460 64 L590 102" marker-end="url(#a)"/>
<path d="M200 166 L270 198" marker-end="url(#a)"/><path d="M560 166 L490 198" marker-end="url(#a)"/>
<path d="M380 258 L380 298" marker-end="url(#a)"/>
<path d="M215 425 C 0 425, 0 260, 30 168" stroke="#f5a524" stroke-dasharray="5 5" marker-end="url(#a)"/>
</g>
<text x="-16" y="300" font-size="11" fill="#f5a524" transform="rotate(-90 -16 300)" text-anchor="middle">
7. only after scoring: update the diaries</text>
</g>
<g opacity="SHORTCUT_OPACITY" stroke="#2ec4b6" stroke-width="2" fill="none">
<path d="M545 38 C 700 38, 700 330, 572 330" marker-end="url(#a)"/></g>
<text x="690" y="200" font-size="12" fill="#2ec4b6" text-anchor="middle" opacity="SHORTCUT_OPACITY">v3: straight</text>
<text x="690" y="216" font-size="12" fill="#2ec4b6" text-anchor="middle" opacity="SHORTCUT_OPACITY">to the head</text>
<g fill="#eef1f4" text-anchor="middle">
<text x="380" y="35" font-size="15" font-weight="600">1. Event: coin i shocks at hour t</text>
<text x="380" y="54" font-size="12" fill="#a9b0b9">one question for every other coin j: will you be next?</text>
<text x="380" y="324" font-size="15" font-weight="600">5. Prediction head (a small neural network)</text>
<text x="380" y="344" font-size="12" fill="#a9b0b9">HEAD_TEXT</text>
<text x="380" y="430" font-size="15" font-weight="600">6. Five risk scores: 1 · 3 · 6 · 12 · 24 hours</text>
</g>
</svg>"""

with tab_built:
    st.markdown("### How we built it: from raw prices to warnings")
    st.markdown("<div class='plain'>Seven steps, in this order. Each box names the notebook that checks it and the "
                "code folder that does it.</div>", unsafe_allow_html=True)
    steps = [
        ("Collect prices", f"Hourly prices of {facts.get('coins', 30)} coins from Binance's public archive, 2020–2026. "
                           "Every file is checked with its fingerprint (checksum).", "Notebook 01", "data/sources"),
        ("Clean and check", "Fill short gaps, fix bad prices, compare with other data sources.", "Notebook 01",
         "data/quality"),
        ("Mark the shocks", f"Remove the daily rush-hour pattern, then flag hours that move more than {thr:.2f}× the "
                            "coin's normal swing.", "Notebook 01", "data/features/targets.py"),
        ("Build the network", "Measure how coins are linked: correlation, spillover, groups, skeleton. This gives 8 "
                              "network numbers per coin.", "Notebook 02", "network/"),
        ("Turn shocks into questions", "One shock becomes one question per other coin: will you shock within 1, 3, 6, "
                                       "12, 24 hours?", "Notebook 03", "data/graphs/temporal_stream.py"),
        ("Train and compare models", "Our TGN against simpler models, all scored on the same locked test years.",
         "Notebook 03", "models/ · training/"),
        ("Explain the warnings", "Why did the model warn? How long does a shock last?", "Notebooks 04 · 05",
         "explain/ · analysis/"),
        ("Stress-test", "Add noise, remove data, attack the model on purpose.", "Notebook 06", "robustness/"),
    ]
    st.markdown("<div class='flow'>" + "".join(
        f"<div class='step'><div class='n'>STEP {i} · {nb}</div><b>{a}</b><span>{b}<br><code>{code}</code></span></div>"
        for i, (a, b, nb, code) in enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)
    say("Here is the whole pipeline in eight steps: collect prices, clean them, mark the shocks, build the network, "
        "turn every shock into questions, train the models, explain them and stress-test them.")

    section("Notebook 01", "What exactly is a shock?",
            f"Every coin has its own normal hourly swing, measured over the <b>past week</b>. A shock is an hour that "
            f"moves more than <b>{thr:.2f} times</b> that swing. Example: if Bitcoin normally moves 0.5% in an hour, "
            f"a move bigger than {thr * 0.5:.2f}% is a shock. First we remove the daily rush-hour pattern: markets are "
            "always busier when US trading opens, and that is not a shock. The past week is used so that the rule "
            "never needs future data.")
    with st.expander("For the examiner: the formula"):
        st.markdown(f"z = (this hour's return, rush-hour pattern removed) ÷ (standard deviation of the previous 168 "
                    f"hours). A shock is |z| > {thr:.2f}. The threshold was chosen on the training years so that about "
                    "4% of coin-hours are shocks. Code: `data/features/targets.py` (`shock_z`, `contagion_labels`).")

    section("Notebook 03", "How we kept the test honest",
            "We split time into three blocks. The model <b>learns</b> on the first, we <b>choose settings</b> on the "
            "second, and the last block stays <b>locked</b> until the very end. It was scored once. A one-week gap "
            "between blocks stops information from leaking across.")
    if facts.get("events_per_split"):
        sp = facts["events_per_split"]
        blocks = pd.DataFrame([
            {"block": "learn (training)", "start": "2020-01-01", "end": "2022-06-30", "events": sp["train"]},
            {"block": "choose settings (validation)", "start": "2022-07-08", "end": "2023-12-24", "events": sp["val"]},
            {"block": "final test (locked)", "start": "2024-01-01", "end": facts.get("end", "2026-09-27"),
             "events": sp["test"]}])
        fb = px.timeline(blocks, x_start="start", x_end="end", y="block", color="block",
                         text=blocks["events"].map(lambda x: f"{x:,} events"),
                         color_discrete_sequence=[GREY, AMBER, TEAL])
        fb.update_layout(showlegend=False, yaxis_title=None, yaxis=dict(autorange="reversed"), margin=dict(t=10))
        draw_chart(fb, 220)
    say("We never let the model see the test years. It learned on 2020 to mid-2022, we tuned it on the next 18 "
        "months, and the last 2.7 years were scored once, at the end.")

    section("v3 improvement", "Where the 5-minute data came from",
            "Hourly prices hide what happens <i>inside</i> the hour, like judging a football match only from the "
            "half-time and full-time scores. So we downloaded <b>5-minute prices</b> from the same free Binance "
            "archive (<code>data.binance.vision</code>). The archive has one zip file per coin per month, each with "
            "a checksum file. A checksum is a fingerprint that proves the file was not damaged. We do not feed the "
            "5-minute prices to the model directly. For each hour we sum them up into a few numbers: how bumpy the "
            "hour was inside, its biggest 5-minute jump, and the move in its last 15 minutes.")
    n5 = sy.five_minute_rows(ROOT)
    c1, c2, c3 = st.columns(3)
    c1.metric("5-minute price bars", f"{n5 / 1e6:.1f} million" if n5 else "—")
    c2.metric("Source", "Binance public archive")
    c3.metric("Turned into", "hourly summary numbers")
    st.caption("Code: `scripts/fetch_5m.py` downloads and checks the files; `data/features/subhour.py` turns them into "
               "hourly numbers.")
    say("To see inside each hour we downloaded about 19 million 5-minute price bars from Binance's free public archive, "
        "each file verified by its checksum, and summarised every hour into a few numbers such as its biggest "
        "5-minute jump.")

    section("Notebook 03", "The model: a Temporal Graph Network (TGN)",
            "A TGN was invented at Twitter in 2020 to follow streams of events, like people replying to each other. "
            "Our stream is shocks: <i>coin i shocked at hour t; will coin j be next?</i> It reads the events strictly "
            "in time order, so it can never use the future.")
    arch = st.segmented_control("Show", ["v2: full TGN (memory on)", "v3: final model (memory off)"],
                                default="v3: final model (memory off)", key="arch")
    is_v3 = arch != "v2: full TGN (memory on)"
    _, in3, _ = _params("tgn_1h_v3_final")
    n3 = in3 - 2 * EMBEDDING_DIM if in3 else 78           # with memory off the two embedding slots are always zero
    w3 = 64 * n3 + 64 + 64 * 5 + 5                        # head weights that multiply a real input
    head_text = (f"{n3} numbers: both coins, the shock, the market, the time → 64 → 5" if is_v3
                 else "diaries + both coins' numbers + the event → 64 → 5")
    svg = (TGN_SVG.replace("MEMORY_OPACITY", "0.18" if is_v3 else "1").replace("SHORTCUT_OPACITY", "1" if is_v3 else "0")
           .replace("HEAD_TEXT", head_text))
    svg64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    st.markdown(f"<img alt='TGN architecture' src='data:image/svg+xml;base64,{svg64}' "
                "style='width:100%; max-width:820px; display:block; margin:0.5rem auto'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown("""**The parts, in plain words**
1. **Event**: "SOL just shocked. ETH, will you be next?"
2. **Memory**: each coin keeps a short diary of what happened to it. A *GRU* is the small network that writes it.
3. **Recent shocks**: the last 10 events each coin took part in.
4. **Attention**: decides which of those recent events matter most.
5. **Prediction head**: a small neural network that reads everything and gives the scores.
6. **Five risk scores**, one per time window.
7. Diaries are updated **only after** the hour is scored: no peeking.""")
    c2.markdown(f"""**v2 and v3, simply**
- **Same question, same answer key, same locked test.** Only the inputs and the model changed.
- **v2** used the full TGN with memory (about 70,000 learned numbers).
- In **v3** we fixed how the memory updates and gave the model better inputs: each coin's jumpiness compared with
  its *own* normal, the 5-minute summaries, the time of day, and the mood of the whole market.
- Then we tested memory on and off. **Off was slightly better**, so the final model uses only the prediction head:
  a small 2-layer network reading **{n3} numbers** about the two coins (8 of them network numbers each), the shock and
  the market, with about {w3:,} learned numbers that matter.
- Notebook 03 had already shown memory did not help at 1 hour (the "memory off" test).""")
    say("The full TGN has a memory per coin and attention over recent events. In version 3 we fixed the memory and gave "
        "the model better inputs, then tested memory on and off. Memory off was slightly better, so the final model "
        "uses only the prediction head. The gain came from better inputs.")

    pv2, in2, cfg2 = _params("tgn_1h_final_v2")
    pv3, _, cfg3 = _params("tgn_1h_v3_final")
    with st.expander("How big is each model, and how was it trained?"):
        c1, c2 = st.columns(2)
        for col, pv, title in ((c1, pv2, "v2 (memory on)"), (c2, pv3, "v3, final (memory off)")):
            with col:
                if pv is None or pv.empty:
                    st.info(f"{title}: the trained model is not in `runs/`.")
                    continue
                st.metric(title, f"{int(pv['parameters'].sum()):,} learned numbers (parameters)")
                fp = px.pie(pv, names="part", values="parameters", hole=0.55,
                            color_discrete_sequence=[TEAL, VIOLET, AMBER, GREY])
                fp.update_traces(textinfo="label+value")
                fp.update_layout(showlegend=False, margin=dict(t=10, b=10))
                if pv is pv3 and in3:
                    st.caption(f"The head is {in3} inputs wide, but with memory off the two {EMBEDDING_DIM}-number "
                               f"memory slots are always zero: {n3} inputs carry information, and about {w3:,} of "
                               "these learned numbers act on them.")
                draw_chart(fp, 280)

        def settings(cfg: dict) -> dict:
            tr, mo = cfg.get("training", {}), cfg.get("model", {})
            return {"optimiser (how it learns)": "Adam", "learning rate (step size)": tr.get("learning_rate"),
                    "loss (what it tries to reduce)": tr.get("loss"), "events per batch": tr.get("batch_size"),
                    "most rounds (epochs)": tr.get("epochs"), "stop after rounds without progress": tr.get("patience"),
                    "memory": "off" if (mo.get("overrides") or {}).get("use_memory") is False else "on"}
        if cfg2 or cfg3:
            st.dataframe(pd.DataFrame({"v2": settings(cfg2), "v3 (final)": settings(cfg3)}).astype(str),
                         width="stretch")
        st.caption("Focal loss makes the model pay more attention to the rare 'yes' answers (about 1 in 10 at 1 hour). "
                   "Each final model was trained 5 times from different random starts and the 5 were averaged.")
    takeaway("prices → shocks → network → questions → model → explanations → stress tests; the final model is a TGN "
             "whose gain came from better inputs, tested once on locked years.")

# --- 3. how coins move (notebook 01) ----------------------------------------------------------------------------------
with tab_moves:
    st.markdown("### How coins move: the facts our model is built on")
    st.markdown("<div class='plain'>Before any model, we studied the market itself. Three facts shaped every later "
                "decision.</div>", unsafe_allow_html=True)

    corr = _net("corr", version)
    if corr is not None:
        section("Notebook 01 · 02", "Fact 1: coins move together",
                "Each square shows how alike two coins' hourly moves are. <b>1</b> means they move identically, "
                "<b>0</b> means no link. Almost every square is warm: when one coin moves, most others move too.")
        sector_of = _sectors()
        order = sorted(corr.index, key=lambda c: (sector_of.get(c, "other"), c))
        cm = corr.loc[order, order]
        fh = px.imshow(cm, color_continuous_scale="Tealgrn", zmin=0, zmax=1, aspect="equal",
                       labels=dict(color="how alike"))
        fh.update_layout(margin=dict(t=10))
        draw_chart(fh, 620)
        off = cm.to_numpy()[~np.eye(len(cm), dtype=bool)]
        st.caption(f"April 2024 – September 2026 (all 30 coins trading), rush-hour pattern removed. Coins are grouped by "
                   f"sector. Average link between two coins: {np.nanmean(off):.2f}.")
        say(f"This heatmap shows how alike every pair of coins moves. Almost everything is warm; the average pair has a "
            f"correlation of {np.nanmean(off):.2f}. Crypto coins move as a pack.")

    ll = _net("lead_lag", version)
    if ll is not None and not ll.empty:
        section("Notebook 01", "Fact 2: no coin leads, not even Bitcoin",
                "If Bitcoin moved first and others followed, we would see a bump at <b>+1 hour</b> on this chart. "
                "There is none. The only peak is at <b>0</b>: the same hour. Coins move <i>together</i>, not one after "
                "another.")
        fl = px.line(ll, x="lag", y="corr", color="leader", color_discrete_map={"BTC": AMBER, "ETH": VIOLET},
                     labels={"lag": "hours later (positive = the other coin moves after BTC / ETH)",
                             "corr": "how alike (correlation)", "leader": ""})
        fl.add_vline(x=0, line_dash="dot", line_color=GREY)
        fl.update_layout(legend=dict(orientation="h", y=1.1), margin=dict(t=40))
        draw_chart(fl, 360)
        peak = ll[ll["lag"] == 0].set_index("leader")["corr"]
        rest = ll[ll["lag"] != 0]["corr"].abs().max()
        st.caption(f"Training years, typical coin. Same hour: {peak.get('BTC', np.nan):.2f} with BTC and "
                   f"{peak.get('ETH', np.nan):.2f} with ETH. Any other hour: never above {rest:.3f}, which is noise. "
                   "Formal checks (Granger tests, shuffled-time comparison) agree.")
        say("We checked whether Bitcoin leads the market. It does not. The link is strong in the same hour and "
            "basically zero one hour later. That is why our model cannot just copy Bitcoin.")

    cj = _story("contagion_jump")
    if not cj.empty:
        section("Notebook 01", "Fact 3: after a shock, more shocks follow",
                "Red: the chance a coin shocks right after <i>another</i> coin shocked. Grey: the chance in any "
                "ordinary window of the same length. The red bar is much higher at 1 hour. This gap is the "
                "contagion our model tries to predict.")
        t = cj[cj["split"] == "test"].copy()
        long = pd.concat([t.assign(case="right after another coin shocks", value=t["rate"]),
                          t.assign(case="in any ordinary window", value=t["base_rate_any_coin"])])
        long["window"] = long["k_hours"].map(dd.horizon_label)
        fc = px.bar(long, x="window", y="value", color="case", barmode="group", text=long["value"].map("{:.0%}".format),
                    color_discrete_map={"right after another coin shocks": RED, "in any ordinary window": GREY},
                    labels={"value": "chance a coin shocks", "window": "", "case": ""})
        fc.update_layout(yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.12), margin=dict(t=40))
        draw_chart(fc, 360)
        lifts = cj[cj["k_hours"] == 1].set_index("split")["lift"]
        st.caption(f"Within 1 hour the chance is {lifts.get('train', np.nan):.1f}× higher in the training years, "
                   f"{lifts.get('val', np.nan):.1f}× in validation and {lifts.get('test', np.nan):.1f}× in the test "
                   "years: stable over time. By 24 hours the gap almost disappears.")
        say(f"This is the contagion itself. After one coin shocks, another coin is about {lifts.get('test', 2.5):.1f} "
            "times more likely to shock in the next hour. The effect is strongest at one hour and fades by a day.")
    takeaway("coins move as a pack, nobody leads, and a shock makes more shocks likely in the next hour: contagion is "
             "real, but fast and market-wide.")

# --- 4. the network (notebook 02) -------------------------------------------------------------------------------------
with tab_net:
    st.markdown("### The market as a network")
    st.markdown("<div class='plain'>Think of the coins as people in a city and the links as roads. Network analysis "
                "asks: who is connected to whom, who passes shocks on, who sits in the middle, and how the city "
                "changes in a crisis. Notebook 02 asks these questions, and its answers feed our model as 8 numbers "
                "per coin.</div>", unsafe_allow_html=True)

    if mm:
        section("Notebook 02 · part 1", f"One 'market tide' explains {mm['share']:.0%} of all movement",
                "We split all the coins' movement into independent directions (eigenvalues). The first one is the "
                "whole market rising or falling together, like a tide lifting every boat. It is 18 times stronger than "
                "anything random noise could create. The others barely rise above noise.")
        c1, c2 = st.columns([3, 2])
        fe = go.Figure(go.Bar(x=list(range(1, 11)), y=mm["eigenvalues"][:10],
                              marker_color=[RED] + [GREY] * 9, text=[f"{v:.1f}" for v in mm["eigenvalues"][:10]]))
        fe.add_hline(y=mm["noise_bound"], line_dash="dot", line_color=TEAL,
                     annotation_text=f"the most pure noise could give ({mm['noise_bound']:.2f})")
        fe.update_layout(xaxis_title="direction of movement, strongest first", yaxis_title="strength (eigenvalue)",
                         margin=dict(t=20))
        with c1:
            draw_chart(fe, 330)
        c2.metric("Movement explained by the market tide", f"{mm['share']:.0%}")
        c2.markdown(f"Strength **{mm['eigenvalues'][0]:.1f}** out of a maximum of {mm['coins']}. Noise limit "
                    f"(Marchenko–Pastur): {mm['noise_bound']:.2f}. Measured over {mm['hours']:,} hours from April "
                    "2024.")
        say(f"About {mm['share']:.0%} of what any coin does in an hour is simply 'the market moved'. That is why crises "
            "hit almost every coin at once.")

    spl = _net("spillover", version)
    if spl is not None:
        section("Notebook 02 · part 2", "Who passes shocks on, and who receives them",
                "Spillover asks: when a coin is surprised, how much of that surprise came from <i>other</i> coins? "
                "Bars to the right are <b>senders</b>: they give more surprise than they receive. Bars to the left "
                "are <b>receivers</b>. This is the Diebold–Yilmaz method, a standard tool in finance.")
        tb = spl["table"].sort_values("net")
        fs = go.Figure(go.Bar(x=tb["net"], y=tb["coin"], orientation="h",
                              marker_color=[TEAL if v > 0 else RED for v in tb["net"]],
                              hovertext=[f"gives {a:.0f}, receives {b:.0f}" for a, b in zip(tb["to"], tb["from"])]))
        fs.update_layout(xaxis_title="sends minus receives (net spillover)", margin=dict(t=10))
        draw_chart(fs, 640)
        top, bottom = spl["table"].iloc[0], spl["table"].iloc[-1]
        st.caption(f"April 2024 – September 2026, 24-hour horizon. Overall, {spl['total']:.0f}% of each coin's "
                   f"surprises come from other coins (the spillover index). Biggest sender: {top['coin']}. "
                   f"Biggest receiver: {bottom['coin']}.")
        say(f"{spl['total']:.0f}% of every coin's surprises come from other coins. Big, well-connected platform coins "
            f"like {top['coin']} pass shocks on; coins with their own stories, like {bottom['coin']}, mostly receive.")

    if P is not None:
        section("Notebook 02 · part 3", "The coin network today",
                "Each dot is a coin, coloured by what it is used for. A line joins two coins that moved closely "
                "together over the past week. <b>Bigger dots are more central</b>: they are linked to coins that are "
                "themselves well linked (eigenvector centrality, the same idea Google used to rank web pages).")
        edges, _ = _graph(S)
        ranks = _rankings(RANKINGS_VERSION)
        sector_of = ranks.set_index("coin")["sector"].to_dict() if not ranks.empty else _sectors()
        central = ranks.set_index("coin")["eigenvector"] if "eigenvector" in ranks else pd.Series(dtype=float)
        coins = sorted(set(P.risk["coin"]) | set(edges["a"]) | set(edges["b"]))
        fig, pos = network_figure(coins, edges, sector_of)
        colors = sector_colors(sector_of.get(c, "other") for c in coins)
        for sec in sorted({sector_of.get(c, "other") for c in coins}):
            cs = [c for c in coins if sector_of.get(c, "other") == sec]
            size = [10 + 14 * float(central.get(c, 1.0)) for c in cs] if not central.empty else 14
            fig.add_trace(go.Scatter(x=[pos[c][0] for c in cs], y=[pos[c][1] for c in cs], mode="markers+text", name=sec,
                                     text=cs, textposition="top center",
                                     marker=dict(size=size, color=colors[sec], line=dict(width=1, color="#333")),
                                     hoverinfo="text", hovertext=[f"{c} ({sec})" for c in cs]))
        fig.update_layout(legend_title="What the coin is used for")
        draw_chart(fig)
        if not ranks.empty:
            with st.expander("Rankings: the most central coins and the biggest senders"):
                cols = {"coin": "coin", "sector": "sector", "eigenvector": "centrality",
                        "dy_net": "sends (+) or receives (−)", "shock_pagerank": "others follow its shocks (PageRank)",
                        "mst_degree": "links in the skeleton"}
                st.dataframe(ranks[[c for c in cols if c in ranks]].rename(columns=cols)
                             .sort_values("centrality", ascending=False).round(3), hide_index=True, width="stretch")
        say("This is the market as a network today. Lines are strong links, bigger dots are more central coins. Notice "
            "how dense it is: almost everything is connected.")

    tree = _net("mst", version)
    if tree is not None:
        section("Notebook 02 · part 4", "The market's skeleton",
                f"The full network has hundreds of links. The skeleton keeps only the <b>{len(tree['edges'])} "
                "strongest links</b> that still connect every coin, with no loops (a minimum spanning tree). Coins with "
                "many links are the hubs that hold the market together.")
        sector_of = _sectors()
        colors = sector_colors(sector_of.get(c, "other") for c in tree["pos"])
        ft = go.Figure()
        for e in tree["edges"].itertuples():
            (x0, y0), (x1, y1) = tree["pos"][e.a], tree["pos"][e.b]
            ft.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="text", showlegend=False,
                                    text=f"{e.a}–{e.b}: correlation {e.corr:.2f}",
                                    line=dict(width=1 + 5 * max(e.corr - 0.4, 0), color="rgba(200,206,214,0.5)")))
        for sec in sorted(colors):
            cs = [c for c in tree["pos"] if sector_of.get(c, "other") == sec]
            ft.add_trace(go.Scatter(x=[tree["pos"][c][0] for c in cs], y=[tree["pos"][c][1] for c in cs],
                                    mode="markers+text", text=cs, textposition="top center", name=sec,
                                    marker=dict(size=[10 + 3 * tree["degree"][c] for c in cs], color=colors[sec],
                                                line=dict(width=1, color="#222")),
                                    hovertext=[f"{c}: {tree['degree'][c]} links" for c in cs], hoverinfo="text"))
        ft.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h", y=-0.02),
                         margin=dict(t=10, b=10))
        draw_chart(ft, 620)
        hubs = sorted(tree["degree"].items(), key=lambda x: -x[1])[:3]
        e0 = tree["edges"].iloc[0]
        st.caption("April 2024 – September 2026. Thicker lines are stronger links; bigger dots have more links. Hubs: "
                   + ", ".join(f"{c} ({d} links)" for c, d in hubs) + f". Tightest pair: {e0['a']}–{e0['b']} "
                   f"(correlation {e0['corr']:.2f}).")
        say(f"The skeleton keeps only the strongest links. {hubs[0][0]} and {hubs[1][0]} are the hubs; meme coins "
            "hang together on their own branch.")

    ranks_c = _rankings(RANKINGS_VERSION)
    com = sn.communities(ranks_c) if not ranks_c.empty else None
    if com is not None:
        section("Notebook 02 · part 5", "Natural groups of coins partly match their sectors",
                "We first removed the market tide, then let an algorithm (Louvain) group coins that still move "
                "together. Do these groups match what the coins are used for? Partly. The match score is "
                f"<b>{com['nmi']:.2f}</b> (0 = no match, 1 = perfect). Shuffling the sectors at random gives "
                f"{com['null_mean']:.2f} on average, so the match is real, not luck.")
        groups = com["groups"]
        st.markdown("<div class='cards'>" + "".join(
            f"<div class='card'><b>Group {i + 1}</b><span>{g}</span></div>" for i, g in enumerate(groups)) + "</div>",
                    unsafe_allow_html=True)
        st.caption(f"Match score: normalised mutual information {com['nmi']:.3f}; chance of a match this good by luck "
                   f"p = {com['p']:.3f} (2,000 random shuffles).")
        say("Once you remove the market tide, coins form natural groups. Meme coins form their own group. The groups "
            "match sectors better than chance, but not perfectly.")

    ts = _story("network_timeseries")
    if not ts.empty:
        section("Notebook 02 · part 6", "In every crisis, the market pulls together",
                "We rebuilt the network every day from 2020 to 2026. Red lines mark the big crises. After each one, "
                "coins moved more alike and the skeleton got shorter: the network tightens under stress.")
        metrics = {"mean_corr": "how alike coins move", "total_spillover": "spillover (%)",
                   "tree_length": "skeleton length (shorter = tighter)", "density": "share of coin pairs linked"}
        pick = st.segmented_control("Measure", list(metrics), default="mean_corr", format_func=metrics.get,
                                    key="net_metric")
        pick = pick or "mean_corr"
        sm = ts.set_index("time")[pick].rolling(14, min_periods=5).mean()
        fn = go.Figure(go.Scatter(x=sm.index, y=sm.values, mode="lines", line=dict(color=TEAL, width=2)))
        cd = _story("crisis_dates")
        for e in cd.itertuples():
            fn.add_vline(x=e.date, line_dash="dot", line_color=RED, opacity=0.6)
            fn.add_annotation(x=e.date, y=1, yref="paper", text=e.name.split("/")[0][:22], showarrow=False,
                              textangle=-90, xanchor="right", yanchor="top", font=dict(color=RED, size=10))
        fn.update_layout(yaxis_title=metrics[pick], margin=dict(t=20))
        draw_chart(fn, 400)
        if not cd.empty:
            cc = sn.crisis_changes(ts, cd)
            show = ["average correlation", "tree length", "share of pairs linked", "spillover index"]
            cc = cc[["crisis"] + [c for c in show if c in cc]]
            up = int((cc["average correlation"] > 0).sum()) if "average correlation" in cc else 0
            shorter = int((cc["tree length"] < 0).sum()) if "tree length" in cc else 0
            st.markdown(f"**Two weeks after vs two weeks before each crisis:** coins moved more alike in **{up} of "
                        f"{len(cc)}** crises, and the skeleton got shorter (tighter) in **{shorter} of {len(cc)}**.")
            st.dataframe(cc.rename(columns={"average correlation": "how alike (change)",
                                            "tree length": "skeleton length (change)",
                                            "share of pairs linked": "pairs linked (change)",
                                            "spillover index": "spillover (change, points)"})
                         .style.format(precision=2).background_gradient(cmap="RdYlGn", axis=None, vmin=-0.5, vmax=0.5),
                         hide_index=True, width="stretch")
            say(f"Every time there is a crisis, the network pulls together. In {up} of {len(cc)} crises coins moved "
                "more alike afterwards. Diversification disappears exactly when you need it.")

    scg = _net("sector_contagion", version)
    if scg is not None and not scg.empty:
        section("Notebook 02 · part 7", "Same sector does not mean more contagion (except meme coins)",
                "You might expect a shock to spread mostly to coins of the same kind. It does not: the chance is the "
                "same for same-sector and different-sector pairs. Shocks spread market-wide. The one exception is "
                "<b>meme coins</b>, which follow each other more often.")
        tst = scg[scg["period"] == "test"].copy()
        tst["kind"] = np.where(tst["pairs"].str.startswith("inside"), "inside one sector", "overall")
        fg = px.bar(tst, x="rate", y="pairs", orientation="h", color="kind", text=tst["rate"].map("{:.1%}".format),
                    color_discrete_map={"overall": TEAL, "inside one sector": GREY},
                    labels={"rate": "chance the other coin shocks within 1 hour", "pairs": "", "kind": ""})
        fg.update_layout(xaxis_tickformat=".0%", yaxis=dict(categoryorder="total ascending"),
                         legend=dict(orientation="h", y=1.1), margin=dict(t=40))
        draw_chart(fg, 380)
        tr = scg[scg["period"] == "train"].set_index("pairs")["rate"]
        st.caption(f"Test years. Training years: same sector {tr.get('same sector', np.nan):.1%}, different sectors "
                   f"{tr.get('different sectors', np.nan):.1%}, any hour {tr.get('any hour (base rate)', np.nan):.1%}.")
        say("We expected shocks to stay inside sectors. They do not: same-sector and cross-sector pairs are equally "
            "likely to follow. Only meme coins follow each other more.")

    ftx = _net("ftx_hours", version)
    if ftx is not None and not ftx.empty:
        section("Notebook 02 · part 8", "A real example: the FTX collapse, hour by hour",
                "FTT was the token of the FTX exchange. On <b>6 November 2022</b> it crashed alone: no other coin "
                "moved. On <b>8 November</b>, when FTX's troubles became public, the crash spread: within hours "
                "almost every coin fell. Contagion needs news to spread, not just a falling coin.")
        ff = go.Figure()
        ff.add_trace(go.Bar(x=ftx["time"], y=ftx["down"], name="coins crashing (shock down)", marker_color=RED))
        ff.add_trace(go.Bar(x=ftx["time"], y=ftx["up"], name="coins jumping (shock up)", marker_color=GREEN))
        ff.add_trace(go.Scatter(x=ftx["time"], y=ftx["ftt_z"], name="FTT's move (in normal swings)", yaxis="y2",
                                line=dict(color=AMBER, width=2)))
        ff.update_layout(barmode="stack", yaxis=dict(title="number of coins"),
                         yaxis2=dict(title="FTT's move", overlaying="y", side="right", showgrid=False),
                         legend=dict(orientation="h", y=1.12), margin=dict(t=40))
        draw_chart(ff, 400)
        st.caption(f"6–11 November 2022, every hour. {int(ftx['trading'].max())} model coins were trading. FTT itself "
                   "is not one of the model's coins; we follow it as the trigger.")
        say("FTX is the clearest example. FTT crashed on the sixth of November and nothing else moved. Two days later "
            "the news broke, and within a few hours almost every coin crashed. That is contagion, hour by hour.")

    section("Notebook 02 · part 9", "What the network gives the model",
            "Network analysis is not just a side study. For every coin, every hour, we compute 8 network numbers over "
            "the past week and feed them to the model.")
    st.dataframe(pd.DataFrame(sn.NETWORK_FEATURES.items(), columns=["network number", "what it means"]),
                 hide_index=True, width="stretch")
    cas = lb[(lb["id"] == "independent cascade")] if not lb.empty else lb
    if not cas.empty:
        c1 = cas[cas["k_hours"] == 1].iloc[0]
        st.markdown(f"We also turned the network into a classic **spreading model** (independent cascade: each "
                    f"shocked coin 'infects' its neighbours with a learned probability). It scores **{c1['auprc']:.3f}** "
                    f"at 1 hour, {c1['lift']:.1f}× better than guessing. That is the bar our deep learning model has to "
                    "beat (tab 5).")
    say("All of this becomes input: eight network numbers per coin per hour. We also built a classic spreading model "
        "from the network, as a baseline our deep learning model has to beat.")
    takeaway("one market tide dominates, big platform coins pass shocks on, the network tightens in every crisis, and "
             "shocks spread market-wide rather than within sectors; these findings shape the model's inputs.")

# --- 5. model & results (notebook 03 + v3) ----------------------------------------------------------------------------
with tab_results:
    st.markdown("### Results: does the model work?")
    st.markdown("<div class='plain'>All models answer the same questions on the same locked test years (January "
                "2024 – September 2026). The score is <b>AUPRC</b>: from 0 to 1, how well a model puts the coins that "
                "really shocked at the top of its list. <b>Guessing</b> scores about 0.10 at 1 hour, because about 1 in "
                "10 answers is 'yes'. So we always show the guessing line.</div>", unsafe_allow_html=True)
    if lb.empty:
        st.info("The result files are missing from `reports/`.")
    else:
        section("Notebook 03 + v3", f"The scoreboard (within {H})",
                "Higher is better. Teal bars are graph deep learning models, amber bars are classic machine learning, "
                "purple is the network spreading model. Change the time window with <b>Look ahead</b> at the top.")
        lbh = lb[lb["k_hours"] == h]
        fig = px.bar(lbh, x="auprc", y="model", orientation="h", color="family", color_discrete_map=FAMILY_COLOR,
                     text=lbh["auprc"].map("{:.3f}".format),
                     labels={"auprc": f"test score within {H} (AUPRC)", "model": "", "family": ""})
        fig.add_vline(x=float(lbh["random"].iloc[0]), line_dash="dot", line_color=GREY)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), legend=dict(orientation="h", y=1.1),
                          margin=dict(t=50, l=10))
        draw_chart(fig, 460)
        st.caption(f"The dotted line is guessing ({lbh['random'].iloc[0]:.3f}).")
        if v3:
            say(f"Our final model, TGN version 3, scores {v3['auprc']:.3f} at one hour, against {v3['random']:.3f} for "
                f"guessing. That is {v3['lift']:.1f} times better, and it is the top of the table.")

    if not bt.empty:
        section("v3", "From v2 to v3: a real improvement",
                "Same questions, same test. v3 scores higher at every time window. To check it is not luck, we "
                "re-ran the comparison 1,000 times on randomly re-drawn test days (a bootstrap). Whole days are "
                "drawn together, because one crash creates hundreds of linked events.")
        c1, c2 = st.columns(2)
        long = pd.concat([bt.assign(model="v2 (report)", score=bt["v2"]), bt.assign(model="v3 (final)", score=bt["v3"])])
        long["window"] = long["k_hours"].map(dd.horizon_label)
        fv = px.bar(long, x="window", y="score", color="model", barmode="group", text=long["score"].map("{:.3f}".format),
                    color_discrete_map={"v2 (report)": GREY, "v3 (final)": TEAL},
                    labels={"score": "test score (AUPRC)", "window": "", "model": ""})
        fv.update_layout(legend=dict(orientation="h", y=1.12), margin=dict(t=40))
        with c1:
            draw_chart(fv, 360)
        fb = go.Figure(go.Scatter(x=bt["k_hours"].map(dd.horizon_label), y=bt["diff"], mode="markers",
                                  marker=dict(size=12, color=TEAL),
                                  error_y=dict(type="data", symmetric=False, array=bt["ci_hi"] - bt["diff"],
                                               arrayminus=bt["diff"] - bt["ci_lo"])))
        fb.add_hline(y=0, line_dash="dot", line_color=RED, annotation_text="no improvement")
        fb.update_layout(yaxis_title="v3 minus v2 (with 95% range)", margin=dict(t=20))
        with c2:
            draw_chart(fb, 360)
        b1 = bt[bt["k_hours"] == 1].iloc[0] if (bt["k_hours"] == 1).any() else bt.iloc[0]
        st.caption(f"At 1 hour: {b1['v2']:.3f} → {b1['v3']:.3f} (+{b1['diff']:.3f}; 95% range {b1['ci_lo']:+.3f} to "
                   f"{b1['ci_hi']:+.3f}). v3 was ahead in {b1['share_v3_better']:.1%} of the 1,000 re-runs. Every range "
                   "stays above zero.")
        say(f"Version 3 beats version 2 at every window. At one hour it goes from {b1['v2']:.3f} to {b1['v3']:.3f}, and "
            f"it wins in {b1['share_v3_better']:.1%} of a thousand re-runs, so the improvement is real.")

    if not lb.empty:
        section("Notebook 03", "How much better than guessing, at every window",
                "Longer windows look better in raw score only because more coins move anyway. Dividing by the "
                "guessing level shows the real skill. It is highest at 1 hour, exactly where contagion is strongest.")
        keep = ["TGN v3 (final model)", "TGN v2 (in the report)", "DyGFormer (transformer)", "Logistic regression",
                "Gradient boosting", "Cascade (classic spreading model)"]
        line_colors = dict(zip(keep, [TEAL, "#7fdcd3", VIOLET, AMBER, "#ff8a3d", GREY]))
        flt = px.line(lb[lb["model"].isin(keep)], x="k_hours", y="lift", color="model", markers=True,
                      color_discrete_map=line_colors, category_orders={"model": keep},
                      labels={"k_hours": "time window (hours)", "lift": "times better than guessing", "model": ""})
        flt.add_hline(y=1, line_dash="dot", line_color=GREY, annotation_text="guessing")
        flt.update_xaxes(type="category")
        draw_chart(flt, 380)

    fs = _story("fair_scores")
    f3 = fs[(fs["model"] == "tgn_1h_v3_final") & (fs["k_hours"] == 1)] if not fs.empty else fs
    if len(f3):
        r = f3.iloc[0]
        section("Fair checks", "Is the score honest?",
                "In a crash hour, one coin can be asked the same question several times (once per shocking coin). "
                "So we also scored each coin-hour only once, and only on coins that were calm the day before. The "
                "model stays well ahead of guessing in every check.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Every question", f"{r['event_lift']:.1f}× guessing")
        c2.metric("Each coin-hour once", f"{r['rh_lift']:.1f}× guessing")
        c3.metric("Calm coins only", f"{r['calm_lift']:.1f}× guessing")
        c4.metric("Top 1% warnings that came true", f"{r['top1_precision']:.0%}")
        st.caption(f"Test years, 1 hour, final model. AUROC (another standard score, 0.5 = guessing): "
                   f"{r['event_auroc']:.2f}.")
        say(f"We checked for tricks in our own score. Counting each coin-hour once, the model is still "
            f"{r['rh_lift']:.1f} times better than guessing, and {r['top1_precision']:.0%} of its strongest warnings "
            "came true.")

    q = _story("fifteen_minutes")
    if not q.empty:
        section("v3", "Contagion is faster than an hour",
                "We asked the same question on 15-minute data. The closer to the shock, the easier it is to predict. "
                "Contagion happens in minutes, which points to the next step for this project.")
        ql = q[q["model"] == "logreg"].copy()
        ql["window"] = ql["minutes"].map(lambda x: f"{x} min")
        fq = px.bar(ql, x="window", y="lift", text=ql["lift"].map("{:.1f}×".format), color_discrete_sequence=[TEAL],
                    labels={"lift": "times better than guessing", "window": ""})
        fq.add_hline(y=1, line_dash="dot", line_color=GREY)
        draw_chart(fq, 320)
        st.caption("Test years, scored once, with a simple model (logistic regression) on the receiving coin's state and "
                   "the market's mood.")

    section("Being honest", "What the results do not show",
            "A simple model (logistic regression) comes close at 1 hour. That tells us most of the hourly signal is "
            "in each coin's own state: is it already busy and jumpy? The graph adds most after market-wide shocks "
            "(notebook 05). And most shocks start with news nobody can predict, so no model gets near a perfect "
            "score.")
    say("We are honest about the limits. A simple model comes close at one hour, because most of the signal is in the "
        "coin's own state. The graph model helps most when the shock is market-wide.")
    takeaway("the final TGN is the best model at every window, about twice as good as guessing, and the v3 gain over v2 "
             "is statistically real; contagion is fast, so finer data is the next step.")

# --- 6. why, how long, how safe (notebooks 04, 05, 06) ------------------------------------------------------------------
with tab_why:
    st.markdown("### Why it warns, how long shocks last, and whether it can be fooled")
    g = _story("attribution_groups")
    if not g.empty:
        section("Notebook 04", "Why the model raises a warning",
                "For the model's strongest warnings we measured how much each input pushed the score up (integrated "
                "gradients: like asking each input 'how much did you change the answer?'). Most of the reason is the "
                "<b>receiving coin itself</b>: it is already trading a lot and jumping around.")
        c1, c2 = st.columns([1, 1.4])
        fg = px.pie(g, names="group", values="share_of_abs", hole=0.55, color_discrete_sequence=[TEAL, AMBER, GREY])
        fg.update_traces(textinfo="label+percent")
        fg.update_layout(showlegend=False, margin=dict(t=10, b=10))
        with c1:
            draw_chart(fg, 300)
        ti = _story("top_inputs")
        if not ti.empty:
            ti = ti.assign(label=[f"{x.split(': ')[0]}: {EVENT_PLAIN.get(x.split(': ')[-1], dd.feature_plain(x.split(': ')[-1]))}"
                                  for x in ti["input"]])
            fi = px.bar(ti.iloc[::-1], x="share_of_abs", y="label", orientation="h", color_discrete_sequence=[TEAL],
                        labels={"share_of_abs": "share of the warning's reasons", "label": ""})
            fi.update_layout(xaxis_tickformat=".0%", margin=dict(t=10))
            with c2:
                draw_chart(fi, 300)
        recv = g.set_index("group")["share_of_abs"].get("receiver coin")
        say(f"When the model warns, about {recv:.0%} of the reason is the receiving coin's own state, mainly a surge in "
            "trading volume and recent jumpiness. Busy, nervous coins are the next to fall." if recv is not None
            else "Most of the reason for a warning is the receiving coin's own state.")

    fd = _story("explainer_fidelity")
    if not fd.empty:
        section("Notebook 04", "Are there chains of events behind a warning? No.",
                "TempME is a 2023 research method that looks for chains of past events behind a prediction, like "
                "'A hit B, then B hit C'. It did barely better than picking events at random. That is a finding, not "
                "a failure: shocks hit the whole market in the same hour, so there are no special chains to find.")
        ff = px.bar(fd, x="characterization", y="explainer", orientation="h",
                    color=fd["explainer"].eq("TempME").map({True: TEAL, False: GREY}), color_discrete_map="identity",
                    labels={"characterization": "how well 3 chosen past events explain the warning (higher is better)",
                            "explainer": ""})
        ff.update_layout(margin=dict(t=10))
        draw_chart(ff, 300)
        say("We ported TempME, a NeurIPS 2023 explainer, to look for chains of events. It found almost none, which "
            "matches the network analysis: shocks hit everything in the same hour.")

    if not pers.empty:
        section("Notebook 05", "How long a shock lasts",
                "After a shock, how many hours until the coin is back to its normal jumpiness? Only hours, for every "
                "kind of shock. That is why we predict 1 to 24 hours ahead: it covers a shock's whole life.")
        pt = pers.copy()
        pt["kind"] = pt["direction"].map({"crash": "drops", "pump": "jumps"}).radd(
            pt["type"].map({"market-wide": "market-wide ", "partial": "partial ", "idiosyncratic": "single-coin "}))
        fp = go.Figure(go.Bar(x=pt["kind"], y=pt["median_hours_to_normal"], marker_color=TEAL,
                              text=pt["median_hours_to_normal"].map("{:.0f} h".format),
                              error_y=dict(type="data", symmetric=False,
                                           array=pt["q75_hours"] - pt["median_hours_to_normal"],
                                           arrayminus=pt["median_hours_to_normal"] - pt["q25_hours"])))
        fp.update_layout(yaxis_title="hours until back to normal", margin=dict(t=10))
        draw_chart(fp, 320)
        st.caption("Bars: the typical shock. Lines: the middle half of shocks. 'Market-wide' means most coins shocked "
                   "together; 'single-coin' means the coin moved on its own.")
        say(f"A shock is short. Within about {pers['median_hours_to_normal'].min():.0f} to "
            f"{pers['median_hours_to_normal'].max():.0f} hours the coin is back to normal. Market-wide crashes drag "
            "the most other coins with them.")

    stx = _story("stress_tests")
    if not stx.empty:
        section("Notebook 06", "Can the model be fooled?",
                "We damaged the inputs on purpose. <b>Random noise</b> and <b>missing events</b> (like a data feed "
                "dropping out) barely matter. A <b>deliberate attack</b>, which nudges every input in the most "
                "harmful direction, cuts the score sharply. The real risk is manipulation, not ordinary data errors.")
        tg = stx[(stx["model"] == "TGN") & (stx["perturbation"] != "none")].copy()
        tg["test"] = tg["perturbation"].map(lambda p: "deliberate attack (one step, FGSM)" if p.startswith("FGSM")
                                            else "deliberate attack (10 steps, PGD)" if p.startswith("PGD")
                                            else "random noise" if p == "input noise" else "missing events")
        fs_ = px.line(tg, x="level", y="change_k1h", color="test", markers=True,
                      color_discrete_map={"deliberate attack (one step, FGSM)": RED,
                                          "deliberate attack (10 steps, PGD)": AMBER,
                                          "random noise": TEAL, "missing events": GREEN},
                      labels={"level": "size of the damage (in typical swings, or share of events missing)",
                              "change_k1h": "change in the 1-hour score", "test": ""})
        fs_.update_layout(yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.12), margin=dict(t=40))
        draw_chart(fs_, 380)
        st.caption("The v2 TGN on the test years.")
        say("Losing half of the past events changes almost nothing, and random noise hurts little. But a deliberate, "
            "targeted attack can fool it, like most neural networks. A real system would need defences.")

    tga = _tgae()
    if tga is not None:
        section("Notebook 06", "Can a network autoencoder spot unusual days? Only the most extreme one.",
                "A temporal graph autoencoder (T-GAE) learns what a normal day's network looks like and flags days it "
                "cannot rebuild well. It did not recognise ordinary stressful days, but it did flag the "
                "<b>10 October 2025 liquidation crash</b> as the most unusual day of the test years.")
        c1, c2 = st.columns([1, 1.5])
        au = pd.DataFrame([{"signal": k, "question": q, "score": v} for q, d in
                           (("same day", tga["same_day"]), ("next day", tga["next_day"])) for k, v in d.items()])
        fa = px.bar(au, x="question", y="score", color="signal", barmode="group", text=au["score"].map("{:.2f}".format),
                    color_discrete_map={"T-GAE anomaly score": VIOLET, "plain turbulence": GREY},
                    labels={"score": "spots stress days (AUROC, 0.5 = guessing)", "question": "", "signal": ""})
        fa.add_hline(y=0.5, line_dash="dot", line_color=RED)
        fa.update_layout(legend=dict(orientation="h", y=1.15), margin=dict(t=40), yaxis_range=[0, 1])
        with c1:
            draw_chart(fa, 340)
        sc = tga["scores"]
        sc = sc[sc["split"] == "test"]
        fsc = go.Figure(go.Scatter(x=sc["time"], y=sc["score"], mode="lines", line=dict(color=VIOLET, width=1.5)))
        top = sc.nlargest(1, "score")
        fsc.add_trace(go.Scatter(x=top["time"], y=top["score"], mode="markers+text", text=["10 Oct 2025 crash"],
                                 textposition="middle left", marker=dict(size=12, color=RED)))
        fsc.update_layout(showlegend=False, yaxis_title="how unusual the day was", margin=dict(t=10))
        with c2:
            draw_chart(fsc, 340)
        say("We also tried an anomaly detector on the network. It does not recognise normal stressful days; simple "
            "turbulence does that better. But it did single out the biggest crash of the test years.")

    section("Wrap-up", "Limits, and what comes next", "")
    c1, c2 = st.columns(2)
    c1.markdown("""**Limits**
- Most shocks start with news nobody can predict, so scores stay modest.
- One exchange (Binance) and one test period; the 30 coins come from a 2026 ranking.
- A simple model is close behind at one hour; memory did not help at hourly speed.
- A deliberate attack can fool the model.""")
    c2.markdown("""**Next steps**
- Run the graph model on 15-minute data, where contagion actually happens.
- Test on a second period, retraining as the market changes.
- Add scheduled news and order-book data.
- Defend against manipulation: compare prices across exchanges, train on attacked examples.""")
    takeaway("warnings come from the receiving coin's own stress, shocks fade within hours, and the model survives noise "
             "and gaps but needs defending against deliberate manipulation.")

# --- 7. crises & what if ------------------------------------------------------------------------------------------------
with tab_crises:
    ev_dir = dd.latest_events_dir(RUNS)
    if ev_dir is None:
        st.info("No past crises yet. Run `cryptograph-train --config configs/experiment/predict_events_1h_v3.yaml`, "
                "then reload this page.")
    else:
        E = _events_index(str(ev_dir))
        st.markdown("### Nine real crises, replayed hour by hour")
        st.markdown("<div class='plain'>Each crisis is replayed through the model at its worst hour. We compare the "
                    "model's five riskiest coins with five coins picked at random. Only the <b>test-period</b> crises "
                    "are true forecasts; the model learned from the others.</div>", unsafe_allow_html=True)
        tl = pd.DataFrame([{"date": pd.Timestamp(e["date"]), "crisis": e["name"], "period": e["split"],
                            "coins hit at worst hour": e["replay_metrics"]["coins_shocking_at_anchor"],
                            "coins trading": e["replay_metrics"]["coins_active_at_anchor"]} for e in E["events"]])
        tl["share"] = tl["coins hit at worst hour"] / tl["coins trading"]
        tl["label"] = tl["crisis"].str.split("/").str[0].str.strip().str.slice(0, 26)
        tl["y"] = [0.35 if i % 2 else -0.35 for i in range(len(tl))]            # alternate rows so labels don't collide
        ft = px.scatter(tl, x="date", y="y", size="share", color="period", text="label",
                        color_discrete_map=SPLIT_COLOR, size_max=34)
        ft.update_traces(textposition=["bottom center" if y < 0 else "top center" for y in tl["y"]],
                         textfont=dict(size=11), hovertemplate="%{text}<extra></extra>")
        ft.update_layout(yaxis=dict(visible=False, range=[-1.2, 1.2]), xaxis_title=None, legend=dict(orientation="h", y=1.15),
                         margin=dict(t=30, b=10))
        draw_chart(ft, 260)

        rows = []
        for e in E["events"]:
            mv = pd.DataFrame(_event(str(ev_dir), e["id"])["model_vs_outcome"])
            r0 = mv[mv["horizon_h"] == h] if len(mv) else mv
            if len(r0):
                r0 = r0.iloc[0]
                _, sentence = dd.crisis_verdict(int(r0["top_hits"]), int(r0["top_n"]), float(r0["expected_by_chance"]),
                                                float(r0["share_shocked"]))
                rows.append({"crisis": f"{pd.Timestamp(e['date']):%b %Y} · {e['name']}", "period": e["split"],
                             "model's 5 that moved": int(r0["top_hits"]), "5 random coins": r0["expected_by_chance"],
                             "coins that moved": r0["share_shocked"],
                             "verdict": SHORT_VERDICT.get(sentence.split(":")[0].split(",")[0], sentence)})
        if rows:
            with st.expander(f"All nine crises at a glance ({H})", expanded=True):
                st.dataframe(pd.DataFrame(rows).style.format({"5 random coins": "{:.1f}", "coins that moved": "{:.0%}"}),
                             hide_index=True, width="stretch")
                st.caption("In big crashes almost every coin moves, so no list of five can do better than random: such "
                           "crises cannot judge any model. The real test is the hundreds of thousands of test-period "
                           "events (tab 5).")

        label_of = {e["id"]: f"{pd.Timestamp(e['date']):%b %Y} · {e['name']}" for e in E["events"]}
        eid = st.selectbox("Choose a past crisis", list(label_of), format_func=label_of.get, key="event")
        ev = _event(str(ev_dir), eid)
        rp, sector_of = ev["replay"], _sectors()
        coins_ev = [x["coin"] for x in rp["sequence"]]
        st.markdown(ev["description"] + (f" [Read more]({ev['source_url']})" if ev["source_url"] else ""))
        if ev.get("split_note"):
            st.caption(ev["split_note"])
        m = rp["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Worst hour (UTC, {ev['anchor_time'][:4]})", f"{pd.Timestamp(ev['anchor_time']):%d %b, %H:%M}",
                  help="The hour of that day when the most coins made a big move at once (UTC).")
        c2.metric("Coins hit in that hour", f"{m['coins_shocking_at_anchor']} of {m['coins_active_at_anchor']}")
        c3.metric("Coins hit within a week", f"{m['reach']:.0%}")

        st.markdown(f"#### Did the model see it coming? (next {H})")
        mv = ev["model_vs_outcome"]
        row = mv[mv["horizon_h"] == h] if len(mv) else mv
        if len(row):
            r0 = row.iloc[0]
            kind, verdict = dd.crisis_verdict(int(r0["top_hits"]), int(r0["top_n"]), float(r0["expected_by_chance"]),
                                              float(r0["share_shocked"]))
            st.markdown(f"In the worst hour, the model flagged **{int(r0['top_n'])} coins** as most at risk. "
                        f"**{int(r0['top_hits'])} of them** made a big move within {H}. Picking "
                        f"{int(r0['top_n'])} coins at random would have caught about "
                        f"**{r0['expected_by_chance']:.1f}**.")
            {"good": st.success, "neutral": st.info, "weak": st.warning}[kind](verdict)
        else:
            st.info(f"We don't know what happened within {H} for this crisis.")

        st.markdown("#### Watch the crisis spread, hour by hour")
        st.caption("Press ▶ Play under the picture. Each dot is a coin; lines join coins that moved closely together "
                   "in the week before.")
        edges_ev, _ = _graph({"as_of_bar": ev["anchor_bar"], "data_version": ev["data_version"],
                              "resolution": ev["resolution"]})
        fig, pos = network_figure(coins_ev, edges_ev, sector_of, height=700)
        xs, ys = [pos[c][0] for c in coins_ev], [pos[c][1] for c in coins_ev]
        states = dd.node_states(rp["frames"], len(coins_ev))

        def node_tr(i):
            return go.Scatter(x=xs, y=ys, mode="markers+text", text=coins_ev, textposition="top center",
                              marker=dict(size=18, color=[STATE_COLOR[s] for s in states[i]],
                                          line=dict(width=1, color="#222")),
                              hovertext=[f"{c}: {STATE_LABEL[s]}" for c, s in zip(coins_ev, states[i])],
                              hoverinfo="text", showlegend=False)

        i0 = -rp["window"][0]                                          # the worst hour
        edge_tr = fig.data[0]
        fig.add_trace(node_tr(i0))
        for s in dd.NODE_STATES:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=STATE_COLOR[s]),
                                     name=STATE_LABEL[s]))
        fig.frames = [go.Frame(data=[edge_tr, node_tr(i)], traces=[0, 1], name=str(f["offset"]))
                      for i, f in enumerate(rp["frames"])]
        play = dict(frame=dict(duration=250, redraw=True), fromcurrent=True, transition=dict(duration=0))
        pause = dict(frame=dict(duration=0, redraw=False), mode="immediate")
        fig.update_layout(
            legend=dict(orientation="h", y=1.0, yanchor="bottom"), margin=dict(t=40, b=120),
            updatemenus=[dict(type="buttons", direction="left", x=0, xanchor="left", y=0, yanchor="top",
                              pad=dict(t=45), bgcolor="#262730", bordercolor="#888", font=dict(color="#fafafa"),
                              showactive=False,
                              buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                       dict(label="⏸ Pause", method="animate", args=[[None], pause])])],
            sliders=[dict(active=i0, x=0.16, len=0.84, y=0, yanchor="top",
                          currentvalue=dict(prefix="hours after the worst hour: "), pad=dict(t=30),
                          steps=[dict(label=f"{f['offset']:+d}", method="animate",
                                      args=[[str(f["offset"])], dict(mode="immediate",
                                                                      frame=dict(duration=0, redraw=True))])
                                 for f in rp["frames"]])])
        draw_chart(fig)

        act = pd.DataFrame(rp["activity"])
        fa = px.bar(act, x="offset", y="coins_shocking", hover_data=["time"],
                    labels={"offset": "hours after the worst hour", "coins_shocking": "coins making a big move"},
                    title="How many coins made a big move, hour by hour")
        fa.add_vline(x=0, line_dash="dash", line_color="#d62728")
        fa.update_layout(margin=dict(t=40, b=10))
        draw_chart(fa, 280)
        say("Here you can watch a real crisis spread across the network, hour by hour. Red is a coin crashing right "
            "now, orange is a coin that already crashed.")

        with st.expander("Details: every coin during this crisis"):
            ot = dd.outcome_table(ev["risk"], h)
            ot = ot.assign(result=ot["outcome"].map({"shocked": "✅ big move", "no shock": "❌ no big move"})
                           .fillna("— unknown"))
            st.dataframe(ot[["coin", "sector", "risk", "result"]].rename(
                columns={"risk": f"model's chance ({H})", "result": "what happened"}).style.format(
                {f"model's chance ({H})": "{:.0%}"}), hide_index=True, width="stretch")
            pick = st.multiselect("Sectors", sorted({sector_of.get(c, "other") for c in coins_ev}),
                                  key="event_sectors", placeholder="All sectors")
            seq = pd.DataFrame(rp["sequence"]).assign(sector=lambda x: x["coin"].map(sector_of).fillna("other"))
            if pick:
                seq = seq[seq["sector"].isin(pick)]
            seq = seq.sort_values(["first_shock_offset_h", "coin"], na_position="last")
            st.dataframe(seq.drop(columns=["active_at_anchor"]).rename(columns={
                "shocking_at_anchor": "big move in the worst hour",
                "first_shock_offset_h": "first big move (hours after)",
                "first_run_h": "lasted (hours)", "shock_hours_after": "big-move hours that week",
                "peak_abs_return_pct": "largest hourly move (%)"}).round(1), hide_index=True, width="stretch")

    st.markdown("### What if? Make one coin crash and watch the spread")
    M = None
    if P is None or _returns(S) is None:
        st.info("The hourly prices or a prediction run are missing, so there is nothing to simulate. Run "
                "`cryptograph-predict` (it needs the trained model in `runs/` and the data in `data/processed/`), then "
                "reload this page.")
    else:
        st.markdown("<div class='plain'>Pick a coin and a price move. A statistical model of how coins moved together "
                    "recently shows how the shock would spread. This is a what-if game, separate from the deep "
                    "learning model's risk scores.</div>", unsafe_allow_html=True)
        ranks_w = _rankings(RANKINGS_VERSION)
        net = ranks_w.set_index("coin")["dy_net"] if "dy_net" in ranks_w else pd.Series(dtype=float)
        c1, c2, c3 = st.columns([1.2, 2, 1.2])
        w_name = c3.selectbox("Learn from", list(wi.WINDOWS), index=1, key="whatif_window",
                              help="Which recent prices to learn from: how coins moved together over this period.")
        try:
            M = _shock_model(S, wi.WINDOWS[w_name], max(k_hours))
        except ValueError as err:
            st.warning(f"Not enough recent prices to simulate ({err}). Try a longer period.")
    if M is not None:
        options = wi.coin_options(M.coins, ranks_w)
        src = c1.selectbox("Coin that moves", options, index=options.index("BTC") if "BTC" in options else 0,
                           key="whatif_coin", help="Coins that tend to pass shocks on are listed first.",
                           format_func=lambda c: f"{c} (passes shocks on)" if net.get(c, 0) > 0 else c)
        shock = c2.slider("Its price move in one hour (%)", -50, 50, -2, 1, key="whatif_shock",
                          help="Negative: a sudden drop. Positive: a sudden jump.")
        sim = wi.simulate(M, src, shock)
        last = int(sim["hour"].max())
        final = wi.at_hour(sim, last)
        others = final[~final["source"]]
        st.markdown(wi.what_if_sentence(sim, src, shock, 0))
        m1, m2, m3 = st.columns(3)
        swing = float(M.usual_swing[M.coins.index(src)])
        m1.metric(f"{src}'s move", f"{abs(np.log1p(shock / 100)) / swing:.0f}× usual" if swing > 0 else "—")
        m2.metric("Coins pushed into a big move", f"{int(others['big'].sum())} of {len(others)}")
        if len(others):
            m3.metric(f"Hardest hit after {dd.horizon_label(last)}",
                      f"{others.iloc[0]['coin']}: {others.iloc[0]['change_pct']:+.1f}%")
        sector_w = ranks_w.set_index("coin")["sector"].to_dict() if not ranks_w.empty else _sectors()
        fig, pos = network_figure(M.coins, pd.DataFrame(columns=["a", "b"]), sector_w, height=600)
        lim = max(sim["change_pct"].abs().max(), 1e-9)
        same_hour = wi.at_hour(sim, 0).set_index("coin")["change_pct"]
        for c in M.coins:
            if c != src and abs(same_hour[c]) > 0.05 * lim:
                fig.add_trace(go.Scatter(x=[pos[src][0], pos[c][0]], y=[pos[src][1], pos[c][1]], mode="lines",
                                         line=dict(width=0.5 + 4 * min(abs(same_hour[c]) / lim, 1.5),
                                                   color="rgba(214,39,40,0.35)" if same_hour[c] < 0
                                                   else "rgba(44,160,44,0.35)"),
                                         hoverinfo="none", showlegend=False))
        xs, ys = [pos[c][0] for c in M.coins], [pos[c][1] for c in M.coins]

        def move_tr(hour):
            t = sim[sim["hour"] == hour].set_index("coin").loc[M.coins]
            return go.Scatter(
                x=xs, y=ys, mode="markers+text", text=M.coins, textposition="top center", showlegend=False,
                marker=dict(size=[10 + 16 * min(abs(v) / lim, 1.5) for v in t["change_pct"]], color=t["change_pct"],
                            colorscale=MOVE_SCALE, cmin=-lim, cmax=lim,
                            colorbar=dict(title="price change (%)", thickness=12),
                            symbol=["star" if s else "circle" for s in t["source"]],
                            line=dict(width=[3 if b else 1 for b in t["big"]],
                                      color=["#fafafa" if b else "#222" for b in t["big"]])),
                hovertext=[f"{c}: {v:+.1f}% so far" + (" (big move)" if b else "")
                           for c, v, b in zip(M.coins, t["change_pct"], t["big"])], hoverinfo="text")

        fig.add_trace(move_tr(0))
        node_i = len(fig.data) - 1
        fig.frames = [go.Frame(data=[move_tr(hh)], traces=[node_i], name=str(hh)) for hh in range(last + 1)]
        play = dict(frame=dict(duration=300, redraw=True), fromcurrent=True, transition=dict(duration=0))
        fig.update_layout(
            margin=dict(t=40, b=120),
            updatemenus=[dict(type="buttons", direction="left", x=0, xanchor="left", y=0, yanchor="top",
                              pad=dict(t=45), bgcolor="#262730", bordercolor="#888", font=dict(color="#fafafa"),
                              showactive=False,
                              buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                       dict(label="⏸ Pause", method="animate",
                                            args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                               mode="immediate")])])],
            sliders=[dict(active=0, x=0.16, len=0.84, y=0, yanchor="top", pad=dict(t=30),
                          currentvalue=dict(prefix="hours after the move: "),
                          steps=[dict(label=str(hh), method="animate",
                                      args=[[str(hh)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                                 for hh in range(last + 1)])])
        draw_chart(fig)
        st.caption("Almost all of the spread happens in the same hour: coins move together at once rather than one "
                   "after another.")
        with st.expander("For the examiner: how this is worked out"):
            st.caption(f"A vector autoregression (VAR) on {M.n_hours:,} hourly returns of {len(M.coins)} coins "
                       f"({pd.Timestamp(M.start):%d %b %Y} to {pd.Timestamp(M.end):%d %b %Y}, {M.lags} lag), read "
                       "through generalised impulse responses (Pesaran and Shin), the same tools as the "
                       "Diebold–Yilmaz spillover analysis. Moves scale in proportion with the shock, so very large "
                       "shocks are a rough guide only.")
        say("Let me crash Bitcoin by 10% and press play. Almost every coin moves in the same hour, and later hours add "
            "little. That is the same-hour contagion we found in the data.")
    takeaway("real crises spread across almost the whole market within hours, and the what-if shows the same thing: "
             "the spread happens in the same hour.")
