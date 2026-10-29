"""CryptoGraph dashboard: the project's story, from the question to the results, with the crisis replays.

    cryptograph-predict                 # writes runs/<run_id>/ (needed for What if? and the market map)
    cryptograph-train --config configs/experiment/predict_events_1h_v3.yaml   # the crisis replays
    streamlit run dashboard/app.py

Written for readers without a data-science background: plain words first, the technical details in "Details"
expanders. Environment: CRYPTOGRAPH_RUNS_DIR (default <project>/runs), CRYPTOGRAPH_ROOT (project root for data/ and
reports/). Every number is read from the project's report files and tracked runs (serving/story.py); the "What if?"
section fits a small VAR on the trailing hourly returns (serving/what_if.py).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptograph.data.config import PROJECT_ROOT          # torch before pandas (Windows DLL order, see README)
from cryptograph.serving import dashboard_data as dd
from cryptograph.serving import story as sy
from cryptograph.serving import what_if as wi

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import plotly.express as px                               # noqa: E402
import plotly.graph_objects as go                         # noqa: E402
import streamlit as st                                    # noqa: E402

ROOT = Path(os.environ.get("CRYPTOGRAPH_ROOT", PROJECT_ROOT))
RUNS = Path(os.environ.get("CRYPTOGRAPH_RUNS_DIR", ROOT / "runs"))

st.set_page_config(page_title="CryptoGraph | crypto shock contagion", page_icon="🕸️", layout="wide")

RED, TEAL, GREY, AMBER, GREEN = "#e5484d", "#2ec4b6", "#8b929c", "#f5a524", "#3ccf7a"
FAMILY_COLOR = {"temporal graph": TEAL, "machine learning": AMBER, "network baseline": "#8e7dff"}
SPLIT_COLOR = {"train": GREY, "validation": AMBER, "test": TEAL}

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
.stTabs [data-baseweb="tab-list"] {{gap: 1.6rem; border-bottom: 1px solid rgba(255, 255, 255, 0.06);}}
.stTabs [data-baseweb="tab"] p {{font-size: 1.02rem;}}
footer {{visibility: hidden;}}
.hero {{padding: 2.2rem 0 1.2rem;}}
.hero .kicker {{color: {TEAL}; letter-spacing: .14em; text-transform: uppercase; font-size: .8rem; font-weight: 600;}}
.hero h1 {{font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.08; margin: .4rem 0 .8rem;}}
.hero p {{color: #b4bac3; font-size: 1.12rem; max-width: 62ch;}}
.kpis {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .9rem; margin: 1rem 0 1.6rem;}}
.kpi {{border: 1px solid rgba(255,255,255,.08); border-radius: .9rem; padding: 1rem 1.1rem; background: rgba(255,255,255,.025);}}
.kpi b {{display: block; font-size: 2rem; font-weight: 600; letter-spacing: -.02em;}}
.kpi span {{color: #9a9fa8; font-size: .92rem;}}
.flow {{display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .6rem; margin: .6rem 0 1.6rem;}}
.step {{border-radius: .8rem; padding: .9rem 1rem; background: rgba(46,196,182,.07); border: 1px solid rgba(46,196,182,.25);
    position: relative;}}
.step .n {{color: {TEAL}; font-weight: 700; font-size: .8rem; letter-spacing: .1em;}}
.step b {{display: block; margin: .2rem 0 .3rem; font-size: 1.05rem;}}
.step span {{color: #a9b0b9; font-size: .88rem;}}
.cards {{display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .9rem; margin: .6rem 0 1.4rem;}}
.card {{border: 1px solid rgba(255,255,255,.08); border-radius: .9rem; padding: 1rem 1.1rem; background: rgba(255,255,255,.025);}}
.card .big {{font-size: 1.7rem; font-weight: 600; color: #fafafa;}}
.card b {{display: block; margin: .25rem 0 .35rem;}}
.card span {{color: #a9b0b9; font-size: .92rem;}}
.takeaway {{border-left: 3px solid {TEAL}; padding: .6rem 1rem; margin: 1.2rem 0; background: rgba(46,196,182,.06);
    border-radius: 0 .6rem .6rem 0; color: #d7dce2;}}
.takeaway b {{color: {TEAL};}}
@media (max-width: 900px) {{.kpis, .cards {{grid-template-columns: repeat(2, minmax(0, 1fr));}}
    .flow {{grid-template-columns: 1fr;}}}}
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
MOVE_SCALE = [[0.0, "#d62728"], [0.5, "#3a3f47"], [1.0, "#2ca02c"]]          # price down red, up green


def draw_chart(fig, height: int | None = None) -> None:
    """Plot on the page's gradient: transparent backgrounds, faint grid lines."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(size=13))
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    st.plotly_chart(fig, width="stretch")


def takeaway(text: str) -> None:
    st.markdown(f"<div class='takeaway'><b>So what:</b> {text}</div>", unsafe_allow_html=True)


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


def fmt_int(x) -> str:
    return f"{int(x):,}" if x is not None else "—"


# --- navbar ------------------------------------------------------------------------------------------------------------
runs = dd.prediction_runs(RUNS)
P = _prediction(str(runs[0])) if runs else None              # the latest prediction run (What if?, market map)
S = P.summary if P is not None else {}
version = S.get("data_version", "v3")
k_hours = list(S.get("k_hours", sy.K_HOURS))
default_h = 6 if 6 in k_hours else k_hours[0]
with st.container(key="navbar"):
    brand, look = st.columns([2, 3], vertical_alignment="center")
    brand.markdown("### CryptoGraph &nbsp;<span style='color:#9a9fa8; font-size:1rem; font-weight:400'>"
                   "crypto shock contagion</span>", unsafe_allow_html=True)
    h = look.segmented_control("Look ahead", k_hours, default=default_h, format_func=dd.horizon_label,
                               key="horizon", width="stretch",
                               help="The time window used by the crisis replays and the results charts.")
h = default_h if h is None else h                         # clicking the selected window again clears it
H = dd.horizon_label(h)

facts = sy.dataset_facts(ROOT, version)
head = _story("headline")
lb = _story("leaderboard")

tab_home, tab_past, tab_net, tab_model, tab_results, tab_explain = st.tabs(
    ["Home", "Past crises", "The network", "The model", "Results", "Explain & stress tests"])

# --- home -----------------------------------------------------------------------------------------------------------------
with tab_home:
    st.markdown("""<div class='hero'><div class='kicker'>Deep learning · temporal graph networks · crypto markets</div>
<h1>When one coin crashes,<br>which coins are hit next, and how soon?</h1>
<p>Crypto trades around the clock and reacts to news within hours. Most research measures how connected coins are.
We turned contagion into a prediction task: after a sudden big move of one coin, predict which other coins make a big
move of their own within 1, 3, 6, 12 or 24 hours, and test it honestly.</p></div>""", unsafe_allow_html=True)

    v3, m15 = head.get("v3"), head.get("m15")
    years = (pd.Timestamp(facts["end"]) - pd.Timestamp(facts["start"])).days / 365.25 if facts.get("start") else None
    kpi = [(f"{facts.get('coins', '—')} coins", f"hourly Binance prices, {years:.1f} years ({facts.get('hours', 0):,} hours)"
            if years else "hourly Binance prices"),
           (f"{facts['events'] / 1e6:.2f} M" if facts.get("events") else "—", "shock events the model learns from"),
           (f"{v3['auprc']:.3f}" if v3 else "—", f"test score at 1 hour: {v3['lift']:.1f}× random guessing"
            if v3 else "test score at 1 hour"),
           (f"{m15['lift']:.1f}×" if m15 else "—", "better than random at 15 minutes: contagion is fast")]
    st.markdown("<div class='kpis'>" + "".join(f"<div class='kpi'><b>{a}</b><span>{b}</span></div>" for a, b in kpi)
                + "</div>", unsafe_allow_html=True)

    st.markdown("#### How it works, in five steps")
    thr = facts.get("shock_threshold") or 2.29
    steps = [("Prices", f"Hourly prices of {facts.get('coins', 30)} coins, every file checksum-verified."),
             ("Shocks", f"A big move: more than {thr:.2f}× the coin's usual hourly swing over the past week."),
             ("Events", "Each shock becomes one question per other coin: will it be next?"),
             ("TGN", "A temporal graph network reads the event stream in time order and scores every question."),
             ("Alerts", "Five chances per coin (1 to 24 hours), explained in plain words.")]
    st.markdown("<div class='flow'>" + "".join(f"<div class='step'><div class='n'>STEP {i}</div><b>{a}</b><span>{b}</span>"
                                               "</div>" for i, (a, b) in enumerate(steps, 1)) + "</div>",
                unsafe_allow_html=True)

    st.markdown("#### What we found")
    cj = _story("contagion_jump")
    cj1 = cj[(cj["split"] == "test") & (cj["k_hours"] == 1)] if not cj.empty else cj
    mm = _market_mode(version, MARKET_WINDOW_START)
    groups = _story("attribution_groups")
    recv = groups.set_index("group")["share_of_abs"].get("receiver coin") if not groups.empty else None
    pers = P.persistence if P is not None else pd.DataFrame()
    cards = [
        (f"{cj1['lift'].iloc[0]:.1f}×" if len(cj1) else "—", "Contagion is real",
         "After a coin shocks, another coin is this much more likely to shock in the very next hour (test years)."),
        (f"{mm['share']:.0%}" if mm else "—", "One market factor",
         "of all hourly movement is the whole market moving together, so crises hit almost every coin."),
        ("same hour", "No coin leads",
         "Coins move together within the hour, not one after another; even Bitcoin does not move first."),
        (f"{recv:.0%}" if recv is not None else "—", "Alerts follow the receiving coin",
         "of an alert's reasons are the receiving coin's own state: volume surge and recent jumpiness."),
        (f"{pers['median_hours_to_normal'].min():.0f}–{pers['median_hours_to_normal'].max():.0f} h"
         if not pers.empty else "—", "Shocks are short", "until a shocked coin is back to normal, for every kind of shock."),
        (f"{v3['rh_lift']:.1f}×" if v3 else "—", "Honest about repeats",
         "better than random even when each coin-hour is counted only once (the fair score)."),
    ]
    st.markdown("<div class='cards'>" + "".join(f"<div class='card'><div class='big'>{a}</div><b>{b}</b><span>{c}</span>"
                                                "</div>" for a, b, c in cards) + "</div>", unsafe_allow_html=True)

    lb1 = lb[lb["k_hours"] == 1] if not lb.empty else lb
    if not lb1.empty:
        st.markdown("#### The scoreboard at 1 hour")
        fig = px.bar(lb1, x="auprc", y="model", orientation="h", color="family", color_discrete_map=FAMILY_COLOR,
                     labels={"auprc": "test score (AUPRC, higher is better)", "model": "", "family": ""},
                     hover_data={"lift": ":.2f", "family": False})
        fig.add_vline(x=float(lb1["random"].iloc[0]), line_dash="dot", line_color=GREY)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), legend=dict(orientation="h", y=1.08),
                          margin=dict(t=40, l=10))
        draw_chart(fig, 420)
        st.caption(f"Test period 2024–26, within 1 hour. The dotted line is random guessing "
                   f"({lb1['random'].iloc[0]:.3f}); colours mark the kind of model.")
    takeaway("contagion between coins is real but fast and mostly market-wide; a temporal graph network ranks the next "
             "coins to be hit about twice as well as chance, and the honest checks hold up. The tabs above tell the story "
             "chapter by chapter.")
    st.caption("Explore: **Past crises** replays nine real crashes · **The network** shows how coins are linked · "
               "**The model** opens the TGN · **Results** is the full scoreboard · **Explain & stress tests** shows why "
               "it alerts and how it breaks. A research project, not financial advice.")

# --- past crises -----------------------------------------------------------------------------------------------------
with tab_past:
    ev_dir = dd.latest_events_dir(RUNS)
    if ev_dir is None:
        st.info("No past crises yet. Run `cryptograph-train --config configs/experiment/predict_events_1h_v3.yaml`, "
                "then reload this page.")
    else:
        E = _events_index(str(ev_dir))
        st.markdown("### Nine real crises, replayed hour by hour")
        st.markdown("Each crisis is replayed through the model at its worst hour, and the model's five riskiest coins are "
                    "compared with five random coins. Only the **test-period** crises are true forecasts; the others "
                    "were used for training or for choosing the model.")
        tl = pd.DataFrame([{"date": pd.Timestamp(e["date"]), "crisis": e["name"], "period": e["split"],
                            "coins hit at worst hour": e["replay_metrics"]["coins_shocking_at_anchor"],
                            "coins trading": e["replay_metrics"]["coins_active_at_anchor"]} for e in E["events"]])
        tl["share"] = tl["coins hit at worst hour"] / tl["coins trading"]
        tl["label"] = tl["crisis"].str.split("/").str[0].str.strip().str.slice(0, 26)
        tl["y"] = [0.35 if i % 2 else -0.35 for i in range(len(tl))]            # alternate rows so labels don't collide
        ft = px.scatter(tl, x="date", y="y", size="share", color="period", text="label",
                        color_discrete_map=SPLIT_COLOR, size_max=34,
)
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
                st.caption("In big crashes almost every coin moves, so five picks cannot be told apart: such crises "
                           "cannot judge any model. The real test is the hundreds of thousands of test-period events "
                           "(Results).")

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

        st.markdown("#### Replay the crisis hour by hour")
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
        takeaway("crises spread across almost the whole market within hours; the replays show it, and they are honest "
                 "about when a crisis cannot judge the model.")

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

# --- the network ------------------------------------------------------------------------------------------------------
with tab_net:
    st.markdown("### The market as a network")
    st.markdown("Coins are the dots; the links are how closely they move together, or how often one coin's shock is "
                "followed by another's. This is where the project started: looking at contagion through networks.")

    cj = _story("contagion_jump")
    if not cj.empty:
        st.markdown("#### 1. Contagion is real, and fast")
        t = cj[cj["split"] == "test"].copy()
        long = pd.concat([t.assign(case="right after another coin shocks", value=t["rate"]),
                          t.assign(case="in any window of the same length", value=t["base_rate_any_coin"])])
        long["window"] = long["k_hours"].map(dd.horizon_label)
        fc = px.bar(long, x="window", y="value", color="case", barmode="group", text=long["value"].map("{:.0%}".format),
                    color_discrete_map={"right after another coin shocks": RED, "in any window of the same length": GREY},
                    labels={"value": "chance a coin makes a big move", "window": "", "case": ""})
        fc.update_layout(yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.12), margin=dict(t=40))
        draw_chart(fc, 360)
        lifts = cj[cj["k_hours"] == 1].set_index("split")["lift"]
        st.caption(f"Test period (2024–26). Within 1 hour the jump is {lifts.get('train', np.nan):.1f}× in the training "
                   f"years, {lifts.get('val', np.nan):.1f}× in validation and {lifts.get('test', np.nan):.1f}× in test: "
                   "stable across periods, and it fades by 24 hours.")

    mm = _market_mode(version, MARKET_WINDOW_START)
    if mm:
        st.markdown("#### 2. One factor moves the whole market")
        c1, c2 = st.columns([3, 2])
        fe = go.Figure(go.Bar(x=list(range(1, 11)), y=mm["eigenvalues"][:10],
                              marker_color=[RED] + [GREY] * 9, text=[f"{v:.1f}" for v in mm["eigenvalues"][:10]]))
        fe.add_hline(y=mm["noise_bound"], line_dash="dot", line_color=TEAL,
                     annotation_text=f"what pure noise could give ({mm['noise_bound']:.2f})")
        fe.update_layout(xaxis_title="direction of movement (eigenvalue rank)", yaxis_title="eigenvalue",
                         margin=dict(t=20))
        with c1:
            draw_chart(fe, 330)
        c2.metric("Movement explained by the market factor", f"{mm['share']:.0%}")
        c2.markdown(f"Across {mm['coins']} coins and {mm['hours']:,} test-period hours (from April 2024, when all were listed; rush-hour pattern removed), the biggest eigenvalue of the "
                    f"correlation matrix is **{mm['eigenvalues'][0]:.1f} of {mm['coins']}**, and it is the only one "
                    "above the random-noise line (Marchenko–Pastur). Most of what any coin does in an hour is "
                    "'the market moved'.")

    ts = _story("network_timeseries")
    if not ts.empty:
        st.markdown("#### 3. In every crisis, the market pulls together")
        metrics = {"mean_corr": "average correlation between coins", "total_spillover": "spillover index (%)",
                   "tree_length": "length of the market's skeleton (shorter = tighter)",
                   "density": "share of coin pairs linked", "modularity": "how clearly coins split into groups"}
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
            ch = []
            for e in cd.itertuples():
                before = ts[(ts["time"] >= e.date - pd.Timedelta(days=14)) & (ts["time"] < e.date)][pick].mean()
                after = ts[(ts["time"] >= e.date) & (ts["time"] < e.date + pd.Timedelta(days=14))][pick].mean()
                ch.append(after - before)
            ups = int(np.sum(np.array(ch) > 0))
            st.caption(f"Daily network snapshots, 14-day average; red lines are the catalogued crises. In the two weeks "
                       f"after a crisis this measure was higher than in the two weeks before in **{ups} of {len(ch)}** "
                       "crises.")

    if P is not None:
        st.markdown("#### 4. The coin network today")
        edges, _ = _graph(S)
        ranks = _rankings(version)
        sector_of = ranks.set_index("coin")["sector"].to_dict() if not ranks.empty else _sectors()
        central = ranks.set_index("coin")["eigenvector"] if "eigenvector" in ranks else pd.Series(dtype=float)
        coins = sorted(set(P.risk["coin"]) | set(edges["a"]) | set(edges["b"]))
        fig, pos = network_figure(coins, edges, sector_of)
        for sec in sorted({sector_of.get(c, "other") for c in coins}):
            cs = [c for c in coins if sector_of.get(c, "other") == sec]
            size = [10 + 14 * float(central.get(c, 1.0)) for c in cs] if not central.empty else 14
            fig.add_trace(go.Scatter(x=[pos[c][0] for c in cs], y=[pos[c][1] for c in cs], mode="markers+text", name=sec,
                                     text=cs, textposition="top center",
                                     marker=dict(size=size, line=dict(width=1, color="#333")), hoverinfo="text",
                                     hovertext=[f"{c} ({sec})" for c in cs]))
        fig.update_layout(legend_title="What the coin is used for")
        draw_chart(fig)
        st.caption("Lines join coins whose prices moved closely together over the past week (the threshold is fitted on "
                   "the training years). Bigger dots are more central in the network.")
        if not ranks.empty:
            with st.expander("Details: network rankings"):
                cols = {"coin": "coin", "sector": "sector", "dy_net": "net spillover (spreads + / absorbs −)",
                        "eigenvector": "centrality", "shock_pagerank": "followed by others (shock PageRank)",
                        "shock_out_lift": "how often others follow"}
                st.dataframe(ranks[[c for c in cols if c in ranks]].rename(columns=cols)
                             .sort_values(cols["dy_net"], ascending=False).round(3), hide_index=True, width="stretch")

    st.markdown("#### 5. What if? A shock spreading through the network")
    M = None
    if P is None or _returns(S) is None:
        st.info("The hourly prices or a prediction run are missing, so there is nothing to simulate. Run "
                "`cryptograph-predict` (it needs the trained model in `runs/` and the data in `data/processed/`), then "
                "reload this page.")
    else:
        ranks_w = _rankings(version)
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
                           key="whatif_coin", help="Coins that tend to spread price swings to others are listed first.",
                           format_func=lambda c: f"{c} (spreads swings)" if net.get(c, 0) > 0 else c)
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
        st.caption("Almost all of the spread happens in the same hour: crypto prices move together at once rather "
                   "than one after another, and later hours add little.")
        with st.expander("Details: how this is worked out"):
            st.caption(f"A vector autoregression on {M.n_hours:,} hourly returns of {len(M.coins)} coins "
                       f"({pd.Timestamp(M.start):%d %b %Y} to {pd.Timestamp(M.end):%d %b %Y}, {M.lags} lag), read "
                       "through generalized impulse responses (Pesaran and Shin), the same tools as our "
                       "Diebold–Yilmaz spillover analysis. Moves scale in proportion with the shock, so very large "
                       "shocks are a rough guide only. A statistical what-if, separate from the model's risk scores.")
    takeaway("coins are tightly linked, one market factor dominates, and in crises the links get even tighter; but at "
             "hourly speed coins move together in the same hour rather than in a chain.")

# --- the model ---------------------------------------------------------------------------------------------------------
TGN_SVG = """<svg viewBox="-34 0 794 470" width="100%" style="max-width:820px" xmlns="http://www.w3.org/2000/svg"
 font-family="Inter, Segoe UI, sans-serif">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#c9ced6"/></marker></defs>
<g fill="none" stroke-width="1.4">
<rect x="215" y="12" width="330" height="52" rx="10" fill="#1d2129" stroke="#e5484d"/>
<rect x="20" y="104" width="240" height="62" rx="10" fill="#1d2129" stroke="#8e7dff"/>
<rect x="500" y="104" width="240" height="62" rx="10" fill="#1d2129" stroke="#8e7dff"/>
<rect x="190" y="200" width="380" height="58" rx="10" fill="#132b2a" stroke="#2ec4b6"/>
<rect x="190" y="300" width="380" height="58" rx="10" fill="#132b2a" stroke="#2ec4b6"/>
<rect x="215" y="400" width="330" height="50" rx="10" fill="#1d2129" stroke="#3ccf7a"/>
</g>
<g fill="#eef1f4" text-anchor="middle">
<text x="380" y="35" font-size="15" font-weight="600">1. Event: coin i shocks at hour t</text>
<text x="380" y="54" font-size="12" fill="#a9b0b9">one question per other coin j · 5 event numbers</text>
<text x="140" y="129" font-size="15" font-weight="600">2. Memory per coin</text>
<text x="140" y="149" font-size="12" fill="#a9b0b9">a 64-number diary (GRU)</text>
<text x="620" y="129" font-size="15" font-weight="600">3. Recent events</text>
<text x="620" y="149" font-size="12" fill="#a9b0b9">each coin's last 10 + time code</text>
<text x="380" y="224" font-size="15" font-weight="600">4. Attention</text>
<text x="380" y="244" font-size="12" fill="#a9b0b9">2 heads weigh the recent events → 64-number summary</text>
<text x="380" y="324" font-size="15" font-weight="600">5. Prediction head</text>
<text x="380" y="344" font-size="12" fill="#a9b0b9">summaries + both coins' features + event → 64 → 5</text>
<text x="380" y="430" font-size="15" font-weight="600">6. Five chances: 1 · 3 · 6 · 12 · 24 hours</text>
</g>
<g stroke="#c9ced6" stroke-width="1.4" fill="none">
<path d="M300 64 L170 102" marker-end="url(#a)"/><path d="M460 64 L590 102" marker-end="url(#a)"/>
<path d="M200 166 L270 198" marker-end="url(#a)"/><path d="M560 166 L490 198" marker-end="url(#a)"/>
<path d="M380 258 L380 298" marker-end="url(#a)"/><path d="M380 358 L380 398" marker-end="url(#a)"/>
<path d="M215 425 C 0 425, 0 260, 30 168" stroke="#f5a524" stroke-dasharray="5 5" marker-end="url(#a)"/>
</g>
<text x="-16" y="300" font-size="11" fill="#f5a524" transform="rotate(-90 -16 300)" text-anchor="middle">
7. after the hour is scored: update memories</text>
</svg>"""

with tab_model:
    st.markdown("### Inside the model: a Temporal Graph Network")
    st.markdown("The **TGN** (Rossi and colleagues, Twitter, 2020) was built for streams of time-stamped interactions, "
                "like users replying to each other. Our stream is shocks: *coin i shocked at hour t; will coin j be "
                "next?* It keeps a memory per coin, reads events strictly in time order, and can never see the future.")

    st.markdown("#### From prices to questions")
    c1, c2 = st.columns([1, 1.1])
    c1.markdown(f"""1. **A shock**: this hour's move, with the market's rush-hour pattern removed, divided by the coin's
typical hourly move over the previous week. More than **{thr:.2f}×**, up or down, is a shock (about 4% of hours).
2. **Events**: every shock creates one question for every other coin trading then.
3. **Five numbers per event**: how closely the two coins moved over the past week (same hour and one hour apart), the
shocking coin's return and volatility, and a shock flag.
4. **Five answers**: did coin j have its own shock within 1, 3, 6, 12 and 24 hours? *Strictly later*: moving in the
same hour is co-movement, not contagion.""")
    if facts.get("events_per_split"):
        sp = facts["events_per_split"]
        c2.metric("Events in total", fmt_int(facts.get("events")))
        c2.dataframe(pd.DataFrame({"period": ["training (2020 – Jun 2022)", "validation (Jul 2022 – 2023)",
                                              "test (2024 – Sep 2026)"],
                                   "events": [fmt_int(sp["train"]), fmt_int(sp["val"]), fmt_int(sp["test"])]}),
                     hide_index=True, width="stretch")
        c2.caption("Fixed dates with a one-week gap at each border; everything is fitted on the training years only, "
                   "and the test years were scored once.")
    st.markdown("Example (illustration): *SOL crashes at 14:00 → SOL→ETH (ETH shocks at 15:00: yes at every window), "
                "SOL→BTC (BTC shocks at 19:00: no at 1 and 3 hours, yes from 6), SOL→DOGE (DOGE moves only at 14:00, "
                "the same hour: no).*")

    st.markdown("#### The architecture")
    svg64 = base64.b64encode(TGN_SVG.encode("utf-8")).decode("ascii")
    st.markdown(f"<img alt='TGN architecture' src='data:image/svg+xml;base64,{svg64}' "
                "style='width:100%; max-width:820px; display:block; margin:0.5rem auto'>", unsafe_allow_html=True)
    st.caption("The dashed line is the only way information flows back: an hour's events are written into the memories "
               "only after they have been scored.")

    st.markdown("#### How big is it?")
    pv2, in2, cfg2 = _params("tgn_1h_final_v2")
    pv3, in3, cfg3 = _params("tgn_1h_v3_final")
    c1, c2 = st.columns(2)
    for col, pv, n_in, title in ((c1, pv2, in2, "The report's TGN (memory on)"),
                                 (c2, pv3, in3, "TGN v3 after the report (memory off)")):
        with col:
            if pv is None or pv.empty:
                st.info(f"{title}: the trained model is not in `runs/`.")
                continue
            st.metric(title, f"{int(pv['parameters'].sum()):,} parameters")
            fp = px.pie(pv, names="part", values="parameters", hole=0.55,
                        color_discrete_sequence=[TEAL, "#8e7dff", AMBER, GREY])
            fp.update_traces(textinfo="label+value")
            fp.update_layout(showlegend=False, margin=dict(t=10, b=10))
            draw_chart(fp, 300)
            if n_in:
                st.caption(f"The prediction head reads {n_in} numbers per question.")
    st.markdown("#### What changed after the report")
    st.markdown("""- **Fair scoring**: the answer depends only on the receiving coin and the hour, so busy crash hours
were counted several times; we now also score each coin-hour once.
- **Memory timing**: memory was written once per batch of about 200 events, so about a third of predictions missed the
previous hour. Fixed to write every hour (it never used future data; it just knew less).
- **Better inputs**: volatility on each coin's own scale (calm years no longer look odd), short-window volatility, the
hour's price range, trade counts, and features from 5-minute bars inside the hour.
- **Result**: the best version has memory switched off, so the gain came from the inputs; it scores higher on test, and
the difference is significant (Results).""")
    with st.expander("Details: training settings"):
        def settings(cfg: dict) -> dict:
            tr, mo = cfg.get("training", {}), cfg.get("model", {})
            return {"optimizer": "Adam", "learning rate": tr.get("learning_rate"), "weight decay": tr.get("weight_decay"),
                    "loss": tr.get("loss"), "batch (events)": tr.get("batch_size"), "max epochs": tr.get("epochs"),
                    "patience": tr.get("patience"), "best epoch chosen on": tr.get("select_on", "1-hour score"),
                    "memory": "off" if (mo.get("overrides") or {}).get("use_memory") is False else "on"}
        if cfg2 or cfg3:
            st.dataframe(pd.DataFrame({"report's TGN": settings(cfg2), "TGN v3": settings(cfg3)}).astype(str),
                         width="stretch")
        st.caption("Focal loss focuses learning on the rare 'yes' cases (only about 10% at one hour). Each final model "
                   "was trained five times with different random starts.")
    takeaway("the TGN keeps a diary per coin and reads events in time order; at hourly speed the diary turned out to "
             "matter less than good inputs describing each coin's current state.")

# --- results ----------------------------------------------------------------------------------------------------------
with tab_results:
    st.markdown("### Results, told honestly")
    st.markdown("All models are scored on exactly the same test events (January 2024 – September 2026). The score is "
                "**AUPRC** (average precision): it rewards putting the real 'yes' cases at the top of the list. Random "
                "guessing scores the share of 'yes' cases, so we always show that line.")
    if lb.empty:
        st.info("The result files are missing from `reports/`.")
    else:
        lbh = lb[lb["k_hours"] == h]
        fig = px.bar(lbh, x="auprc", y="model", orientation="h", color="family", color_discrete_map=FAMILY_COLOR,
                     text=lbh["auprc"].map("{:.3f}".format),
                     labels={"auprc": f"test AUPRC within {H}", "model": "", "family": ""})
        fig.add_vline(x=float(lbh["random"].iloc[0]), line_dash="dot", line_color=GREY)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), legend=dict(orientation="h", y=1.1),
                          margin=dict(t=50, l=10))
        draw_chart(fig, 440)
        st.caption(f"Within {H}. The dotted line is random guessing ({lbh['random'].iloc[0]:.3f}). Change the window "
                   "with 'Look ahead' at the top.")

        st.markdown("#### How far ahead of random, at every window")
        keep = ["TGN v3, after the report", "TGN, the report's model", "DyGFormer (Transformer)",
                "Logistic regression (4-number rules)", "Gradient boosting", "Independent cascade"]
        line_colors = dict(zip(keep, [TEAL, "#7fdcd3", "#8e7dff", AMBER, "#ff8a3d", GREY]))
        fl = px.line(lb[lb["model"].isin(keep)], x="k_hours", y="lift", color="model", markers=True,
                     color_discrete_map=line_colors, category_orders={"model": keep},
                     labels={"k_hours": "window (hours)", "lift": "times better than random", "model": ""})
        fl.add_hline(y=1, line_dash="dot", line_color=GREY, annotation_text="random")
        fl.update_xaxes(type="category")
        draw_chart(fl, 380)
        st.caption("Longer windows look better in raw AUPRC only because more coins move anyway; dividing by the random "
                   "level shows the real skill, which is highest at one hour.")

    bt = _story("bootstrap_v3_vs_v2")
    if not bt.empty:
        st.markdown("#### Is the improvement real? A significance test")
        fb = go.Figure(go.Scatter(x=bt["k_hours"].astype(str), y=bt["diff"], mode="markers",
                                  marker=dict(size=12, color=TEAL),
                                  error_y=dict(type="data", symmetric=False, array=bt["ci_hi"] - bt["diff"],
                                               arrayminus=bt["diff"] - bt["ci_lo"])))
        fb.add_hline(y=0, line_dash="dot", line_color=RED, annotation_text="no difference")
        fb.update_layout(xaxis_title="window (hours)", yaxis_title="TGN v3 minus the report's TGN (AUPRC)",
                         margin=dict(t=20))
        fb.update_xaxes(type="category")
        draw_chart(fb, 320)
        b1 = bt[bt["k_hours"] == 1].iloc[0] if (bt["k_hours"] == 1).any() else bt.iloc[0]
        st.caption(f"Whole test days resampled 1,000 times (one crash creates hundreds of linked events). At 1 hour: "
                   f"+{b1['diff']:.3f} (95% interval {b1['ci_lo']:+.3f} to {b1['ci_hi']:+.3f}); v3 was ahead in "
                   f"{b1['share_v3_better']:.1%} of resamples. Every interval is above zero.")

    fs = _story("fair_scores")
    if not fs.empty:
        st.markdown("#### The fair score, and the riskiest alerts")
        f1 = fs[fs["k_hours"] == 1].copy()
        f1["model"] = f1["model"].map(lambda m: sy.MODELS.get(m, (m,))[0])
        st.dataframe(f1[["model", "event_auprc", "event_lift", "rh_auprc", "rh_lift", "calm_auprc", "calm_lift",
                         "top1_precision"]].rename(columns={
            "event_auprc": "every event", "event_lift": "× random", "rh_auprc": "each coin-hour once",
            "rh_lift": "× random ", "calm_auprc": "calm coins only", "calm_lift": "× random  ",
            "top1_precision": "top 1% alerts that came true"}).style.format(
            {c: "{:.3f}" for c in ["every event", "each coin-hour once", "calm coins only"]}
            | {c: "{:.2f}×" for c in ["× random", "× random ", "× random  "]} | {"top 1% alerts that came true": "{:.0%}"}),
            hide_index=True, width="stretch")
        st.caption("Test period, 1 hour. The label depends only on the receiving coin and the hour, so busy hours repeat "
                   "once per shocking coin; counting each coin-hour once, the models stay well ahead of random. 'Calm "
                   "coins' had no shock in the previous day.")

    q = _story("fifteen_minutes")
    if not q.empty:
        st.markdown("#### Contagion is faster than an hour")
        ql = q[q["model"] == "logreg"].copy()
        ql["window"] = ql["minutes"].map(lambda x: f"{x} min")
        fq = px.bar(ql, x="window", y="lift", text=ql["lift"].map("{:.1f}×".format), color_discrete_sequence=[TEAL],
                    labels={"lift": "times better than random", "window": ""})
        fq.add_hline(y=1, line_dash="dot", line_color=GREY)
        draw_chart(fq, 320)
        st.caption("The same question on 15-minute bars (from checksum-verified 5-minute Binance data), scored once on "
                   "test with a simple model on the receiving coin's state and market-wide numbers. The closer to the "
                   "shock, the more predictable: contagion is a minutes-scale process, where a graph model should be "
                   "tested next.")
    takeaway("the TGN beats the classic network-infection model at every window, the improved v3 is significantly "
             "better than the report's model, and the honest re-scorings keep it about 2× ahead of chance; a simple "
             "model stays close, which says the hourly signal sits mostly in each coin's own state.")

# --- explain & stress tests -------------------------------------------------------------------------------------------
with tab_explain:
    st.markdown("### Why the model raises an alert, and how it breaks")
    g = _story("attribution_groups")
    if not g.empty:
        st.markdown("#### What drives an alert")
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
                        labels={"share_of_abs": "share of the alert's reasons", "label": ""})
            fi.update_layout(xaxis_tickformat=".0%", margin=dict(t=10))
            with c2:
                draw_chart(fi, 300)
        st.caption("Integrated gradients on the model's riskiest 1% of test alerts: how much each input pushed the score, "
                   "compared with an average situation. The model mostly flags coins that are already busy and jumpy "
                   "when another coin shocks.")

    fd = _story("explainer_fidelity")
    if not fd.empty:
        st.markdown("#### Are there chains of events behind an alert?")
        ff = px.bar(fd, x="characterization", y="explainer", orientation="h",
                    color=fd["explainer"].eq("TempME").map({True: TEAL, False: GREY}), color_discrete_map="identity",
                    labels={"characterization": "how well 3 chosen past events explain the alert (higher is better)",
                            "explainer": ""})
        ff.update_layout(margin=dict(t=10))
        draw_chart(ff, 300)
        st.caption("TempME (NeurIPS 2023) looks for chains of past events behind each alert. It did barely better than "
                   "picking events at random: shocks hit the market in the same hour, not along chains, so there are no "
                   "special chains to find. A finding about the market, not a bug.")

    if P is not None and not P.persistence.empty:
        st.markdown("#### How long a shock lasts")
        pt = P.persistence.copy()
        pt["kind"] = pt["direction"].map({"crash": "drops", "pump": "jumps"}).radd(
            pt["type"].map({"market-wide": "market-wide ", "partial": "partial ", "idiosyncratic": "single-coin "}))
        fp = go.Figure(go.Bar(x=pt["kind"], y=pt["median_hours_to_normal"], marker_color=TEAL,
                              error_y=dict(type="data", symmetric=False,
                                           array=pt["q75_hours"] - pt["median_hours_to_normal"],
                                           arrayminus=pt["median_hours_to_normal"] - pt["q25_hours"])))
        fp.update_layout(yaxis_title="hours until back to normal", margin=dict(t=10))
        draw_chart(fp, 320)
        st.caption("Bars: the typical case; lines: the middle half of cases. Shocks fade within hours, so 1 to 24-hour "
                   "windows cover a shock's whole life.")

    stx = _story("stress_tests")
    if not stx.empty:
        st.markdown("#### Can it be fooled?")
        tg = stx[(stx["model"] == "TGN") & (stx["perturbation"] != "none")].copy()
        tg["test"] = tg["perturbation"].map(lambda p: "deliberate attack (FGSM)" if p.startswith("FGSM")
                                            else "deliberate attack (PGD, 10 steps)" if p.startswith("PGD")
                                            else "random noise" if p == "input noise" else "missing events")
        fs_ = px.line(tg, x="level", y="change_k1h", color="test", markers=True,
                      color_discrete_map={"deliberate attack (FGSM)": RED, "deliberate attack (PGD, 10 steps)": AMBER,
                                          "random noise": TEAL, "missing events": GREEN},
                      labels={"level": "size of the change (standard deviations, or share of events missing)",
                              "change_k1h": "change in the 1-hour score", "test": ""})
        fs_.update_layout(yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.12), margin=dict(t=40))
        draw_chart(fs_, 380)
        st.caption("The report's TGN on the test events. Losing up to half of the past events or adding random noise "
                   "barely matters; a deliberate attack that nudges every input in the most harmful direction cuts the "
                   "score sharply. The risk is manipulation, not ordinary data errors: the 'Secure' in CryptoGraph-Secure.")

    st.markdown("#### Limits, and what comes next")
    c1, c2 = st.columns(2)
    c1.markdown("""**Limits**
- Overall discrimination is modest: most shocks are triggered by news nobody can predict.
- One market (Binance) and one test period; the 30 coins come from a 2026 ranking (survivorship bias).
- A simple model is close behind at one hour; memory did not help at hourly speed.
- Vulnerable to deliberate input manipulation.""")
    c2.markdown("""**Next steps**
- Run the graph model on 15-minute data, where contagion actually happens.
- A second test period, retraining as the market changes.
- Scheduled-news and order-book data; a money-based test of the alerts.
- Defences against manipulation: checking inputs across exchanges, adversarial training.""")
    takeaway("the model's alerts are driven by each coin's own stress, shocks fade within hours, and the model is robust "
             "to noise and gaps but must be defended against deliberate manipulation.")
