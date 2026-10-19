"""CryptoGraph dashboard: after a sudden big move of one coin, which other coins are likely to follow?

    cryptograph-predict                 # writes runs/<run_id>/ (risk, drivers, alerts, shock context)
    streamlit run dashboard/app.py      # shows the latest prediction run

Written for readers without a data-science background: plain words first, the technical details in "Details"
expanders. Environment: CRYPTOGRAPH_RUNS_DIR (default <project>/runs), CRYPTOGRAPH_ROOT (project root for data/).
Nothing is computed here beyond reading and plotting; every number comes from a tracked run.
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptograph.data.config import PROJECT_ROOT          # torch before pandas (Windows DLL order, see README)
from cryptograph.serving import dashboard_data as dd

import pandas as pd                                       # noqa: E402
import plotly.express as px                               # noqa: E402
import plotly.graph_objects as go                         # noqa: E402
import streamlit as st                                    # noqa: E402

ROOT = Path(os.environ.get("CRYPTOGRAPH_ROOT", PROJECT_ROOT))
RUNS = Path(os.environ.get("CRYPTOGRAPH_RUNS_DIR", ROOT / "runs"))

st.set_page_config(page_title="CryptoGraph | who moves next", page_icon="🕸️", layout="wide")

# Minimal dark look: a black-to-grey gradient behind everything (base colours in .streamlit/config.toml)
st.markdown("""<style>
.stApp {background: linear-gradient(160deg, #07080a 0%, #111317 45%, #1c1f25 100%); background-attachment: fixed;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: rgba(10, 11, 14, 0.72); border-right: 1px solid rgba(255, 255, 255, 0.06);}
[data-testid="stVerticalBlockBorderWrapper"] {border-color: rgba(255, 255, 255, 0.08) !important;
    background: rgba(255, 255, 255, 0.02);}
h1 {font-weight: 600; letter-spacing: -0.02em;}
h2, h3, h4 {font-weight: 500;}
[data-testid="stMetricValue"] {font-weight: 400;}
[data-testid="stMetricLabel"] p {color: #9a9fa8;}
.stTabs [data-baseweb="tab-list"] {gap: 1.5rem; border-bottom: 1px solid rgba(255, 255, 255, 0.06);}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)


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


@st.cache_data(show_spinner=False)
def _events_index(path: str) -> dict:
    return dd.load_events_index(path)


@st.cache_data(show_spinner=False)
def _event(path: str, event_id: int) -> dict:
    return dd.load_event(path, event_id)


STATE_LABEL = dict(zip(dd.NODE_STATES, ("big move now", "moved earlier", "calm so far", "not trading yet")))
STATE_COLOR = dict(zip(dd.NODE_STATES, ("#d62728", "#ff9f43", "#9aa5b1", "#3a3f47")))
UP, DOWN = ("<span style='color:#d62728; font-weight:700'>▲</span>",
            "<span style='color:#2ca02c; font-weight:700'>▼</span>")
LEVEL_COLOR = {label: colour for label, colour, _ in dd.RISK_LEVELS} | {"Unknown": "#9aa5b1"}
when = lambda ts: pd.Timestamp(ts).strftime("%d %b %Y, %H:%M UTC")   # noqa: E731


def draw_chart(fig) -> None:
    """Plot on the page's gradient: transparent backgrounds, faint grid lines."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    st.plotly_chart(fig, width="stretch")


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


# --- header and sidebar ----------------------------------------------------------------------------------------------
st.title("CryptoGraph")
st.markdown("#### Who moves next?")
st.markdown("When one cryptocurrency makes a sudden big move, others often follow within hours. This page shows "
            "**which coins are most likely to make a big move of their own next**, and why.")
runs = dd.prediction_runs(RUNS)
if not runs:
    st.info("There is nothing to show yet. Run `cryptograph-predict` (it needs the trained model in `runs/` and the "
            f"data in `data/processed/v2/`), then reload this page. Looked in `{RUNS}`.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    with st.expander("Advanced"):
        run = st.selectbox("Prediction run", runs, format_func=lambda p: p.name)
    P = _prediction(str(run))
    k_hours = list(P.summary["k_hours"])
    h = st.select_slider("Look ahead", options=k_hours, value=6 if 6 in k_hours else k_hours[0],
                         format_func=dd.horizon_label, key="horizon",
                         help="How far ahead to look: the chance of a big move within this many hours.")
    st.caption(f"Model run {P.summary['target_run']}")
    st.caption("A research tool, not financial advice.")

S = P.summary
risk_h = P.risk[P.risk["horizon_h"] == h].copy()
typical = float(risk_h["base_rate"].iloc[0]) if "base_rate" in risk_h and risk_h["base_rate"].notna().any() else None
risk_h["level"] = [dd.risk_level(x)[0] for x in (risk_h["lift"] if "lift" in risk_h else [None] * len(risk_h))]
risk_h = risk_h.sort_values(["risk", "score"], ascending=False)
top = risk_h.iloc[0]
H = dd.horizon_label(h)

st.subheader("What just happened")
with st.container(border=True):
    st.markdown(f"**{when(S['as_of'])}** (the latest data)")
    for src in P.sources.to_dict("records"):
        recent = P.context.set_index("coin")["shocks_24h"].get(src["coin"]) if "shocks_24h" in P.context else None
        st.markdown(dd.shock_sentence(src, dd.typical_duration(P.persistence, src["type"], src["direction"]),
                                      None if recent is None or pd.isna(recent) else int(recent)))
    if P.sources.empty:
        st.markdown(f"Sudden big moves by: {', '.join(S['source_coins'])}.")
    if S["hours_since_as_of"]:
        st.caption(f"The data runs {S['hours_since_as_of']} hours past this moment.")

c1, c2, c3 = st.columns(3)
c1.metric(f"Most likely to move next ({H})", f"{top['coin']}: {top['risk']:.0%}",
          help=f"The coin with the highest chance of a big move within {H}.")
if typical is not None:
    c2.metric("A typical coin", f"{typical:.0%} chance",
              help=f"How often a coin made a big move within {H} after a shock elsewhere, in past data.")
n_high = int(risk_h["level"].isin(["Much higher than usual", "Higher than usual"]).sum())
c3.metric("Coins at higher-than-usual risk", f"{n_high} of {len(risk_h)}")
if not S["calibrated"]:
    st.warning("These numbers are raw model scores, not chances (the model run has no validation predictions).")

tab_risk, tab_why, tab_past, tab_map, tab_about = st.tabs(
    ["Who's at risk", "Why", "Past crises", "Market map", "How it works"])

# --- who's at risk -------------------------------------------------------------------------------------------------
with tab_risk:
    st.markdown(f"The chance that each coin makes **a big move of its own within the next {H}**, most likely first.")
    fig = px.bar(risk_h.iloc[::-1], x="risk", y="coin", orientation="h", color="level",
                 color_discrete_map=LEVEL_COLOR, category_orders={"level": [x[0] for x in dd.RISK_LEVELS]},
                 labels={"risk": f"chance of a big move within {H}", "coin": "", "level": ""},
                 hover_data={"risk": ":.0%", "level": True, "coin": False})
    if typical is not None:
        fig.add_vline(x=typical, line_dash="dot", line_color="#bbb", annotation_text="typical coin",
                      annotation_position="bottom right")
    # most likely coin on top
    fig.update_layout(height=max(460, 24 * len(risk_h) + 90), xaxis_tickformat=".0%",
                      yaxis=dict(categoryorder="array", categoryarray=risk_h["coin"].tolist()[::-1]),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", title=None), margin=dict(t=90))
    draw_chart(fig)
    st.caption("A **big move** is an hourly price change several times larger than the coin's usual swings over the "
               "past week, up or down. Colours compare each coin with a typical coin.")
    with st.expander("Details: every time window"):
        m = dd.risk_matrix(P.risk)
        hm = px.imshow(m.values, x=[dd.horizon_label(c) for c in m.columns], y=m.index, color_continuous_scale="Reds",
                       aspect="auto", zmin=0, labels=dict(color="chance"))
        hm.update_traces(text=[[f"{v:.0%}" for v in r] for r in m.values], texttemplate="%{text}")
        hm.update_layout(height=max(380, 22 * len(m)), margin=dict(t=10))
        draw_chart(hm)
        st.dataframe(risk_h[["coin", "risk", "lift", "score", "source", "event_time"]].rename(
            columns={"risk": "chance", "lift": "× typical coin", "score": "raw model score",
                     "source": "after the move of", "event_time": "at"}).round(3), hide_index=True,
            width="stretch")

# --- why --------------------------------------------------------------------------------------------------------------
with tab_why:
    coin = st.selectbox("Choose a coin", risk_h["coin"].tolist(), key="why_coin",
                        help="Coins are listed from most to least likely to move next.")
    r = risk_h[risk_h["coin"] == coin].iloc[0]
    label, colour = dd.risk_level(r.get("lift"))
    rel = f", **{r['lift']:.1f} times** a typical coin" if pd.notna(r.get("lift")) else ""
    st.markdown(f"### {coin}: {r['risk']:.0%} chance of a big move within {H}")
    st.markdown(f"<span style='color:{colour}; font-weight:600'>{label}</span>{rel}. This follows the big move of "
                f"**{r['source']}**.", unsafe_allow_html=True)
    drivers = P.drivers.get(f"{coin}|{h}", [])
    if drivers:
        st.markdown("**What the model weighed most:**")
        for d in drivers:
            arrow = UP if d["attribution"] > 0 else DOWN
            st.markdown(f"- {arrow} {dd.plain_driver(d, coin, r['source'], P.context)}", unsafe_allow_html=True)
        st.caption("▲ pushes the chance up, ▼ pushes it down. These are the inputs that changed this coin's "
                   "score "
                   "the most, compared with an average situation (they explain the model, not the market).")
    else:
        st.info("No explanation was saved for this coin at this time window (an older prediction run).")

# --- past crises -----------------------------------------------------------------------------------------------------
PERIOD_BANNER = {
    "test": (st.success, "**A real test.** The model never saw this crisis while it was built, so this shows how it "
                         "would have done live."),
    "validation": (st.warning, "**Not a fully fair test.** This crisis was used to fine-tune the model."),
    "train": (st.warning, "**Not a fair test.** The model learned from this crisis, so it looks better here than it "
                          "would in real life."),
}
with tab_past:
    ev_dir = dd.latest_events_dir(RUNS)
    if ev_dir is None:
        st.info("No past crises yet. Run `cryptograph-train --config configs/experiment/predict_events_1h.yaml`, then "
                "reload this page.")
    else:
        E = _events_index(str(ev_dir))
        label_of = {e["id"]: f"{pd.Timestamp(e['date']):%b %Y} · {e['name']}" for e in E["events"]}
        eid = st.selectbox("Choose a past crisis", list(label_of), format_func=label_of.get, key="event")
        ev = _event(str(ev_dir), eid)
        rp, sector_of = ev["replay"], _sectors()
        coins_ev = [x["coin"] for x in rp["sequence"]]
        banner, text = PERIOD_BANNER.get(ev["split"], (st.warning, ev["split_note"]))
        banner(text)
        st.markdown(ev["description"] + (f" [Read more]({ev['source_url']})" if ev["source_url"] else ""))
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
        ot = dd.outcome_table(ev["risk"], h)
        ot = ot.assign(result=ot["outcome"].map({"shocked": "✅ big move", "no shock": "❌ no big move"})
                       .fillna("— unknown"))
        st.dataframe(ot[["coin", "sector", "risk", "result"]].rename(
            columns={"risk": f"model's chance ({H})", "result": "what happened"}).style.format(
            {f"model's chance ({H})": "{:.0%}"}), hide_index=True, width="stretch")

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
        fa.update_layout(height=280, margin=dict(t=40, b=10))
        draw_chart(fa)

        with st.expander("Details: every coin during this crisis"):
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
                "peak_abs_return_pct": "largest hourly move (%)"}).round(1), hide_index=True,
                width="stretch")
            st.dataframe(dd.outcome_table(ev["risk"], h, pick).style.format(
                {"risk": "{:.1%}", "score": "{:.3f}", "lift": "{:.2f}×"}, na_rep="—"), hide_index=True,
                width="stretch")
            st.caption("The model's chance is grouped into bands, so neighbouring coins can share a value; the raw "
                       "score sets their order.")

# --- market map -------------------------------------------------------------------------------------------------------
with tab_map:
    edges, snap_bar = _graph(S)
    ranks = _rankings(S.get("data_version", "v2"))
    sector_of = ranks.set_index("coin")["sector"].to_dict() if not ranks.empty else _sectors()
    coins = sorted(set(P.risk["coin"]) | set(edges["a"]) | set(edges["b"]))
    st.markdown("Each dot is a coin, grouped by what it is used for. **Lines join coins whose prices moved closely "
                f"together over the past week.** Bigger dots have a higher chance of a big move within {H}; ★ marks "
                "the coin that just moved.")
    fig, pos = network_figure(coins, edges, sector_of)
    chance = risk_h.set_index("coin")["risk"]
    for sec in sorted({sector_of.get(c, "other") for c in coins}):
        cs = [c for c in coins if sector_of.get(c, "other") == sec]
        fig.add_trace(go.Scatter(
            x=[pos[c][0] for c in cs], y=[pos[c][1] for c in cs], mode="markers+text", name=sec, text=cs,
            textposition="top center",
            marker=dict(size=[10 + 60 * float(chance.get(c, 0)) for c in cs], line=dict(width=1, color="#333"),
                        symbol=["star" if c in S["source_coins"] else "circle" for c in cs]),
            hovertext=[f"{c} ({sec}): {chance.get(c, float('nan')):.0%} chance within {H}" for c in cs],
            hoverinfo="text"))
    fig.update_layout(legend_title="What the coin is used for")
    draw_chart(fig)
    if not ranks.empty:
        with st.expander("Details: network rankings"):
            st.caption("Measured on the test period (notebook 02): which coins tend to spread price swings to others "
                       "(positive net spillover) and whose big moves are most often followed by others.")
            cols = {"coin": "coin", "sector": "sector", "dy_net": "net spillover (spreads + / absorbs −)",
                    "shock_pagerank": "followed by others (shock PageRank)",
                    "shock_out_lift": "how often others follow (× chance)"}
            st.dataframe(ranks[[c for c in cols if c in ranks]].rename(columns=cols)
                         .sort_values(cols["dy_net"], ascending=False).round(3), hide_index=True,
                         width="stretch")

# --- how it works -----------------------------------------------------------------------------------------------------
with tab_about:
    st.markdown(dd.about_text(S.get("data"), k_hours))
    tm = dd.model_test_metrics(P.model_metrics, k_hours)
    if not tm.empty:
        st.markdown(f"**How good is it?** On {dd.test_period_phrase(S.get('data'))} it never saw, scored once:")
        st.dataframe(tm.assign(window=tm["horizon_h"].map(dd.horizon_label)).rename(columns={
            "test_auprc": "model's precision score", "positive_rate": "random guess",
            "lift": "times better than random"})
            [["window", "model's precision score", "random guess", "times better than random"]].round(2),
            hide_index=True, width="stretch")
        st.caption("The precision score is the average share of correct alerts when the model's alerts are ranked from "
                   "most to least confident (average precision); a random guess scores the share of coins that move.")
    if not P.persistence.empty:
        st.markdown("**How long do big moves last?** Hours until a coin's price swings are back to normal, for past "
                    "big moves of each kind (the bar shows the typical case; the line, the middle half of cases):")
        pt = P.persistence.copy()
        pt["kind"] = pt["direction"].map({"crash": "drops", "pump": "jumps"}).radd(
            pt["type"].map({"market-wide": "market-wide ", "partial": "partial ", "idiosyncratic": "single-coin "}))
        cur = {f"{s['type']} {s['direction']}" for s in P.sources.to_dict("records")}
        fig = go.Figure(go.Bar(
            x=pt["kind"], y=pt["median_hours_to_normal"],
            error_y=dict(type="data", symmetric=False, array=pt["q75_hours"] - pt["median_hours_to_normal"],
                         arrayminus=pt["median_hours_to_normal"] - pt["q25_hours"]),
            marker_color=["#d62728" if f"{t} {d}" in cur else "#7f7f7f" for t, d in zip(pt["type"], pt["direction"])]))
        fig.update_layout(yaxis_title="hours", height=360, margin=dict(t=10))
        draw_chart(fig)
        st.caption("Red: the kind of move that just happened. Market-wide = many coins at once; single-coin = one coin "
                   "on its own.")
    st.caption("A research project, not financial advice. Past patterns do not guarantee future moves.")
