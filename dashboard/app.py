"""CryptoGraph dashboard: contagion risk per coin after the most recent shocks, from `cryptograph-predict` output.

    cryptograph-predict                 # writes runs/<run_id>/ (risk, alerts, drivers, shock context)
    streamlit run dashboard/app.py      # shows the latest prediction run (or pick one in the sidebar)

Environment: CRYPTOGRAPH_RUNS_DIR (default <project>/runs), CRYPTOGRAPH_ROOT (project root for data/processed).
Nothing is computed here beyond reading and plotting; every number comes from a tracked prediction run.
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

st.set_page_config(page_title="CryptoGraph | contagion risk", page_icon="🕸️", layout="wide")


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


STATE_COLOR = dict(zip(dd.NODE_STATES, ("#d62728", "#ff9f43", "#9aa5b1", "#3a3f47")))


st.title("🕸️ CryptoGraph: shock contagion risk")
runs = dd.prediction_runs(RUNS)
if not runs:
    st.info("No prediction runs yet. Run `cryptograph-predict` (needs the final TGN run `tgn_1h_final_v2` seed 0 in "
            f"`runs/` and the processed data in `data/processed/v2/`), then reload. Looked in `{RUNS}`.")
    st.stop()

with st.sidebar:
    st.header("Prediction run")
    run = st.selectbox("Run", runs, format_func=lambda p: p.name)
    P = _prediction(str(run))
    k_hours = list(P.summary["k_hours"])
    h = st.radio("Horizon (hours)", k_hours, index=k_hours.index(6) if 6 in k_hours else 0, horizontal=True)
    n_top = st.slider("Alerts shown", 3, max(3, len(P.alerts)), min(10, max(3, len(P.alerts))))
    st.caption(f"Model: `{P.summary['target_run']}`")
    st.caption("Research output, not investment advice.")

S = P.summary
short = lambda ts: pd.Timestamp(ts).strftime("%d %b %H:%M")   # noqa: E731  (year in the label: fits narrow columns)
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"As of (UTC, {S['as_of'][:4]})", short(S["as_of"]))
c2.metric("Shocking coins", ", ".join(S["source_coins"]))
c3.metric("Events scored", f"{S['n_events']:,}")
c4.metric(f"Data ends (UTC, {S['data_end'][:4]})", short(S["data_end"]), delta=f"{S['hours_since_as_of']} h after as-of"
          if S["hours_since_as_of"] else "same hour", delta_color="off")
PLURAL = {"crash": "crashes", "pump": "pumps"}
if not S["calibrated"]:
    st.warning("The model run has no validation predictions, so risks are raw model scores (not probabilities).")

for src in P.sources.to_dict("records"):
    typ = dd.typical_duration(P.persistence, src["type"], src["direction"])
    line = (f"**{src['coin']}**: {src['type']} {src['direction']} (z {src['z']:+.1f}; "
            f"{src['breadth']:.0%} of the other coins shocked in the same hour).")
    if typ:
        line += (f" Earlier {src['type']} {PLURAL.get(src['direction'], src['direction'])} lasted a median "
                 f"**{typ['median_hours_to_normal']:.0f} h**"
                 f" (half within {typ['q25_hours']:.0f}–{typ['q75_hours']:.0f} h) and were followed by "
                 f"{typ['followers_6h']:.1f} other coins shocking within 6 h.")
    st.markdown(line)

tab_risk, tab_events, tab_alerts, tab_dur, tab_net, tab_model = st.tabs(
    ["Risk per coin", "Event explorer", "Riskiest alerts", "Shock duration", "Network", "How good is the model"])

with tab_risk:
    rk = P.risk[P.risk["horizon_h"] == h].sort_values("risk", ascending=False)
    fig = px.bar(rk, x="coin", y="risk", color="source", hover_data=["score", "event_time", "lift"],
                 title=f"P(coin shocks within {h} h), from its riskiest recent event")
    if "base_rate" in rk and rk["base_rate"].notna().any():
        fig.add_hline(y=float(rk["base_rate"].iloc[0]), line_dash="dot",
                      annotation_text=f"validation base rate {rk['base_rate'].iloc[0]:.1%}")
    fig.update_layout(yaxis_tickformat=".0%", height=420)
    st.plotly_chart(fig, use_container_width=True)
    m = dd.risk_matrix(P.risk)
    hm = px.imshow(m.values, x=[f"{c} h" for c in m.columns], y=m.index, color_continuous_scale="Reds",
                   aspect="auto", zmin=0, title="Risk by coin and horizon")
    hm.update_traces(text=[[f"{v:.0%}" for v in r] for r in m.values], texttemplate="%{text}")
    hm.update_layout(height=max(380, 22 * len(m)))
    st.plotly_chart(hm, use_container_width=True)

with tab_events:
    ev_dir = dd.latest_events_dir(RUNS)
    if ev_dir is None:
        st.info("No event run yet. Run `cryptograph-train --config configs/experiment/predict_events_1h.yaml`, then "
                "reload.")
    else:
        E = _events_index(str(ev_dir))
        label = {e["id"]: f"{e['date']} · {e['name']} ({e['split']})" for e in E["events"]}
        eid = st.selectbox("Crisis", list(label), format_func=label.get, key="event")
        ev = _event(str(ev_dir), eid)
        rp, sector_of = ev["replay"], _sectors()
        coins_ev = [x["coin"] for x in rp["sequence"]]
        note = ev["split_note"].split(": ", 1)[-1]
        (st.success if ev["split"] == "test" else st.warning)(
            f"**{ev['split'].capitalize()} period:** {note}.")
        st.markdown(ev["description"] + (f" [Source]({ev['source_url']})" if ev["source_url"] else ""))
        m = rp["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Crisis hour (UTC, {ev['anchor_time'][:4]})", short(ev["anchor_time"]))
        c2.metric("Coins shocking then", f"{m['coins_shocking_at_anchor']} of {m['coins_active_at_anchor']}")
        c3.metric(f"Reach within {rp['window'][1]} h", f"{m['reach']:.0%}")
        c4.metric("First follower after", "—" if m["hours_to_first_follower"] is None
                  else f"{m['hours_to_first_follower']} h")

        # replay: the correlation network at the crisis hour, coins coloured by shock state, hour by hour
        edges_ev, _ = _graph({"as_of_bar": ev["anchor_bar"], "data_version": ev["data_version"],
                              "resolution": ev["resolution"]})
        pos = dd.circle_layout(coins_ev, sector_of)
        xs, ys = [pos[c][0] for c in coins_ev], [pos[c][1] for c in coins_ev]
        ex, ey = [], []
        for e in edges_ev.itertuples():
            ex += [pos[e.a][0], pos[e.b][0], None]
            ey += [pos[e.a][1], pos[e.b][1], None]
        edge_tr = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="#666"), hoverinfo="none",
                             showlegend=False)
        states = dd.node_states(rp["frames"], len(coins_ev))

        def node_tr(i):
            return go.Scatter(x=xs, y=ys, mode="markers+text", text=coins_ev, textposition="top center",
                              marker=dict(size=18, color=[STATE_COLOR[s] for s in states[i]],
                                          line=dict(width=1, color="#222")),
                              hovertext=[f"{c} ({sector_of.get(c, 'other')}): {s}"
                                         for c, s in zip(coins_ev, states[i])],
                              hoverinfo="text", showlegend=False)

        i0 = -rp["window"][0]                                          # the crisis hour
        frames = [go.Frame(data=[edge_tr, node_tr(i)], traces=[0, 1], name=str(f["offset"]))
                  for i, f in enumerate(rp["frames"])]
        legend = [go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=STATE_COLOR[s]), name=s)
                  for s in dd.NODE_STATES]
        play = dict(frame=dict(duration=250, redraw=True), fromcurrent=True, transition=dict(duration=0))
        pause = dict(frame=dict(duration=0, redraw=False), mode="immediate")
        fig = go.Figure(data=[edge_tr, node_tr(i0)] + legend, frames=frames)
        fig.update_layout(
            height=700, xaxis=dict(visible=False, range=[-1.35, 1.35]), yaxis=dict(visible=False, range=[-1.25, 1.25]),
            legend=dict(orientation="h", y=1.0, yanchor="bottom"), margin=dict(t=60, b=120),
            title="Replay: hours from the crisis hour (edges: correlation over the 7 days before)",
            updatemenus=[dict(type="buttons", direction="left", x=0, xanchor="left", y=0, yanchor="top",
                              pad=dict(t=45), bgcolor="#262730", bordercolor="#888", font=dict(color="#fafafa"),
                              buttons=[
                                  dict(label="▶ Play", method="animate", args=[None, play]),
                                  dict(label="⏸ Pause", method="animate", args=[[None], pause])])],
            sliders=[dict(active=i0, x=0.16, len=0.84, y=0, yanchor="top",
                          currentvalue=dict(prefix="hours from the crisis hour: "), pad=dict(t=30),
                          steps=[dict(label=f"{f['offset']:+d}", method="animate",
                                      args=[[str(f["offset"])], dict(mode="immediate",
                                                                      frame=dict(duration=0, redraw=True))])
                                 for f in rp["frames"]])])
        st.plotly_chart(fig, use_container_width=True)

        act = pd.DataFrame(rp["activity"])
        fa = px.bar(act, x="offset", y="coins_shocking", hover_data=["time"],
                    labels={"offset": "hours from the crisis hour", "coins_shocking": "coins shocking"},
                    title="Coins shocking per hour")
        fa.add_vline(x=0, line_dash="dash", line_color="#d62728")
        fa.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fa, use_container_width=True)

        all_sectors = sorted({sector_of.get(c, "other") for c in coins_ev})
        pick = st.multiselect("Sectors", all_sectors, key="event_sectors", placeholder="All sectors")

        st.subheader("What the model said at the crisis hour, and what happened")
        mv = ev["model_vs_outcome"]
        row = mv[mv["horizon_h"] == h] if len(mv) else mv
        if len(row):
            r0 = row.iloc[0]
            st.markdown(f"Of the model's **{int(r0['top_n'])} riskiest coins**, **{int(r0['top_hits'])} shocked** "
                        f"within {h} h; {int(r0['top_n'])} random coins would average "
                        f"{r0['expected_by_chance']:.1f} ({r0['share_shocked']:.0%} of the "
                        f"{int(r0['coins_scored'])} scored coins shocked).")
        else:
            st.info(f"No observed outcomes at {h} h for this event.")
        ot = dd.outcome_table(ev["risk"], h, pick)
        st.dataframe(ot.style.format({"risk": "{:.1%}", "lift": "{:.2f}×"}, na_rep="—"), hide_index=True,
                     use_container_width=True)

        st.subheader("Shock sequence per coin")
        seq = pd.DataFrame(rp["sequence"]).assign(sector=lambda x: x["coin"].map(sector_of).fillna("other"))
        if pick:
            seq = seq[seq["sector"].isin(pick)]
        seq = seq.sort_values(["first_shock_offset_h", "coin"], na_position="last")
        seq = seq.rename(columns={"first_shock_offset_h": "first shock (h after)", "first_run_h": "run (h)",
                                  "shock_hours_after": "shock hours after", "peak_abs_return_pct": "peak |return| %",
                                  "shocking_at_anchor": "shocking at crisis hour"})
        st.dataframe(seq.drop(columns=["active_at_anchor"]).round(2), hide_index=True, use_container_width=True)

        with st.expander("How to read this"):
            st.markdown(
                "- **Crisis hour**: the hour of the event date with the most coins shocking (the date alone can be "
                "hours from the crash). A **shock** is an hourly move beyond 2.29 standard deviations of the coin's "
                "trailing 7-day volatility (the project's leakage-free label).\n"
                "- **Edges** are correlations over the 7 days before: coins that moved together, not proof that one "
                "caused the other.\n"
                "- **Model vs outcome**: the risk each coin had at the crisis hour (calibrated on the validation "
                "period) and whether it then shocked within the horizon. Only **test-period** events are genuine "
                "forecasts; the model learned from training-period events, and validation events were used to "
                "choose and calibrate it.\n"
                "- When nearly every coin shocks (most crises), any ranking scores well: compare the hits with what "
                "random picks would get.")

with tab_alerts:
    st.caption("The riskiest coin × horizon pairs. Drivers are integrated gradients on the model's prediction head: "
               "how much each input pushed this risk up (+) or down (−) relative to an average input.")
    for a in P.alerts[:n_top]:
        lift = f", {a['lift']:.1f}× base rate" if a.get("lift") == a.get("lift") and a.get("lift") else ""
        with st.expander(f"{a['coin']} within {a['horizon_h']} h: {a['risk']:.1%}{lift} (after {a['source']} at "
                         f"{a['event_time'][:16]})"):
            d = dd.drivers_frame(a)
            if d.empty:
                st.write("No drivers recorded.")
                continue
            f = go.Figure(go.Bar(x=d["attribution"], y=d["input"], orientation="h",
                                 marker_color=["#d62728" if v > 0 else "#1f77b4" for v in d["attribution"]]))
            f.update_layout(height=60 + 34 * len(d), margin=dict(l=10, r=10, t=10, b=10),
                            yaxis=dict(autorange="reversed"), xaxis_title="contribution to the logit")
            st.plotly_chart(f, use_container_width=True)

with tab_dur:
    if P.persistence.empty:
        st.info("No shock-duration history in this run (predict.shock_context: false).")
    else:
        pt = P.persistence.copy()
        pt["group"] = pt["type"] + " " + pt["direction"]
        cur = {f"{s['type']} {s['direction']}" for s in P.sources.to_dict("records")}
        fig = go.Figure(go.Bar(
            x=pt["group"], y=pt["median_hours_to_normal"],
            error_y=dict(type="data", symmetric=False, array=pt["q75_hours"] - pt["median_hours_to_normal"],
                         arrayminus=pt["median_hours_to_normal"] - pt["q25_hours"]),
            marker_color=["#d62728" if g in cur else "#7f7f7f" for g in pt["group"]]))
        fig.update_layout(title="Hours until a coin's volatility is back to normal (median, quartiles); red = now",
                          yaxis_title="hours", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pt.drop(columns="group").round(2), hide_index=True, use_container_width=True)
        st.caption("Fresh shocks before the as-of time whose following week is fully observed (notebook 05 "
                   "definitions). Across all types, volatility is typically normal again within 8–10 hours.")

with tab_net:
    edges, snap_bar = _graph(S)
    ranks = _rankings(S.get("data_version", "v2"))
    coins = sorted(set(P.risk["coin"]) | set(edges["a"]) | set(edges["b"]))
    if not coins:
        st.info("No graph snapshot available.")
    else:
        sector = (ranks.set_index("coin")["sector"] if not ranks.empty else pd.Series(_sectors(), dtype=object))
        pos = dd.circle_layout(coins, sector.to_dict())
        risk_h = P.risk[P.risk["horizon_h"] == h].set_index("coin")["risk"]
        ex, ey = [], []
        for e in edges.itertuples():
            ex += [pos[e.a][0], pos[e.b][0], None]
            ey += [pos[e.a][1], pos[e.b][1], None]
        fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.7, color="#999"), hoverinfo="none",
                                   name="correlation edges", showlegend=False))
        for sec in sorted(sector.reindex(coins).fillna("other").unique()):
            cs = [c for c in coins if sector.get(c, "other") == sec]
            fig.add_trace(go.Scatter(
                x=[pos[c][0] for c in cs], y=[pos[c][1] for c in cs], mode="markers+text", name=sec,
                text=cs, textposition="top center",
                marker=dict(size=[10 + 60 * float(risk_h.get(c, 0)) for c in cs], line=dict(width=1, color="#333"),
                            symbol=["star" if c in S["source_coins"] else "circle" for c in cs]),
                hovertext=[f"{c}: {risk_h.get(c, float('nan')):.1%} within {h} h" for c in cs], hoverinfo="text"))
        fig.update_layout(height=560, xaxis=dict(visible=False, range=[-1.35, 1.35]),
                          yaxis=dict(visible=False, range=[-1.25, 1.25]), legend_title="sector",
                          title=f"Correlation network (trailing 7 days, snapshot bar {snap_bar}); node size = risk "
                                f"within {h} h; ★ = shocking now")
        st.plotly_chart(fig, use_container_width=True)
        if not ranks.empty:
            st.caption("Network rankings on the test window (notebook 02): who transmits volatility (Diebold–Yilmaz "
                       "net) and whose shocks are followed by others (shock PageRank).")
            cols = ["coin", "sector", "dy_net", "dy_pagerank", "shock_pagerank", "shock_out_lift", "community"]
            st.dataframe(ranks[[c for c in cols if c in ranks]].sort_values("dy_net", ascending=False).round(3),
                         hide_index=True, use_container_width=True)

with tab_model:
    tm = dd.model_test_metrics(P.model_metrics, k_hours)
    if tm.empty:
        st.info("The model's training run (metrics.json) is not in the runs folder.")
    else:
        st.caption("Test-period scores of the served model (one evaluation, notebook 03). AUPRC is precision averaged "
                   "over recall; the positive rate is what a random guess would score.")
        st.dataframe(tm.round(3), hide_index=True, use_container_width=True)
    st.markdown(
        "- **Risk** is the model's score after isotonic calibration on the validation period, so it reads as a "
        "probability (\"of past events scored like this, this share were followed by a shock of the coin\").\n"
        "- **Drivers** are exact decompositions of the score (integrated gradients; completeness gap "
        f"{S.get('ig_completeness_max_gap', float('nan')):.1e}).\n"
        "- The model is strongest after market-wide and partial shocks; after idiosyncratic shocks a coin's own "
        "state carries most of the signal (notebook 05).")
