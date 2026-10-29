"""The dashboard's story readers (serving/story.py): report files in, tidy frames out, nothing when files are missing."""

import numpy as np
import pandas as pd

from cryptograph.serving import story as sy


def _write(root, rel, df):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def test_readers_return_nothing_without_report_files(tmp_path):
    assert sy.leaderboard(tmp_path).empty and sy.contagion_jump(tmp_path).empty
    assert sy.headline(tmp_path) == {} and sy.dataset_facts(tmp_path) == {}
    assert sy.network_timeseries(tmp_path).empty and sy.stress_tests(tmp_path).empty
    assert sy.attribution_groups(tmp_path).empty and sy.bootstrap_v3_vs_v2(tmp_path).empty


def test_leaderboard_joins_the_report_files_and_keeps_known_models(tmp_path):
    _write(tmp_path, "reports/cascade_baseline.csv", pd.DataFrame(
        {"resolution": "1h", "split": "test", "k_hours": [1, 1], "model": ["independent cascade", "pairwise p_ij"],
         "auprc": [0.16, 0.11], "positive_rate": [0.1, 0.1]}))
    _write(tmp_path, "reports/fair_eval_v3.csv", pd.DataFrame(
        {"model": ["tgn_1h_v3_final", "tgn_1h_v3_final"], "split": ["test", "val"], "k_hours": [1, 1],
         "event_auprc": [0.23, 0.18], "event_random": [0.1, 0.09], "event_lift": [2.3, 2.0], "event_auroc": [0.7, 0.67],
         "rh_lift": [2.6, 2.1], "top1_precision": [0.33, 0.22]}))
    _write(tmp_path, "reports/final_1h_test.csv", pd.DataFrame(
        {"inputs": ["v2", "v9"], "model": ["logreg", "logreg"], "k_hours": [1, 1], "event_auprc": [0.21, 0.5],
         "random": [0.1, 0.1]}))
    lb = sy.leaderboard(tmp_path)
    assert lb["id"].tolist() == ["independent cascade", "v2|logreg", "tgn_1h_v3_final"]   # unknown ids dropped
    assert np.allclose(lb["lift"], [1.6, 2.1, 2.3])
    assert set(lb["family"]) == {"network baseline", "machine learning", "temporal graph"}
    h = sy.headline(tmp_path)
    assert np.isclose(h["v3"]["auprc"], 0.23) and np.isclose(h["v3"]["rh_lift"], 2.6)   # test split only


def test_contagion_jump_is_the_rate_over_the_base_rate(tmp_path):
    _write(tmp_path, "reports/hourly_1h_label_balance.csv", pd.DataFrame(
        {"label": ["shock_next(|z|>2.29)", "contagion_k1h", "contagion_k24h"], "split": "test", "n": 10, "positives": 1,
         "rate": [0.04, 0.10, 0.60], "base_rate_any_coin": [np.nan, 0.04, 0.50]}))
    cj = sy.contagion_jump(tmp_path)
    assert cj["k_hours"].tolist() == [1, 24] and np.allclose(cj["lift"], [2.5, 1.2])


def test_network_story_communities_and_crisis_changes():
    from cryptograph.serving import story_network as sn

    rk = pd.DataFrame({"coin": list("ABCDEF"), "sector": list("xxxyyy"), "community": [0, 0, 0, 1, 1, 1]})
    c = sn.communities(rk, n_null=200)
    assert np.isclose(c["nmi"], 1.0) and c["p"] < 0.1 and len(c["groups"]) == 2      # groups = sectors exactly
    assert sn.communities(pd.DataFrame()) is None
    ts = pd.DataFrame({"time": pd.date_range("2022-01-01", periods=40), "mean_corr": [0.3] * 20 + [0.5] * 20})
    cc = sn.crisis_changes(ts, pd.DataFrame({"date": [pd.Timestamp("2022-01-21")], "name": ["Crash / detail"]}))
    assert cc["crisis"].iloc[0] == "Jan 2022 · Crash" and np.isclose(cc["average correlation"].iloc[0], 0.2)


def test_market_mode_finds_one_common_factor():
    rng = np.random.default_rng(0)
    market = rng.normal(size=(2000, 1))
    r = market + 0.5 * rng.normal(size=(2000, 10))           # ten coins sharing one factor
    mm = sy.market_mode(r, np.ones(2000, bool))
    assert mm["coins"] == 10 and mm["eigenvalues"][0] > 7 and mm["share"] > 0.7
    assert mm["eigenvalues"][1] < mm["noise_bound"] < mm["eigenvalues"][0]
    assert sy.market_mode(r, np.zeros(2000, bool)) is None   # too few hours
