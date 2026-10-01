import networkx as nx
import numpy as np

from cryptograph.network.cascade import IndependentCascade, rolling_transmission
from cryptograph.network.centrality import eigenvector_centrality, pagerank, strength, transmitter_pagerank
from cryptograph.network.communities import align_labels, louvain, modularity, nmi, remove_market_mode
from cryptograph.network.features import NODE_FEATURES, NetworkParams, compute_network_features
from cryptograph.network.mst import corr_distance, mst_degree, mst_edges, normalized_tree_length
from cryptograph.network.spillover import spillover_summary, var_ols, window_gfevd
from cryptograph.network.topology import threshold_adjacency, topology_metrics
from cryptograph.data.features.targets import future_any

ALL = np.ones(5, bool)


def test_pagerank_matches_networkx():
    rng = np.random.default_rng(0)
    W = rng.uniform(0, 1, (5, 5)) * (rng.uniform(size=(5, 5)) > 0.4)
    np.fill_diagonal(W, 0)
    G = nx.DiGraph()
    G.add_nodes_from(range(5))
    G.add_weighted_edges_from((i, j, W[i, j]) for i in range(5) for j in range(5) if W[i, j] > 0)
    ref = nx.pagerank(G, alpha=0.85, weight="weight", tol=1e-12)
    assert np.allclose(pagerank(W, ALL), [ref[i] for i in range(5)], atol=1e-8)


def test_transmitter_pagerank_ranks_the_coin_others_depend_on():
    theta = np.zeros((5, 5))
    theta[1:, 0] = 0.5                       # coins 1..4 are driven by coin 0
    theta[1, 2] = 0.1
    pr = transmitter_pagerank(theta, ALL)
    assert pr.argmax() == 0 and abs(np.nansum(pr) - 1) < 1e-9
    act = np.array([True, True, True, False, True])
    assert np.isnan(transmitter_pagerank(theta, act)[3])


def test_strength_and_eigenvector():
    C = np.full((4, 4), 0.2)
    C[0, 1:] = C[1:, 0] = 0.8
    np.fill_diagonal(C, np.nan)
    s = strength(C, np.ones(4, bool))
    e = eigenvector_centrality(C, np.ones(4, bool))
    assert s.argmax() == 0 and np.isclose(s[0], 0.8) and e.argmax() == 0 and np.isclose(e.sum(), 1)


def test_mst_on_known_example():
    C = np.array([[1, .9, .1, .1], [.9, 1, .8, .1], [.1, .8, 1, .7], [.1, .1, .7, 1.]])
    e = mst_edges(C, np.ones(4, bool))
    assert sorted((min(i, j), max(i, j)) for i, j, _ in e) == [(0, 1), (1, 2), (2, 3)]
    d = corr_distance(np.array([.9, .8, .7]))
    assert np.isclose(normalized_tree_length(C, np.ones(4, bool)), d.mean())
    assert mst_degree(C, np.ones(4, bool)).tolist() == [1, 2, 2, 1]
    act = np.array([True, True, False, True])
    assert len(mst_edges(C, act)) == 2


def _two_block_corr(seed=0, T=3000):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=T)
    b1, b2 = rng.normal(size=T), rng.normal(size=T)
    X = np.stack([m + 0.7 * (b1 if i < 4 else b2) + 0.5 * rng.normal(size=T) for i in range(8)], axis=1)
    C = np.corrcoef(X.T)
    np.fill_diagonal(C, np.nan)
    return C, np.array([0] * 4 + [1] * 4)


def test_louvain_after_market_mode_recovers_blocks():
    C, truth = _two_block_corr()
    act = np.ones(8, bool)
    R = remove_market_mode(C, act)
    lab = louvain(R, act, seed=1)
    assert nmi(lab, truth) == 1.0 and modularity(R, lab) > 0.2
    assert np.array_equal(lab, louvain(R, act, seed=1))
    lab_raw = louvain(np.nan_to_num(C), act, seed=1)          # the market mode hides the blocks
    assert len(np.unique(lab_raw)) <= len(np.unique(lab))


def test_align_labels_keeps_ids_stable():
    prev = np.array([0, 0, 1, 1, -1])
    cur = np.array([1, 1, 0, 0, 2])
    assert align_labels(prev, cur).tolist() == [0, 0, 1, 1, 2]


def test_topology_on_complete_and_empty_graphs():
    C = np.full((5, 5), 0.9)
    np.fill_diagonal(C, np.nan)
    m = topology_metrics(threshold_adjacency(C, ALL, 0.5), ALL, np.array([0, 0, 1, 1, 1]))
    assert m["density"] == 1 and m["avg_clustering"] == 1 and m["avg_shortest_path"] == 1 and m["max_core"] == 4
    e = topology_metrics(threshold_adjacency(C, ALL, 0.95), ALL)
    assert e["density"] == 0 and e["n_components"] == 5


def test_var_ols_and_gfevd():
    rng = np.random.default_rng(0)
    T, A = 20000, np.array([[0.5, 0.0], [0.3, 0.2]])
    X = np.zeros((T, 2))
    for t in range(1, T):
        X[t] = A @ X[t - 1] + rng.normal(size=2)
    coefs, sigma = var_ols(X, 1)
    assert np.allclose(coefs[0], A, atol=0.03) and np.allclose(sigma, np.eye(2), atol=0.05)
    th = window_gfevd(X, H=10)
    assert np.allclose(th.sum(axis=1), 100) and th[1, 0] > th[0, 1]      # 0 drives 1
    s = spillover_summary(th)
    assert s["net"][0] > 0 > s["net"][1]


def _shock_data(T=4000, seed=0):
    rng = np.random.default_rng(seed)
    shock = rng.uniform(size=(T, 3)) < 0.04
    follow = np.roll(shock[:, 0], 1) & (rng.uniform(size=T) < 0.8)     # coin 1 follows coin 0 one bar later
    shock[:, 1] |= follow
    return shock, np.ones_like(shock)


def test_cascade_fitted_on_train_only_and_finds_the_link():
    shock, valid = _shock_data()
    train = np.zeros(len(shock), bool)
    train[:2000 - 3] = True                     # train sample bars: label window (t, t+3] ends inside train
    m = IndependentCascade.fit(shock, valid, train, [1, 3], prior_strength=10)
    assert m.P[1][0, 1] > 0.6 and m.P[1][0, 2] < 0.15 and m.P[1][1, 0] < 0.15
    s2 = shock.copy()
    s2[2000:] = ~s2[2000:]                                             # change everything after train
    m2 = IndependentCascade.fit(s2, valid, train, [1, 3], prior_strength=10)
    assert np.allclose(m.P[1], m2.P[1]) and np.allclose(m.P[3], m2.P[3])
    shocked = np.array([[True, False, False], [False, False, True]])
    ic = m.cascade(1, shocked, np.array([1, 1]))
    assert ic[0] > 0.6 and ic[1] < 0.15


def test_rolling_transmission_uses_only_completed_windows():
    shock, valid = _shock_data()
    src = shock & valid
    anyf, allv = future_any(src, valid, 1)
    P, lift = rolling_transmission(src, anyf, allv, 1500, 1000, 1, 10)
    s2 = src.copy()
    s2[1501:] = True                                                   # bars after t_end
    a2, v2 = future_any(s2, valid, 1)
    P2, _ = rolling_transmission(s2, a2, v2, 1500, 1000, 1, 10)
    assert np.allclose(P, P2, equal_nan=True) and lift[0, 1] > 0.5


def test_network_features_are_causal():
    rng = np.random.default_rng(0)
    T, N = 1200, 6
    m = rng.normal(size=(T, 1))
    r = m + rng.normal(size=(T, N))
    r[:200, 5] = np.nan                                                # a late listing
    active = np.isfinite(r)
    shock = np.abs(r) > 2.0
    valid = active.copy()
    snap_t = np.arange(99, T, 50)
    p = NetworkParams(corr_window=100, spill_window=200, spill_lags=1, spill_horizon=5, cascade_window=400,
                      cascade_k=2, cascade_prior=10, threshold=0.4)
    sec = np.array([0, 0, 1, 1, 2, 2])
    X1, c1, g1 = compute_network_features(r, shock, valid, active, snap_t, sec, p)
    assert X1.shape == (len(snap_t), N, len(NODE_FEATURES))
    t0 = 700
    r2, s2 = r.copy(), shock.copy()
    r2[t0 + 1:] = r2[t0 + 1:] * 3 + 1
    s2[t0 + 1:] = ~s2[t0 + 1:]
    X2, c2, g2 = compute_network_features(r2, s2, valid, active, snap_t, sec, p)
    keep = snap_t <= t0
    assert np.allclose(X1[keep], X2[keep], equal_nan=True) and np.array_equal(c1[keep], c2[keep])
    assert g1[keep].equals(g2[keep])
    assert np.isnan(X1[snap_t < 200][:, 5, 0]).all() and np.isfinite(X1[-1, :, 0]).all()
