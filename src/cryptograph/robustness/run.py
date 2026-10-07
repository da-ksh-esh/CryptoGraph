"""Robustness run behind notebook 06: stress tests of the final TGN and gradient boosting on the test period.

Checks before any perturbation: the clean TGN pass must reproduce the run's saved test predictions, and the refitted
boosting model (same trees, same data, deterministic) must reproduce its saved validation AUPRC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ..data.config import PROJECT_ROOT
from ..explain.run import _find_run
from ..explain.tgn_capture import load_trained
from .stress import auprc_by_k, tgn_test_pass


def _fit_gbm(run_dir, seed: int):
    import json

    from sklearn.ensemble import HistGradientBoostingClassifier

    from ..training.gbm_task import gbm_design
    from ..training.metrics import auprc

    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    met = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    g = gbm_design(cfg)
    tr = cfg["training"]
    params = dict(learning_rate=tr.get("learning_rate", 0.05), max_leaf_nodes=tr.get("max_leaf_nodes", 31),
                  min_samples_leaf=tr.get("min_samples_leaf", 200), l2_regularization=tr.get("l2", 1.0),
                  categorical_features=g["cat"], early_stopping=False, random_state=cfg["seed"])
    models, val_err = [], 0.0
    for k, kh in enumerate(g["k_hours"]):
        rows = {}
        for s in ("train", "val"):
            a, b = g["bounds"][s]
            m = g["YM"][a:b, k] & g["wmask"][s][g["t_np"][a:b]]
            rows[s] = np.arange(a, b)[m]
        clf = HistGradientBoostingClassifier(max_iter=met["trees"][k], **params).fit(g["X"][rows["train"]],
                                                                                    g["Y"][rows["train"], k])
        v = auprc(g["Y"][rows["val"], k], clf.predict_proba(g["X"][rows["val"]])[:, 1])
        val_err = max(val_err, abs(v - met[f"val_auprc_k{kh}h"]))
        models.append(clf)
    return models, g, val_err


def robustness_task(cfg: dict, seed: int, device: torch.device, tracker) -> dict[str, Any]:
    rc = cfg["robustness"]
    target = _find_run(rc["tgn_experiment"], rc.get("tgn_seed", 0))
    runner, D = load_trained(target)
    bounds, K = D["bounds"], D["k_hours"]
    tracker.log(f"TGN {target.name}")
    clean = tgn_test_pass(runner, bounds, seed=seed)
    saved = torch.load(target / "predictions_test.pt", weights_only=False)
    assert np.array_equal(saved["position"].numpy(), clean["pos"]), "clean pass must score the saved events"
    rebuild = float(np.abs(saved["prob"].float().numpy() - clean["prob"]).max())
    tracker.log(f"clean pass vs saved test predictions: max |dp| {rebuild:.1e}")
    assert rebuild < 5e-3, "clean pass does not reproduce the saved predictions"
    y, m, pos = clean["y"], clean["mask"], clean["pos"]

    gbm_dir = _find_run(rc["gbm_experiment"], 0, required="metrics.json")
    models, g, val_err = _fit_gbm(gbm_dir, seed)
    tracker.log(f"boosting {gbm_dir.name} refitted; max |val AUPRC - saved| {val_err:.1e}")
    assert val_err < 1e-5, "refitted boosting model differs from the saved run"
    Xt = g["X"][pos].copy()
    npb = g["n_perturbable"]

    def gbm_prob(X):
        return np.stack([mdl.predict_proba(X)[:, 1] for mdl in models], axis=1)

    gb_clean = gbm_prob(Xt)
    rows = []

    def add(model, kind, level, prob, ref):
        rows.append({"model": model, "perturbation": kind, "level": level,
                     **{f"auprc_k{k}h": v for k, v in zip(K, auprc_by_k(prob, y, m))},
                     "mean_abs_dp": float(np.abs(prob - ref)[m].mean())})

    add("TGN", "none", 0.0, clean["prob"], clean["prob"])
    add("gradient boosting", "none", 0.0, gb_clean, gb_clean)
    rng = np.random.default_rng(seed)
    for sigma in rc.get("noise_levels", [0.1, 0.25, 0.5, 1.0]):
        r = tgn_test_pass(runner, bounds, noise=sigma, seed=seed)
        add("TGN", "input noise", sigma, r["prob"], clean["prob"])
        Xn = Xt.copy()
        Xn[:, :npb] += sigma * rng.standard_normal((len(Xn), npb)).astype(np.float32)
        add("gradient boosting", "input noise", sigma, gbm_prob(Xn), gb_clean)
        tracker.log(f"noise {sigma}: done")
    for p in rc.get("drop_levels", [0.1, 0.25, 0.5]):
        r = tgn_test_pass(runner, bounds, drop=p, seed=seed)
        add("TGN", "missing events", p, r["prob"], clean["prob"])
        add("gradient boosting", "missing events", p, gb_clean, gb_clean)     # no event memory: unaffected
        tracker.log(f"drop {p}: done")
    for eps in rc.get("fgsm_levels", [0.05, 0.1, 0.25]):
        r = tgn_test_pass(runner, bounds, fgsm=eps, seed=seed, keep_inputs=True)
        add("TGN", "FGSM attack (white-box)", eps, r["prob"], clean["prob"])
        Xa = Xt.copy()
        Xa[:, :npb] = np.concatenate([r["x_src"], r["x_dst"], r["msg"]], axis=1)
        add("gradient boosting", "FGSM transferred from TGN", eps, gbm_prob(Xa), gb_clean)
        tracker.log(f"fgsm {eps}: done")
    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "reports" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stress_tests.csv", index=False)
    df.to_csv(tracker.dir / "stress_tests.csv", index=False)
    metrics: dict[str, Any] = {"tgn_run": target.name, "gbm_run": gbm_dir.name, "clean_rebuild_max_dp": rebuild,
                               "gbm_refit_val_err": val_err, "evaluated_test": True}
    for _, r in df.iterrows():
        key = f"{r['model'].split()[0].lower()}_{r['perturbation'].split()[0].lower()}_{r['level']:g}_auprc_k1h"
        metrics[key] = float(r["auprc_k1h"])
    return metrics
