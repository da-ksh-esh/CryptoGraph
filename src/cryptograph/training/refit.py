"""Refit on train + validation, then score the test period once (stream models: TGN, DyGFormer).

The number of epochs is fixed in the config, chosen beforehand from the validation curves of the matching
`*_final` runs; there is no early stopping because validation is now training data. The validation split ends
with the 7-day embargo before the test period, and validation samples only use label windows inside validation,
so no test information enters training.
"""

from __future__ import annotations

from typing import Any

import torch

from .tgn_task import _metrics


def refit_and_score(runner, d, bounds: dict, wmask: dict, epochs: int, opt, loss_fn, k_hours, tracker,
                    checkpoint: dict) -> dict[str, Any]:
    wmask["trainval"] = wmask["train"] | wmask["val"]       # the runner holds this same dict
    tv = (bounds["train"][0], bounds["val"][1])
    tracker.log(f"refit on train + val: events {tv[1] - tv[0]:,}, {epochs} epochs (fixed, no early stopping)")
    history = []
    for epoch in range(epochs):
        runner.reset()
        tl, *_ = runner.run(*tv, "trainval", opt, loss_fn)
        history.append({"epoch": epoch, "train_loss": tl})
        tracker.log(f"epoch {epoch}: train+val loss {tl:.5f}")
    tracker.save_checkpoint(checkpoint | {"model_state": runner.model.state_dict(), "refit_train_val": True})
    runner.reset()                                          # replay train + val (no gradient), then test once
    runner.run(*tv, "trainval")
    loss, lg, y, m = runner.run(*bounds["test"], "test", loss_fn=loss_fn)
    metrics: dict[str, Any] = {"epochs_run": epochs, "refit_train_val": True, "evaluated_test": True,
                               "test_loss": loss, **_metrics("test", lg, y, m, k_hours)}
    if lg is not None:
        pos = runner.last_positions
        torch.save({"position": pos, "t": d.t[pos], "src": d.src[pos], "dst": d.dst[pos],
                    "prob": torch.sigmoid(lg).half(), "y": y, "mask": m, "k_hours": k_hours},
                   tracker.dir / "predictions_test.pt")
    metrics["history"] = history
    tracker.log(f"test AUPRC k1h {metrics.get(f'test_auprc_k{k_hours[0]}h', float('nan')):.4f}")
    return metrics


def is_refit(tr: dict) -> bool:
    return bool(tr.get("refit_train_val", False))
