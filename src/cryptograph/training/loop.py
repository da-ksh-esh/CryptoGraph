"""Generic training / evaluation loop (skeleton; model-specific code supplies `step_fn`).

`step_fn(model, batch, device) -> (loss, outputs)` keeps this loop independent of the
model type (LSTM windows, graph snapshots, TGN event batches).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

StepFn = Callable[[torch.nn.Module, Any, torch.device], tuple]


def train_epoch(model: torch.nn.Module, loader: Iterable, step_fn: StepFn, optimizer: torch.optim.Optimizer,
                device: torch.device, grad_clip: float | None = 1.0) -> float:
    """One pass over `loader`; returns the mean loss."""
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        loss, _ = step_fn(model, batch, device)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: Iterable, step_fn: StepFn, device: torch.device) -> tuple[float, list]:
    """Mean loss and the list of per-batch outputs (for metrics)."""
    model.eval()
    total, n, outputs = 0.0, 0, []
    for batch in loader:
        loss, out = step_fn(model, batch, device)
        total += float(loss)
        n += 1
        outputs.append(out)
    return total / max(n, 1), outputs
