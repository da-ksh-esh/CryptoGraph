import numpy as np
import torch

from cryptograph.training.gbm_task import _scored_positions
from cryptograph.training.refit import refit_and_score


def test_scored_positions_match_stream_batches():
    # bars 0,0,1,1,1,2 ; batch size 2 never splits a bar -> batches [0,2), [2,5), [5,6)
    t = np.array([0, 0, 1, 1, 1, 2])
    ym = np.zeros((6, 1), bool)
    ym[3] = True                                      # only the middle batch has a counted label
    assert _scored_positions(t, ym, 0, 6, 2).tolist() == [2, 3, 4]
    # offset slice: events 2..5, labels given relative to a=2
    assert _scored_positions(t, ym[2:], 2, 6, 2).tolist() == [2, 3, 4]


class _Tracker:
    def __init__(self, tmp):
        self.dir, self.logs, self.ckpt = tmp, [], None

    def log(self, s):
        self.logs.append(s)

    def save_checkpoint(self, c):
        self.ckpt = c


class _Runner:
    """Records which ranges and splits are run; returns fixed logits on the test range."""

    def __init__(self):
        self.calls, self.model = [], torch.nn.Linear(1, 1)

    def reset(self):
        self.calls.append("reset")

    def run(self, a, b, split, optimizer=None, loss_fn=None, max_batches=None):
        self.calls.append((a, b, split, optimizer is not None))
        if split != "test":
            return 0.1, None, None, None
        n = b - a
        self.last_positions = torch.arange(a, b)
        y = torch.zeros(n, 1, dtype=torch.bool)
        y[0] = True
        return 0.2, torch.linspace(1, -1, n)[:, None], y, torch.ones(n, 1, dtype=torch.bool)


class _D:
    t = torch.arange(30)
    src = torch.zeros(30, dtype=torch.long)
    dst = torch.ones(30, dtype=torch.long)


def test_refit_trains_on_train_plus_val_only_then_scores_test_once(tmp_path):
    runner, tracker = _Runner(), _Tracker(tmp_path)
    bounds = {"train": (0, 10), "val": (10, 20), "test": (20, 30)}
    wmask = {s: np.ones(30, bool) for s in ("train", "val", "test")}
    m = refit_and_score(runner, _D(), bounds, wmask, 3, object(), None, [1], tracker, {})
    train_calls = [c for c in runner.calls if c != "reset" and c[3]]
    assert train_calls == [(0, 20, "trainval", True)] * 3          # gradient steps never touch the test range
    assert [c for c in runner.calls if c != "reset" and c[2] == "test"] == [(20, 30, "test", False)]
    assert m["refit_train_val"] and m["evaluated_test"] and m["test_auprc_k1h"] == 1.0
    assert (tmp_path / "predictions_test.pt").exists()
    assert tracker.ckpt["refit_train_val"]
