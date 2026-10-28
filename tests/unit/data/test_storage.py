import pickle
from collections import OrderedDict

import pytest
import torch

from cryptograph.data.storage import load_pt


class Payload:
    """Stands in for any object a tampered file could carry."""


def test_load_pt_reads_tensors_and_the_pyg_stream_but_refuses_other_objects(tmp_path):
    from torch_geometric.data import TemporalData

    torch.save({"x": torch.arange(3), "names": ["a", "b"], "nested": OrderedDict(k=1.5)}, tmp_path / "plain.pt")
    out = load_pt(tmp_path / "plain.pt")
    assert torch.equal(out["x"], torch.arange(3)) and out["names"] == ["a", "b"] and out["nested"]["k"] == 1.5
    d = TemporalData(src=torch.tensor([0, 1]), dst=torch.tensor([1, 0]), t=torch.tensor([3, 4]), msg=torch.ones(2, 5))
    torch.save({"full": d, "label_k_hours": [1, 3]}, tmp_path / "events.pt")
    e = load_pt(tmp_path / "events.pt")
    assert isinstance(e["full"], TemporalData) and e["full"].t.tolist() == [3, 4]
    torch.save({"bad": Payload()}, tmp_path / "tampered.pt")
    with pytest.raises(pickle.UnpicklingError):
        load_pt(tmp_path / "tampered.pt")
