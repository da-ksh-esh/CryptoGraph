import importlib

import pytest


@pytest.mark.parametrize("mod", ["centrality", "communities", "mst", "topology", "cascade"])
def test_network_modules_import(mod):
    importlib.import_module(f"cryptograph.network.{mod}")
