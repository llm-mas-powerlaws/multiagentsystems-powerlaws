"""
topologies/__init__.py — Exports all 7 topology classes and a factory.
"""
from .base import BaseTopology, MASState
from .chain import ChainTopology
from .star import StarTopology
from .tree import TreeTopology
from .full_mesh import FullMeshTopology
from .sparse_mesh import SparseMeshTopology
from .hybrid import HybridModularTopology
from .dynamic_reputation import DynamicReputationTopology
from loggers.schemas import TopologyName

ALL_TOPOLOGIES = [
    TopologyName.CHAIN, TopologyName.STAR, TopologyName.TREE,
    TopologyName.FULL_MESH, TopologyName.SPARSE_MESH,
    TopologyName.HYBRID_MODULAR, TopologyName.DYNAMIC_REPUTATION,
]

_REGISTRY = {
    TopologyName.CHAIN:              ChainTopology,
    TopologyName.STAR:               StarTopology,
    TopologyName.TREE:               TreeTopology,
    TopologyName.FULL_MESH:          FullMeshTopology,
    TopologyName.SPARSE_MESH:        SparseMeshTopology,
    TopologyName.HYBRID_MODULAR:     HybridModularTopology,
    TopologyName.DYNAMIC_REPUTATION: DynamicReputationTopology,
}

def get_topology(name, **kwargs) -> BaseTopology:
    if isinstance(name, str):
        name = TopologyName(name)
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown topology: {name}")
    return cls(**kwargs)
