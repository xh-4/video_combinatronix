"""
Core channel algebra module.

Contains the fundamental State class and basic operations.
"""

from .state import State, StateArray, EMPTY, DELTA, PHI, PSI
from .operations import (
    gate, admit, overlay, weave, comp, neg_i, neg_q,
    compose, pipe
)
from .nested import NestedState
from .parallel import ParallelChannels
from .lattice import (
    partial_order, are_comparable, meet, join, lattice_distance,
    complement, is_atom, is_coatom, ChannelLattice, get_lattice,
    lattice_operations
)

__all__ = [
    'State', 'StateArray', 'EMPTY', 'DELTA', 'PHI', 'PSI',
    'gate', 'admit', 'overlay', 'weave', 'comp', 'neg_i', 'neg_q',
    'compose', 'pipe',
    'NestedState', 'ParallelChannels',
    'partial_order', 'are_comparable', 'meet', 'join', 'lattice_distance',
    'complement', 'is_atom', 'is_coatom', 'ChannelLattice', 'get_lattice',
    'lattice_operations',
]
