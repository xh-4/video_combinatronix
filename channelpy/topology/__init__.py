"""
Topology module for channel algebra.

Contains topological data analysis and persistent homology components.
"""

from .persistence import PersistenceDiagram, compute_betti_numbers

__all__ = [
    'PersistenceDiagram', 'compute_betti_numbers',
]







