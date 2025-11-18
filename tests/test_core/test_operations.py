"""
Tests for core.operations module
"""
import pytest
import numpy as np
from channelpy.core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from channelpy.core.operations import gate, admit, overlay, weave, comp, neg_i, neg_q


def test_gate_operation():
    """Test gate operation"""
    assert gate(DELTA) == EMPTY  # δ → ∅
    assert gate(PSI) == PSI      # ψ → ψ
    assert gate(PHI) == PHI      # φ → φ
    assert gate(EMPTY) == EMPTY  # ∅ → ∅


def test_admit_operation():
    """Test admit operation"""
    assert admit(DELTA) == PSI   # δ → ψ
    assert admit(PSI) == PSI     # ψ → ψ
    assert admit(PHI) == PHI     # φ → φ
    assert admit(EMPTY) == EMPTY # ∅ → ∅


def test_overlay_operation():
    """Test overlay operation"""
    assert overlay(DELTA, PHI) == PSI   # δ | φ = ψ
    assert overlay(EMPTY, PSI) == PSI   # ∅ | ψ = ψ
    assert overlay(EMPTY, EMPTY) == EMPTY  # ∅ | ∅ = ∅


def test_weave_operation():
    """Test weave operation"""
    assert weave(PSI, DELTA) == DELTA   # ψ & δ = δ
    assert weave(PHI, DELTA) == EMPTY   # φ & δ = ∅
    assert weave(PSI, PSI) == PSI       # ψ & ψ = ψ


def test_comp_operation():
    """Test complement operation"""
    assert comp(EMPTY) == PSI    # ∅ ↔ ψ
    assert comp(PSI) == EMPTY   # ψ ↔ ∅
    assert comp(DELTA) == PHI   # δ ↔ φ
    assert comp(PHI) == DELTA   # φ ↔ δ


def test_neg_i_operation():
    """Test neg_i operation"""
    assert neg_i(PSI) == PHI    # ψ → φ
    assert neg_i(PHI) == PSI    # φ → ψ
    assert neg_i(DELTA) == EMPTY  # δ → ∅
    assert neg_i(EMPTY) == DELTA  # ∅ → δ


def test_neg_q_operation():
    """Test neg_q operation"""
    assert neg_q(PSI) == DELTA  # ψ → δ
    assert neg_q(DELTA) == PSI  # δ → ψ
    assert neg_q(PHI) == EMPTY  # φ → ∅
    assert neg_q(EMPTY) == PHI  # ∅ → φ


def test_array_operations():
    """Test operations on StateArray"""
    states = StateArray.from_bits(i=[1, 0, 1], q=[1, 1, 0])
    gated = gate(states)
    admitted = admit(states)
    
    assert gated[0] == PSI   # ψ → ψ
    assert gated[1] == PHI   # φ → φ
    assert gated[2] == EMPTY # δ → ∅
    
    assert admitted[0] == PSI   # ψ → ψ
    assert admitted[1] == PHI   # φ → φ
    assert admitted[2] == PSI   # δ → ψ







