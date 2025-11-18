"""
Tests for core.state module
"""
import pytest
import numpy as np
from channelpy.core.state import State, StateArray, EMPTY, DELTA, PHI, PSI


def test_state_creation():
    """Test basic state creation"""
    state = State(1, 1)
    assert state.i == 1
    assert state.q == 1
    assert state == PSI


def test_state_equality():
    """Test state equality"""
    assert State(1, 1) == PSI
    assert State(0, 0) == EMPTY
    assert State(1, 0) != State(0, 1)


def test_state_string_representation():
    """Test string conversion"""
    assert str(EMPTY) == '∅'
    assert str(DELTA) == 'δ'
    assert str(PHI) == 'φ'
    assert str(PSI) == 'ψ'


def test_state_from_name():
    """Test creating state from name"""
    assert State.from_name('psi') == PSI
    assert State.from_name('ψ') == PSI
    assert State.from_name('empty') == EMPTY


def test_state_array_creation():
    """Test StateArray creation"""
    states = StateArray.from_bits(i=[1, 0, 1], q=[1, 1, 0])
    assert len(states) == 3
    assert states[0] == PSI
    assert states[1] == PHI
    assert states[2] == DELTA


def test_state_array_count():
    """Test state counting"""
    states = StateArray.from_bits(
        i=[1, 0, 1, 1, 0],
        q=[1, 1, 0, 1, 0]
    )
    counts = states.count_by_state()
    assert counts[PSI] == 2
    assert counts[PHI] == 1
    assert counts[DELTA] == 1
    assert counts[EMPTY] == 1


def test_state_invalid_bits():
    """Test validation of bit values"""
    with pytest.raises(ValueError):
        State(2, 0)  # Invalid i
    with pytest.raises(ValueError):
        State(0, -1)  # Invalid q







