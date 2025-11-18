"""
Core channel algebra operations
"""
from typing import Union, List
import numpy as np
from .state import State, StateArray, EMPTY, DELTA, PHI, PSI


def gate(state: Union[State, StateArray]) -> Union[State, StateArray]:
    """
    Gate operation: Remove elements not validated by membership
    
    Rule: If q=0, set i=0
    
    ∅ → ∅
    δ → ∅  (puncture removed)
    φ → φ  (hole preserved)
    ψ → ψ  (resonant preserved)
    
    Examples
    --------
    >>> gate(DELTA)
    ∅
    >>> gate(PSI)
    ψ
    """
    if isinstance(state, State):
        return State(state.i & state.q, state.q)
    elif isinstance(state, StateArray):
        return StateArray(state.i & state.q, state.q)
    else:
        raise TypeError(f"Expected State or StateArray, got {type(state)}")


def admit(state: Union[State, StateArray]) -> Union[State, StateArray]:
    """
    Admit operation: Grant membership to present elements
    
    Rule: If i=1, set q=1
    
    ∅ → ∅
    δ → ψ  (puncture validated)
    φ → φ  (hole remains)
    ψ → ψ  (already resonant)
    
    Examples
    --------
    >>> admit(DELTA)
    ψ
    >>> admit(EMPTY)
    ∅
    """
    if isinstance(state, State):
        return State(state.i, state.q | state.i)
    elif isinstance(state, StateArray):
        return StateArray(state.i, state.q | state.i)
    else:
        raise TypeError(f"Expected State or StateArray, got {type(state)}")


def overlay(state1: Union[State, StateArray], 
            state2: Union[State, StateArray]) -> Union[State, StateArray]:
    """
    Overlay operation: Bitwise OR (union)
    
    Takes maximum information from both states
    
    Examples
    --------
    >>> overlay(DELTA, PHI)
    ψ
    >>> overlay(EMPTY, PSI)
    ψ
    """
    if isinstance(state1, State) and isinstance(state2, State):
        return State(state1.i | state2.i, state1.q | state2.q)
    elif isinstance(state1, StateArray) and isinstance(state2, StateArray):
        return StateArray(state1.i | state2.i, state1.q | state2.q)
    else:
        raise TypeError("Both arguments must be same type (State or StateArray)")


def weave(state1: Union[State, StateArray], 
          state2: Union[State, StateArray]) -> Union[State, StateArray]:
    """
    Weave operation: Bitwise AND (intersection)
    
    Keeps only common information
    
    Examples
    --------
    >>> weave(PSI, DELTA)
    δ
    >>> weave(PHI, DELTA)
    ∅
    """
    if isinstance(state1, State) and isinstance(state2, State):
        return State(state1.i & state2.i, state1.q & state2.q)
    elif isinstance(state1, StateArray) and isinstance(state2, StateArray):
        return StateArray(state1.i & state2.i, state1.q & state2.q)
    else:
        raise TypeError("Both arguments must be same type (State or StateArray)")


def comp(state: Union[State, StateArray]) -> Union[State, StateArray]:
    """
    Complement operation: Flip both bits
    
    ∅ ↔ ψ
    δ ↔ φ
    
    Examples
    --------
    >>> comp(EMPTY)
    ψ
    >>> comp(DELTA)
    φ
    """
    if isinstance(state, State):
        return State(1 - state.i, 1 - state.q)
    elif isinstance(state, StateArray):
        return StateArray(1 - state.i, 1 - state.q)
    else:
        raise TypeError(f"Expected State or StateArray, got {type(state)}")


def neg_i(state: Union[State, StateArray]) -> Union[State, StateArray]:
    """Flip i-bit only"""
    if isinstance(state, State):
        return State(1 - state.i, state.q)
    elif isinstance(state, StateArray):
        return StateArray(1 - state.i, state.q)
    else:
        raise TypeError(f"Expected State or StateArray, got {type(state)}")


def neg_q(state: Union[State, StateArray]) -> Union[State, StateArray]:
    """Flip q-bit only"""
    if isinstance(state, State):
        return State(state.i, 1 - state.q)
    elif isinstance(state, StateArray):
        return StateArray(state.i, 1 - state.q)
    else:
        raise TypeError(f"Expected State or StateArray, got {type(state)}")


# Functional composition helpers

def compose(*operations):
    """
    Compose operations right-to-left
    
    Examples
    --------
    >>> admit_then_gate = compose(gate, admit)
    >>> admit_then_gate(DELTA)
    ψ
    """
    def composed(state):
        result = state
        for op in reversed(operations):
            result = op(result)
        return result
    return composed


def pipe(*operations):
    """
    Compose operations left-to-right
    
    Examples
    --------
    >>> process = pipe(admit, gate)
    >>> process(DELTA)
    ψ
    """
    def piped(state):
        result = state
        for op in operations:
            result = op(result)
        return result
    return piped







