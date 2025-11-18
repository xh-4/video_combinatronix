"""
Core channel state representation
"""
from typing import Optional, Union, Tuple
from enum import Enum
import numpy as np


class State:
    """
    Channel algebra state with two bits: i (presence) and q (membership)
    
    Four possible states:
    - EMPTY (∅): i=0, q=0  - Absent
    - DELTA (δ): i=1, q=0  - Present but not member (puncture)
    - PHI   (φ): i=0, q=1  - Not present but expected (hole)
    - PSI   (ψ): i=1, q=1  - Present and member (resonant)
    
    Examples
    --------
    >>> state = State(i=1, q=1)  # ψ state
    >>> print(state)
    ψ
    >>> state == PSI
    True
    """
    
    __slots__ = ('_i', '_q')
    
    def __init__(self, i: int, q: int):
        """
        Initialize channel state
        
        Parameters
        ----------
        i : int
            Presence bit (0 or 1)
        q : int
            Membership bit (0 or 1)
            
        Raises
        ------
        ValueError
            If i or q not in {0, 1}
        """
        if i not in (0, 1) or q not in (0, 1):
            raise ValueError(f"Bits must be 0 or 1, got i={i}, q={q}")
        
        self._i = i
        self._q = q
    
    @property
    def i(self) -> int:
        """Presence bit"""
        return self._i
    
    @property
    def q(self) -> int:
        """Membership bit"""
        return self._q
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, State):
            return False
        return self.i == other.i and self.q == other.q
    
    def __hash__(self) -> int:
        return hash((self.i, self.q))
    
    def __repr__(self) -> str:
        return f"State(i={self.i}, q={self.q})"
    
    def __str__(self) -> str:
        """Unicode representation"""
        symbols = {
            (0, 0): '∅',
            (1, 0): 'δ',
            (0, 1): 'φ',
            (1, 1): 'ψ'
        }
        return symbols[(self.i, self.q)]
    
    def to_bits(self) -> Tuple[int, int]:
        """Return as (i, q) tuple"""
        return (self.i, self.q)
    
    def to_int(self) -> int:
        """Convert to integer 0-3"""
        return self.i * 2 + self.q
    
    @classmethod
    def from_int(cls, value: int) -> 'State':
        """Create state from integer 0-3"""
        if not 0 <= value <= 3:
            raise ValueError(f"Value must be 0-3, got {value}")
        i = value // 2
        q = value % 2
        return cls(i, q)
    
    @classmethod
    def from_name(cls, name: str) -> 'State':
        """Create state from name"""
        name_map = {
            'empty': (0, 0), '∅': (0, 0),
            'delta': (1, 0), 'δ': (1, 0),
            'phi': (0, 1), 'φ': (0, 1),
            'psi': (1, 1), 'ψ': (1, 1),
        }
        if name.lower() not in name_map:
            raise ValueError(f"Unknown state name: {name}")
        i, q = name_map[name.lower()]
        return cls(i, q)
    
    def to_complex(self) -> complex:
        """
        Convert to complex number: i + iq
        Useful for quantum/phase space interpretations
        """
        return self.i + 1j * self.q


# Pre-defined state constants
EMPTY = State(0, 0)
DELTA = State(1, 0)
PHI = State(0, 1)
PSI = State(1, 1)


class StateArray:
    """
    Efficient array of states using numpy
    
    Examples
    --------
    >>> states = StateArray.from_bits(i=[1,0,1], q=[1,1,0])
    >>> states[0]
    ψ
    >>> len(states)
    3
    """
    
    def __init__(self, i: np.ndarray, q: np.ndarray):
        """
        Initialize state array
        
        Parameters
        ----------
        i : np.ndarray
            Array of presence bits
        q : np.ndarray
            Array of membership bits
        """
        i = np.asarray(i, dtype=np.int8)
        q = np.asarray(q, dtype=np.int8)
        
        if i.shape != q.shape:
            raise ValueError("i and q must have same shape")
        
        self._i = i
        self._q = q
    
    @classmethod
    def from_bits(cls, i, q) -> 'StateArray':
        """Create from bit arrays"""
        return cls(i, q)
    
    @classmethod
    def from_states(cls, states: list) -> 'StateArray':
        """Create from list of State objects"""
        i = np.array([s.i for s in states], dtype=np.int8)
        q = np.array([s.q for s in states], dtype=np.int8)
        return cls(i, q)
    
    @property
    def i(self) -> np.ndarray:
        """Presence bit array"""
        return self._i
    
    @property
    def q(self) -> np.ndarray:
        """Membership bit array"""
        return self._q
    
    def __len__(self) -> int:
        return len(self._i)
    
    def __getitem__(self, idx) -> Union[State, 'StateArray']:
        if isinstance(idx, int):
            return State(int(self._i[idx]), int(self._q[idx]))
        else:
            return StateArray(self._i[idx], self._q[idx])
    
    def to_ints(self) -> np.ndarray:
        """Convert to integer array 0-3"""
        return self._i * 2 + self._q
    
    def to_strings(self) -> np.ndarray:
        """Convert to string array"""
        symbols = np.array(['∅', 'φ', 'δ', 'ψ'])
        return symbols[self.to_ints()]
    
    def count_by_state(self) -> dict:
        """Count occurrences of each state"""
        ints = self.to_ints()
        return {
            EMPTY: np.sum(ints == 0),
            PHI: np.sum(ints == 1),
            DELTA: np.sum(ints == 2),
            PSI: np.sum(ints == 3)
        }







