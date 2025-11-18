"""
Parallel channel systems (independent dimensions)
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from .state import State, StateArray


class ParallelChannels:
    """
    Multiple independent channel states
    
    Examples
    --------
    >>> channels = ParallelChannels(
    ...     technical=State(1, 1),
    ...     business=State(1, 0),
    ...     team=State(0, 1)
    ... )
    >>> channels['technical']
    ψ
    >>> channels.all_names()
    ['technical', 'business', 'team']
    """
    
    def __init__(self, **channels):
        """
        Initialize parallel channels
        
        Parameters
        ----------
        **channels : State
            Named channel states
        """
        self._channels = {}
        
        for name, state in channels.items():
            if not isinstance(state, State):
                raise TypeError(f"Channel values must be State, got {type(state)}")
            self._channels[name] = state
    
    def __getitem__(self, name: str) -> State:
        """Get channel state by name"""
        return self._channels[name]
    
    def __setitem__(self, name: str, state: State):
        """Set channel state by name"""
        if not isinstance(state, State):
            raise TypeError(f"Value must be State, got {type(state)}")
        self._channels[name] = state
    
    def __len__(self) -> int:
        """Number of channels"""
        return len(self._channels)
    
    def all_names(self) -> List[str]:
        """List of channel names"""
        return list(self._channels.keys())
    
    def all_states(self) -> List[State]:
        """List of all states"""
        return list(self._channels.values())
    
    def to_dict(self) -> Dict[str, State]:
        """Convert to dictionary"""
        return self._channels.copy()
    
    def count_psi(self) -> int:
        """Count channels in ψ state"""
        from .state import PSI
        return sum(1 for s in self._channels.values() if s == PSI)
    
    def all_psi(self) -> bool:
        """Check if all channels are ψ"""
        from .state import PSI
        return all(s == PSI for s in self._channels.values())
    
    def any_empty(self) -> bool:
        """Check if any channel is ∅"""
        from .state import EMPTY
        return any(s == EMPTY for s in self._channels.values())
    
    def __str__(self) -> str:
        parts = [f"{name}:{state}" for name, state in self._channels.items()]
        return f"({', '.join(parts)})"
    
    def __repr__(self) -> str:
        items = ', '.join(
            f"{name}={repr(state)}" 
            for name, state in self._channels.items()
        )
        return f"ParallelChannels({items})"







