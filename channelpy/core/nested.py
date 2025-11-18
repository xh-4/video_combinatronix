"""
Nested channel states (hierarchical structure)
"""
from typing import Union, List, Optional, Dict
from .state import State, EMPTY, DELTA, PHI, PSI


class NestedState:
    """
    Nested channel state with multiple levels
    
    Each level is a State, forming a tree structure
    
    Examples
    --------
    >>> # Two-level nested state
    >>> state = NestedState(
    ...     level0=State(1, 1),
    ...     level1=State(0, 1)
    ... )
    >>> print(state)
    ψ.φ
    >>> state.depth
    1
    """
    
    def __init__(self, **levels):
        """
        Initialize nested state
        
        Parameters
        ----------
        **levels : State
            Keyword arguments level0, level1, level2, etc.
        """
        # Validate and store levels
        self._levels = {}
        level_nums = []
        
        for key, value in levels.items():
            if not key.startswith('level'):
                raise ValueError(f"Keys must be 'levelN', got '{key}'")
            
            try:
                level_num = int(key[5:])
            except ValueError:
                raise ValueError(f"Invalid level key: '{key}'")
            
            if not isinstance(value, State):
                raise TypeError(f"Level values must be State, got {type(value)}")
            
            level_nums.append(level_num)
            self._levels[level_num] = value
        
        # Check contiguous levels starting from 0
        if level_nums:
            level_nums.sort()
            if level_nums[0] != 0:
                raise ValueError("Levels must start at 0")
            for i in range(len(level_nums) - 1):
                if level_nums[i+1] != level_nums[i] + 1:
                    raise ValueError("Levels must be contiguous")
        
        self._depth = max(level_nums) if level_nums else -1
    
    @property
    def depth(self) -> int:
        """Maximum level index"""
        return self._depth
    
    @property
    def num_levels(self) -> int:
        """Number of levels"""
        return self._depth + 1
    
    @property
    def total_bits(self) -> int:
        """Total number of bits"""
        return 2 * self.num_levels
    
    @property
    def total_states(self) -> int:
        """Total number of possible states"""
        return 2 ** self.total_bits
    
    def get_level(self, level: int) -> State:
        """Get state at specific level"""
        if level not in self._levels:
            raise IndexError(f"Level {level} does not exist")
        return self._levels[level]
    
    def set_level(self, level: int, state: State):
        """Set state at specific level"""
        if level < 0 or level > self._depth + 1:
            raise IndexError(f"Invalid level: {level}")
        self._levels[level] = state
        if level > self._depth:
            self._depth = level
    
    def all_levels(self) -> List[State]:
        """Return list of all level states"""
        return [self._levels[i] for i in range(self.num_levels)]
    
    def all_psi(self) -> bool:
        """Check if all levels are ψ"""
        return all(s == PSI for s in self.all_levels())
    
    def any_empty(self) -> bool:
        """Check if any level is ∅"""
        return any(s == EMPTY for s in self.all_levels())
    
    def count_psi(self) -> int:
        """Count number of ψ levels"""
        return sum(1 for s in self.all_levels() if s == PSI)
    
    def path_string(self) -> str:
        """
        Return path as string
        
        Examples
        --------
        >>> state = NestedState(level0=PSI, level1=PHI, level2=DELTA)
        >>> state.path_string()
        'ψ.φ.δ'
        """
        return '.'.join(str(s) for s in self.all_levels())
    
    def path_matches(self, pattern: str) -> bool:
        """
        Check if path matches pattern
        
        Pattern can include wildcards (*)
        
        Examples
        --------
        >>> state = NestedState(level0=PSI, level1=PHI)
        >>> state.path_matches("ψ.*")
        True
        >>> state.path_matches("*.φ")
        True
        """
        pattern_parts = pattern.split('.')
        path_parts = self.path_string().split('.')
        
        if len(pattern_parts) != len(path_parts):
            return False
        
        for pattern_part, path_part in zip(pattern_parts, path_parts):
            if pattern_part != '*' and pattern_part != path_part:
                return False
        
        return True
    
    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.path_string() == other
        elif isinstance(other, NestedState):
            return self._levels == other._levels
        return False
    
    def __str__(self) -> str:
        return self.path_string()
    
    def __repr__(self) -> str:
        levels_str = ', '.join(
            f"level{i}={repr(s)}" 
            for i, s in self._levels.items()
        )
        return f"NestedState({levels_str})"
    
    @classmethod
    def from_path(cls, path: str) -> 'NestedState':
        """
        Create nested state from path string
        
        Examples
        --------
        >>> state = NestedState.from_path("ψ.φ.δ")
        >>> state.depth
        2
        """
        parts = path.split('.')
        levels = {}
        for i, part in enumerate(parts):
            levels[f'level{i}'] = State.from_name(part)
        return cls(**levels)







