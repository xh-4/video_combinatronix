"""
Lattice structure for channel states

Defines partial order and lattice operations
"""
from typing import Optional, Set, List, Tuple
from .state import State, EMPTY, DELTA, PHI, PSI


def partial_order(s1: State, s2: State) -> bool:
    """
    Check if s1 ≤ s2 in channel lattice
    
    s1 ≤ s2 iff s1.i ≤ s2.i AND s1.q ≤ s2.q
    
    Examples
    --------
    >>> partial_order(EMPTY, DELTA)  # True
    >>> partial_order(DELTA, EMPTY)  # False
    >>> partial_order(EMPTY, PSI)    # True
    """
    return s1.i <= s2.i and s1.q <= s2.q


def are_comparable(s1: State, s2: State) -> bool:
    """
    Check if states are comparable in the lattice
    
    Two states are comparable if one is less than or equal to the other
    
    Examples
    --------
    >>> are_comparable(EMPTY, PSI)   # True
    >>> are_comparable(DELTA, PHI)   # False
    """
    return partial_order(s1, s2) or partial_order(s2, s1)


def meet(s1: State, s2: State) -> State:
    """
    Greatest lower bound (AND operation)
    
    Returns the largest state that is less than or equal to both inputs
    
    Examples
    --------
    >>> meet(DELTA, PHI)  # EMPTY
    >>> meet(PSI, DELTA)  # DELTA
    """
    return State(min(s1.i, s2.i), min(s1.q, s2.q))


def join(s1: State, s2: State) -> State:
    """
    Least upper bound (OR operation)
    
    Returns the smallest state that is greater than or equal to both inputs
    
    Examples
    --------
    >>> join(DELTA, PHI)  # PSI
    >>> join(EMPTY, PSI)  # PSI
    """
    return State(max(s1.i, s2.i), max(s1.q, s2.q))


def lattice_distance(s1: State, s2: State) -> int:
    """
    Hamming distance in the lattice
    
    Returns the number of bit positions that differ
    
    Examples
    --------
    >>> lattice_distance(EMPTY, PSI)  # 2
    >>> lattice_distance(DELTA, PHI)  # 2
    >>> lattice_distance(EMPTY, DELTA)  # 1
    """
    return abs(s1.i - s2.i) + abs(s1.q - s2.q)


def complement(state: State) -> State:
    """
    Lattice complement (dual)
    
    In a Boolean lattice, complement flips both bits
    
    Examples
    --------
    >>> complement(EMPTY)  # PSI
    >>> complement(PSI)   # EMPTY
    >>> complement(DELTA) # PHI
    >>> complement(PHI)   # DELTA
    """
    return State(1 - state.i, 1 - state.q)


def is_atom(state: State) -> bool:
    """
    Check if state is an atom (immediate successor of bottom)
    
    Examples
    --------
    >>> is_atom(DELTA)  # True
    >>> is_atom(PHI)    # True
    >>> is_atom(EMPTY)  # False
    >>> is_atom(PSI)    # False
    """
    return lattice_distance(state, EMPTY) == 1


def is_coatom(state: State) -> bool:
    """
    Check if state is a coatom (immediate predecessor of top)
    
    Examples
    --------
    >>> is_coatom(DELTA)  # True
    >>> is_coatom(PHI)    # True
    >>> is_coatom(EMPTY)  # False
    >>> is_coatom(PSI)    # False
    """
    return lattice_distance(state, PSI) == 1


class ChannelLattice:
    """
    Complete lattice structure for channel states
    
    Provides comprehensive lattice operations and analysis
    
    Examples
    --------
    >>> lattice = ChannelLattice()
    >>> lattice.bottom
    ∅
    >>> lattice.top
    ψ
    >>> lattice.get_successors(EMPTY)
    [δ, φ]
    >>> lattice.get_predecessors(PSI)
    [δ, φ]
    """
    
    def __init__(self):
        self.elements = [EMPTY, DELTA, PHI, PSI]
        self.bottom = EMPTY
        self.top = PSI
        
        # Pre-compute lattice structure
        self._successors = self._compute_successors()
        self._predecessors = self._compute_predecessors()
        self._chains = self._compute_all_chains()
    
    def _compute_successors(self) -> dict:
        """Pre-compute immediate successors for each state"""
        successors = {}
        for state in self.elements:
            successors[state] = []
            for other in self.elements:
                if (partial_order(state, other) and 
                    lattice_distance(state, other) == 1):
                    successors[state].append(other)
        return successors
    
    def _compute_predecessors(self) -> dict:
        """Pre-compute immediate predecessors for each state"""
        predecessors = {}
        for state in self.elements:
            predecessors[state] = []
            for other in self.elements:
                if (partial_order(other, state) and 
                    lattice_distance(other, state) == 1):
                    predecessors[state].append(other)
        return predecessors
    
    def _compute_all_chains(self) -> dict:
        """Pre-compute all possible chains in the lattice"""
        chains = {}
        for start in self.elements:
            for end in self.elements:
                if partial_order(start, end):
                    chains[(start, end)] = self._find_chains_recursive(start, end, [start])
        return chains
    
    def _find_chains_recursive(self, current: State, target: State, path: List[State]) -> List[List[State]]:
        """Recursively find all chains from current to target"""
        if current == target:
            return [path.copy()]
        
        chains = []
        for successor in self._successors[current]:
            if partial_order(successor, target) and successor not in path:
                new_path = path + [successor]
                chains.extend(self._find_chains_recursive(successor, target, new_path))
        
        return chains
    
    def get_successors(self, state: State) -> List[State]:
        """Get immediate successors in lattice"""
        return self._successors[state].copy()
    
    def get_predecessors(self, state: State) -> List[State]:
        """Get immediate predecessors"""
        return self._predecessors[state].copy()
    
    def compute_chain(self, start: State, end: State) -> List[List[State]]:
        """
        Find all chains from start to end
        
        Returns
        -------
        chains : List[List[State]]
            List of all possible chains from start to end
        """
        if not partial_order(start, end):
            return []
        
        return self._chains.get((start, end), [])
    
    def get_shortest_chain(self, start: State, end: State) -> List[State]:
        """
        Get shortest chain from start to end
        
        Returns
        -------
        chain : List[State]
            Shortest chain, or empty list if no chain exists
        """
        chains = self.compute_chain(start, end)
        if not chains:
            return []
        
        return min(chains, key=len)
    
    def get_longest_chain(self, start: State, end: State) -> List[State]:
        """
        Get longest chain from start to end
        
        Returns
        -------
        chain : List[State]
            Longest chain, or empty list if no chain exists
        """
        chains = self.compute_chain(start, end)
        if not chains:
            return []
        
        return max(chains, key=len)
    
    def get_atoms(self) -> List[State]:
        """Get all atoms (immediate successors of bottom)"""
        return [state for state in self.elements if is_atom(state)]
    
    def get_coatoms(self) -> List[State]:
        """Get all coatoms (immediate predecessors of top)"""
        return [state for state in self.elements if is_coatom(state)]
    
    def get_height(self, state: State) -> int:
        """
        Get height of state in lattice (distance from bottom)
        
        Examples
        --------
        >>> lattice.get_height(EMPTY)  # 0
        >>> lattice.get_height(DELTA)  # 1
        >>> lattice.get_height(PSI)   # 2
        """
        return lattice_distance(state, self.bottom)
    
    def get_depth(self, state: State) -> int:
        """
        Get depth of state in lattice (distance from top)
        
        Examples
        --------
        >>> lattice.get_depth(PSI)    # 0
        >>> lattice.get_depth(DELTA)  # 1
        >>> lattice.get_depth(EMPTY)  # 2
        """
        return lattice_distance(state, self.top)
    
    def get_level(self, state: State) -> int:
        """
        Get level of state (same as height for this lattice)
        
        Level 0: EMPTY
        Level 1: DELTA, PHI
        Level 2: PSI
        """
        return self.get_height(state)
    
    def get_states_at_level(self, level: int) -> List[State]:
        """Get all states at a specific level"""
        return [state for state in self.elements if self.get_level(state) == level]
    
    def is_chain(self, states: List[State]) -> bool:
        """
        Check if a list of states forms a chain
        
        A chain is a totally ordered subset of the lattice
        """
        if len(states) < 2:
            return True
        
        for i in range(len(states) - 1):
            if not partial_order(states[i], states[i + 1]):
                return False
        
        return True
    
    def get_antichain(self, states: List[State]) -> List[State]:
        """
        Get maximal antichain from a set of states
        
        An antichain is a set of pairwise incomparable elements
        """
        antichain = []
        for state in states:
            is_incomparable = True
            for other in antichain:
                if are_comparable(state, other):
                    is_incomparable = False
                    break
            
            if is_incomparable:
                antichain.append(state)
        
        return antichain
    
    def get_interval(self, start: State, end: State) -> List[State]:
        """
        Get interval [start, end] in the lattice
        
        Returns all states s such that start ≤ s ≤ end
        """
        if not partial_order(start, end):
            return []
        
        interval = []
        for state in self.elements:
            if partial_order(start, state) and partial_order(state, end):
                interval.append(state)
        
        return interval
    
    def get_cover_relations(self) -> List[Tuple[State, State]]:
        """
        Get all cover relations in the lattice
        
        A cover relation is a direct predecessor-successor pair
        """
        covers = []
        for state in self.elements:
            for successor in self.get_successors(state):
                covers.append((state, successor))
        return covers
    
    def is_distributive(self) -> bool:
        """
        Check if the lattice is distributive
        
        A lattice is distributive if:
        x ∧ (y ∨ z) = (x ∧ y) ∨ (x ∧ z)
        """
        # The channel lattice is distributive (it's a Boolean lattice)
        return True
    
    def is_complemented(self) -> bool:
        """
        Check if the lattice is complemented
        
        A lattice is complemented if every element has a complement
        """
        # The channel lattice is complemented
        return True
    
    def get_complement(self, state: State) -> State:
        """Get the complement of a state"""
        return complement(state)
    
    def get_dual(self) -> 'ChannelLattice':
        """
        Get the dual lattice (order reversed)
        
        Returns a new lattice with reversed partial order
        """
        dual = ChannelLattice()
        # The dual is the same lattice with reversed order
        return dual
    
    def visualize(self) -> str:
        """
        Create a text visualization of the lattice
        
        Returns
        -------
        viz : str
            ASCII art representation of the lattice
        """
        lines = []
        lines.append("Channel Lattice Structure:")
        lines.append("")
        lines.append("    ψ (PSI)")
        lines.append("   / \\")
        lines.append("  δ   φ")
        lines.append("   \\ /")
        lines.append("    ∅ (EMPTY)")
        lines.append("")
        lines.append("Cover relations:")
        for start, end in self.get_cover_relations():
            lines.append(f"  {start} → {end}")
        
        return "\n".join(lines)
    
    def get_lattice_properties(self) -> dict:
        """
        Get comprehensive lattice properties
        
        Returns
        -------
        properties : dict
            Dictionary of lattice properties and metrics
        """
        return {
            'size': len(self.elements),
            'height': self.get_height(self.top),
            'width': len(self.get_antichain(self.elements)),
            'is_distributive': self.is_distributive(),
            'is_complemented': self.is_complemented(),
            'is_modular': True,  # All distributive lattices are modular
            'is_boolean': True,  # This is a Boolean lattice
            'atoms': len(self.get_atoms()),
            'coatoms': len(self.get_coatoms()),
            'cover_relations': len(self.get_cover_relations())
        }


# Global lattice instance
_lattice = ChannelLattice()


def get_lattice() -> ChannelLattice:
    """Get the global channel lattice instance"""
    return _lattice


def lattice_operations(states: List[State]) -> dict:
    """
    Perform common lattice operations on a list of states
    
    Parameters
    ----------
    states : List[State]
        List of states to operate on
        
    Returns
    -------
    operations : dict
        Dictionary of lattice operations results
    """
    if not states:
        return {}
    
    # Meet (greatest lower bound)
    meet_result = states[0]
    for state in states[1:]:
        meet_result = meet(meet_result, state)
    
    # Join (least upper bound)
    join_result = states[0]
    for state in states[1:]:
        join_result = join(join_result, state)
    
    # Antichain
    antichain = _lattice.get_antichain(states)
    
    # Chain check
    is_chain = _lattice.is_chain(states)
    
    return {
        'meet': meet_result,
        'join': join_result,
        'antichain': antichain,
        'is_chain': is_chain,
        'size': len(states),
        'atoms': [s for s in states if is_atom(s)],
        'coatoms': [s for s in states if is_coatom(s)]
    }







