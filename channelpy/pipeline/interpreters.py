"""
State → Decision interpreters for channel pipelines

This module provides interpreters for Stage 3 of the pipeline (States → Decisions)
"""
from typing import Union, Dict, List, Tuple, Callable, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from ..core.nested import NestedState
from ..core.parallel import ParallelChannels


class BaseInterpreter(ABC):
    """
    Abstract base class for all interpreters
    
    Interpreters convert channel states to actionable decisions
    """
    
    def __init__(self):
        self.is_fitted = False
    
    @abstractmethod
    def interpret(self, states):
        """
        Interpret states to decisions
        
        Parameters
        ----------
        states : State, StateArray, NestedState, or ParallelChannels
            Channel states to interpret
            
        Returns
        -------
        decision : any
            Interpretation result (action, label, score, etc.)
        """
        pass
    
    def fit(self, states, y=None):
        """
        Optionally learn interpretation rules from data
        
        Parameters
        ----------
        states : array of states
            Training states
        y : array, optional
            Training labels
            
        Returns
        -------
        self : BaseInterpreter
        """
        self.is_fitted = True
        return self
    
    def __call__(self, states):
        """Make interpreter callable"""
        return self.interpret(states)


class RuleBasedInterpreter(BaseInterpreter):
    """
    Rule-based interpreter using explicit if-then rules
    
    Rules are Python functions or lambdas that map states to decisions
    
    Examples
    --------
    >>> interpreter = RuleBasedInterpreter()
    >>> interpreter.add_rule(PSI, "APPROVE")
    >>> interpreter.add_rule(EMPTY, "REJECT")
    >>> interpreter.add_rule(lambda s: s.i == 1, "REVIEW")
    >>> decision = interpreter.interpret(State(1, 1))
    'APPROVE'
    """
    
    def __init__(self):
        super().__init__()
        self.rules = []  # List of (condition, action) tuples
        self.default_action = "NO_ACTION"
    
    def add_rule(self, condition: Union[State, Callable], action: Any, 
                 priority: int = 0):
        """
        Add interpretation rule
        
        Parameters
        ----------
        condition : State or callable
            If State: exact match required
            If callable: function(state) -> bool
        action : any
            Action to return when condition matches
        priority : int
            Higher priority rules checked first
        """
        self.rules.append((condition, action, priority))
        # Sort by priority (descending)
        self.rules.sort(key=lambda x: x[2], reverse=True)
    
    def set_default(self, action: Any):
        """Set default action when no rules match"""
        self.default_action = action
    
    def interpret(self, state: State) -> Any:
        """Interpret using rules"""
        for condition, action, priority in self.rules:
            if self._check_condition(state, condition):
                return action
        
        return self.default_action
    
    def _check_condition(self, state: State, condition) -> bool:
        """Check if condition matches state"""
        if isinstance(condition, State):
            # Exact match
            return state == condition
        elif callable(condition):
            # Function predicate
            return condition(state)
        else:
            return False
    
    def interpret_batch(self, states: StateArray) -> List[Any]:
        """Interpret multiple states"""
        return [self.interpret(state) for state in states]


class LookupTableInterpreter(BaseInterpreter):
    """
    Fast lookup table for state → decision mapping
    
    Pre-computed decision for each possible state
    
    Examples
    --------
    >>> interpreter = LookupTableInterpreter()
    >>> interpreter.build_table({
    ...     PSI: {'action': 'BUY', 'confidence': 0.9},
    ...     DELTA: {'action': 'HOLD', 'confidence': 0.5},
    ...     PHI: {'action': 'RESEARCH', 'confidence': 0.3},
    ...     EMPTY: {'action': 'SELL', 'confidence': 0.8}
    ... })
    >>> interpreter.interpret(PSI)
    {'action': 'BUY', 'confidence': 0.9}
    """
    
    def __init__(self):
        super().__init__()
        self.table = {}
    
    def build_table(self, mapping: Dict[State, Any]):
        """
        Build lookup table
        
        Parameters
        ----------
        mapping : dict
            State → Decision mapping
        """
        self.table = mapping.copy()
        self.is_fitted = True
    
    def interpret(self, state: State) -> Any:
        """Look up decision"""
        if state not in self.table:
            return {
                'action': 'UNKNOWN',
                'confidence': 0.0,
                'reason': 'State not in lookup table'
            }
        
        return self.table[state]
    
    def interpret_batch(self, states: StateArray) -> List[Any]:
        """Batch lookup"""
        return [self.interpret(state) for state in states]


class FSMInterpreter(BaseInterpreter):
    """
    Finite State Machine interpreter for sequential states
    
    Maintains internal mode and transitions between modes based on state inputs
    
    Examples
    --------
    >>> fsm = FSMInterpreter(initial_mode='WAITING')
    >>> fsm.add_transition('WAITING', PSI, 'ACTIVE', action='START')
    >>> fsm.add_transition('ACTIVE', EMPTY, 'WAITING', action='STOP')
    >>> 
    >>> action = fsm.process(PSI)  # 'START', mode becomes 'ACTIVE'
    >>> action = fsm.process(PSI)  # 'CONTINUE', stays in 'ACTIVE'
    >>> action = fsm.process(EMPTY)  # 'STOP', mode becomes 'WAITING'
    """
    
    def __init__(self, initial_mode: str = 'INIT'):
        """
        Parameters
        ----------
        initial_mode : str
            Starting mode
        """
        super().__init__()
        self.initial_mode = initial_mode
        self.current_mode = initial_mode
        self.transitions = {}  # (mode, state) -> (next_mode, action)
        self.mode_history = []
    
    def add_transition(self, from_mode: str, state_input: State, 
                       to_mode: str, action: Any):
        """
        Add state machine transition
        
        Parameters
        ----------
        from_mode : str
            Current mode
        state_input : State
            State that triggers transition
        to_mode : str
            Next mode
        action : any
            Action to take during transition
        """
        key = (from_mode, state_input)
        self.transitions[key] = (to_mode, action)
    
    def process(self, state: State) -> Any:
        """
        Process state through FSM
        
        Parameters
        ----------
        state : State
            Input state
            
        Returns
        -------
        action : any
            Action for this transition
        """
        key = (self.current_mode, state)
        
        if key in self.transitions:
            next_mode, action = self.transitions[key]
            self.mode_history.append((self.current_mode, state, action))
            self.current_mode = next_mode
            return action
        else:
            # No explicit transition, stay in current mode
            return f"CONTINUE_{self.current_mode}"
    
    def reset(self):
        """Reset FSM to initial mode"""
        self.current_mode = self.initial_mode
        self.mode_history = []
    
    def get_mode(self) -> str:
        """Get current mode"""
        return self.current_mode
    
    def interpret(self, state: State) -> Any:
        """Alias for process()"""
        return self.process(state)


class PatternMatcher(BaseInterpreter):
    """
    Pattern-based interpreter for state sequences
    
    Matches patterns in sequences of states
    
    Examples
    --------
    >>> matcher = PatternMatcher()
    >>> matcher.add_pattern([PSI, PSI, PSI], "STRONG_TREND")
    >>> matcher.add_pattern([PSI, DELTA, EMPTY], "DEGRADATION")
    >>> matcher.add_pattern([EMPTY, PHI, PSI], "RECOVERY")
    >>> 
    >>> decision = matcher.interpret_sequence([PSI, PSI, PSI])
    'STRONG_TREND'
    """
    
    def __init__(self, max_pattern_length: int = 10):
        """
        Parameters
        ----------
        max_pattern_length : int
            Maximum length of patterns to match
        """
        super().__init__()
        self.patterns = {}  # tuple of states -> action
        self.max_pattern_length = max_pattern_length
        self.state_history = []
    
    def add_pattern(self, pattern: List[State], action: Any):
        """
        Add pattern to match
        
        Parameters
        ----------
        pattern : list of State
            Sequence of states
        action : any
            Action when pattern matches
        """
        pattern_tuple = tuple(pattern)
        self.patterns[pattern_tuple] = action
    
    def interpret_sequence(self, states: List[State]) -> Any:
        """
        Match pattern in state sequence
        
        Checks for longest matching pattern
        """
        states_tuple = tuple(states)
        
        # Check from longest to shortest patterns
        for length in range(min(len(states), self.max_pattern_length), 0, -1):
            suffix = states_tuple[-length:]
            if suffix in self.patterns:
                return self.patterns[suffix]
        
        return "NO_PATTERN_MATCH"
    
    def update_and_interpret(self, new_state: State) -> Any:
        """
        Add new state to history and check for patterns
        
        Useful for streaming data
        """
        self.state_history.append(new_state)
        
        # Keep only recent history
        if len(self.state_history) > self.max_pattern_length:
            self.state_history.pop(0)
        
        return self.interpret_sequence(self.state_history)
    
    def interpret(self, state: State) -> Any:
        """Single state interpretation (uses history)"""
        return self.update_and_interpret(state)
    
    def reset_history(self):
        """Clear state history"""
        self.state_history = []


class NestedStateInterpreter(BaseInterpreter):
    """
    Interpret nested (hierarchical) states
    
    Examples
    --------
    >>> interpreter = NestedStateInterpreter()
    >>> 
    >>> # High-level failure overrides everything
    >>> nested_state = NestedState(level0=EMPTY, level1=PSI)
    >>> interpreter.interpret(nested_state)
    {'action': 'STOP', 'reason': 'Level 0 failure'}
    >>> 
    >>> # All good at all levels
    >>> nested_state = NestedState(level0=PSI, level1=PSI, level2=PSI)
    >>> interpreter.interpret(nested_state)
    {'action': 'PERFECT', 'confidence': 1.0}
    """
    
    def __init__(self):
        super().__init__()
        self.level_rules = {}  # level -> {state -> action}
        self.cascade_rules = []  # List of cascading interpretation rules
    
    def add_level_rule(self, level: int, state: State, action: Any):
        """Add rule for specific level"""
        if level not in self.level_rules:
            self.level_rules[level] = {}
        self.level_rules[level][state] = action
    
    def add_cascade_rule(self, pattern: str, action: Any):
        """
        Add cascading rule
        
        Pattern is path string like "ψ.δ.*"
        """
        self.cascade_rules.append((pattern, action))
    
    def interpret(self, nested_state: NestedState) -> Any:
        """Interpret nested state"""
        
        # Check cascade rules first
        for pattern, action in self.cascade_rules:
            if nested_state.path_matches(pattern):
                return action
        
        # Check level-by-level (higher levels first)
        for level in range(nested_state.depth, -1, -1):
            level_state = nested_state.get_level(level)
            
            if level in self.level_rules:
                if level_state in self.level_rules[level]:
                    return self.level_rules[level][level_state]
        
        # Default interpretation
        if nested_state.all_psi():
            return {'action': 'PERFECT', 'confidence': 1.0}
        elif nested_state.any_empty():
            return {'action': 'FAILURE', 'confidence': 0.0}
        else:
            psi_ratio = nested_state.count_psi() / nested_state.num_levels
            return {'action': 'PARTIAL', 'confidence': psi_ratio}


class ParallelChannelInterpreter(BaseInterpreter):
    """
    Interpret multiple parallel channels
    
    Aggregates decisions from independent channels
    
    Examples
    --------
    >>> interpreter = ParallelChannelInterpreter(strategy='conservative')
    >>> channels = ParallelChannels(
    ...     technical=PSI,
    ...     business=DELTA,
    ...     team=PSI
    ... )
    >>> interpreter.interpret(channels)
    {'action': 'PROCEED_WITH_CAUTION', 'weak_channel': 'business'}
    """
    
    def __init__(self, strategy: str = 'conservative'):
        """
        Parameters
        ----------
        strategy : str
            Aggregation strategy:
            - 'conservative': All must be good
            - 'optimistic': Any good is enough
            - 'majority': Majority vote
            - 'weighted': Weighted combination
        """
        super().__init__()
        self.strategy = strategy
        self.channel_weights = {}
        self.channel_interpreters = {}
    
    def set_channel_weight(self, channel_name: str, weight: float):
        """Set importance weight for channel"""
        self.channel_weights[channel_name] = weight
    
    def set_channel_interpreter(self, channel_name: str, 
                                interpreter: BaseInterpreter):
        """Set custom interpreter for specific channel"""
        self.channel_interpreters[channel_name] = interpreter
    
    def interpret(self, channels: ParallelChannels) -> Dict:
        """Interpret parallel channels"""
        
        # Interpret each channel
        channel_decisions = {}
        for name in channels.all_names():
            state = channels[name]
            
            # Use custom interpreter if available
            if name in self.channel_interpreters:
                decision = self.channel_interpreters[name].interpret(state)
            else:
                decision = self._default_interpret(state)
            
            channel_decisions[name] = decision
        
        # Aggregate decisions
        return self._aggregate_decisions(channel_decisions)
    
    def _default_interpret(self, state: State) -> Dict:
        """Default single-channel interpretation"""
        if state == PSI:
            return {'score': 1.0, 'status': 'GOOD'}
        elif state == DELTA:
            return {'score': 0.5, 'status': 'ISSUES'}
        elif state == PHI:
            return {'score': 0.3, 'status': 'INCOMPLETE'}
        else:  # EMPTY
            return {'score': 0.0, 'status': 'BAD'}
    
    def _aggregate_decisions(self, channel_decisions: Dict) -> Dict:
        """Aggregate decisions from all channels"""
        
        if self.strategy == 'conservative':
            # All channels must be good
            min_score = min(d['score'] for d in channel_decisions.values())
            worst_channel = min(
                channel_decisions.items(), 
                key=lambda x: x[1]['score']
            )
            
            return {
                'action': 'APPROVE' if min_score >= 0.7 else 'REJECT',
                'overall_score': min_score,
                'weakest_channel': worst_channel[0],
                'channel_details': channel_decisions
            }
        
        elif self.strategy == 'optimistic':
            # Any channel being good is sufficient
            max_score = max(d['score'] for d in channel_decisions.values())
            best_channel = max(
                channel_decisions.items(),
                key=lambda x: x[1]['score']
            )
            
            return {
                'action': 'APPROVE' if max_score >= 0.7 else 'REJECT',
                'overall_score': max_score,
                'strongest_channel': best_channel[0],
                'channel_details': channel_decisions
            }
        
        elif self.strategy == 'weighted':
            # Weighted average
            total_weight = sum(
                self.channel_weights.get(name, 1.0)
                for name in channel_decisions.keys()
            )
            
            weighted_score = sum(
                self.channel_weights.get(name, 1.0) * decision['score']
                for name, decision in channel_decisions.items()
            ) / total_weight
            
            return {
                'action': 'APPROVE' if weighted_score >= 0.7 else 'REJECT',
                'overall_score': weighted_score,
                'channel_details': channel_decisions
            }
        
        else:  # majority
            # Majority vote
            good_count = sum(
                1 for d in channel_decisions.values() 
                if d['score'] >= 0.7
            )
            total_count = len(channel_decisions)
            
            return {
                'action': 'APPROVE' if good_count > total_count / 2 else 'REJECT',
                'overall_score': good_count / total_count,
                'votes': f'{good_count}/{total_count}',
                'channel_details': channel_decisions
            }


class ThresholdBasedInterpreter(BaseInterpreter):
    """
    Interpret based on threshold crossings
    
    Useful when state confidence needs to be quantified
    
    Examples
    --------
    >>> interpreter = ThresholdBasedInterpreter(
    ...     low_threshold=0.3,
    ...     high_threshold=0.7
    ... )
    >>> # Converts states to scores and applies thresholds
    """
    
    def __init__(self, low_threshold: float = 0.3, 
                 high_threshold: float = 0.7):
        """
        Parameters
        ----------
        low_threshold : float
            Threshold for rejection
        high_threshold : float
            Threshold for approval
        """
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def state_to_score(self, state: State) -> float:
        """Convert state to numeric score"""
        score_map = {
            EMPTY: 0.0,
            PHI: 0.33,
            DELTA: 0.67,
            PSI: 1.0
        }
        return score_map.get(state, 0.0)
    
    def interpret(self, state: State) -> Dict:
        """Interpret based on thresholds"""
        score = self.state_to_score(state)
        
        if score >= self.high_threshold:
            action = 'APPROVE'
            confidence = 'HIGH'
        elif score <= self.low_threshold:
            action = 'REJECT'
            confidence = 'HIGH'
        else:
            action = 'REVIEW'
            confidence = 'MEDIUM'
        
        return {
            'action': action,
            'score': score,
            'confidence': confidence,
            'state': str(state)
        }


# Convenience function

def quick_interpret(state: State, domain: str = 'generic') -> str:
    """
    Quick interpretation for common domains
    
    Parameters
    ----------
    state : State
        State to interpret
    domain : str
        Domain context ('generic', 'trading', 'medical', etc.)
        
    Returns
    -------
    action : str
        Recommended action
    """
    if domain == 'generic':
        mapping = {
            PSI: 'PROCEED',
            DELTA: 'INVESTIGATE',
            PHI: 'WAIT',
            EMPTY: 'STOP'
        }
    elif domain == 'trading':
        mapping = {
            PSI: 'BUY',
            DELTA: 'HOLD',
            PHI: 'RESEARCH',
            EMPTY: 'SELL'
        }
    elif domain == 'medical':
        mapping = {
            PSI: 'TREAT',
            DELTA: 'TEST',
            PHI: 'MONITOR',
            EMPTY: 'HEALTHY'
        }
    else:
        mapping = {
            PSI: 'YES',
            DELTA: 'MAYBE',
            PHI: 'INCOMPLETE',
            EMPTY: 'NO'
        }
    
    return mapping.get(state, 'UNKNOWN')







