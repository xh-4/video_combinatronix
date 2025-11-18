"""
Tests for pipeline.interpreters module
"""
import pytest
import numpy as np
from channelpy.core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from channelpy.core.nested import NestedState
from channelpy.core.parallel import ParallelChannels
from channelpy.pipeline.interpreters import (
    RuleBasedInterpreter, LookupTableInterpreter, FSMInterpreter,
    PatternMatcher, NestedStateInterpreter, ParallelChannelInterpreter,
    ThresholdBasedInterpreter, quick_interpret
)


def test_rule_based_interpreter():
    """Test RuleBasedInterpreter"""
    interpreter = RuleBasedInterpreter()
    
    # Add rules
    interpreter.add_rule(PSI, "APPROVE", priority=1)
    interpreter.add_rule(EMPTY, "REJECT", priority=2)
    interpreter.add_rule(lambda s: s.i == 1, "REVIEW", priority=0)
    interpreter.set_default("UNKNOWN")
    
    # Test interpretations
    assert interpreter.interpret(PSI) == "APPROVE"
    assert interpreter.interpret(EMPTY) == "REJECT"
    assert interpreter.interpret(DELTA) == "REVIEW"  # i=1
    assert interpreter.interpret(PHI) == "UNKNOWN"   # No match
    
    # Test batch interpretation
    states = StateArray.from_bits(i=[1, 0, 1], q=[1, 0, 0])
    decisions = interpreter.interpret_batch(states)
    assert decisions == ["APPROVE", "UNKNOWN", "REVIEW"]


def test_lookup_table_interpreter():
    """Test LookupTableInterpreter"""
    interpreter = LookupTableInterpreter()
    
    # Build lookup table
    table = {
        PSI: {'action': 'BUY', 'confidence': 0.9},
        DELTA: {'action': 'HOLD', 'confidence': 0.5},
        PHI: {'action': 'RESEARCH', 'confidence': 0.3},
        EMPTY: {'action': 'SELL', 'confidence': 0.8}
    }
    interpreter.build_table(table)
    
    # Test lookups
    assert interpreter.interpret(PSI) == {'action': 'BUY', 'confidence': 0.9}
    assert interpreter.interpret(DELTA) == {'action': 'HOLD', 'confidence': 0.5}
    
    # Test unknown state
    unknown_state = State(1, 1)  # Same as PSI but different object
    result = interpreter.interpret(unknown_state)
    assert result['action'] == 'UNKNOWN'
    assert result['confidence'] == 0.0


def test_fsm_interpreter():
    """Test FSMInterpreter"""
    fsm = FSMInterpreter(initial_mode='WAITING')
    
    # Add transitions
    fsm.add_transition('WAITING', PSI, 'ACTIVE', action='START')
    fsm.add_transition('ACTIVE', EMPTY, 'WAITING', action='STOP')
    fsm.add_transition('ACTIVE', PSI, 'ACTIVE', action='CONTINUE')
    
    # Test state machine
    assert fsm.get_mode() == 'WAITING'
    
    action = fsm.process(PSI)  # Should transition to ACTIVE
    assert action == 'START'
    assert fsm.get_mode() == 'ACTIVE'
    
    action = fsm.process(PSI)  # Should stay in ACTIVE
    assert action == 'CONTINUE'
    assert fsm.get_mode() == 'ACTIVE'
    
    action = fsm.process(EMPTY)  # Should transition back to WAITING
    assert action == 'STOP'
    assert fsm.get_mode() == 'WAITING'
    
    # Test reset
    fsm.reset()
    assert fsm.get_mode() == 'WAITING'


def test_pattern_matcher():
    """Test PatternMatcher"""
    matcher = PatternMatcher(max_pattern_length=5)
    
    # Add patterns
    matcher.add_pattern([PSI, PSI, PSI], "STRONG_TREND")
    matcher.add_pattern([PSI, DELTA, EMPTY], "DEGRADATION")
    matcher.add_pattern([EMPTY, PHI, PSI], "RECOVERY")
    
    # Test pattern matching
    assert matcher.interpret_sequence([PSI, PSI, PSI]) == "STRONG_TREND"
    assert matcher.interpret_sequence([PSI, DELTA, EMPTY]) == "DEGRADATION"
    assert matcher.interpret_sequence([EMPTY, PHI, PSI]) == "RECOVERY"
    assert matcher.interpret_sequence([PSI, PSI]) == "NO_PATTERN_MATCH"
    
    # Test streaming updates
    matcher.reset_history()
    assert matcher.update_and_interpret(PSI) == "NO_PATTERN_MATCH"
    assert matcher.update_and_interpret(PSI) == "NO_PATTERN_MATCH"
    assert matcher.update_and_interpret(PSI) == "STRONG_TREND"


def test_nested_state_interpreter():
    """Test NestedStateInterpreter"""
    interpreter = NestedStateInterpreter()
    
    # Add level rules
    interpreter.add_level_rule(0, EMPTY, {'action': 'STOP', 'reason': 'Level 0 failure'})
    interpreter.add_level_rule(1, PSI, {'action': 'GOOD', 'reason': 'Level 1 success'})
    
    # Add cascade rule
    interpreter.add_cascade_rule("ψ.δ.*", {'action': 'PARTIAL', 'reason': 'Mixed signals'})
    
    # Test nested state interpretation
    nested_state = NestedState(level0=EMPTY, level1=PSI)
    result = interpreter.interpret(nested_state)
    assert result['action'] == 'STOP'
    assert 'Level 0 failure' in result['reason']
    
    # Test cascade rule
    nested_state = NestedState(level0=PSI, level1=DELTA, level2=PHI)
    result = interpreter.interpret(nested_state)
    assert result['action'] == 'PARTIAL'
    assert 'Mixed signals' in result['reason']
    
    # Test default interpretation
    nested_state = NestedState(level0=PSI, level1=PSI, level2=PSI)
    result = interpreter.interpret(nested_state)
    assert result['action'] == 'PERFECT'
    assert result['confidence'] == 1.0


def test_parallel_channel_interpreter():
    """Test ParallelChannelInterpreter"""
    interpreter = ParallelChannelInterpreter(strategy='conservative')
    
    # Set channel weights
    interpreter.set_channel_weight('technical', 0.4)
    interpreter.set_channel_weight('business', 0.3)
    interpreter.set_channel_weight('team', 0.3)
    
    # Test parallel channels
    channels = ParallelChannels(
        technical=PSI,
        business=DELTA,
        team=PSI
    )
    
    result = interpreter.interpret(channels)
    assert 'action' in result
    assert 'overall_score' in result
    assert 'channel_details' in result
    assert 'technical' in result['channel_details']
    assert 'business' in result['channel_details']
    assert 'team' in result['channel_details']
    
    # Test different strategies
    interpreter_opt = ParallelChannelInterpreter(strategy='optimistic')
    result_opt = interpreter_opt.interpret(channels)
    assert result_opt['action'] in ['APPROVE', 'REJECT']


def test_threshold_based_interpreter():
    """Test ThresholdBasedInterpreter"""
    interpreter = ThresholdBasedInterpreter(low_threshold=0.3, high_threshold=0.7)
    
    # Test state to score conversion
    assert interpreter.state_to_score(EMPTY) == 0.0
    assert interpreter.state_to_score(PHI) == 0.33
    assert interpreter.state_to_score(DELTA) == 0.67
    assert interpreter.state_to_score(PSI) == 1.0
    
    # Test interpretations
    result = interpreter.interpret(PSI)
    assert result['action'] == 'APPROVE'
    assert result['confidence'] == 'HIGH'
    assert result['score'] == 1.0
    
    result = interpreter.interpret(EMPTY)
    assert result['action'] == 'REJECT'
    assert result['confidence'] == 'HIGH'
    assert result['score'] == 0.0
    
    result = interpreter.interpret(DELTA)
    assert result['action'] == 'REVIEW'
    assert result['confidence'] == 'MEDIUM'
    assert result['score'] == 0.67


def test_quick_interpret():
    """Test quick_interpret convenience function"""
    # Test generic domain
    assert quick_interpret(PSI, 'generic') == 'PROCEED'
    assert quick_interpret(DELTA, 'generic') == 'INVESTIGATE'
    assert quick_interpret(PHI, 'generic') == 'WAIT'
    assert quick_interpret(EMPTY, 'generic') == 'STOP'
    
    # Test trading domain
    assert quick_interpret(PSI, 'trading') == 'BUY'
    assert quick_interpret(DELTA, 'trading') == 'HOLD'
    assert quick_interpret(PHI, 'trading') == 'RESEARCH'
    assert quick_interpret(EMPTY, 'trading') == 'SELL'
    
    # Test medical domain
    assert quick_interpret(PSI, 'medical') == 'TREAT'
    assert quick_interpret(DELTA, 'medical') == 'TEST'
    assert quick_interpret(PHI, 'medical') == 'MONITOR'
    assert quick_interpret(EMPTY, 'medical') == 'HEALTHY'
    
    # Test unknown domain
    assert quick_interpret(PSI, 'unknown') == 'YES'
    assert quick_interpret(EMPTY, 'unknown') == 'NO'


def test_interpreter_callable():
    """Test that interpreters are callable"""
    interpreter = RuleBasedInterpreter()
    interpreter.add_rule(PSI, "GOOD")
    
    # Should work with __call__
    result = interpreter(PSI)
    assert result == "GOOD"


def test_interpreter_fit():
    """Test interpreter fit method"""
    interpreter = RuleBasedInterpreter()
    
    # Should be able to fit (even if no-op)
    result = interpreter.fit([PSI, DELTA], ['GOOD', 'BAD'])
    assert result is interpreter
    assert interpreter.is_fitted







