"""
Interpreters Example using ChannelPy

This example demonstrates the various interpreters available in ChannelPy
for converting channel states into actionable decisions.
"""

import numpy as np
from channelpy import (
    State, StateArray, EMPTY, DELTA, PHI, PSI,
    RuleBasedInterpreter, LookupTableInterpreter, FSMInterpreter,
    PatternMatcher, ThresholdBasedInterpreter, quick_interpret
)
from channelpy.core.nested import NestedState
from channelpy.core.parallel import ParallelChannels
from channelpy.pipeline.interpreters import (
    NestedStateInterpreter, ParallelChannelInterpreter
)


def demonstrate_rule_based_interpreter():
    """Demonstrate rule-based interpretation"""
    print("=== Rule-Based Interpreter ===")
    
    interpreter = RuleBasedInterpreter()
    
    # Add business rules
    interpreter.add_rule(PSI, "APPROVE", priority=1)
    interpreter.add_rule(EMPTY, "REJECT", priority=2)
    interpreter.add_rule(lambda s: s.i == 1 and s.q == 0, "REVIEW", priority=0)  # DELTA
    interpreter.add_rule(lambda s: s.i == 0 and s.q == 1, "WAIT", priority=0)    # PHI
    interpreter.set_default("UNKNOWN")
    
    # Test different states
    test_states = [PSI, DELTA, PHI, EMPTY]
    for state in test_states:
        decision = interpreter.interpret(state)
        print(f"  {state} → {decision}")
    print()


def demonstrate_lookup_table_interpreter():
    """Demonstrate lookup table interpretation"""
    print("=== Lookup Table Interpreter ===")
    
    interpreter = LookupTableInterpreter()
    
    # Build trading decision table
    trading_table = {
        PSI: {'action': 'BUY', 'confidence': 0.9, 'reason': 'Strong signal'},
        DELTA: {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Mixed signal'},
        PHI: {'action': 'RESEARCH', 'confidence': 0.3, 'reason': 'Incomplete data'},
        EMPTY: {'action': 'SELL', 'confidence': 0.8, 'reason': 'No signal'}
    }
    interpreter.build_table(trading_table)
    
    # Test trading decisions
    test_states = [PSI, DELTA, PHI, EMPTY]
    for state in test_states:
        decision = interpreter.interpret(state)
        print(f"  {state} → {decision['action']} (confidence: {decision['confidence']})")
    print()


def demonstrate_fsm_interpreter():
    """Demonstrate finite state machine interpretation"""
    print("=== FSM Interpreter ===")
    
    fsm = FSMInterpreter(initial_mode='IDLE')
    
    # Define system states and transitions
    fsm.add_transition('IDLE', PSI, 'ACTIVE', action='START_PROCESSING')
    fsm.add_transition('ACTIVE', PSI, 'ACTIVE', action='CONTINUE_PROCESSING')
    fsm.add_transition('ACTIVE', DELTA, 'WARNING', action='SLOW_DOWN')
    fsm.add_transition('WARNING', PSI, 'ACTIVE', action='RESUME_NORMAL')
    fsm.add_transition('WARNING', EMPTY, 'IDLE', action='STOP_PROCESSING')
    fsm.add_transition('ACTIVE', EMPTY, 'IDLE', action='STOP_PROCESSING')
    
    # Simulate a processing sequence
    sequence = [PSI, PSI, DELTA, PSI, EMPTY]
    print("  Processing sequence:")
    for i, state in enumerate(sequence):
        action = fsm.process(state)
        mode = fsm.get_mode()
        print(f"    Step {i+1}: {state} → {action} (mode: {mode})")
    
    print(f"  Final mode: {fsm.get_mode()}")
    print()


def demonstrate_pattern_matcher():
    """Demonstrate pattern matching interpretation"""
    print("=== Pattern Matcher ===")
    
    matcher = PatternMatcher(max_pattern_length=5)
    
    # Add trend patterns
    matcher.add_pattern([PSI, PSI, PSI], "STRONG_UPTREND")
    matcher.add_pattern([EMPTY, EMPTY, EMPTY], "STRONG_DOWNTREND")
    matcher.add_pattern([PSI, DELTA, EMPTY], "DEGRADING_TREND")
    matcher.add_pattern([EMPTY, PHI, PSI], "RECOVERY_TREND")
    matcher.add_pattern([PSI, PSI], "WEAK_UPTREND")
    
    # Test pattern recognition
    test_sequences = [
        [PSI, PSI, PSI],
        [EMPTY, EMPTY, EMPTY],
        [PSI, DELTA, EMPTY],
        [EMPTY, PHI, PSI],
        [PSI, PSI],
        [PSI, PHI, DELTA]  # No pattern
    ]
    
    print("  Pattern recognition:")
    for sequence in test_sequences:
        pattern = matcher.interpret_sequence(sequence)
        print(f"    {[str(s) for s in sequence]} → {pattern}")
    
    # Test streaming pattern detection
    print("  Streaming pattern detection:")
    matcher.reset_history()
    stream = [PSI, PSI, PSI, DELTA, EMPTY]
    for i, state in enumerate(stream):
        pattern = matcher.update_and_interpret(state)
        print(f"    Step {i+1}: {state} → {pattern}")
    print()


def demonstrate_nested_state_interpreter():
    """Demonstrate nested state interpretation"""
    print("=== Nested State Interpreter ===")
    
    interpreter = NestedStateInterpreter()
    
    # Add level-specific rules
    interpreter.add_level_rule(0, EMPTY, {'action': 'CRITICAL_FAILURE', 'level': 0})
    interpreter.add_level_rule(1, EMPTY, {'action': 'MODERATE_FAILURE', 'level': 1})
    interpreter.add_level_rule(2, EMPTY, {'action': 'MINOR_FAILURE', 'level': 2})
    
    # Add cascade rules for specific patterns
    interpreter.add_cascade_rule("ψ.ψ.ψ", {'action': 'PERFECT_SYSTEM', 'confidence': 1.0})
    interpreter.add_cascade_rule("ψ.δ.*", {'action': 'PARTIAL_SUCCESS', 'confidence': 0.7})
    interpreter.add_cascade_rule("*.φ.*", {'action': 'INCOMPLETE_DATA', 'confidence': 0.3})
    
    # Test nested states
    test_cases = [
        NestedState(level0=PSI, level1=PSI, level2=PSI),
        NestedState(level0=PSI, level1=DELTA, level2=PHI),
        NestedState(level0=EMPTY, level1=PSI, level2=PSI),
        NestedState(level0=PSI, level1=PHI, level2=PSI),
        NestedState(level0=PSI, level1=PSI, level2=EMPTY),
    ]
    
    print("  Nested state interpretation:")
    for i, nested_state in enumerate(test_cases):
        result = interpreter.interpret(nested_state)
        print(f"    Case {i+1}: {nested_state} → {result['action']} (conf: {result.get('confidence', 'N/A')})")
    print()


def demonstrate_parallel_channel_interpreter():
    """Demonstrate parallel channel interpretation"""
    print("=== Parallel Channel Interpreter ===")
    
    # Test different strategies
    strategies = ['conservative', 'optimistic', 'majority', 'weighted']
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        interpreter = ParallelChannelInterpreter(strategy=strategy)
        
        # Set channel weights for weighted strategy
        if strategy == 'weighted':
            interpreter.set_channel_weight('technical', 0.5)
            interpreter.set_channel_weight('business', 0.3)
            interpreter.set_channel_weight('team', 0.2)
        
        # Test different channel combinations
        test_cases = [
            ParallelChannels(technical=PSI, business=PSI, team=PSI),
            ParallelChannels(technical=PSI, business=DELTA, team=PSI),
            ParallelChannels(technical=EMPTY, business=PSI, team=PSI),
            ParallelChannels(technical=DELTA, business=DELTA, team=DELTA),
        ]
        
        for i, channels in enumerate(test_cases):
            result = interpreter.interpret(channels)
            print(f"    Case {i+1}: {channels} → {result['action']} (score: {result['overall_score']:.2f})")
        print()


def demonstrate_threshold_based_interpreter():
    """Demonstrate threshold-based interpretation"""
    print("=== Threshold-Based Interpreter ===")
    
    interpreter = ThresholdBasedInterpreter(low_threshold=0.3, high_threshold=0.7)
    
    # Test all states
    test_states = [EMPTY, PHI, DELTA, PSI]
    print("  Threshold-based decisions:")
    for state in test_states:
        result = interpreter.interpret(state)
        print(f"    {state} → {result['action']} (score: {result['score']:.2f}, conf: {result['confidence']})")
    print()


def demonstrate_quick_interpret():
    """Demonstrate quick interpretation for different domains"""
    print("=== Quick Interpret (Domain-Specific) ===")
    
    domains = ['generic', 'trading', 'medical', 'quality_control']
    test_states = [PSI, DELTA, PHI, EMPTY]
    
    for domain in domains:
        print(f"  Domain: {domain}")
        for state in test_states:
            action = quick_interpret(state, domain)
            print(f"    {state} → {action}")
        print()


def demonstrate_complete_decision_system():
    """Demonstrate a complete decision system using multiple interpreters"""
    print("=== Complete Decision System ===")
    
    # Create a multi-level decision system
    print("  Multi-level decision system:")
    
    # Level 1: Quick screening
    quick_screener = lambda state: quick_interpret(state, 'trading')
    
    # Level 2: Detailed analysis
    detailed_analyzer = LookupTableInterpreter()
    detailed_analyzer.build_table({
        PSI: {'action': 'STRONG_BUY', 'confidence': 0.9, 'risk': 'LOW'},
        DELTA: {'action': 'HOLD', 'confidence': 0.6, 'risk': 'MEDIUM'},
        PHI: {'action': 'RESEARCH', 'confidence': 0.3, 'risk': 'HIGH'},
        EMPTY: {'action': 'STRONG_SELL', 'confidence': 0.8, 'risk': 'LOW'}
    })
    
    # Level 3: Pattern recognition
    pattern_analyzer = PatternMatcher()
    pattern_analyzer.add_pattern([PSI, PSI], "MOMENTUM_BUILDING")
    pattern_analyzer.add_pattern([EMPTY, EMPTY], "MOMENTUM_BREAKING")
    
    # Test the complete system
    test_sequence = [PSI, PSI, DELTA, PHI, EMPTY, EMPTY]
    print("    Decision sequence:")
    
    for i, state in enumerate(test_sequence):
        # Level 1: Quick screening
        quick_decision = quick_screener(state)
        
        # Level 2: Detailed analysis
        detailed_decision = detailed_analyzer.interpret(state)
        
        # Level 3: Pattern analysis
        pattern_decision = pattern_analyzer.update_and_interpret(state)
        
        print(f"      Step {i+1}: {state}")
        print(f"        Quick: {quick_decision}")
        print(f"        Detailed: {detailed_decision['action']} (conf: {detailed_decision['confidence']})")
        print(f"        Pattern: {pattern_decision}")
        print()


def main():
    """Main demonstration function"""
    print("ChannelPy Interpreters Example")
    print("=" * 40)
    
    demonstrate_rule_based_interpreter()
    demonstrate_lookup_table_interpreter()
    demonstrate_fsm_interpreter()
    demonstrate_pattern_matcher()
    demonstrate_nested_state_interpreter()
    demonstrate_parallel_channel_interpreter()
    demonstrate_threshold_based_interpreter()
    demonstrate_quick_interpret()
    demonstrate_complete_decision_system()
    
    print("Interpreters example completed successfully!")


if __name__ == "__main__":
    main()







