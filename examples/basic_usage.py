"""
ChannelPy Basic Usage Example

This example demonstrates the fundamental concepts of ChannelPy:
- State creation and operations
- State arrays
- Basic pipeline usage
- Simple interpretation
"""

import numpy as np
from channelpy import (
    State, StateArray, PSI, DELTA, PHI, EMPTY,
    gate, admit, overlay, weave, comp, neg_i, neg_q,
    ChannelPipeline, StandardScaler, ThresholdEncoder, RuleBasedInterpreter
)


def basic_states():
    """Demonstrate basic state operations"""
    print("=== Basic State Operations ===")
    
    # Create states
    print("1. Creating states:")
    state1 = State(1, 1)  # ψ (present and member)
    state2 = State(1, 0)  # δ (present but not member)
    state3 = State(0, 1)  # φ (not present but expected)
    state4 = State(0, 0)  # ∅ (absent)
    
    print(f"   State 1: {state1} (i={state1.i}, q={state1.q})")
    print(f"   State 2: {state2} (i={state2.i}, q={state2.q})")
    print(f"   State 3: {state3} (i={state3.i}, q={state3.q})")
    print(f"   State 4: {state4} (i={state4.i}, q={state4.q})")
    
    # State operations
    print("\n2. State operations:")
    print(f"   gate(δ): {gate(state2)}")  # δ → ∅
    print(f"   admit(δ): {admit(state2)}")  # δ → ψ
    print(f"   overlay(δ, φ): {overlay(state2, state3)}")  # δ | φ = ψ
    print(f"   weave(ψ, δ): {weave(state1, state2)}")  # ψ & δ = δ
    print(f"   comp(ψ): {comp(state1)}")  # ψ → ∅
    print(f"   neg_i(ψ): {neg_i(state1)}")  # ψ → φ
    print(f"   neg_q(ψ): {neg_q(state1)}")  # ψ → δ


def basic_state_arrays():
    """Demonstrate state arrays"""
    print("\n=== State Arrays ===")
    
    # Create state array from bits
    print("1. Creating state array from bits:")
    i_bits = [1, 0, 1, 0, 1, 0]
    q_bits = [1, 1, 0, 0, 1, 0]
    states = StateArray.from_bits(i=i_bits, q=q_bits)
    
    print(f"   i-bits: {i_bits}")
    print(f"   q-bits: {q_bits}")
    print(f"   States: {[str(s) for s in states]}")
    
    # Count states
    print("\n2. State counting:")
    counts = states.count_by_state()
    print(f"   State counts: {counts}")
    
    # Access individual states
    print("\n3. Accessing states:")
    print(f"   First state: {states[0]}")
    print(f"   Last state: {states[-1]}")
    print(f"   States 1-3: {[str(s) for s in states[1:4]]}")


def basic_pipeline():
    """Demonstrate basic pipeline usage"""
    print("\n=== Basic Pipeline ===")
    
    # Generate some data
    print("1. Generating data:")
    np.random.seed(42)
    X = np.random.randn(100, 2) * 2 + 1
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    print(f"   Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Create pipeline
    print("\n2. Creating pipeline:")
    pipeline = ChannelPipeline()
    
    # Add preprocessor
    pipeline.add_preprocessor(StandardScaler())
    print("   Added StandardScaler preprocessor")
    
    # Add encoder
    pipeline.add_encoder(ThresholdEncoder(threshold_i=0.0, threshold_q=0.5))
    print("   Added ThresholdEncoder")
    
    # Add interpreter
    interpreter = RuleBasedInterpreter()
    interpreter.add_rule(PSI, "POSITIVE", priority=1)
    interpreter.add_rule(EMPTY, "NEGATIVE", priority=2)
    interpreter.add_rule(DELTA, "UNCERTAIN", priority=3)
    interpreter.add_rule(PHI, "INCOMPLETE", priority=4)
    pipeline.add_interpreter(interpreter)
    print("   Added RuleBasedInterpreter")
    
    # Fit and transform
    print("\n3. Processing data:")
    pipeline.fit(X, y)
    decisions, states = pipeline.transform(X)
    
    print(f"   Processed {len(decisions[0])} samples")
    print(f"   Generated {len(states[0])} states")
    
    # Show results
    print("\n4. Results:")
    state_array = states[0]
    print(f"   State distribution: {state_array.count_by_state()}")
    
    # Show decision distribution
    decision_counts = {}
    for decision in decisions[0]:
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    print(f"   Decision distribution: {decision_counts}")
    
    # Show first few examples
    print("\n5. First 10 examples:")
    for i in range(10):
        print(f"   Sample {i}: {state_array[i]} -> {decisions[0][i]}")


def basic_interpretation():
    """Demonstrate basic interpretation"""
    print("\n=== Basic Interpretation ===")
    
    # Create some states
    print("1. Creating test states:")
    test_states = [PSI, DELTA, PHI, EMPTY, PSI, DELTA]
    print(f"   Test states: {[str(s) for s in test_states]}")
    
    # Create interpreter
    print("\n2. Creating interpreter:")
    interpreter = RuleBasedInterpreter()
    interpreter.add_rule(PSI, "EXCELLENT", priority=1)
    interpreter.add_rule(DELTA, "GOOD", priority=2)
    interpreter.add_rule(PHI, "FAIR", priority=3)
    interpreter.add_rule(EMPTY, "POOR", priority=4)
    interpreter.set_default("UNKNOWN")
    print("   Created RuleBasedInterpreter with 4 rules")
    
    # Interpret states
    print("\n3. Interpreting states:")
    for i, state in enumerate(test_states):
        decision = interpreter.interpret(state)
        print(f"   {state} -> {decision}")
    
    # Batch interpretation
    print("\n4. Batch interpretation:")
    decisions = [interpreter.interpret(state) for state in test_states]
    print(f"   All decisions: {decisions}")


def main():
    """Main function"""
    print("ChannelPy Basic Usage Example")
    print("=" * 40)
    
    basic_states()
    basic_state_arrays()
    basic_pipeline()
    basic_interpretation()
    
    print("\n" + "=" * 40)
    print("Basic usage example completed!")
    print("\nNext steps:")
    print("- Try the quick start: python examples/quick_start.py")
    print("- Run the complete tutorial: python examples/complete_tutorial.py")
    print("- Experiment with different encoders and interpreters")


if __name__ == "__main__":
    main()







