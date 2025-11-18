"""
ChannelPy Quick Start Example

This example demonstrates the basic usage of ChannelPy in just a few lines of code.
Perfect for getting started quickly.
"""

import numpy as np
from channelpy import (
    State, StateArray, PSI, DELTA, PHI, EMPTY,
    ChannelPipeline, StandardScaler, ThresholdEncoder, RuleBasedInterpreter,
    make_classification_data, plot_states
)


def quick_start():
    """Quick start example showing basic ChannelPy usage"""
    
    print("ChannelPy Quick Start")
    print("=" * 30)
    
    # 1. Create some states
    print("1. Creating states:")
    states = [PSI, DELTA, PHI, EMPTY]
    for state in states:
        print(f"   {state} (i={state.i}, q={state.q})")
    
    # 2. Generate some data
    print("\n2. Generating data:")
    X, y = make_classification_data(n_samples=50, n_features=2, n_classes=2)
    print(f"   Generated {X.shape[0]} samples with {X.shape[1]} features")
    
    # 3. Create a simple pipeline
    print("\n3. Creating pipeline:")
    pipeline = ChannelPipeline()
    pipeline.add_preprocessor(StandardScaler())
    pipeline.add_encoder(ThresholdEncoder(threshold_i=0.0, threshold_q=0.5))
    
    # Add a simple interpreter
    interpreter = RuleBasedInterpreter()
    interpreter.add_rule(PSI, "GOOD", priority=1)
    interpreter.add_rule(DELTA, "OKAY", priority=2)
    interpreter.add_rule(PHI, "WAIT", priority=3)
    interpreter.add_rule(EMPTY, "BAD", priority=4)
    pipeline.add_interpreter(interpreter)
    
    print("   Pipeline created with preprocessor, encoder, and interpreter")
    
    # 4. Process the data
    print("\n4. Processing data:")
    pipeline.fit(X, y)
    decisions, states = pipeline.transform(X)
    
    print(f"   Processed {len(decisions[0])} samples")
    print(f"   Generated {len(states[0])} states")
    
    # 5. Show results
    print("\n5. Results:")
    state_array = states[0]
    print(f"   State distribution: {state_array.count_by_state()}")
    
    # Show first 10 decisions
    print(f"   First 10 decisions: {decisions[0][:10]}")
    
    # 6. Create a simple plot
    print("\n6. Creating visualization:")
    try:
        plot_states(state_array, title="Quick Start - State Sequence")
        print("   Plot created successfully!")
    except Exception as e:
        print(f"   Plot creation failed: {e}")
    
    print("\nQuick start completed! 🎉")
    print("\nNext steps:")
    print("- Try the complete tutorial: python examples/complete_tutorial.py")
    print("- Explore different encoders and interpreters")
    print("- Experiment with your own data")


if __name__ == "__main__":
    quick_start()







