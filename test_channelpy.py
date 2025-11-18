#!/usr/bin/env python3
"""
Quick test script to verify ChannelPy library functionality
"""

import numpy as np
from channelpy import State, StateArray, EMPTY, DELTA, PHI, PSI
from channelpy import gate, admit, overlay, weave, comp
from channelpy.adaptive import StreamingAdaptiveThreshold
from channelpy.visualization import plot_states, plot_state_distribution


def test_basic_states():
    """Test basic state creation and operations"""
    print("Testing basic states...")
    
    # Test state creation
    state1 = State(1, 1)  # ψ
    state2 = State(1, 0)  # δ
    state3 = State(0, 1)  # φ
    state4 = State(0, 0)  # ∅
    
    assert state1 == PSI
    assert state2 == DELTA
    assert state3 == PHI
    assert state4 == EMPTY
    
    print(f"✓ State creation: {state1}, {state2}, {state3}, {state4}")
    
    # Test operations
    assert gate(DELTA) == EMPTY  # δ → ∅
    assert admit(DELTA) == PSI   # δ → ψ
    assert overlay(DELTA, PHI) == PSI  # δ | φ = ψ
    assert weave(PSI, DELTA) == DELTA  # ψ & δ = δ
    assert comp(EMPTY) == PSI    # ∅ → ψ
    
    print("✓ Basic operations working")


def test_state_array():
    """Test StateArray functionality"""
    print("Testing StateArray...")
    
    states = StateArray.from_bits(i=[1, 0, 1, 0], q=[1, 1, 0, 0])
    assert len(states) == 4
    assert states[0] == PSI
    assert states[1] == PHI
    assert states[2] == DELTA
    assert states[3] == EMPTY
    
    counts = states.count_by_state()
    assert counts[PSI] == 1
    assert counts[PHI] == 1
    assert counts[DELTA] == 1
    assert counts[EMPTY] == 1
    
    print("✓ StateArray working")


def test_adaptive_thresholds():
    """Test adaptive threshold functionality"""
    print("Testing adaptive thresholds...")
    
    threshold = StreamingAdaptiveThreshold(window_size=100)
    
    # Generate some test data
    data = np.random.normal(0, 1, 200)
    
    for value in data:
        threshold.update(value)
    
    stats = threshold.get_stats()
    assert stats['n_samples'] == 200
    assert 'mean' in stats
    assert 'std' in stats
    assert 'threshold_i' in stats
    assert 'threshold_q' in stats
    
    # Test encoding
    state = threshold.encode(0.5)
    assert isinstance(state, State)
    
    print("✓ Adaptive thresholds working")


def test_pipeline():
    """Test pipeline functionality"""
    print("Testing pipeline...")
    
    from channelpy import ChannelPipeline, ThresholdEncoder
    
    # Create a simple pipeline
    pipeline = ChannelPipeline()
    pipeline.add_encoder(ThresholdEncoder(threshold_i=0.5, threshold_q=0.75))
    
    # Test data
    X = np.random.random((100, 1))
    y = np.random.randint(0, 2, 100)
    
    # Fit pipeline
    pipeline.fit(X, y)
    
    # Transform
    decisions, states = pipeline.transform(X)
    
    assert len(decisions) == 1  # One interpreter
    assert len(states) == 1     # One encoder
    
    print("✓ Pipeline working")


def main():
    """Run all tests"""
    print("ChannelPy Library Test")
    print("=" * 30)
    
    try:
        test_basic_states()
        test_state_array()
        test_adaptive_thresholds()
        test_pipeline()
        
        print("\n🎉 All tests passed! ChannelPy is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()







