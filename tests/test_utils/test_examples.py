"""
Tests for utils.examples module
"""
import pytest
import numpy as np
from channelpy.utils.examples import (
    generate_sample_trading_data, generate_sample_medical_data, generate_sample_signal_data,
    generate_sample_text_data, generate_sample_state_sequence, generate_sample_nested_states,
    generate_sample_parallel_channels, create_sample_pipeline_data, analyze_state_distribution,
    find_state_patterns, calculate_state_transitions
)
from channelpy.core.state import StateArray, EMPTY, DELTA, PHI, PSI


def test_generate_sample_trading_data():
    """Test trading data generation"""
    prices, volumes = generate_sample_trading_data(n_samples=100)
    
    assert len(prices) == 100
    assert len(volumes) == 100
    assert all(np.isfinite(prices))
    assert all(np.isfinite(volumes))
    assert all(volumes > 0)  # Volumes should be positive


def test_generate_sample_medical_data():
    """Test medical data generation"""
    X, y, feature_names = generate_sample_medical_data(n_patients=100, n_features=5)
    
    assert X.shape == (100, 5)
    assert len(y) == 100
    assert len(feature_names) == 5
    assert all(np.isfinite(X))
    assert set(y) <= {0, 1}  # Binary labels
    assert all(isinstance(name, str) for name in feature_names)


def test_generate_sample_signal_data():
    """Test signal data generation"""
    time, signal = generate_sample_signal_data(n_samples=200, n_frequencies=2)
    
    assert len(time) == 200
    assert len(signal) == 200
    assert all(np.isfinite(time))
    assert all(np.isfinite(signal))
    assert time[0] == 0  # Should start at 0


def test_generate_sample_text_data():
    """Test text data generation"""
    documents, labels = generate_sample_text_data(n_documents=50, n_words_per_doc=20)
    
    assert len(documents) == 50
    assert len(labels) == 50
    assert all(isinstance(doc, str) for doc in documents)
    assert set(labels) <= {0, 1}  # Binary labels
    assert all(len(doc.split()) == 20 for doc in documents)


def test_generate_sample_state_sequence():
    """Test state sequence generation"""
    states = generate_sample_state_sequence(n_states=50)
    
    assert len(states) == 50
    assert all(isinstance(state, type(PSI)) for state in states)
    assert all(state in [EMPTY, DELTA, PHI, PSI] for state in states)


def test_generate_sample_nested_states():
    """Test nested state generation"""
    nested_states = generate_sample_nested_states(n_sequences=20, max_depth=3)
    
    assert len(nested_states) == 20
    assert all(nested_state.depth >= 1 for nested_state in nested_states)
    assert all(nested_state.depth <= 3 for nested_state in nested_states)


def test_generate_sample_parallel_channels():
    """Test parallel channel generation"""
    channels = generate_sample_parallel_channels(n_sequences=20)
    
    assert len(channels) == 20
    assert all(len(channel) == 4 for channel in channels)  # Default 4 channels
    assert all('technical' in channel.all_names() for channel in channels)


def test_create_sample_pipeline_data():
    """Test sample pipeline data creation"""
    data = create_sample_pipeline_data()
    
    assert 'trading' in data
    assert 'medical' in data
    assert 'signal' in data
    assert 'text' in data
    assert 'states' in data
    
    # Check trading data
    assert 'prices' in data['trading']
    assert 'volumes' in data['trading']
    
    # Check medical data
    assert 'X' in data['medical']
    assert 'y' in data['medical']
    assert 'feature_names' in data['medical']
    
    # Check states data
    assert 'sequence' in data['states']
    assert 'nested' in data['states']
    assert 'parallel' in data['states']


def test_analyze_state_distribution():
    """Test state distribution analysis"""
    states = StateArray.from_bits(i=[1, 0, 1, 1, 0], q=[1, 1, 0, 1, 0])
    analysis = analyze_state_distribution(states)
    
    assert 'total_states' in analysis
    assert 'counts' in analysis
    assert 'percentages' in analysis
    assert 'most_common' in analysis
    assert 'least_common' in analysis
    
    assert analysis['total_states'] == 5
    assert sum(analysis['counts'].values()) == 5
    assert abs(sum(analysis['percentages'].values()) - 100) < 1e-10


def test_find_state_patterns():
    """Test state pattern finding"""
    states = StateArray.from_bits(i=[1, 1, 0, 1, 1], q=[1, 1, 1, 1, 1])
    patterns = find_state_patterns(states, pattern_length=2)
    
    assert isinstance(patterns, dict)
    assert all(isinstance(pattern, tuple) for pattern in patterns.keys())
    assert all(isinstance(count, int) for count in patterns.values())
    assert all(count > 0 for count in patterns.values())


def test_calculate_state_transitions():
    """Test state transition calculation"""
    states = StateArray.from_bits(i=[1, 0, 1, 0], q=[1, 1, 0, 0])
    transitions = calculate_state_transitions(states)
    
    assert isinstance(transitions, dict)
    assert all(isinstance(transition, tuple) for transition in transitions.keys())
    assert all(len(transition) == 2 for transition in transitions.keys())
    assert all(isinstance(count, int) for count in transitions.values())
    assert all(count > 0 for count in transitions.values())
    
    # Should have 3 transitions for 4 states
    assert len(transitions) == 3


def test_data_consistency():
    """Test that generated data is consistent"""
    # Test trading data consistency
    prices, volumes = generate_sample_trading_data(n_samples=100)
    assert len(prices) == len(volumes)
    
    # Test medical data consistency
    X, y, feature_names = generate_sample_medical_data(n_patients=50, n_features=3)
    assert X.shape[0] == len(y)
    assert X.shape[1] == len(feature_names)
    
    # Test signal data consistency
    time, signal = generate_sample_signal_data(n_samples=100)
    assert len(time) == len(signal)
    
    # Test text data consistency
    documents, labels = generate_sample_text_data(n_documents=20)
    assert len(documents) == len(labels)


def test_reproducibility():
    """Test that data generation is reproducible with same seed"""
    # Generate data twice with same parameters
    prices1, volumes1 = generate_sample_trading_data(n_samples=50)
    prices2, volumes2 = generate_sample_trading_data(n_samples=50)
    
    # Should be identical due to fixed seed
    np.testing.assert_array_equal(prices1, prices2)
    np.testing.assert_array_equal(volumes1, volumes2)







