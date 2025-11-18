"""
Utils Example using ChannelPy

This example demonstrates the utilities available in ChannelPy
for validation, serialization, and data generation.
"""

import numpy as np
import tempfile
import os
from channelpy import State, StateArray, EMPTY, DELTA, PHI, PSI
from channelpy.core.nested import NestedState
from channelpy.core.parallel import ParallelChannels
from channelpy.utils.validation import (
    ValidationError, validate_threshold, validate_array_shape, validate_pipeline_data,
    validate_input, require_fitted, ValidationContext
)
from channelpy.utils.serialization import (
    save, load, to_dict, from_dict, ChannelPyEncoder
)
from channelpy.utils.examples import (
    generate_sample_trading_data, generate_sample_medical_data, create_sample_pipeline_data,
    analyze_state_distribution, find_state_patterns, calculate_state_transitions
)


def demonstrate_validation():
    """Demonstrate validation utilities"""
    print("=== Validation Utilities ===")
    
    # Basic validation
    print("  Basic validation:")
    validate_threshold(0.5)  # Valid
    validate_threshold(1.0)  # Valid
    
    try:
        validate_threshold("invalid")  # Invalid
    except ValidationError as e:
        print(f"    Caught expected error: {e}")
    
    # Array validation
    print("  Array validation:")
    arr = np.array([[1, 2], [3, 4]])
    validate_array_shape(arr, expected_shape=(2, 2))  # Valid
    validate_array_shape(arr, min_dims=2)  # Valid
    
    try:
        validate_array_shape(arr, expected_shape=(3, 2))  # Invalid
    except ValidationError as e:
        print(f"    Caught expected error: {e}")
    
    # Pipeline data validation
    print("  Pipeline data validation:")
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    validate_pipeline_data(X, y)  # Valid
    
    try:
        X_invalid = np.array([[1, 2], [np.nan, 4]])
        validate_pipeline_data(X_invalid)  # Invalid
    except ValidationError as e:
        print(f"    Caught expected error: {e}")
    
    print()


def demonstrate_validation_decorators():
    """Demonstrate validation decorators"""
    print("=== Validation Decorators ===")
    
    # Function with input validation
    @validate_input(threshold=validate_threshold, window_size=lambda x: x > 0)
    def process_data(threshold, window_size, data):
        return f"Processing with threshold={threshold}, window={window_size}"
    
    # Valid call
    result = process_data(0.5, 10, [1, 2, 3])
    print(f"  Valid call result: {result}")
    
    # Invalid call
    try:
        process_data("invalid", 10, [1, 2, 3])
    except ValidationError as e:
        print(f"    Caught expected error: {e}")
    
    # Class with fitted validation
    class MockModel:
        def __init__(self, fitted=True):
            self.is_fitted = fitted
        
        @require_fitted
        def predict(self, X):
            return f"Predicted {len(X)} samples"
    
    # Fitted model
    fitted_model = MockModel(True)
    result = fitted_model.predict([1, 2, 3])
    print(f"  Fitted model result: {result}")
    
    # Not fitted model
    try:
        unfitted_model = MockModel(False)
        unfitted_model.predict([1, 2, 3])
    except ValidationError as e:
        print(f"    Caught expected error: {e}")
    
    print()


def demonstrate_validation_context():
    """Demonstrate validation context manager"""
    print("=== Validation Context Manager ===")
    
    with ValidationContext("Data processing pipeline"):
        try:
            validate_threshold(0.5)  # Valid
            validate_threshold("invalid")  # Invalid
        except ValidationError as e:
            print(f"  Context error: {e}")
    
    print()


def demonstrate_serialization():
    """Demonstrate serialization utilities"""
    print("=== Serialization Utilities ===")
    
    # Create sample objects
    state = State(1, 1)  # ψ
    state_array = StateArray.from_bits(i=[1, 0, 1], q=[1, 1, 0])
    nested_state = NestedState(level0=PSI, level1=DELTA, level2=PHI)
    channels = ParallelChannels(technical=PSI, business=DELTA, team=PHI)
    
    # Test JSON serialization
    print("  JSON serialization:")
    json_str = state.to_json() if hasattr(state, 'to_json') else "Not implemented"
    print(f"    State JSON: {json_str}")
    
    # Test dictionary conversion
    print("  Dictionary conversion:")
    state_dict = to_dict(state)
    print(f"    State dict: {state_dict}")
    
    reconstructed_state = from_dict(state_dict)
    print(f"    Reconstructed: {reconstructed_state}")
    print(f"    Equal: {reconstructed_state == state}")
    
    # Test file serialization
    print("  File serialization:")
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_path = f.name
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pkl_path = f.name
    
    try:
        # Save objects
        save(state, json_path, format='json')
        save(state_array, pkl_path, format='pickle')
        
        # Load objects
        loaded_state = load(json_path)
        loaded_array = load(pkl_path)
        
        print(f"    Loaded state: {loaded_state}")
        print(f"    Loaded array length: {len(loaded_array)}")
        print(f"    State equal: {loaded_state == state}")
        print(f"    Array equal: {loaded_array[0] == state_array[0]}")
        
    finally:
        os.unlink(json_path)
        os.unlink(pkl_path)
    
    print()


def demonstrate_data_generation():
    """Demonstrate data generation utilities"""
    print("=== Data Generation Utilities ===")
    
    # Generate trading data
    print("  Trading data:")
    prices, volumes = generate_sample_trading_data(n_samples=100)
    print(f"    Prices: {len(prices)} points, range: [{prices.min():.2f}, {prices.max():.2f}]")
    print(f"    Volumes: {len(volumes)} points, range: [{volumes.min():.2f}, {volumes.max():.2f}]")
    
    # Generate medical data
    print("  Medical data:")
    X, y, feature_names = generate_sample_medical_data(n_patients=50, n_features=5)
    print(f"    Features: {X.shape}, Labels: {len(y)}")
    print(f"    Feature names: {feature_names}")
    print(f"    Disease prevalence: {y.mean():.2f}")
    
    # Generate state sequences
    print("  State sequences:")
    states = generate_sample_state_sequence(n_states=20)
    print(f"    Generated {len(states)} states")
    print(f"    First 5 states: {[str(s) for s in states[:5]]}")
    
    # Generate nested states
    print("  Nested states:")
    nested_states = generate_sample_nested_states(n_sequences=5, max_depth=3)
    for i, nested in enumerate(nested_states):
        print(f"    Nested {i+1}: {nested} (depth: {nested.depth})")
    
    # Generate parallel channels
    print("  Parallel channels:")
    channels_list = generate_sample_parallel_channels(n_sequences=3)
    for i, channels in enumerate(channels_list):
        print(f"    Channels {i+1}: {channels}")
    
    print()


def demonstrate_data_analysis():
    """Demonstrate data analysis utilities"""
    print("=== Data Analysis Utilities ===")
    
    # Generate sample state data
    states = StateArray.from_bits(
        i=[1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        q=[1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    )
    
    print(f"  Sample states: {[str(s) for s in states]}")
    
    # Analyze distribution
    print("  State distribution analysis:")
    analysis = analyze_state_distribution(states)
    print(f"    Total states: {analysis['total_states']}")
    print(f"    Counts: {analysis['counts']}")
    print(f"    Percentages: {analysis['percentages']}")
    print(f"    Most common: {analysis['most_common']}")
    print(f"    Least common: {analysis['least_common']}")
    
    # Find patterns
    print("  Pattern analysis:")
    patterns = find_state_patterns(states, pattern_length=2)
    print(f"    Found {len(patterns)} patterns of length 2:")
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        pattern_str = ''.join(str(s) for s in pattern)
        print(f"      {pattern_str}: {count} times")
    
    # Calculate transitions
    print("  Transition analysis:")
    transitions = calculate_state_transitions(states)
    print(f"    Found {len(transitions)} transitions:")
    for transition, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True):
        transition_str = f"{transition[0]} → {transition[1]}"
        print(f"      {transition_str}: {count} times")
    
    print()


def demonstrate_complete_pipeline():
    """Demonstrate complete pipeline with validation and serialization"""
    print("=== Complete Pipeline Example ===")
    
    # Generate sample data
    print("  Generating sample data...")
    sample_data = create_sample_pipeline_data()
    
    # Validate data
    print("  Validating data...")
    with ValidationContext("Pipeline data validation"):
        validate_pipeline_data(sample_data['medical']['X'], sample_data['medical']['y'])
        validate_pipeline_data(sample_data['trading']['prices'])
    
    # Create state sequence from trading data
    print("  Creating state sequence...")
    prices = sample_data['trading']['prices']
    price_changes = np.diff(prices, prepend=prices[0])
    
    # Simple threshold-based encoding
    threshold = np.median(price_changes)
    states = StateArray.from_bits(
        i=(price_changes > threshold).astype(int),
        q=(price_changes > threshold * 1.5).astype(int)
    )
    
    print(f"    Generated {len(states)} states from {len(prices)} price points")
    
    # Analyze the states
    print("  Analyzing states...")
    analysis = analyze_state_distribution(states)
    print(f"    State distribution: {analysis['percentages']}")
    
    # Find patterns
    patterns = find_state_patterns(states, pattern_length=3)
    print(f"    Found {len(patterns)} patterns of length 3")
    
    # Save results
    print("  Saving results...")
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        results_path = f.name
    
    try:
        results = {
            'states': to_dict(states),
            'analysis': analysis,
            'patterns': {str(k): v for k, v in patterns.items()},
            'data_info': {
                'n_prices': len(prices),
                'n_states': len(states),
                'threshold': float(threshold)
            }
        }
        
        save(results, results_path, format='json')
        print(f"    Results saved to: {results_path}")
        
        # Load and verify
        loaded_results = load(results_path)
        print(f"    Loaded results contain {len(loaded_results)} keys")
        
    finally:
        os.unlink(results_path)
    
    print()


def main():
    """Main demonstration function"""
    print("ChannelPy Utils Example")
    print("=" * 40)
    
    demonstrate_validation()
    demonstrate_validation_decorators()
    demonstrate_validation_context()
    demonstrate_serialization()
    demonstrate_data_generation()
    demonstrate_data_analysis()
    demonstrate_complete_pipeline()
    
    print("Utils example completed successfully!")


if __name__ == "__main__":
    main()







