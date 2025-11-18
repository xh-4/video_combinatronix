"""
Complete ChannelPy Tutorial

This tutorial demonstrates the complete ChannelPy workflow from data generation
through preprocessing, encoding, interpretation, and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from channelpy import (
    State, StateArray, EMPTY, DELTA, PHI, PSI,
    ChannelPipeline, StandardScaler, ThresholdEncoder, RuleBasedInterpreter,
    StreamingAdaptiveThreshold, plot_states, plot_state_distribution
)
from channelpy.examples.datasets import (
    make_classification_data, make_trading_data, make_medical_data,
    make_state_sequence, generate_streaming_data
)
from channelpy.utils.examples import analyze_state_distribution, find_state_patterns


def tutorial_basic_states():
    """Tutorial: Basic State Operations"""
    print("=== Tutorial: Basic State Operations ===")
    
    # Create states
    print("1. Creating states:")
    state1 = State(1, 1)  # ψ (present and member)
    state2 = State(1, 0)  # δ (present but not member)
    state3 = State(0, 1)  # φ (not present but expected)
    state4 = State(0, 0)  # ∅ (absent)
    
    print(f"   State 1: {state1} (i={state1.i}, q={state1.q})")
    print(f"   State 2: {state2} (i={state1.i}, q={state2.q})")
    print(f"   State 3: {state3} (i={state3.i}, q={state3.q})")
    print(f"   State 4: {state4} (i={state4.i}, q={state4.q})")
    
    # State operations
    print("\n2. State operations:")
    from channelpy import gate, admit, overlay, weave
    
    print(f"   gate(δ): {gate(state2)}")  # δ → ∅
    print(f"   admit(δ): {admit(state2)}")  # δ → ψ
    print(f"   overlay(δ, φ): {overlay(state2, state3)}")  # δ | φ = ψ
    print(f"   weave(ψ, δ): {weave(state1, state2)}")  # ψ & δ = δ
    
    # State arrays
    print("\n3. State arrays:")
    states = StateArray.from_bits(i=[1, 0, 1, 0], q=[1, 1, 0, 0])
    print(f"   States: {[str(s) for s in states]}")
    print(f"   Counts: {states.count_by_state()}")
    
    print()


def tutorial_data_generation():
    """Tutorial: Data Generation"""
    print("=== Tutorial: Data Generation ===")
    
    # Classification data
    print("1. Classification data:")
    X, y = make_classification_data(n_samples=100, n_features=2, n_classes=2)
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Trading data
    print("\n2. Trading data:")
    df = make_trading_data(n_samples=50, volatility=0.02)
    print(f"   DataFrame shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")
    
    # Medical data
    print("\n3. Medical data:")
    symptoms, tests, labels = make_medical_data(n_samples=100)
    print(f"   Symptoms shape: {symptoms.shape}")
    print(f"   Test results: {tests.sum()}/{len(tests)} positive")
    print(f"   True prevalence: {labels.mean():.2f}")
    
    # State sequences
    print("\n4. State sequences:")
    states = make_state_sequence(length=20)
    print(f"   State sequence: {[str(s) for s in states]}")
    analysis = analyze_state_distribution(states)
    print(f"   Distribution: {analysis['percentages']}")
    
    print()


def tutorial_preprocessing():
    """Tutorial: Data Preprocessing"""
    print("=== Tutorial: Data Preprocessing ===")
    
    # Generate noisy data
    X, y = make_classification_data(n_samples=100, noise=0.3)
    print(f"1. Original data shape: {X.shape}")
    print(f"   Data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   Data mean: {X.mean(axis=0)}")
    
    # Standard scaling
    print("\n2. Standard scaling:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    print(f"   Scaled mean: {X_scaled.mean(axis=0)}")
    print(f"   Scaled std: {X_scaled.std(axis=0)}")
    
    # Missing data handling
    print("\n3. Missing data handling:")
    from channelpy import MissingDataHandler
    
    # Add some missing values
    X_missing = X.copy()
    X_missing[0, 0] = np.nan
    X_missing[5, 1] = np.nan
    print(f"   Missing values: {np.isnan(X_missing).sum()}")
    
    handler = MissingDataHandler(strategy='median')
    X_filled, missing_mask = handler.fit_transform(X_missing)
    print(f"   After filling: {np.isnan(X_filled).sum()} missing")
    print(f"   Filled values: {missing_mask.sum()}")
    
    print()


def tutorial_encoding():
    """Tutorial: Feature Encoding to States"""
    print("=== Tutorial: Feature Encoding to States ===")
    
    # Generate data
    X, y = make_classification_data(n_samples=100, n_features=2)
    
    # Simple threshold encoding
    print("1. Simple threshold encoding:")
    encoder = ThresholdEncoder(threshold_i=0.0, threshold_q=0.5)
    states = encoder(X)
    print(f"   Encoded {len(states)} states")
    print(f"   State distribution: {states.count_by_state()}")
    
    # Learned threshold encoding
    print("\n2. Learned threshold encoding:")
    from channelpy import LearnedThresholdEncoder
    
    encoder_learned = LearnedThresholdEncoder(method='statistical')
    encoder_learned.fit(X, y)
    states_learned = encoder_learned(X)
    print(f"   Learned thresholds: i={encoder_learned.threshold_i:.2f}, q={encoder_learned.threshold_q:.2f}")
    print(f"   State distribution: {states_learned.count_by_state()}")
    
    # Dual feature encoding
    print("\n3. Dual feature encoding:")
    from channelpy import DualFeatureEncoder
    
    # Create two separate features
    X_i = X[:, 0]  # First feature for i-bit
    X_q = X[:, 1]  # Second feature for q-bit
    
    dual_encoder = DualFeatureEncoder()
    dual_encoder.fit(X_i, X_q, y)
    states_dual = dual_encoder(X_i, X_q)
    print(f"   Dual encoding distribution: {states_dual.count_by_state()}")
    
    print()


def tutorial_interpretation():
    """Tutorial: State Interpretation"""
    print("=== Tutorial: State Interpretation ===")
    
    # Generate states
    states = make_state_sequence(length=20)
    print(f"1. State sequence: {[str(s) for s in states]}")
    
    # Rule-based interpretation
    print("\n2. Rule-based interpretation:")
    interpreter = RuleBasedInterpreter()
    interpreter.add_rule(PSI, "APPROVE", priority=1)
    interpreter.add_rule(DELTA, "REVIEW", priority=2)
    interpreter.add_rule(PHI, "WAIT", priority=3)
    interpreter.add_rule(EMPTY, "REJECT", priority=4)
    interpreter.set_default("UNKNOWN")
    
    decisions = [interpreter.interpret(state) for state in states]
    print(f"   Decisions: {decisions[:10]}...")  # Show first 10
    
    # Lookup table interpretation
    print("\n3. Lookup table interpretation:")
    from channelpy import LookupTableInterpreter
    
    lookup_interpreter = LookupTableInterpreter()
    lookup_interpreter.build_table({
        PSI: {'action': 'BUY', 'confidence': 0.9},
        DELTA: {'action': 'HOLD', 'confidence': 0.5},
        PHI: {'action': 'RESEARCH', 'confidence': 0.3},
        EMPTY: {'action': 'SELL', 'confidence': 0.8}
    })
    
    lookup_decisions = [lookup_interpreter.interpret(state) for state in states[:5]]
    print(f"   Lookup decisions: {lookup_decisions}")
    
    # Threshold-based interpretation
    print("\n4. Threshold-based interpretation:")
    from channelpy import ThresholdBasedInterpreter
    
    threshold_interpreter = ThresholdBasedInterpreter(low_threshold=0.3, high_threshold=0.7)
    threshold_decisions = [threshold_interpreter.interpret(state) for state in states[:5]]
    print(f"   Threshold decisions: {threshold_decisions}")
    
    print()


def tutorial_adaptive_thresholds():
    """Tutorial: Adaptive Thresholds"""
    print("=== Tutorial: Adaptive Thresholds ===")
    
    # Generate streaming data
    print("1. Streaming data with adaptive thresholds:")
    stream = generate_streaming_data(base_value=100, volatility=0.02, drift=0.001)
    
    # Create adaptive threshold
    adaptive_threshold = StreamingAdaptiveThreshold(window_size=50)
    
    # Process first 100 points
    states = []
    values = []
    for i, value in enumerate(stream):
        if i >= 100:
            break
        
        adaptive_threshold.update(value)
        state = adaptive_threshold.encode(value)
        states.append(state)
        values.append(value)
    
    print(f"   Processed {len(states)} values")
    print(f"   Final thresholds: i={adaptive_threshold.threshold_i:.2f}, q={adaptive_threshold.threshold_q:.2f}")
    
    # Analyze states
    state_array = StateArray.from_states(states)
    analysis = analyze_state_distribution(state_array)
    print(f"   State distribution: {analysis['percentages']}")
    
    print()


def tutorial_complete_pipeline():
    """Tutorial: Complete Pipeline"""
    print("=== Tutorial: Complete Pipeline ===")
    
    # Generate data
    X, y = make_classification_data(n_samples=200, n_features=3, n_classes=2)
    print(f"1. Generated data: {X.shape}, labels: {y.shape}")
    
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
    interpreter.add_rule(PSI, "CLASS_1", priority=1)
    interpreter.add_rule(EMPTY, "CLASS_0", priority=2)
    interpreter.add_rule(DELTA, "UNCERTAIN", priority=3)
    interpreter.add_rule(PHI, "INCOMPLETE", priority=4)
    pipeline.add_interpreter(interpreter)
    print("   Added RuleBasedInterpreter")
    
    # Fit pipeline
    print("\n3. Fitting pipeline:")
    pipeline.fit(X, y)
    print("   Pipeline fitted successfully")
    
    # Transform data
    print("\n4. Transforming data:")
    decisions, states = pipeline.transform(X)
    print(f"   Generated {len(decisions)} decisions")
    print(f"   Generated {len(states)} state arrays")
    
    # Analyze results
    print("\n5. Analyzing results:")
    state_array = states[0]  # First (and only) encoder
    analysis = analyze_state_distribution(state_array)
    print(f"   State distribution: {analysis['percentages']}")
    
    # Check decision distribution
    decision_counts = {}
    for decision in decisions[0]:  # First (and only) interpreter
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    print(f"   Decision distribution: {decision_counts}")
    
    print()


def tutorial_visualization():
    """Tutorial: Visualization"""
    print("=== Tutorial: Visualization ===")
    
    # Generate state sequence
    states = make_state_sequence(length=100)
    print(f"1. Generated {len(states)} states for visualization")
    
    # Plot states
    print("\n2. Creating state plots:")
    fig1, ax1 = plot_states(states, title="State Sequence Over Time")
    plt.tight_layout()
    plt.savefig('tutorial_states.png', dpi=150, bbox_inches='tight')
    print("   Saved: tutorial_states.png")
    
    # Plot distribution
    fig2, ax2 = plot_state_distribution(states, title="State Distribution")
    plt.tight_layout()
    plt.savefig('tutorial_distribution.png', dpi=150, bbox_inches='tight')
    print("   Saved: tutorial_distribution.png")
    
    # Find patterns
    print("\n3. Pattern analysis:")
    patterns = find_state_patterns(states, pattern_length=3)
    print(f"   Found {len(patterns)} patterns of length 3")
    
    # Show top patterns
    top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
    for pattern, count in top_patterns:
        pattern_str = ''.join(str(s) for s in pattern)
        print(f"   {pattern_str}: {count} times")
    
    print()


def tutorial_real_world_example():
    """Tutorial: Real-World Example (Trading System)"""
    print("=== Tutorial: Real-World Example (Trading System) ===")
    
    # Generate trading data
    df = make_trading_data(n_samples=252, volatility=0.02, drift=0.0005)
    print(f"1. Generated {len(df)} trading days")
    print(f"   Price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")
    
    # Create trading pipeline
    print("\n2. Creating trading pipeline:")
    trading_pipeline = ChannelPipeline()
    
    # Preprocess: normalize prices and volumes
    from channelpy import normalize
    trading_pipeline.add_preprocessor(lambda x: normalize(x))
    
    # Encode: use price changes and volume
    def trading_encoder(features):
        prices = features[:, 0]  # Price
        volumes = features[:, 1]  # Volume
        
        # Price momentum (i-bit)
        price_changes = np.diff(prices, prepend=prices[0])
        price_threshold = np.percentile(price_changes, 50)
        i_bits = (price_changes > price_threshold).astype(int)
        
        # Volume confirmation (q-bit)
        volume_threshold = np.percentile(volumes, 75)
        q_bits = (volumes > volume_threshold).astype(int)
        
        return StateArray.from_bits(i=i_bits, q=q_bits)
    
    trading_pipeline.add_encoder(trading_encoder)
    
    # Interpret: trading decisions
    trading_interpreter = RuleBasedInterpreter()
    trading_interpreter.add_rule(PSI, "STRONG_BUY", priority=1)
    trading_interpreter.add_rule(DELTA, "WEAK_BUY", priority=2)
    trading_interpreter.add_rule(PHI, "HOLD", priority=3)
    trading_interpreter.add_rule(EMPTY, "SELL", priority=4)
    trading_pipeline.add_interpreter(trading_interpreter)
    
    # Prepare features
    features = np.column_stack([df['close'], df['volume']])
    
    # Fit and transform
    print("\n3. Processing trading data:")
    trading_pipeline.fit(features)
    decisions, states = trading_pipeline.transform(features)
    
    # Analyze results
    print("\n4. Trading analysis:")
    state_array = states[0]
    analysis = analyze_state_distribution(state_array)
    print(f"   State distribution: {analysis['percentages']}")
    
    # Count trading signals
    trading_decisions = decisions[0]
    signal_counts = {}
    for signal in trading_decisions:
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    print(f"   Trading signals: {signal_counts}")
    
    # Calculate returns for each signal
    returns = df['close'].pct_change().dropna()
    signal_returns = {}
    for signal in set(trading_decisions):
        mask = np.array(trading_decisions) == signal
        if mask.sum() > 0:
            signal_returns[signal] = returns[mask].mean()
    
    print(f"   Average returns by signal: {signal_returns}")
    
    print()


def main():
    """Main tutorial function"""
    print("ChannelPy Complete Tutorial")
    print("=" * 50)
    
    tutorial_basic_states()
    tutorial_data_generation()
    tutorial_preprocessing()
    tutorial_encoding()
    tutorial_interpretation()
    tutorial_adaptive_thresholds()
    tutorial_complete_pipeline()
    tutorial_visualization()
    tutorial_real_world_example()
    
    print("Tutorial completed successfully!")
    print("\nNext steps:")
    print("1. Explore the generated plots (tutorial_*.png)")
    print("2. Try modifying the parameters in each tutorial section")
    print("3. Experiment with different datasets and encoders")
    print("4. Build your own custom interpreters for specific domains")


if __name__ == "__main__":
    main()







