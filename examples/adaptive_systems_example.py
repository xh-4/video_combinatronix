"""
Adaptive Systems Example

This example demonstrates the complete adaptive system capabilities:
1. Streaming adaptive thresholds
2. Multi-dimensional feature scoring
3. Persistent homology analysis
4. Integration of all adaptive components
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

# Import all adaptive components
from channelpy import (
    StreamingAdaptiveThreshold, FeatureScorer, ScoreDimension,
    relevance_scorer, confidence_scorer, freshness_scorer, 
    stability_scorer, density_scorer,
    create_trading_scorer, create_medical_scorer, create_signal_scorer,
    PersistenceDiagram, compute_betti_numbers, compute_persistence_entropy,
    StateArray, EMPTY, DELTA, PHI, PSI
)


def generate_streaming_data(n_samples: int = 1000, 
                           trend: float = 0.001,
                           noise_level: float = 0.1,
                           regime_changes: int = 3) -> np.ndarray:
    """Generate realistic streaming data with trends and regime changes"""
    np.random.seed(42)
    
    # Base trend
    t = np.arange(n_samples)
    trend_component = trend * t
    
    # Regime changes
    regime_length = n_samples // regime_changes
    regime_means = np.random.normal(0, 0.5, regime_changes)
    regime_component = np.repeat(regime_means, regime_length)[:n_samples]
    
    # Noise
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Combine components
    data = trend_component + regime_component + noise
    
    return data


def demonstrate_streaming_adaptive_thresholds():
    """Demonstrate streaming adaptive threshold system"""
    print("=== Streaming Adaptive Thresholds ===")
    
    # Generate streaming data
    data = generate_streaming_data(n_samples=2000, trend=0.002, noise_level=0.15)
    
    # Create adaptive threshold
    threshold = StreamingAdaptiveThreshold(
        window_size=500,
        adaptation_rate=0.01
    )
    
    # Process streaming data
    states = []
    threshold_history = []
    
    print("Processing streaming data...")
    for i, value in enumerate(data):
        threshold.update(value)
        state = threshold.encode(value)
        states.append(state)
        threshold_history.append(threshold.threshold_i)
        
        # Print progress every 500 samples
        if (i + 1) % 500 == 0:
            stats = threshold.get_stats()
            print(f"  Sample {i+1}: mean={stats['mean']:.3f}, "
                  f"std={stats['std']:.3f}, threshold_i={stats['threshold_i']:.3f}")
    
    # Convert to StateArray
    states_array = StateArray(states)
    
    # Analyze results
    print(f"\nFinal Statistics:")
    final_stats = threshold.get_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value:.3f}")
    
    print(f"\nState Distribution:")
    state_counts = states_array.count_by_state()
    for state, count in state_counts.items():
        print(f"  {state}: {count} ({count/len(states)*100:.1f}%)")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Data and thresholds
    axes[0].plot(data, alpha=0.7, label='Data')
    axes[0].plot(threshold_history, 'r-', linewidth=2, label='Adaptive Threshold')
    axes[0].set_title('Streaming Data with Adaptive Threshold')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: State sequence
    state_ints = states_array.to_ints()
    axes[1].plot(state_ints, 'o-', markersize=2, alpha=0.7)
    axes[1].set_title('Channel States Over Time')
    axes[1].set_ylabel('State (int)')
    axes[1].set_ylim(-0.5, 3.5)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(['∅', 'δ', 'φ', 'ψ'])
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Threshold evolution
    axes[2].plot(threshold_history, 'g-', linewidth=2)
    axes[2].set_title('Threshold Evolution')
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Threshold Value')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('streaming_adaptive_thresholds.png', dpi=150, bbox_inches='tight')
    print("Saved: streaming_adaptive_thresholds.png")
    
    return states_array, threshold_history


def demonstrate_feature_scoring():
    """Demonstrate multi-dimensional feature scoring"""
    print("\n=== Multi-Dimensional Feature Scoring ===")
    
    # Create different scorers for different domains
    trading_scorer = create_trading_scorer()
    medical_scorer = create_medical_scorer()
    signal_scorer = create_signal_scorer()
    
    # Generate sample data with context
    np.random.seed(42)
    n_samples = 100
    
    # Trading context
    trading_values = np.random.normal(0.5, 0.2, n_samples)
    trading_returns = np.random.normal(0.1, 0.3, n_samples)
    trading_context = {
        'historical_values': trading_values[:-10].tolist(),
        'historical_outcomes': trading_returns[:-10].tolist(),
        'sample_size': len(trading_values),
        'age_seconds': 0,
        'missing_rate': 0.05,
        'noise_level': 0.1
    }
    
    # Medical context
    medical_values = np.random.normal(0.3, 0.15, n_samples)
    medical_diagnoses = (medical_values > 0.4).astype(float)
    medical_context = {
        'historical_values': medical_values[:-10].tolist(),
        'historical_outcomes': medical_diagnoses[:-10].tolist(),
        'sample_size': len(medical_values),
        'recent_values': medical_values[-5:].tolist(),
        'missing_rate': 0.02,
        'noise_level': 0.05
    }
    
    # Signal processing context
    signal_values = np.random.normal(0.0, 0.3, n_samples)
    signal_context = {
        'historical_values': signal_values[:-10].tolist(),
        'historical_outcomes': np.abs(signal_values[:-10]).tolist(),
        'sample_size': len(signal_values),
        'recent_values': signal_values[-5:].tolist(),
        'all_values': signal_values.tolist(),
        'bandwidth': 0.1,
        'noise_level': 0.2
    }
    
    # Score features
    print("Scoring features across different domains...")
    
    # Trading scores
    trading_scores = []
    for value in trading_values[-10:]:
        score, dim_scores = trading_scorer.score_and_aggregate(value, trading_context)
        trading_scores.append(score)
    
    # Medical scores
    medical_scores = []
    for value in medical_values[-10:]:
        score, dim_scores = medical_scorer.score_and_aggregate(value, medical_context)
        medical_scores.append(score)
    
    # Signal scores
    signal_scores = []
    for value in signal_values[-10:]:
        score, dim_scores = signal_scorer.score_and_aggregate(value, signal_context)
        signal_scores.append(score)
    
    # Analyze results
    print(f"\nTrading Scores: mean={np.mean(trading_scores):.3f}, std={np.std(trading_scores):.3f}")
    print(f"Medical Scores: mean={np.mean(medical_scores):.3f}, std={np.std(medical_scores):.3f}")
    print(f"Signal Scores: mean={np.mean(signal_scores):.3f}, std={np.std(signal_scores):.3f}")
    
    # Demonstrate score explanation
    print(f"\nDetailed Score Explanation (Trading):")
    example_value = trading_values[-1]
    explanation = trading_scorer.explain_score(example_value, trading_context)
    print(explanation)
    
    # Plot scoring results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Trading scores
    axes[0, 0].plot(trading_scores, 'o-', color='blue', alpha=0.7)
    axes[0, 0].set_title('Trading Feature Scores')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Medical scores
    axes[0, 1].plot(medical_scores, 'o-', color='red', alpha=0.7)
    axes[0, 1].set_title('Medical Feature Scores')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Signal scores
    axes[1, 0].plot(signal_scores, 'o-', color='green', alpha=0.7)
    axes[1, 0].set_title('Signal Processing Scores')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Score comparison
    axes[1, 1].plot(trading_scores, 'o-', label='Trading', alpha=0.7)
    axes[1, 1].plot(medical_scores, 'o-', label='Medical', alpha=0.7)
    axes[1, 1].plot(signal_scores, 'o-', label='Signal', alpha=0.7)
    axes[1, 1].set_title('Score Comparison Across Domains')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_scoring_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: feature_scoring_analysis.png")
    
    return trading_scorer, medical_scorer, signal_scorer


def demonstrate_persistence_analysis():
    """Demonstrate persistent homology analysis"""
    print("\n=== Persistent Homology Analysis ===")
    
    # Generate data with different topological structures
    np.random.seed(42)
    
    # Dataset 1: Circle
    t = np.linspace(0, 2*np.pi, 100)
    circle_data = np.column_stack([
        0.5 * np.cos(t) + np.random.normal(0, 0.05, 100),
        0.5 * np.sin(t) + np.random.normal(0, 0.05, 100)
    ])
    
    # Dataset 2: Two clusters
    cluster1 = np.random.normal([0, 0], 0.1, (50, 2))
    cluster2 = np.random.normal([1, 1], 0.1, (50, 2))
    cluster_data = np.vstack([cluster1, cluster2])
    
    # Dataset 3: Random points
    random_data = np.random.uniform(-1, 1, (100, 2))
    
    datasets = {
        'Circle': circle_data,
        'Two Clusters': cluster_data,
        'Random': random_data
    }
    
    # Analyze each dataset
    diagrams = {}
    betti_numbers = {}
    persistence_entropy = {}
    
    for name, data in datasets.items():
        print(f"\nAnalyzing {name} dataset...")
        
        # Create persistence diagram
        diagram = PersistenceDiagram()
        diagram.compute(data, max_dim=1)
        diagrams[name] = diagram
        
        # Get Betti numbers
        betti = diagram.get_betti_numbers(epsilon=0.1)
        betti_numbers[name] = betti
        print(f"  Betti numbers: {betti}")
        
        # Get persistence entropy
        entropy = diagram.get_persistence_entropy()
        persistence_entropy[name] = entropy
        print(f"  Persistence entropy: {entropy}")
        
        # Get summary
        summary = diagram.get_summary()
        print(f"  Total persistence: {summary['total_persistence']}")
        print(f"  Number of features: {summary['num_features']}")
    
    # Plot persistence diagrams
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (name, diagram) in enumerate(diagrams.items()):
        if diagram.is_computed:
            diagram.plot(title=f'{name} Persistence Diagram', ax=axes[i])
        else:
            axes[i].text(0.5, 0.5, f'{name}\n(No data)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{name} Persistence Diagram')
    
    plt.tight_layout()
    plt.savefig('persistence_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: persistence_analysis.png")
    
    # Plot Betti curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, diagram in diagrams.items():
        if diagram.is_computed:
            diagram.plot_betti_curves(ax=ax)
    
    ax.set_title('Betti Number Curves Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig('betti_curves_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: betti_curves_comparison.png")
    
    return diagrams, betti_numbers, persistence_entropy


def demonstrate_integrated_adaptive_system():
    """Demonstrate integrated adaptive system"""
    print("\n=== Integrated Adaptive System ===")
    
    # Generate complex streaming data
    data = generate_streaming_data(n_samples=1500, trend=0.001, noise_level=0.2, regime_changes=4)
    
    # Create integrated system
    threshold = StreamingAdaptiveThreshold(window_size=300, adaptation_rate=0.02)
    scorer = create_trading_scorer()
    
    # Process data with integrated system
    states = []
    scores = []
    context_history = []
    
    print("Processing data with integrated adaptive system...")
    
    for i, value in enumerate(data):
        # Update threshold
        threshold.update(value)
        state = threshold.encode(value)
        states.append(state)
        
        # Create context for scoring
        if i > 50:  # Need some history
            context = {
                'historical_values': data[max(0, i-50):i].tolist(),
                'historical_outcomes': np.diff(data[max(0, i-50):i+1]).tolist(),
                'sample_size': min(50, i),
                'age_seconds': 0,
                'missing_rate': 0.02,
                'noise_level': 0.1,
                'recent_values': data[max(0, i-10):i].tolist()
            }
            
            # Score the feature
            score, dim_scores = scorer.score_and_aggregate(value, context)
            scores.append(score)
            context_history.append(context.copy())
            
            # Record for analysis
            scorer.record_score(value, context, outcome=value)
        else:
            scores.append(0.5)  # Neutral score
    
    # Convert to arrays
    states_array = StateArray(states)
    scores_array = np.array(scores)
    
    # Analyze results
    print(f"\nIntegrated System Results:")
    print(f"  Total samples: {len(data)}")
    print(f"  State distribution: {states_array.count_by_state()}")
    print(f"  Average score: {np.mean(scores_array):.3f}")
    print(f"  Score std: {np.std(scores_array):.3f}")
    
    # Get dimension statistics
    print(f"\nDimension Statistics:")
    for dim_name in ['strength', 'confidence', 'timeliness', 'stability']:
        stats = scorer.get_dimension_statistics(dim_name)
        if stats:
            print(f"  {dim_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    # Plot integrated results
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: Data and thresholds
    axes[0].plot(data, alpha=0.7, label='Data')
    axes[0].plot(threshold.threshold_i * np.ones(len(data)), 'r-', linewidth=2, label='Threshold')
    axes[0].set_title('Integrated Adaptive System: Data and Thresholds')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: States
    state_ints = states_array.to_ints()
    axes[1].plot(state_ints, 'o-', markersize=2, alpha=0.7)
    axes[1].set_title('Channel States')
    axes[1].set_ylabel('State')
    axes[1].set_ylim(-0.5, 3.5)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(['∅', 'δ', 'φ', 'ψ'])
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Scores
    axes[2].plot(scores_array, 'g-', linewidth=2, alpha=0.8)
    axes[2].set_title('Feature Scores Over Time')
    axes[2].set_ylabel('Score')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Combined view
    # Normalize data for plotting
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    axes[3].plot(data_norm, alpha=0.5, label='Data (normalized)')
    axes[3].plot(scores_array, 'g-', linewidth=2, label='Scores')
    axes[3].plot(np.array(state_ints) / 3, 'r-', alpha=0.7, label='States (normalized)')
    axes[3].set_title('Combined View: Data, States, and Scores')
    axes[3].set_xlabel('Sample')
    axes[3].set_ylabel('Normalized Value')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('integrated_adaptive_system.png', dpi=150, bbox_inches='tight')
    print("Saved: integrated_adaptive_system.png")
    
    return states_array, scores_array, scorer


def demonstrate_real_world_application():
    """Demonstrate real-world application"""
    print("\n=== Real-World Application: Trading System ===")
    
    # Simulate realistic trading data
    np.random.seed(42)
    n_days = 1000
    
    # Generate price data with trends and volatility
    returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Price series
    
    # Generate technical indicators
    sma_20 = np.convolve(prices, np.ones(20)/20, mode='valid')
    rsi = np.random.uniform(20, 80, n_days)  # Simplified RSI
    
    # Create trading system
    threshold = StreamingAdaptiveThreshold(window_size=100, adaptation_rate=0.01)
    scorer = create_trading_scorer()
    
    # Process trading data
    trading_signals = []
    scores = []
    positions = []
    
    print("Processing trading data...")
    
    for i in range(20, n_days):  # Start after SMA calculation
        # Current price and indicators
        current_price = prices[i]
        current_sma = sma_20[i-20]
        current_rsi = rsi[i]
        
        # Create signal (price relative to SMA)
        signal = (current_price - current_sma) / current_sma
        
        # Update threshold and encode
        threshold.update(signal)
        state = threshold.encode(signal)
        trading_signals.append(state)
        
        # Score the signal
        if i > 50:
            context = {
                'historical_values': [prices[j] for j in range(i-50, i)],
                'historical_outcomes': [returns[j] for j in range(i-50, i)],
                'sample_size': 50,
                'age_seconds': 0,
                'missing_rate': 0.01,
                'noise_level': 0.02
            }
            
            score, _ = scorer.score_and_aggregate(signal, context)
            scores.append(score)
            
            # Simple position logic based on score and state
            if score > 0.7 and state in [PHI, PSI]:
                position = 1  # Long
            elif score < 0.3 and state in [EMPTY, DELTA]:
                position = -1  # Short
            else:
                position = 0  # Neutral
            
            positions.append(position)
        else:
            scores.append(0.5)
            positions.append(0)
    
    # Analyze trading performance
    positions_array = np.array(positions)
    returns_array = np.array(returns[20:])
    
    # Calculate strategy returns
    strategy_returns = positions_array * returns_array
    
    # Performance metrics
    total_return = np.sum(strategy_returns)
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    win_rate = np.mean(strategy_returns > 0)
    
    print(f"\nTrading System Performance:")
    print(f"  Total return: {total_return:.3f}")
    print(f"  Sharpe ratio: {sharpe_ratio:.3f}")
    print(f"  Win rate: {win_rate:.3f}")
    print(f"  Average score: {np.mean(scores):.3f}")
    print(f"  Position distribution: {np.bincount(positions_array + 1)}")
    
    # Plot trading results
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: Price and SMA
    axes[0].plot(prices, label='Price')
    axes[0].plot(range(20, n_days), sma_20, label='SMA 20')
    axes[0].set_title('Price and Moving Average')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Trading signals
    signal_ints = [s.to_int() for s in trading_signals]
    axes[1].plot(signal_ints, 'o-', markersize=2, alpha=0.7)
    axes[1].set_title('Trading Signals (Channel States)')
    axes[1].set_ylabel('State')
    axes[1].set_ylim(-0.5, 3.5)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(['∅', 'δ', 'φ', 'ψ'])
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Scores
    axes[2].plot(scores, 'g-', linewidth=2, alpha=0.8)
    axes[2].set_title('Feature Scores')
    axes[2].set_ylabel('Score')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Positions and returns
    axes[3].plot(positions_array, 'r-', linewidth=2, label='Positions')
    axes[3].plot(strategy_returns, 'b-', alpha=0.7, label='Strategy Returns')
    axes[3].set_title('Positions and Returns')
    axes[3].set_xlabel('Day')
    axes[3].set_ylabel('Value')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trading_system_application.png', dpi=150, bbox_inches='tight')
    print("Saved: trading_system_application.png")


def main():
    """Main adaptive systems example function"""
    print("ChannelPy Adaptive Systems Example")
    print("=" * 50)
    
    # 1. Streaming adaptive thresholds
    states_array, threshold_history = demonstrate_streaming_adaptive_thresholds()
    
    # 2. Feature scoring
    trading_scorer, medical_scorer, signal_scorer = demonstrate_feature_scoring()
    
    # 3. Persistence analysis
    diagrams, betti_numbers, persistence_entropy = demonstrate_persistence_analysis()
    
    # 4. Integrated adaptive system
    integrated_states, integrated_scores, integrated_scorer = demonstrate_integrated_adaptive_system()
    
    # 5. Real-world application
    demonstrate_real_world_application()
    
    print("\n" + "=" * 50)
    print("Adaptive systems example completed successfully!")
    print("\nGenerated files:")
    print("- streaming_adaptive_thresholds.png: Streaming threshold adaptation")
    print("- feature_scoring_analysis.png: Multi-dimensional feature scoring")
    print("- persistence_analysis.png: Persistent homology analysis")
    print("- betti_curves_comparison.png: Betti number curves")
    print("- integrated_adaptive_system.png: Integrated adaptive system")
    print("- trading_system_application.png: Real-world trading application")
    print("\nNext steps:")
    print("- Use adaptive thresholds for streaming data")
    print("- Implement multi-dimensional feature scoring")
    print("- Apply persistent homology for topological analysis")
    print("- Integrate all components for advanced adaptive systems")


if __name__ == "__main__":
    main()







