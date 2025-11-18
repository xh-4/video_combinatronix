"""
Topology-Aware Adaptive Thresholding Example

This example demonstrates the KEY INNOVATION of ChannelPy: topology-aware
adaptive thresholding that responds to the shape of data distributions.

Features demonstrated:
1. Topology analysis of different distribution types
2. Adaptive threshold strategies for each topology
3. Real-time topology change detection
4. Integration with feature scoring
5. Real-world applications
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

# Import topology-aware components
from channelpy import (
    TopologyFeatures, TopologyAnalyzer, TopologyAdaptiveThreshold,
    FeatureScorer, create_trading_scorer, create_medical_scorer,
    StateArray, EMPTY, DELTA, PHI, PSI
)


def generate_multimodal_data(n_samples: int = 1000, 
                           n_modes: int = 3,
                           noise_level: float = 0.1) -> np.ndarray:
    """Generate multimodal data with specified number of modes"""
    np.random.seed(42)
    
    # Create modes
    modes = np.random.uniform(-2, 2, n_modes)
    weights = np.random.dirichlet(np.ones(n_modes))
    
    # Generate samples from mixture
    data = []
    for _ in range(n_samples):
        mode_idx = np.random.choice(n_modes, p=weights)
        sample = np.random.normal(modes[mode_idx], noise_level)
        data.append(sample)
    
    return np.array(data)


def generate_skewed_data(n_samples: int = 1000, 
                        skewness: float = 2.0) -> np.ndarray:
    """Generate skewed data using log-normal distribution"""
    np.random.seed(42)
    
    if skewness > 0:
        # Right-skewed: log-normal
        data = np.random.lognormal(0, skewness, n_samples)
    else:
        # Left-skewed: negative log-normal
        data = -np.random.lognormal(0, abs(skewness), n_samples)
    
    # Normalize to have mean 0
    data = data - np.mean(data)
    return data


def generate_heavy_tailed_data(n_samples: int = 1000,
                             tail_weight: float = 0.1) -> np.ndarray:
    """Generate heavy-tailed data using mixture of normal and Cauchy"""
    np.random.seed(42)
    
    # Mixture: mostly normal, some Cauchy (heavy tails)
    normal_samples = int(n_samples * (1 - tail_weight))
    cauchy_samples = n_samples - normal_samples
    
    normal_data = np.random.normal(0, 1, normal_samples)
    cauchy_data = np.random.standard_cauchy(cauchy_samples)
    
    data = np.concatenate([normal_data, cauchy_data])
    np.random.shuffle(data)
    
    return data


def generate_clustered_data(n_samples: int = 1000,
                          n_clusters: int = 3,
                          cluster_separation: float = 2.0) -> np.ndarray:
    """Generate clustered data with gaps between clusters"""
    np.random.seed(42)
    
    # Create cluster centers
    centers = np.linspace(-cluster_separation, cluster_separation, n_clusters)
    
    # Generate samples
    data = []
    samples_per_cluster = n_samples // n_clusters
    
    for center in centers:
        cluster_data = np.random.normal(center, 0.3, samples_per_cluster)
        data.extend(cluster_data)
    
    # Add remaining samples to first cluster
    remaining = n_samples - len(data)
    if remaining > 0:
        extra_data = np.random.normal(centers[0], 0.3, remaining)
        data.extend(extra_data)
    
    return np.array(data)


def demonstrate_topology_analysis():
    """Demonstrate topology analysis of different distribution types"""
    print("=== Topology Analysis of Different Distributions ===")
    
    # Generate different types of data
    datasets = {
        'Normal': np.random.normal(0, 1, 1000),
        'Multimodal': generate_multimodal_data(1000, 3),
        'Right-Skewed': generate_skewed_data(1000, 2.0),
        'Left-Skewed': generate_skewed_data(1000, -1.5),
        'Heavy-Tailed': generate_heavy_tailed_data(1000, 0.2),
        'Clustered': generate_clustered_data(1000, 3, 2.0)
    }
    
    # Analyze each dataset
    analyzer = TopologyAnalyzer()
    results = {}
    
    for name, data in datasets.items():
        print(f"\nAnalyzing {name} distribution...")
        
        features = analyzer.analyze(data)
        results[name] = {'data': data, 'features': features}
        
        print(f"  Modality: {features.modality}")
        print(f"  Skewness: {features.skewness:.3f}")
        print(f"  Kurtosis: {features.kurtosis:.3f}")
        print(f"  Local maxima: {features.local_maxima}")
        print(f"  Gaps: {len(features.gaps)}")
        print(f"  Density variance: {features.density_variance:.3f}")
        print(f"  Connected components: {features.connected_components}")
    
    # Plot topology analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(results.items()):
        data = result['data']
        features = result['features']
        
        # Plot histogram
        axes[i].hist(data, bins=50, density=True, alpha=0.7, color='skyblue')
        
        # Plot KDE if available
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_grid = np.linspace(data.min(), data.max(), 1000)
            density = kde(x_grid)
            axes[i].plot(x_grid, density, 'k-', linewidth=2, label='Density')
        except ImportError:
            pass
        
        # Mark local maxima
        for mode in features.local_maxima:
            axes[i].axvline(mode, color='red', linestyle='--', alpha=0.7)
        
        # Mark gaps
        for gap_start, gap_end in features.gaps:
            axes[i].axvspan(gap_start, gap_end, alpha=0.3, color='yellow')
        
        axes[i].set_title(f'{name}\nModality: {features.modality}, '
                         f'Skew: {features.skewness:.2f}, '
                         f'Kurt: {features.kurtosis:.2f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('topology_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: topology_analysis.png")
    
    return results


def demonstrate_adaptive_thresholds():
    """Demonstrate topology-aware adaptive thresholds"""
    print("\n=== Topology-Aware Adaptive Thresholds ===")
    
    # Generate complex streaming data with regime changes
    np.random.seed(42)
    n_samples = 2000
    
    # Create data with changing topology
    data = []
    
    # Phase 1: Normal distribution
    data.extend(np.random.normal(0, 1, 500))
    
    # Phase 2: Multimodal distribution
    data.extend(generate_multimodal_data(500, 2))
    
    # Phase 3: Skewed distribution
    data.extend(generate_skewed_data(500, 1.5))
    
    # Phase 4: Heavy-tailed distribution
    data.extend(generate_heavy_tailed_data(500, 0.3))
    
    data = np.array(data)
    
    # Create topology-aware adaptive threshold
    threshold = TopologyAdaptiveThreshold(
        window_size=300,
        adaptation_rate=0.02,
        topology_update_interval=50
    )
    
    # Process data
    states = []
    threshold_history = []
    topology_changes = []
    strategies = []
    
    print("Processing data with topology-aware adaptive thresholds...")
    
    for i, value in enumerate(data):
        threshold.update(value)
        state = threshold.encode(value)
        states.append(state)
        
        # Record threshold history
        threshold_history.append({
            'i': threshold.threshold_i,
            'q': threshold.threshold_q
        })
        
        # Check for topology changes
        if threshold.topology_changed():
            topology_changes.append(i)
            print(f"  Topology change detected at sample {i}")
        
        # Record strategy
        strategies.append(threshold._get_current_strategy())
        
        # Print progress
        if (i + 1) % 500 == 0:
            info = threshold.get_thresholds()
            print(f"  Sample {i+1}: strategy={info['adaptation_strategy']}, "
                  f"threshold_i={info['threshold_i']:.3f}, "
                  f"threshold_q={info['threshold_q']:.3f}")
    
    # Convert to StateArray
    states_array = StateArray(states)
    
    # Analyze results
    print(f"\nResults:")
    print(f"  Total samples: {len(data)}")
    print(f"  Topology changes: {len(topology_changes)}")
    print(f"  State distribution: {states_array.count_by_state()}")
    
    # Strategy distribution
    strategy_counts = {}
    for strategy in strategies:
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print(f"  Strategy distribution: {strategy_counts}")
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: Data with thresholds
    axes[0].plot(data, alpha=0.7, label='Data')
    threshold_i_history = [t['i'] for t in threshold_history]
    threshold_q_history = [t['q'] for t in threshold_history]
    axes[0].plot(threshold_i_history, 'r-', linewidth=2, label='Threshold i')
    axes[0].plot(threshold_q_history, 'orange', linewidth=2, label='Threshold q')
    
    # Mark topology changes
    for change_point in topology_changes:
        axes[0].axvline(change_point, color='red', linestyle='--', alpha=0.7)
    
    axes[0].set_title('Data with Topology-Aware Adaptive Thresholds')
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
    
    # Plot 3: Strategies over time
    strategy_map = {'normal': 0, 'multimodal': 1, 'skewed': 2, 
                   'heavy_tailed': 3, 'clustered': 4}
    strategy_ints = [strategy_map.get(s, 0) for s in strategies]
    axes[2].plot(strategy_ints, 'g-', linewidth=2, alpha=0.8)
    axes[2].set_title('Adaptation Strategy Over Time')
    axes[2].set_ylabel('Strategy')
    axes[2].set_ylim(-0.5, 4.5)
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(['Normal', 'Multimodal', 'Skewed', 
                            'Heavy-Tailed', 'Clustered'])
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Threshold evolution
    axes[3].plot(threshold_i_history, 'r-', linewidth=2, label='Threshold i')
    axes[3].plot(threshold_q_history, 'orange', linewidth=2, label='Threshold q')
    axes[3].set_title('Threshold Evolution')
    axes[3].set_xlabel('Sample')
    axes[3].set_ylabel('Threshold Value')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('topology_adaptive_thresholds.png', dpi=150, bbox_inches='tight')
    print("Saved: topology_adaptive_thresholds.png")
    
    return states_array, threshold_history, topology_changes


def demonstrate_integration_with_scoring():
    """Demonstrate integration with feature scoring"""
    print("\n=== Integration with Feature Scoring ===")
    
    # Generate data with known patterns
    np.random.seed(42)
    n_samples = 1500
    
    # Create data with changing patterns
    data = []
    for i in range(n_samples):
        if i < 500:
            # Normal phase
            value = np.random.normal(0, 1)
        elif i < 1000:
            # Trending phase
            value = 0.01 * i + np.random.normal(0, 0.5)
        else:
            # Volatile phase
            value = np.random.normal(0, 2) + 0.5 * np.sin(i * 0.1)
        
        data.append(value)
    
    data = np.array(data)
    
    # Create integrated system
    threshold = TopologyAdaptiveThreshold(
        window_size=200,
        adaptation_rate=0.01,
        topology_update_interval=100
    )
    
    # Create feature scorer
    scorer = create_trading_scorer()
    
    # Process data
    states = []
    scores = []
    contexts = []
    
    print("Processing data with integrated topology-aware scoring...")
    
    for i, value in enumerate(data):
        # Update threshold
        threshold.update(value)
        state = threshold.encode(value)
        states.append(state)
        
        # Create context for scoring
        if i > 100:  # Need some history
            context = {
                'historical_values': data[max(0, i-100):i].tolist(),
                'historical_outcomes': np.diff(data[max(0, i-100):i+1]).tolist(),
                'sample_size': min(100, i),
                'age_seconds': 0,
                'missing_rate': 0.01,
                'noise_level': 0.1,
                'recent_values': data[max(0, i-10):i].tolist()
            }
            
            # Score the feature
            score, dim_scores = scorer.score_and_aggregate(value, context)
            scores.append(score)
            contexts.append(context.copy())
            
            # Record for analysis
            scorer.record_score(value, context, outcome=value)
        else:
            scores.append(0.5)  # Neutral score
    
    # Analyze results
    print(f"\nIntegrated System Results:")
    print(f"  Total samples: {len(data)}")
    print(f"  Average score: {np.mean(scores):.3f}")
    print(f"  Score std: {np.std(scores):.3f}")
    
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
    info = threshold.get_thresholds()
    axes[0].axhline(info['threshold_i'], color='red', linestyle='--', 
                    label=f'Threshold i = {info["threshold_i"]:.3f}')
    axes[0].axhline(info['threshold_q'], color='orange', linestyle='--', 
                    label=f'Threshold q = {info["threshold_q"]:.3f}')
    axes[0].set_title('Data with Integrated Topology-Aware Scoring')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: States
    states_array = StateArray(states)
    state_ints = states_array.to_ints()
    axes[1].plot(state_ints, 'o-', markersize=2, alpha=0.7)
    axes[1].set_title('Channel States')
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
    
    # Plot 4: Combined view
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    axes[3].plot(data_norm, alpha=0.5, label='Data (normalized)')
    axes[3].plot(scores, 'g-', linewidth=2, label='Scores')
    axes[3].plot(np.array(state_ints) / 3, 'r-', alpha=0.7, label='States (normalized)')
    axes[3].set_title('Combined View: Data, States, and Scores')
    axes[3].set_xlabel('Sample')
    axes[3].set_ylabel('Normalized Value')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('topology_scoring_integration.png', dpi=150, bbox_inches='tight')
    print("Saved: topology_scoring_integration.png")
    
    return states_array, scores, scorer


def demonstrate_real_world_application():
    """Demonstrate real-world application: financial data analysis"""
    print("\n=== Real-World Application: Financial Data Analysis ===")
    
    # Generate realistic financial data
    np.random.seed(42)
    n_days = 1000
    
    # Generate price data with different market regimes
    returns = []
    
    # Bull market (days 0-300)
    returns.extend(np.random.normal(0.002, 0.015, 300))
    
    # Sideways market (days 300-600)
    returns.extend(np.random.normal(0.000, 0.010, 300))
    
    # Bear market (days 600-800)
    returns.extend(np.random.normal(-0.001, 0.020, 200))
    
    # Volatile market (days 800-1000)
    returns.extend(np.random.normal(0.000, 0.025, 200))
    
    returns = np.array(returns)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create topology-aware system
    threshold = TopologyAdaptiveThreshold(
        window_size=100,
        adaptation_rate=0.02,
        topology_update_interval=50
    )
    
    # Process financial data
    signals = []
    states = []
    strategies = []
    
    print("Processing financial data with topology-aware system...")
    
    for i, price in enumerate(prices):
        # Use price change as signal
        if i > 0:
            price_change = (price - prices[i-1]) / prices[i-1]
            
            # Update threshold
            threshold.update(price_change)
            state = threshold.encode(price_change)
            signals.append(price_change)
            states.append(state)
            strategies.append(threshold._get_current_strategy())
    
    # Analyze results
    states_array = StateArray(states)
    
    print(f"\nFinancial Analysis Results:")
    print(f"  Total days: {len(prices)}")
    print(f"  State distribution: {states_array.count_by_state()}")
    print(f"  Strategy distribution: {dict(zip(*np.unique(strategies, return_counts=True)))}")
    
    # Calculate performance metrics
    signals_array = np.array(signals)
    positive_signals = signals_array[signals_array > 0]
    negative_signals = signals_array[signals_array < 0]
    
    print(f"  Positive signals: {len(positive_signals)} ({len(positive_signals)/len(signals)*100:.1f}%)")
    print(f"  Negative signals: {len(negative_signals)} ({len(negative_signals)/len(signals)*100:.1f}%)")
    print(f"  Average return: {np.mean(signals_array):.4f}")
    print(f"  Volatility: {np.std(signals_array):.4f}")
    
    # Plot financial analysis
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: Price series
    axes[0].plot(prices, 'b-', linewidth=1, alpha=0.8)
    axes[0].set_title('Price Series with Market Regimes')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Mark regime changes
    regime_changes = [300, 600, 800]
    for change in regime_changes:
        axes[0].axvline(change, color='red', linestyle='--', alpha=0.7)
    
    # Plot 2: Returns
    axes[1].plot(signals_array, 'g-', linewidth=1, alpha=0.8)
    axes[1].set_title('Daily Returns')
    axes[1].set_ylabel('Return')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: States
    state_ints = states_array.to_ints()
    axes[2].plot(state_ints, 'o-', markersize=2, alpha=0.7)
    axes[2].set_title('Channel States')
    axes[2].set_ylabel('State')
    axes[2].set_ylim(-0.5, 3.5)
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(['∅', 'δ', 'φ', 'ψ'])
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Strategies
    strategy_map = {'normal': 0, 'multimodal': 1, 'skewed': 2, 
                   'heavy_tailed': 3, 'clustered': 4}
    strategy_ints = [strategy_map.get(s, 0) for s in strategies]
    axes[3].plot(strategy_ints, 'purple', linewidth=2, alpha=0.8)
    axes[3].set_title('Adaptation Strategy Over Time')
    axes[3].set_xlabel('Day')
    axes[3].set_ylabel('Strategy')
    axes[3].set_ylim(-0.5, 4.5)
    axes[3].set_yticks(range(5))
    axes[3].set_yticklabels(['Normal', 'Multimodal', 'Skewed', 
                            'Heavy-Tailed', 'Clustered'])
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('financial_topology_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: financial_topology_analysis.png")


def demonstrate_topology_visualization():
    """Demonstrate topology visualization capabilities"""
    print("\n=== Topology Visualization ===")
    
    # Generate complex data
    data = generate_multimodal_data(1000, 3, 0.15)
    
    # Create threshold system
    threshold = TopologyAdaptiveThreshold(
        window_size=500,
        adaptation_rate=0.01,
        topology_update_interval=100
    )
    
    # Process data
    for value in data:
        threshold.update(value)
    
    # Create visualization
    try:
        fig = threshold.plot_topology_and_thresholds()
        plt.savefig('topology_visualization.png', dpi=150, bbox_inches='tight')
        print("Saved: topology_visualization.png")
    except ImportError:
        print("Matplotlib or scipy not available for visualization")


def main():
    """Main topology-aware adaptive thresholding example"""
    print("ChannelPy Topology-Aware Adaptive Thresholding Example")
    print("=" * 60)
    
    # 1. Topology analysis
    topology_results = demonstrate_topology_analysis()
    
    # 2. Adaptive thresholds
    states_array, threshold_history, topology_changes = demonstrate_adaptive_thresholds()
    
    # 3. Integration with scoring
    integrated_states, integrated_scores, integrated_scorer = demonstrate_integration_with_scoring()
    
    # 4. Real-world application
    demonstrate_real_world_application()
    
    # 5. Topology visualization
    demonstrate_topology_visualization()
    
    print("\n" + "=" * 60)
    print("Topology-aware adaptive thresholding example completed successfully!")
    print("\nGenerated files:")
    print("- topology_analysis.png: Analysis of different distribution types")
    print("- topology_adaptive_thresholds.png: Adaptive thresholding demonstration")
    print("- topology_scoring_integration.png: Integration with feature scoring")
    print("- financial_topology_analysis.png: Real-world financial application")
    print("- topology_visualization.png: Topology visualization")
    print("\nKey innovations demonstrated:")
    print("- Topology-aware threshold adaptation")
    print("- Real-time distribution shape analysis")
    print("- Adaptive strategies for different topologies")
    print("- Integration with multi-dimensional scoring")
    print("- Real-world applications in financial data")
    print("\nNext steps:")
    print("- Use topology-aware thresholds for your data")
    print("- Integrate with feature scoring systems")
    print("- Apply to real-world streaming data")
    print("- Customize adaptation strategies for your domain")


if __name__ == "__main__":
    main()







