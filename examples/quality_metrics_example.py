"""
Quality Metrics Example

This example demonstrates quality metrics for channel encodings:
1. Basic quality metrics
2. Distribution quality analysis
3. Discrimination power evaluation
4. Threshold stability analysis
5. Encoding robustness testing
6. Comprehensive quality reporting
"""

import numpy as np
import matplotlib.pyplot as plt
from channelpy.metrics.quality import (
    encoding_accuracy, state_distribution_quality, discrimination_power,
    threshold_stability, information_content, encoding_consistency,
    state_transition_quality, channel_correlation, encoding_robustness,
    comprehensive_quality_report, compare_encodings
)
from channelpy import StateArray, EMPTY, DELTA, PHI, PSI
from channelpy.examples.datasets import make_classification_data


def generate_test_data():
    """Generate test data for quality metrics"""
    print("=== Generating Test Data ===")
    
    # Generate classification data
    X, y = make_classification_data(n_samples=500, n_features=2, n_classes=2, noise=0.3)
    
    # Generate different encodings
    encodings = {}
    
    # Encoding 1: Simple threshold
    threshold1 = np.median(X[:, 0])
    states1 = StateArray.from_bits(
        i=(X[:, 0] > threshold1).astype(int),
        q=(X[:, 0] > threshold1 * 1.5).astype(int)
    )
    encodings['simple_threshold'] = states1
    
    # Encoding 2: Percentile-based
    threshold2 = np.percentile(X[:, 0], 50)
    q_threshold2 = np.percentile(X[:, 0], 75)
    states2 = StateArray.from_bits(
        i=(X[:, 0] > threshold2).astype(int),
        q=(X[:, 0] > q_threshold2).astype(int)
    )
    encodings['percentile_based'] = states2
    
    # Encoding 3: Balanced
    threshold3 = np.percentile(X[:, 0], 25)
    q_threshold3 = np.percentile(X[:, 0], 75)
    states3 = StateArray.from_bits(
        i=(X[:, 0] > threshold3).astype(int),
        q=(X[:, 0] > q_threshold3).astype(int)
    )
    encodings['balanced'] = states3
    
    # Encoding 4: Random (for comparison)
    states4 = StateArray.from_bits(
        i=np.random.randint(0, 2, len(X)),
        q=np.random.randint(0, 2, len(X))
    )
    encodings['random'] = states4
    
    print(f"Generated {len(X)} samples with {len(encodings)} different encodings")
    for name, states in encodings.items():
        print(f"  {name}: {states.count_by_state()}")
    
    return X, y, encodings


def demonstrate_basic_metrics(encodings):
    """Demonstrate basic quality metrics"""
    print("\n=== Basic Quality Metrics ===")
    
    for name, states in encodings.items():
        print(f"\n{name.upper()} Encoding:")
        
        # Information content
        info_content = information_content(states)
        print(f"  Information content: {info_content:.3f}")
        
        # Distribution quality
        dist_quality = state_distribution_quality(states)
        print(f"  Balance score: {dist_quality['balance_score']:.3f}")
        print(f"  PSI ratio: {dist_quality['psi_ratio']:.3f}")
        print(f"  PSI quality: {dist_quality['psi_quality']:.3f}")
        print(f"  Overall quality: {dist_quality['overall_quality']:.3f}")
        
        # Transition quality
        trans_quality = state_transition_quality(states)
        print(f"  Transition regularity: {trans_quality['transition_regularity']:.3f}")
        print(f"  Number of transitions: {trans_quality['num_transitions']}")


def demonstrate_discrimination_power(encodings, labels):
    """Demonstrate discrimination power analysis"""
    print("\n=== Discrimination Power Analysis ===")
    
    for name, states in encodings.items():
        discrimination = discrimination_power(states, labels)
        print(f"{name}: {discrimination:.3f}")
    
    # Find best encoding for discrimination
    best_discrimination = 0.0
    best_encoding = None
    
    for name, states in encodings.items():
        discrimination = discrimination_power(states, labels)
        if discrimination > best_discrimination:
            best_discrimination = discrimination
            best_encoding = name
    
    print(f"\nBest encoding for discrimination: {best_encoding} ({best_discrimination:.3f})")


def demonstrate_threshold_stability():
    """Demonstrate threshold stability analysis"""
    print("\n=== Threshold Stability Analysis ===")
    
    # Simulate different threshold histories
    stable_history = np.random.normal(0.5, 0.01, 100)  # Stable thresholds
    unstable_history = np.random.normal(0.5, 0.1, 100)  # Unstable thresholds
    adaptive_history = np.cumsum(np.random.normal(0, 0.05, 100)) + 0.5  # Adaptive thresholds
    
    histories = {
        'stable': stable_history.tolist(),
        'unstable': unstable_history.tolist(),
        'adaptive': adaptive_history.tolist()
    }
    
    for name, history in histories.items():
        stability = threshold_stability(history)
        print(f"\n{name.upper()} Thresholds:")
        print(f"  Mean: {stability['mean']:.3f}")
        print(f"  Std: {stability['std']:.3f}")
        print(f"  Coefficient of variation: {stability['coefficient_of_variation']:.3f}")
        print(f"  Stability score: {stability['stability_score']:.3f}")


def demonstrate_encoding_consistency(encodings, X):
    """Demonstrate encoding consistency analysis"""
    print("\n=== Encoding Consistency Analysis ===")
    
    for name, states in encodings.items():
        # Use median as threshold for consistency check
        threshold = np.median(X[:, 0])
        
        consistency = encoding_consistency(states, X[:, 0], threshold)
        print(f"\n{name.upper()} Encoding:")
        print(f"  i-bit consistency: {consistency['i_bit_consistency']:.3f}")
        print(f"  q-bit consistency: {consistency['q_bit_consistency']:.3f}")
        print(f"  Overall consistency: {consistency['overall_consistency']:.3f}")


def demonstrate_channel_correlation(encodings):
    """Demonstrate channel correlation analysis"""
    print("\n=== Channel Correlation Analysis ===")
    
    # Create multiple channels
    channels = {
        'channel_1': encodings['simple_threshold'],
        'channel_2': encodings['percentile_based'],
        'channel_3': encodings['balanced']
    }
    
    correlation = channel_correlation(channels)
    print(f"Average correlation: {correlation['average_correlation']:.3f}")
    print("Pairwise correlations:")
    for pair, corr in correlation['correlations'].items():
        print(f"  {pair}: {corr:.3f}")


def demonstrate_encoding_robustness(encodings, X):
    """Demonstrate encoding robustness testing"""
    print("\n=== Encoding Robustness Testing ===")
    
    # Test robustness for each encoding
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    
    for name, states in encodings.items():
        print(f"\n{name.upper()} Encoding:")
        
        # Use median as threshold
        threshold = np.median(X[:, 0])
        
        robustness = encoding_robustness(states, X[:, 0], noise_levels)
        print(f"  Average robustness: {robustness['average_robustness']:.3f}")
        print(f"  Robustness trend: {robustness['robustness_trend']:.3f}")
        
        # Show robustness at different noise levels
        for noise, score in zip(noise_levels, robustness['robustness_scores']):
            print(f"    Noise {noise:.2f}: {score:.3f}")


def demonstrate_comprehensive_reporting(encodings, X, y):
    """Demonstrate comprehensive quality reporting"""
    print("\n=== Comprehensive Quality Reporting ===")
    
    # Generate comprehensive report for each encoding
    for name, states in encodings.items():
        print(f"\n{name.upper()} Encoding Report:")
        
        # Use median as threshold
        threshold = np.median(X[:, 0])
        
        # Generate threshold history (simulate)
        threshold_history = np.random.normal(threshold, 0.05, 50).tolist()
        
        report = comprehensive_quality_report(
            states=states,
            labels=y,
            feature_values=X[:, 0],
            threshold=threshold,
            threshold_history=threshold_history
        )
        
        print(f"  Overall Score: {report['overall_score']:.3f}")
        print(f"  Distribution Quality: {report['distribution_quality']['overall_quality']:.3f}")
        print(f"  Information Content: {report['basic_metrics']['information_content']:.3f}")
        
        if 'discrimination_analysis' in report:
            print(f"  Discrimination Power: {report['discrimination_analysis']['discrimination_power']:.3f}")
        
        if 'consistency_analysis' in report:
            print(f"  Consistency: {report['consistency_analysis']['overall_consistency']:.3f}")
        
        if 'stability_analysis' in report:
            print(f"  Stability: {report['stability_analysis']['stability_score']:.3f}")
        
        print(f"  Transition Quality: {report['transition_quality']['overall_quality']:.3f}")


def demonstrate_encoding_comparison(encodings, y):
    """Demonstrate encoding comparison"""
    print("\n=== Encoding Comparison ===")
    
    comparison = compare_encodings(encodings, y)
    
    print("Encoding Rankings:")
    for i, name in enumerate(comparison['ranking']):
        score = comparison['results'][name]['overall_score']
        print(f"  {i+1}. {name}: {score:.3f}")
    
    print(f"\nBest Encoding: {comparison['best_encoding']}")
    
    # Show detailed comparison
    print("\nDetailed Comparison:")
    for name, results in comparison['results'].items():
        print(f"\n{name.upper()}:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.3f}")


def create_quality_visualization(encodings, comparison):
    """Create visualization of quality metrics"""
    print("\n=== Creating Quality Visualization ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Distribution quality comparison
    names = list(encodings.keys())
    dist_qualities = []
    trans_qualities = []
    info_contents = []
    
    for name in names:
        states = encodings[name]
        dist_quality = state_distribution_quality(states)
        trans_quality = state_transition_quality(states)
        info_content = information_content(states)
        
        dist_qualities.append(dist_quality['overall_quality'])
        trans_qualities.append(trans_quality['overall_quality'])
        info_contents.append(info_content)
    
    x = np.arange(len(names))
    width = 0.25
    
    axes[0, 0].bar(x - width, dist_qualities, width, label='Distribution Quality', alpha=0.7)
    axes[0, 0].bar(x, trans_qualities, width, label='Transition Quality', alpha=0.7)
    axes[0, 0].bar(x + width, info_contents, width, label='Information Content', alpha=0.7)
    
    axes[0, 0].set_title('Quality Metrics Comparison')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: State distribution for each encoding
    for i, (name, states) in enumerate(encodings.items()):
        counts = states.count_by_state()
        state_names = ['∅', 'δ', 'φ', 'ψ']
        values = [counts[EMPTY], counts[DELTA], counts[PHI], counts[PSI]]
        
        axes[0, 1].bar([j + i*0.2 for j in range(4)], values, 
                      width=0.2, label=name, alpha=0.7)
    
    axes[0, 1].set_title('State Distribution by Encoding')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xlabel('State')
    axes[0, 1].set_xticks(range(4))
    axes[0, 1].set_xticklabels(state_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Overall scores
    overall_scores = [comparison['results'][name]['overall_score'] for name in names]
    colors = ['green' if score == max(overall_scores) else 'skyblue' for score in overall_scores]
    
    axes[0, 2].bar(names, overall_scores, color=colors, alpha=0.7)
    axes[0, 2].set_title('Overall Quality Scores')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Discrimination power
    discrimination_powers = []
    for name in names:
        states = encodings[name]
        discrimination = discrimination_power(states, y)
        discrimination_powers.append(discrimination)
    
    axes[1, 0].bar(names, discrimination_powers, color='orange', alpha=0.7)
    axes[1, 0].set_title('Discrimination Power')
    axes[1, 0].set_ylabel('Discrimination Power')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Robustness testing
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    for name, states in encodings.items():
        robustness = encoding_robustness(states, X[:, 0], noise_levels)
        axes[1, 1].plot(noise_levels, robustness['robustness_scores'], 
                       marker='o', label=name, linewidth=2)
    
    axes[1, 1].set_title('Encoding Robustness')
    axes[1, 1].set_xlabel('Noise Level')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Quality radar chart
    metrics = ['Distribution', 'Transition', 'Information', 'Discrimination']
    values = [dist_qualities, trans_qualities, info_contents, discrimination_powers]
    
    # Normalize values to 0-1 range
    normalized_values = []
    for metric_values in values:
        max_val = max(metric_values) if max(metric_values) > 0 else 1
        normalized = [v / max_val for v in metric_values]
        normalized_values.append(normalized)
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, name in enumerate(names):
        values_radar = [normalized_values[j][i] for j in range(len(metrics))]
        values_radar += values_radar[:1]  # Complete the circle
        
        axes[1, 2].plot(angles, values_radar, 'o-', linewidth=2, label=name)
        axes[1, 2].fill(angles, values_radar, alpha=0.25)
    
    axes[1, 2].set_xticks(angles[:-1])
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Quality Radar Chart')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quality_metrics_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: quality_metrics_analysis.png")


def demonstrate_real_world_application():
    """Demonstrate real-world application of quality metrics"""
    print("\n=== Real-World Application ===")
    
    # Simulate real-world scenario: choosing best encoding for a specific task
    print("Scenario: Choosing best encoding for classification task")
    
    # Generate data with known structure
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Test different encodings
    encodings = {}
    
    # Encoding 1: Feature 1 only
    threshold1 = np.median(X[:, 0])
    states1 = StateArray.from_bits(
        i=(X[:, 0] > threshold1).astype(int),
        q=(X[:, 0] > threshold1 * 1.5).astype(int)
    )
    encodings['feature_1_only'] = states1
    
    # Encoding 2: Feature 2 only
    threshold2 = np.median(X[:, 1])
    states2 = StateArray.from_bits(
        i=(X[:, 1] > threshold2).astype(int),
        q=(X[:, 1] > threshold2 * 1.5).astype(int)
    )
    encodings['feature_2_only'] = states2
    
    # Encoding 3: Combined features
    combined = X[:, 0] + X[:, 1]
    threshold3 = np.median(combined)
    states3 = StateArray.from_bits(
        i=(combined > threshold3).astype(int),
        q=(combined > threshold3 * 1.5).astype(int)
    )
    encodings['combined_features'] = states3
    
    # Compare encodings
    comparison = compare_encodings(encodings, y)
    
    print("Encoding comparison for classification task:")
    for name, results in comparison['results'].items():
        print(f"  {name}: {results['overall_score']:.3f}")
    
    print(f"\nBest encoding: {comparison['best_encoding']}")
    
    # Show why this encoding is best
    best_encoding = comparison['best_encoding']
    best_results = comparison['results'][best_encoding]
    
    print(f"\nWhy {best_encoding} is best:")
    print(f"  Discrimination power: {best_results['discrimination_power']:.3f}")
    print(f"  Distribution quality: {best_results['distribution_quality']:.3f}")
    print(f"  Transition quality: {best_results['transition_quality']:.3f}")


def main():
    """Main quality metrics example function"""
    print("ChannelPy Quality Metrics Example")
    print("=" * 50)
    
    # 1. Generate test data
    X, y, encodings = generate_test_data()
    
    # 2. Basic metrics
    demonstrate_basic_metrics(encodings)
    
    # 3. Discrimination power
    demonstrate_discrimination_power(encodings, y)
    
    # 4. Threshold stability
    demonstrate_threshold_stability()
    
    # 5. Encoding consistency
    demonstrate_encoding_consistency(encodings, X)
    
    # 6. Channel correlation
    demonstrate_channel_correlation(encodings)
    
    # 7. Encoding robustness
    demonstrate_encoding_robustness(encodings, X)
    
    # 8. Comprehensive reporting
    demonstrate_comprehensive_reporting(encodings, X, y)
    
    # 9. Encoding comparison
    demonstrate_encoding_comparison(encodings, y)
    
    # 10. Visualization
    comparison = compare_encodings(encodings, y)
    create_quality_visualization(encodings, comparison)
    
    # 11. Real-world application
    demonstrate_real_world_application()
    
    print("\n" + "=" * 50)
    print("Quality metrics example completed successfully!")
    print("\nGenerated files:")
    print("- quality_metrics_analysis.png: Comprehensive quality analysis")
    print("\nNext steps:")
    print("- Use quality metrics to evaluate your encodings")
    print("- Compare different encoding strategies")
    print("- Optimize encodings based on quality metrics")
    print("- Integrate quality assessment into your pipelines")


if __name__ == "__main__":
    main()







