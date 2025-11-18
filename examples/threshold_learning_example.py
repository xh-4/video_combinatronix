"""
Threshold Learning Example

This example demonstrates various threshold learning methods:
1. Statistical thresholds
2. Supervised threshold learning
3. Domain-specific thresholds
4. Optimal threshold finding
5. Multi-feature threshold learning
6. Threshold stability analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from channelpy.adaptive.thresholds import (
    StatisticalThresholds, SupervisedThresholds, DomainThresholds,
    OptimalThresholds, MultiFeatureThresholdLearner, ThresholdStabilityAnalyzer
)
from channelpy.examples.datasets import make_classification_data, make_trading_data
from channelpy import StateArray, PSI, DELTA, PHI, EMPTY


def generate_test_data():
    """Generate test data for threshold learning"""
    print("=== Generating Test Data ===")
    
    # Generate classification data
    X, y = make_classification_data(n_samples=500, n_features=2, n_classes=2, noise=0.3)
    
    # Generate trading data
    df = make_trading_data(n_samples=100, volatility=0.02)
    
    print(f"Classification data: {X.shape}, labels: {y.shape}")
    print(f"Trading data: {df.shape}")
    print(f"Price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")
    
    return X, y, df


def demonstrate_statistical_thresholds(X):
    """Demonstrate statistical threshold learning"""
    print("\n=== Statistical Thresholds ===")
    
    # Test different percentile combinations
    percentile_combinations = [
        (50, 75),   # Median and 75th percentile
        (25, 50),   # 25th and 50th percentile
        (75, 90),   # 75th and 90th percentile
    ]
    
    for perc_i, perc_q in percentile_combinations:
        learner = StatisticalThresholds(percentile_i=perc_i, percentile_q=perc_q)
        learner.fit(X[:, 0])  # Use first feature
        
        thresh_i, thresh_q = learner.get_thresholds()
        print(f"  Percentiles ({perc_i}, {perc_q}): i={thresh_i:.2f}, q={thresh_q:.2f}")
        
        # Show state distribution
        states = StateArray.from_bits(
            i=(X[:, 0] > thresh_i).astype(int),
            q=(X[:, 0] > thresh_q).astype(int)
        )
        print(f"    State distribution: {states.count_by_state()}")


def demonstrate_supervised_thresholds(X, y):
    """Demonstrate supervised threshold learning"""
    print("\n=== Supervised Thresholds ===")
    
    # Test different metrics
    metrics = ['mutual_info', 'correlation', 'accuracy']
    
    for metric in metrics:
        learner = SupervisedThresholds(metric=metric, n_candidates=20)
        learner.fit(X[:, 0], y)
        
        thresh_i, thresh_q = learner.get_thresholds()
        print(f"  Metric '{metric}': i={thresh_i:.2f}, q={thresh_q:.2f}")
        
        # Show state distribution
        states = StateArray.from_bits(
            i=(X[:, 0] > thresh_i).astype(int),
            q=(X[:, 0] > thresh_q).astype(int)
        )
        print(f"    State distribution: {states.count_by_state()}")


def demonstrate_domain_thresholds(df):
    """Demonstrate domain-specific thresholds"""
    print("\n=== Domain-Specific Thresholds ===")
    
    # Trading domain
    print("1. Trading domain:")
    trading_learner = DomainThresholds(domain='trading')
    
    # Test with RSI-like values (0-100)
    rsi_values = np.random.uniform(0, 100, 100)
    trading_learner.fit(rsi_values, feature_name='rsi')
    thresh_i, thresh_q = trading_learner.get_thresholds()
    print(f"   RSI thresholds: i={thresh_i:.1f}, q={thresh_q:.1f}")
    
    # Test with volume values
    volume_values = df['volume'].values
    trading_learner.fit(volume_values, feature_name='volume')
    thresh_i, thresh_q = trading_learner.get_thresholds()
    print(f"   Volume thresholds: i={thresh_i:.0f}, q={thresh_q:.0f}")
    
    # Medical domain
    print("\n2. Medical domain:")
    medical_learner = DomainThresholds(domain='medical')
    
    # Test with temperature values
    temp_values = np.random.normal(37, 1, 100)  # Normal body temperature
    medical_learner.fit(temp_values, feature_name='temperature')
    thresh_i, thresh_q = medical_learner.get_thresholds()
    print(f"   Temperature thresholds: i={thresh_i:.1f}°C, q={thresh_q:.1f}°C")
    
    # Custom domain rules
    print("\n3. Custom domain rules:")
    custom_learner = DomainThresholds(domain='generic')
    custom_learner.set_rule('custom_feature', threshold_i=0.5, threshold_q=0.8)
    custom_learner.fit(np.random.rand(100), feature_name='custom_feature')
    thresh_i, thresh_q = custom_learner.get_thresholds()
    print(f"   Custom thresholds: i={thresh_i:.1f}, q={thresh_q:.1f}")


def demonstrate_optimal_thresholds(X, y):
    """Demonstrate optimal threshold finding"""
    print("\n=== Optimal Thresholds ===")
    
    # Default objective (maximize correlation)
    print("1. Default objective (correlation):")
    learner = OptimalThresholds()
    learner.fit(X[:, 0], y)
    thresh_i, thresh_q = learner.get_thresholds()
    print(f"   Optimal thresholds: i={thresh_i:.2f}, q={thresh_q:.2f}")
    
    # Custom objective function
    print("\n2. Custom objective (entropy):")
    def entropy_objective(threshold, X, y):
        bits = (X > threshold).astype(int)
        p = np.mean(bits)
        if p == 0 or p == 1:
            return 0
        entropy = -p * np.log(p) - (1-p) * np.log(1-p)
        return entropy
    
    learner = OptimalThresholds(objective=entropy_objective)
    learner.fit(X[:, 0], y)
    thresh_i, thresh_q = learner.get_thresholds()
    print(f"   Entropy-optimal thresholds: i={thresh_i:.2f}, q={thresh_q:.2f}")
    
    # Show state distribution
    states = StateArray.from_bits(
        i=(X[:, 0] > thresh_i).astype(int),
        q=(X[:, 0] > thresh_q).astype(int)
    )
    print(f"   State distribution: {states.count_by_state()}")


def demonstrate_multi_feature_learning(X, y):
    """Demonstrate multi-feature threshold learning"""
    print("\n=== Multi-Feature Threshold Learning ===")
    
    # Create multi-feature learner
    multi_learner = MultiFeatureThresholdLearner()
    
    # Add different learners for different features
    multi_learner.add_feature('feature_1', StatisticalThresholds(percentile_i=50, percentile_q=75))
    multi_learner.add_feature('feature_2', SupervisedThresholds(metric='mutual_info'))
    
    # Prepare data dictionary
    X_dict = {
        'feature_1': X[:, 0],
        'feature_2': X[:, 1]
    }
    
    # Fit all learners
    multi_learner.fit(X_dict, y)
    
    # Get thresholds for each feature
    print("Learned thresholds:")
    for feature_name in ['feature_1', 'feature_2']:
        thresh_i, thresh_q = multi_learner.get_thresholds(feature_name)
        print(f"  {feature_name}: i={thresh_i:.2f}, q={thresh_q:.2f}")
    
    # Get all thresholds
    all_thresholds = multi_learner.get_all_thresholds()
    print(f"\nAll thresholds: {all_thresholds}")


def demonstrate_stability_analysis(X, y):
    """Demonstrate threshold stability analysis"""
    print("\n=== Threshold Stability Analysis ===")
    
    # Test different learners
    learners = [
        ('Statistical', StatisticalThresholds(percentile_i=50, percentile_q=75)),
        ('Supervised', SupervisedThresholds(metric='mutual_info')),
        ('Optimal', OptimalThresholds())
    ]
    
    analyzer = ThresholdStabilityAnalyzer(n_bootstrap=50, confidence=0.95)
    
    for name, learner in learners:
        print(f"\n{name} Thresholds:")
        stability = analyzer.analyze(X[:, 0], y, learner)
        
        print(f"  Threshold i: {stability['mean_threshold_i']:.2f} ± {stability['std_threshold_i']:.2f}")
        print(f"  Threshold q: {stability['mean_threshold_q']:.2f} ± {stability['std_threshold_q']:.2f}")
        print(f"  Stability score: {stability['stability_score']:.3f}")
        print(f"  CI for i: [{stability['ci_threshold_i'][0]:.2f}, {stability['ci_threshold_i'][1]:.2f}]")
        print(f"  CI for q: [{stability['ci_threshold_q'][0]:.2f}, {stability['ci_threshold_q'][1]:.2f}]")


def compare_threshold_methods(X, y):
    """Compare different threshold learning methods"""
    print("\n=== Threshold Method Comparison ===")
    
    methods = [
        ('Statistical (50,75)', StatisticalThresholds(percentile_i=50, percentile_q=75)),
        ('Statistical (25,50)', StatisticalThresholds(percentile_i=25, percentile_q=50)),
        ('Supervised (MI)', SupervisedThresholds(metric='mutual_info')),
        ('Supervised (Corr)', SupervisedThresholds(metric='correlation')),
        ('Optimal', OptimalThresholds())
    ]
    
    print(f"{'Method':<20} {'Threshold i':<12} {'Threshold q':<12} {'States':<20}")
    print("-" * 70)
    
    for name, learner in methods:
        try:
            learner.fit(X[:, 0], y)
            thresh_i, thresh_q = learner.get_thresholds()
            
            # Generate states
            states = StateArray.from_bits(
                i=(X[:, 0] > thresh_i).astype(int),
                q=(X[:, 0] > thresh_q).astype(int)
            )
            state_counts = states.count_by_state()
            
            print(f"{name:<20} {thresh_i:<12.2f} {thresh_q:<12.2f} {str(state_counts):<20}")
            
        except Exception as e:
            print(f"{name:<20} {'ERROR':<12} {'ERROR':<12} {str(e):<20}")


def visualize_threshold_learning(X, y):
    """Create visualizations for threshold learning"""
    print("\n=== Creating Visualizations ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Data distribution
    axes[0, 0].hist(X[:, 0], bins=30, alpha=0.7, color='blue', label='Feature 1')
    axes[0, 0].set_title('Feature Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Threshold comparison
    methods = [
        ('Statistical', StatisticalThresholds(percentile_i=50, percentile_q=75)),
        ('Supervised', SupervisedThresholds(metric='mutual_info')),
        ('Optimal', OptimalThresholds())
    ]
    
    colors = ['red', 'green', 'blue']
    for i, (name, learner) in enumerate(methods):
        try:
            learner.fit(X[:, 0], y)
            thresh_i, thresh_q = learner.get_thresholds()
            
            axes[0, 1].axvline(thresh_i, color=colors[i], linestyle='--', alpha=0.7, label=f'{name} i-threshold')
            axes[0, 1].axvline(thresh_q, color=colors[i], linestyle='-', alpha=0.7, label=f'{name} q-threshold')
        except:
            continue
    
    axes[0, 1].hist(X[:, 0], bins=30, alpha=0.3, color='gray')
    axes[0, 1].set_title('Threshold Comparison')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: State distribution for different methods
    state_distributions = {}
    for name, learner in methods:
        try:
            learner.fit(X[:, 0], y)
            thresh_i, thresh_q = learner.get_thresholds()
            
            states = StateArray.from_bits(
                i=(X[:, 0] > thresh_i).astype(int),
                q=(X[:, 0] > thresh_q).astype(int)
            )
            state_distributions[name] = states.count_by_state()
        except:
            continue
    
    # Plot state distributions
    if state_distributions:
        method_names = list(state_distributions.keys())
        state_names = ['∅', 'δ', 'φ', 'ψ']
        
        x = np.arange(len(state_names))
        width = 0.25
        
        for i, (method, counts) in enumerate(state_distributions.items()):
            values = [counts.get(state, 0) for state in [EMPTY, DELTA, PHI, PSI]]
            axes[1, 0].bar(x + i*width, values, width, label=method, alpha=0.7)
        
        axes[1, 0].set_title('State Distribution by Method')
        axes[1, 0].set_xlabel('State')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(state_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Stability analysis
    analyzer = ThresholdStabilityAnalyzer(n_bootstrap=30, confidence=0.95)
    learner = StatisticalThresholds(percentile_i=50, percentile_q=75)
    stability = analyzer.analyze(X[:, 0], y, learner)
    
    # Plot confidence intervals
    axes[1, 1].errorbar([0, 1], 
                       [stability['mean_threshold_i'], stability['mean_threshold_q']],
                       yerr=[stability['std_threshold_i'], stability['std_threshold_q']],
                       fmt='o', capsize=5, capthick=2, markersize=8)
    axes[1, 1].set_title('Threshold Stability')
    axes[1, 1].set_xlabel('Threshold Type')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['i-threshold', 'q-threshold'])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_learning_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: threshold_learning_analysis.png")


def demonstrate_real_world_example(df):
    """Demonstrate threshold learning with real-world trading data"""
    print("\n=== Real-World Example: Trading Data ===")
    
    # Calculate RSI-like indicator
    prices = df['close'].values
    returns = np.diff(prices, prepend=prices[0])
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    
    # Simple RSI calculation
    window = 14
    rsi_values = []
    for i in range(len(prices)):
        if i < window:
            rsi_values.append(50)  # Neutral
        else:
            avg_gain = np.mean(gains[i-window+1:i+1])
            avg_loss = np.mean(losses[i-window+1:i+1])
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
    
    rsi_values = np.array(rsi_values)
    
    print(f"RSI range: [{rsi_values.min():.1f}, {rsi_values.max():.1f}]")
    
    # Test different threshold methods on RSI
    methods = [
        ('Statistical', StatisticalThresholds(percentile_i=50, percentile_q=75)),
        ('Domain', DomainThresholds(domain='trading')),
        ('Supervised', SupervisedThresholds(metric='mutual_info'))
    ]
    
    # Create binary labels based on price movement
    future_returns = np.diff(prices[1:], prepend=prices[1])
    labels = (future_returns > 0).astype(int)
    
    print("\nThreshold learning on RSI data:")
    for name, learner in methods:
        try:
            if name == 'Domain':
                learner.fit(rsi_values, feature_name='rsi')
            else:
                learner.fit(rsi_values, labels)
            
            thresh_i, thresh_q = learner.get_thresholds()
            print(f"  {name}: i={thresh_i:.1f}, q={thresh_q:.1f}")
            
            # Generate states
            states = StateArray.from_bits(
                i=(rsi_values > thresh_i).astype(int),
                q=(rsi_values > thresh_q).astype(int)
            )
            print(f"    State distribution: {states.count_by_state()}")
            
        except Exception as e:
            print(f"  {name}: Error - {e}")


def main():
    """Main threshold learning example function"""
    print("ChannelPy Threshold Learning Example")
    print("=" * 50)
    
    # 1. Generate test data
    X, y, df = generate_test_data()
    
    # 2. Demonstrate statistical thresholds
    demonstrate_statistical_thresholds(X)
    
    # 3. Demonstrate supervised thresholds
    demonstrate_supervised_thresholds(X, y)
    
    # 4. Demonstrate domain-specific thresholds
    demonstrate_domain_thresholds(df)
    
    # 5. Demonstrate optimal thresholds
    demonstrate_optimal_thresholds(X, y)
    
    # 6. Demonstrate multi-feature learning
    demonstrate_multi_feature_learning(X, y)
    
    # 7. Demonstrate stability analysis
    demonstrate_stability_analysis(X, y)
    
    # 8. Compare methods
    compare_threshold_methods(X, y)
    
    # 9. Create visualizations
    visualize_threshold_learning(X, y)
    
    # 10. Real-world example
    demonstrate_real_world_example(df)
    
    print("\n" + "=" * 50)
    print("Threshold learning example completed successfully!")
    print("\nGenerated files:")
    print("- threshold_learning_analysis.png: Comprehensive analysis plots")
    print("\nNext steps:")
    print("- Experiment with different threshold learning methods")
    print("- Try custom objective functions for optimal thresholds")
    print("- Test stability analysis with different datasets")
    print("- Integrate threshold learning into your pipelines")


if __name__ == "__main__":
    main()







