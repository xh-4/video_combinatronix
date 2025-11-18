"""
Multi-Scale Adaptive Thresholding Example

This example demonstrates the sophisticated multi-scale threshold tracking system
that maintains thresholds at multiple timescales to detect regime changes and
adapt appropriately to different volatility regimes.

Features demonstrated:
1. Multi-scale threshold tracking (fast, medium, slow)
2. Regime change detection
3. Adaptive threshold selection based on regime
4. Divergence analysis between scales
5. Real-world applications with regime detection
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

# Import multiscale components
from channelpy import (
    MultiScaleAdaptiveThreshold, RegimeType, RegimeChange,
    StateArray, EMPTY, DELTA, PHI, PSI
)


def generate_regime_data(n_samples: int = 5000) -> np.ndarray:
    """Generate data with multiple regime changes"""
    np.random.seed(42)
    
    data = []
    
    # Regime 1: Stable (samples 0-1000)
    data.extend(np.random.normal(0, 0.5, 1000))
    
    # Regime 2: Trending (samples 1000-2000)
    trend = np.linspace(0, 2, 1000)
    noise = np.random.normal(0, 0.3, 1000)
    data.extend(trend + noise)
    
    # Regime 3: Volatile (samples 2000-3000)
    data.extend(np.random.normal(0, 2.0, 1000))
    
    # Regime 4: Mean reverting (samples 3000-4000)
    # Generate mean-reverting process
    mean_reverting = [0]
    for i in range(999):
        prev = mean_reverting[-1]
        new_val = 0.9 * prev + np.random.normal(0, 0.5)
        mean_reverting.append(new_val)
    data.extend(mean_reverting)
    
    # Regime 5: Transitioning (samples 4000-5000)
    # Gradual change from one state to another
    transition = np.linspace(0, 1, 1000)
    noise = np.random.normal(0, 0.4, 1000)
    data.extend(transition + noise)
    
    return np.array(data)


def generate_financial_data(n_days: int = 2000) -> np.ndarray:
    """Generate realistic financial data with market regimes"""
    np.random.seed(42)
    
    returns = []
    
    # Bull market (days 0-500)
    returns.extend(np.random.normal(0.002, 0.015, 500))
    
    # Sideways market (days 500-1000)
    returns.extend(np.random.normal(0.000, 0.010, 500))
    
    # Bear market (days 1000-1500)
    returns.extend(np.random.normal(-0.001, 0.020, 500))
    
    # High volatility (days 1500-2000)
    returns.extend(np.random.normal(0.000, 0.030, 500))
    
    returns = np.array(returns)
    prices = 100 * np.exp(np.cumsum(returns))
    
    return prices, returns


def demonstrate_basic_multiscale():
    """Demonstrate basic multiscale threshold tracking"""
    print("=== Basic Multi-Scale Threshold Tracking ===")
    
    # Generate data with regime changes
    data = generate_regime_data(3000)
    
    # Create multiscale tracker
    tracker = MultiScaleAdaptiveThreshold(
        use_topology=True,
        fast_window=100,
        medium_window=500,
        slow_window=2000
    )
    
    # Process data
    states = []
    regime_changes = []
    
    print("Processing data with multi-scale tracking...")
    
    for i, value in enumerate(data):
        tracker.update(value)
        state = tracker.encode_adaptive(value)
        states.append(state)
        
        # Check for regime changes
        if tracker.regime_changed():
            change = tracker.get_last_regime_change()
            regime_changes.append(change)
            print(f"  Regime change at sample {i}: {change.from_regime.value} → {change.to_regime.value} "
                  f"(confidence: {change.confidence:.3f})")
        
        # Print progress
        if (i + 1) % 1000 == 0:
            regime_info = tracker.get_regime_info()
            print(f"  Sample {i+1}: regime={regime_info['current_regime']}, "
                  f"changes={regime_info['num_regime_changes']}")
    
    # Analyze results
    states_array = StateArray(states)
    
    print(f"\nResults:")
    print(f"  Total samples: {len(data)}")
    print(f"  Regime changes detected: {len(regime_changes)}")
    print(f"  Final regime: {tracker.get_current_regime().value}")
    print(f"  State distribution: {states_array.count_by_state()}")
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: Data with regime annotations
    axes[0].plot(data, alpha=0.7, label='Data')
    
    # Mark regime changes
    for change in regime_changes:
        axes[0].axvline(change.timestamp, color='red', linestyle='--', alpha=0.7)
    
    axes[0].set_title('Data with Regime Change Detection')
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
    
    # Plot 3: Thresholds across scales
    thresholds = tracker.get_all_thresholds()
    axes[2].axhline(thresholds['fast']['threshold_i'], 
                    label='Fast i', linestyle='--', color='lightblue', linewidth=2)
    axes[2].axhline(thresholds['fast']['threshold_q'], 
                    label='Fast q', linestyle='--', color='blue', linewidth=2)
    axes[2].axhline(thresholds['medium']['threshold_i'], 
                    label='Medium i', linestyle='--', color='lightgreen', linewidth=2)
    axes[2].axhline(thresholds['medium']['threshold_q'], 
                    label='Medium q', linestyle='--', color='green', linewidth=2)
    axes[2].axhline(thresholds['slow']['threshold_i'], 
                    label='Slow i', linestyle='--', color='lightcoral', linewidth=2)
    axes[2].axhline(thresholds['slow']['threshold_q'], 
                    label='Slow q', linestyle='--', color='red', linewidth=2)
    axes[2].set_title('Multi-Scale Thresholds')
    axes[2].set_ylabel('Threshold Value')
    axes[2].legend(loc='upper right', ncol=3)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Regime timeline
    regime_colors = {
        RegimeType.STABLE: 'green',
        RegimeType.TRANSITIONING: 'yellow',
        RegimeType.VOLATILE: 'red',
        RegimeType.TRENDING: 'blue',
        RegimeType.MEAN_REVERTING: 'purple',
        RegimeType.UNKNOWN: 'gray'
    }
    
    for i, change in enumerate(regime_changes):
        start = change.timestamp
        end = regime_changes[i+1].timestamp if i+1 < len(regime_changes) else len(data)
        
        axes[3].axvspan(start, end, 
                        color=regime_colors.get(change.to_regime, 'gray'),
                        alpha=0.3,
                        label=change.to_regime.value if i == 0 else "")
    
    axes[3].set_title('Regime Evolution')
    axes[3].set_xlabel('Sample')
    axes[3].set_ylabel('Regime')
    axes[3].set_ylim(-0.5, 0.5)
    axes[3].set_yticks([])
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiscale_basic.png', dpi=150, bbox_inches='tight')
    print("Saved: multiscale_basic.png")
    
    return states_array, regime_changes, tracker


def demonstrate_divergence_analysis():
    """Demonstrate divergence analysis between scales"""
    print("\n=== Divergence Analysis Between Scales ===")
    
    # Generate data with known regime changes
    data = generate_regime_data(2000)
    
    # Create tracker
    tracker = MultiScaleAdaptiveThreshold(
        use_topology=True,
        fast_window=50,
        medium_window=200,
        slow_window=1000
    )
    
    # Process data and collect divergence history
    divergence_history = []
    regime_history = []
    
    print("Processing data and analyzing divergences...")
    
    for i, value in enumerate(data):
        tracker.update(value)
        
        # Collect divergence data
        if tracker.divergence_history:
            divergence_history.append(tracker.divergence_history[-1])
        
        # Collect regime data
        if tracker.regime_changed():
            regime_history.append(tracker.get_last_regime_change())
        
        # Print progress
        if (i + 1) % 500 == 0:
            regime_info = tracker.get_regime_info()
            print(f"  Sample {i+1}: regime={regime_info['current_regime']}, "
                  f"divergences={len(divergence_history)}")
    
    # Analyze divergences
    if divergence_history:
        fast_medium_divs = [d['fast_medium'] for d in divergence_history]
        fast_slow_divs = [d['fast_slow'] for d in divergence_history]
        medium_slow_divs = [d['medium_slow'] for d in divergence_history]
        
        print(f"\nDivergence Analysis:")
        print(f"  Fast-Medium divergence: mean={np.mean(fast_medium_divs):.3f}, std={np.std(fast_medium_divs):.3f}")
        print(f"  Fast-Slow divergence: mean={np.mean(fast_slow_divs):.3f}, std={np.std(fast_slow_divs):.3f}")
        print(f"  Medium-Slow divergence: mean={np.mean(medium_slow_divs):.3f}, std={np.std(medium_slow_divs):.3f}")
    
    # Plot divergence analysis
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Divergences over time
    if divergence_history:
        updates = [d['update_count'] for d in divergence_history]
        
        axes[0].plot(updates, fast_medium_divs, label='Fast-Medium', linewidth=2, alpha=0.8)
        axes[0].plot(updates, fast_slow_divs, label='Fast-Slow', linewidth=2, alpha=0.8)
        axes[0].plot(updates, medium_slow_divs, label='Medium-Slow', linewidth=2, alpha=0.8)
        
        # Add threshold lines
        axes[0].axhline(0.5, color='orange', linestyle=':', alpha=0.5, label='Low threshold')
        axes[0].axhline(2.0, color='red', linestyle=':', alpha=0.5, label='High threshold')
        
        axes[0].set_title('Scale Divergences Over Time')
        axes[0].set_ylabel('Divergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Divergence distributions
    if divergence_history:
        axes[1].hist(fast_medium_divs, bins=30, alpha=0.7, label='Fast-Medium', color='blue')
        axes[1].hist(fast_slow_divs, bins=30, alpha=0.7, label='Fast-Slow', color='red')
        axes[1].hist(medium_slow_divs, bins=30, alpha=0.7, label='Medium-Slow', color='green')
        
        axes[1].set_title('Divergence Distributions')
        axes[1].set_xlabel('Divergence')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Regime changes
    if regime_history:
        regime_timestamps = [change.timestamp for change in regime_history]
        regime_types = [change.to_regime.value for change in regime_history]
        
        # Create regime timeline
        regime_colors = {
            'stable': 'green',
            'transitioning': 'yellow',
            'volatile': 'red',
            'trending': 'blue',
            'mean_reverting': 'purple',
            'unknown': 'gray'
        }
        
        for i, (timestamp, regime_type) in enumerate(zip(regime_timestamps, regime_types)):
            start = timestamp
            end = regime_timestamps[i+1] if i+1 < len(regime_timestamps) else len(data)
            
            axes[2].axvspan(start, end, 
                           color=regime_colors.get(regime_type, 'gray'),
                           alpha=0.3,
                           label=regime_type if i == 0 else "")
        
        axes[2].set_title('Regime Changes')
        axes[2].set_xlabel('Sample')
        axes[2].set_ylabel('Regime')
        axes[2].set_ylim(-0.5, 0.5)
        axes[2].set_yticks([])
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiscale_divergence_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: multiscale_divergence_analysis.png")
    
    return divergence_history, regime_history


def demonstrate_financial_application():
    """Demonstrate financial application with market regime detection"""
    print("\n=== Financial Application: Market Regime Detection ===")
    
    # Generate financial data
    prices, returns = generate_financial_data(2000)
    
    # Create multiscale tracker for financial data
    tracker = MultiScaleAdaptiveThreshold(
        use_topology=True,
        fast_window=50,    # ~1 month
        medium_window=200, # ~4 months
        slow_window=1000   # ~2 years
    )
    
    # Process financial data
    states = []
    regime_changes = []
    trading_signals = []
    
    print("Processing financial data with multi-scale regime detection...")
    
    for i, (price, return_val) in enumerate(zip(prices, returns)):
        # Update with return
        tracker.update(return_val)
        state = tracker.encode_adaptive(return_val)
        states.append(state)
        
        # Check for regime changes
        if tracker.regime_changed():
            change = tracker.get_last_regime_change()
            regime_changes.append(change)
            print(f"  Market regime change at day {i}: {change.from_regime.value} → {change.to_regime.value} "
                  f"(confidence: {change.confidence:.3f})")
        
        # Generate trading signal based on regime
        if tracker.get_current_regime() == RegimeType.TRENDING:
            # Trending market: follow the trend
            if return_val > 0:
                signal = 1  # Buy
            else:
                signal = -1  # Sell
        elif tracker.get_current_regime() == RegimeType.MEAN_REVERTING:
            # Mean reverting: contrarian
            if return_val > 0:
                signal = -1  # Sell (expect reversion)
            else:
                signal = 1   # Buy (expect reversion)
        else:
            # Other regimes: neutral
            signal = 0
        
        trading_signals.append(signal)
        
        # Print progress
        if (i + 1) % 500 == 0:
            regime_info = tracker.get_regime_info()
            print(f"  Day {i+1}: regime={regime_info['current_regime']}, "
                  f"changes={regime_info['num_regime_changes']}")
    
    # Analyze results
    states_array = StateArray(states)
    signals_array = np.array(trading_signals)
    
    print(f"\nFinancial Analysis Results:")
    print(f"  Total days: {len(prices)}")
    print(f"  Market regime changes: {len(regime_changes)}")
    print(f"  Final regime: {tracker.get_current_regime().value}")
    print(f"  State distribution: {states_array.count_by_state()}")
    
    # Trading signal analysis
    buy_signals = np.sum(signals_array == 1)
    sell_signals = np.sum(signals_array == -1)
    neutral_signals = np.sum(signals_array == 0)
    
    print(f"  Trading signals: Buy={buy_signals}, Sell={sell_signals}, Neutral={neutral_signals}")
    
    # Calculate strategy performance (simplified)
    strategy_returns = signals_array * returns
    total_return = np.sum(strategy_returns)
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    
    print(f"  Strategy total return: {total_return:.3f}")
    print(f"  Strategy Sharpe ratio: {sharpe_ratio:.3f}")
    
    # Plot financial analysis
    fig, axes = plt.subplots(5, 1, figsize=(15, 16))
    
    # Plot 1: Price series
    axes[0].plot(prices, 'b-', linewidth=1, alpha=0.8)
    axes[0].set_title('Price Series with Market Regimes')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Mark regime changes
    for change in regime_changes:
        axes[0].axvline(change.timestamp, color='red', linestyle='--', alpha=0.7)
    
    # Plot 2: Returns
    axes[1].plot(returns, 'g-', linewidth=1, alpha=0.8)
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
    
    # Plot 4: Trading signals
    axes[3].plot(signals_array, 'purple', linewidth=2, alpha=0.8)
    axes[3].set_title('Trading Signals (1=Buy, -1=Sell, 0=Neutral)')
    axes[3].set_ylabel('Signal')
    axes[3].set_ylim(-1.5, 1.5)
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Regime timeline
    regime_colors = {
        RegimeType.STABLE: 'green',
        RegimeType.TRANSITIONING: 'yellow',
        RegimeType.VOLATILE: 'red',
        RegimeType.TRENDING: 'blue',
        RegimeType.MEAN_REVERTING: 'purple',
        RegimeType.UNKNOWN: 'gray'
    }
    
    for i, change in enumerate(regime_changes):
        start = change.timestamp
        end = regime_changes[i+1].timestamp if i+1 < len(regime_changes) else len(prices)
        
        axes[4].axvspan(start, end, 
                        color=regime_colors.get(change.to_regime, 'gray'),
                        alpha=0.3,
                        label=change.to_regime.value if i == 0 else "")
    
    axes[4].set_title('Market Regime Evolution')
    axes[4].set_xlabel('Day')
    axes[4].set_ylabel('Regime')
    axes[4].set_ylim(-0.5, 0.5)
    axes[4].set_yticks([])
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiscale_financial_application.png', dpi=150, bbox_inches='tight')
    print("Saved: multiscale_financial_application.png")
    
    return states_array, regime_changes, trading_signals


def demonstrate_regime_adaptation():
    """Demonstrate how thresholds adapt to different regimes"""
    print("\n=== Regime Adaptation Demonstration ===")
    
    # Generate data with clear regime changes
    data = generate_regime_data(1500)
    
    # Create tracker
    tracker = MultiScaleAdaptiveThreshold(
        use_topology=True,
        fast_window=100,
        medium_window=500,
        slow_window=1500
    )
    
    # Process data and track threshold evolution
    threshold_evolution = []
    regime_evolution = []
    
    print("Processing data and tracking threshold adaptation...")
    
    for i, value in enumerate(data):
        tracker.update(value)
        
        # Record thresholds and regime
        thresholds = tracker.get_all_thresholds()
        regime = tracker.get_current_regime()
        
        threshold_evolution.append({
            'sample': i,
            'fast_i': thresholds['fast']['threshold_i'],
            'fast_q': thresholds['fast']['threshold_q'],
            'medium_i': thresholds['medium']['threshold_i'],
            'medium_q': thresholds['medium']['threshold_q'],
            'slow_i': thresholds['slow']['threshold_i'],
            'slow_q': thresholds['slow']['threshold_q']
        })
        
        regime_evolution.append({
            'sample': i,
            'regime': regime.value
        })
        
        # Print regime changes
        if tracker.regime_changed():
            change = tracker.get_last_regime_change()
            print(f"  Regime change at sample {i}: {change.from_regime.value} → {change.to_regime.value}")
    
    # Analyze threshold adaptation
    print(f"\nThreshold Adaptation Analysis:")
    
    # Calculate threshold stability
    fast_i_std = np.std([t['fast_i'] for t in threshold_evolution])
    medium_i_std = np.std([t['medium_i'] for t in threshold_evolution])
    slow_i_std = np.std([t['slow_i'] for t in threshold_evolution])
    
    print(f"  Fast threshold stability (std): {fast_i_std:.3f}")
    print(f"  Medium threshold stability (std): {medium_i_std:.3f}")
    print(f"  Slow threshold stability (std): {slow_i_std:.3f}")
    
    # Plot threshold adaptation
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Threshold evolution
    samples = [t['sample'] for t in threshold_evolution]
    fast_i = [t['fast_i'] for t in threshold_evolution]
    medium_i = [t['medium_i'] for t in threshold_evolution]
    slow_i = [t['slow_i'] for t in threshold_evolution]
    
    axes[0].plot(samples, fast_i, label='Fast', linewidth=2, alpha=0.8)
    axes[0].plot(samples, medium_i, label='Medium', linewidth=2, alpha=0.8)
    axes[0].plot(samples, slow_i, label='Slow', linewidth=2, alpha=0.8)
    axes[0].set_title('Threshold Evolution Across Scales')
    axes[0].set_ylabel('Threshold i')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Threshold differences
    fast_medium_diff = [abs(f - m) for f, m in zip(fast_i, medium_i)]
    fast_slow_diff = [abs(f - s) for f, s in zip(fast_i, slow_i)]
    medium_slow_diff = [abs(m - s) for m, s in zip(medium_i, slow_i)]
    
    axes[1].plot(samples, fast_medium_diff, label='Fast-Medium', linewidth=2, alpha=0.8)
    axes[1].plot(samples, fast_slow_diff, label='Fast-Slow', linewidth=2, alpha=0.8)
    axes[1].plot(samples, medium_slow_diff, label='Medium-Slow', linewidth=2, alpha=0.8)
    axes[1].set_title('Threshold Differences (Divergences)')
    axes[1].set_ylabel('Absolute Difference')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Regime evolution
    regime_colors = {
        'stable': 'green',
        'transitioning': 'yellow',
        'volatile': 'red',
        'trending': 'blue',
        'mean_reverting': 'purple',
        'unknown': 'gray'
    }
    
    for i, regime_data in enumerate(regime_evolution):
        regime = regime_data['regime']
        color = regime_colors.get(regime, 'gray')
        axes[2].scatter(regime_data['sample'], 0, c=color, alpha=0.7, s=10)
    
    axes[2].set_title('Regime Evolution')
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Regime')
    axes[2].set_ylim(-0.5, 0.5)
    axes[2].set_yticks([])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiscale_regime_adaptation.png', dpi=150, bbox_inches='tight')
    print("Saved: multiscale_regime_adaptation.png")
    
    return threshold_evolution, regime_evolution


def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive multiscale analysis"""
    print("\n=== Comprehensive Multiscale Analysis ===")
    
    # Generate complex data
    data = generate_regime_data(4000)
    
    # Create comprehensive tracker
    tracker = MultiScaleAdaptiveThreshold(
        use_topology=True,
        fast_window=100,
        medium_window=500,
        slow_window=2000
    )
    
    # Process data
    states = []
    regime_changes = []
    
    print("Processing data with comprehensive multiscale analysis...")
    
    for i, value in enumerate(data):
        tracker.update(value)
        state = tracker.encode_adaptive(value)
        states.append(state)
        
        if tracker.regime_changed():
            change = tracker.get_last_regime_change()
            regime_changes.append(change)
    
    # Get comprehensive analysis
    regime_info = tracker.get_regime_info()
    thresholds = tracker.get_all_thresholds()
    
    print(f"\nComprehensive Analysis Results:")
    print(f"  Total samples: {len(data)}")
    print(f"  Regime changes: {regime_info['num_regime_changes']}")
    print(f"  Current regime: {regime_info['current_regime']}")
    print(f"  Update count: {regime_info['update_count']}")
    
    # Threshold analysis
    print(f"\nThreshold Analysis:")
    for scale, thresh in thresholds.items():
        print(f"  {scale.capitalize()}: i={thresh['threshold_i']:.3f}, q={thresh['threshold_q']:.3f}")
    
    # Create comprehensive visualization
    try:
        fig = tracker.plot_multiscale()
        plt.savefig('multiscale_comprehensive.png', dpi=150, bbox_inches='tight')
        print("Saved: multiscale_comprehensive.png")
    except ImportError:
        print("Matplotlib not available for comprehensive visualization")
    
    return states, regime_changes, tracker


def main():
    """Main multiscale adaptive thresholding example"""
    print("ChannelPy Multi-Scale Adaptive Thresholding Example")
    print("=" * 60)
    
    # 1. Basic multiscale tracking
    states_array, regime_changes, tracker = demonstrate_basic_multiscale()
    
    # 2. Divergence analysis
    divergence_history, regime_history = demonstrate_divergence_analysis()
    
    # 3. Financial application
    financial_states, financial_regimes, trading_signals = demonstrate_financial_application()
    
    # 4. Regime adaptation
    threshold_evolution, regime_evolution = demonstrate_regime_adaptation()
    
    # 5. Comprehensive analysis
    comprehensive_states, comprehensive_regimes, comprehensive_tracker = demonstrate_comprehensive_analysis()
    
    print("\n" + "=" * 60)
    print("Multi-scale adaptive thresholding example completed successfully!")
    print("\nGenerated files:")
    print("- multiscale_basic.png: Basic multi-scale threshold tracking")
    print("- multiscale_divergence_analysis.png: Divergence analysis between scales")
    print("- multiscale_financial_application.png: Financial market regime detection")
    print("- multiscale_regime_adaptation.png: Threshold adaptation to regimes")
    print("- multiscale_comprehensive.png: Comprehensive multiscale analysis")
    print("\nKey features demonstrated:")
    print("- Multi-scale threshold tracking (fast, medium, slow)")
    print("- Regime change detection and classification")
    print("- Adaptive threshold selection based on regime")
    print("- Divergence analysis between scales")
    print("- Real-world financial applications")
    print("- Comprehensive regime adaptation")
    print("\nNext steps:")
    print("- Use multi-scale tracking for your data")
    print("- Implement regime-aware threshold selection")
    print("- Apply to financial or other time-series data")
    print("- Customize scale parameters for your domain")


if __name__ == "__main__":
    main()







