"""
Preprocessing Example using ChannelPy

This example demonstrates the various preprocessors available in ChannelPy
for data cleaning, feature extraction, and preparation for channel encoding.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from channelpy import (
    State, StateArray, ChannelPipeline, ThresholdEncoder,
    StandardScaler, RobustScaler, MissingDataHandler, OutlierDetector,
    TimeSeriesFeatureExtractor, StatisticalFeatureExtractor, normalize
)
from channelpy.visualization import plot_states, plot_state_distribution


def generate_sample_data(n_samples=1000, n_features=3):
    """Generate sample dataset with various data quality issues"""
    np.random.seed(42)
    
    # Generate base data
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some structure
    X[:, 0] += np.linspace(0, 2, n_samples)  # Trend
    X[:, 1] *= 1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, n_samples))  # Seasonality
    X[:, 2] += 0.5 * np.random.randn(n_samples)  # Noise
    
    # Add missing values (5% missing)
    missing_mask = np.random.random((n_samples, n_features)) < 0.05
    X[missing_mask] = np.nan
    
    # Add outliers (2% outliers)
    outlier_mask = np.random.random((n_samples, n_features)) < 0.02
    X[outlier_mask] += np.random.choice([-5, 5], size=np.sum(outlier_mask))
    
    return X, missing_mask, outlier_mask


def demonstrate_scalers():
    """Demonstrate different scaling methods"""
    print("=== Scaling Demonstration ===")
    
    # Generate data with outliers
    X = np.random.normal(0, 1, (100, 2))
    X = np.vstack([X, [[10, 15], [-8, -12]]])  # Add outliers
    
    # Standard scaler
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(X)
    
    # Robust scaler
    scaler_robust = RobustScaler()
    X_robust = scaler_robust.fit_transform(X)
    
    print(f"Original data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Standard scaled range: [{X_std.min():.2f}, {X_std.max():.2f}]")
    print(f"Robust scaled range: [{X_robust.min():.2f}, {X_robust.max():.2f}]")
    print(f"Standard scaled mean: {X_std.mean(axis=0)}")
    print(f"Robust scaled mean: {X_robust.mean(axis=0)}")
    print()


def demonstrate_missing_data_handling():
    """Demonstrate missing data handling strategies"""
    print("=== Missing Data Handling ===")
    
    # Generate data with missing values
    X = np.random.normal(0, 1, (100, 3))
    missing_indices = np.random.choice(100, size=20, replace=False)
    X[missing_indices, 0] = np.nan
    
    strategies = ['mean', 'median', 'constant']
    
    for strategy in strategies:
        handler = MissingDataHandler(strategy=strategy, fill_value=0.0)
        X_filled, missing_mask = handler.fit_transform(X)
        
        n_missing = np.sum(missing_mask)
        print(f"{strategy.capitalize()} strategy: {n_missing} missing values filled")
        print(f"  Mean of filled data: {X_filled.mean(axis=0)}")
    print()


def demonstrate_outlier_detection():
    """Demonstrate outlier detection methods"""
    print("=== Outlier Detection ===")
    
    # Generate data with outliers
    X = np.random.normal(0, 1, (100, 2))
    X = np.vstack([X, [[5, 6], [-4, -5], [7, 8]]])  # Add outliers
    
    methods = ['iqr', 'zscore']
    actions = ['flag', 'clip', 'remove']
    
    for method in methods:
        for action in actions:
            detector = OutlierDetector(method=method, action=action, threshold=1.5)
            X_processed, outlier_mask = detector.fit_transform(X)
            
            n_outliers = np.sum(outlier_mask)
            print(f"{method.upper()} + {action}: {n_outliers} outliers, shape {X_processed.shape}")
    print()


def demonstrate_time_series_features():
    """Demonstrate time series feature extraction"""
    print("=== Time Series Feature Extraction ===")
    
    # Generate time series data
    t = np.linspace(0, 10, 200)
    price = 100 + 2*t + 5*np.sin(t) + np.random.normal(0, 1, 200)
    
    extractor = TimeSeriesFeatureExtractor(
        window_size=20,
        features=['mean', 'std', 'trend', 'momentum', 'volatility']
    )
    
    features = extractor.fit_transform(price)
    
    print("Extracted features:")
    for name, values in features.items():
        print(f"  {name}: shape {values.shape}, mean {values.mean():.3f}")
    print()


def demonstrate_statistical_features():
    """Demonstrate statistical feature extraction"""
    print("=== Statistical Feature Extraction ===")
    
    # Generate data
    X = np.random.normal(0, 1, (100, 3))
    
    extractor = StatisticalFeatureExtractor(
        features=['mean', 'std', 'skew', 'kurtosis', 'percentile_25', 'percentile_75']
    )
    
    features = extractor.fit_transform(X)
    
    print("Statistical features:")
    for name, value in features.items():
        print(f"  {name}: {value:.3f}")
    print()


def demonstrate_complete_pipeline():
    """Demonstrate a complete preprocessing pipeline"""
    print("=== Complete Preprocessing Pipeline ===")
    
    # Generate messy data
    X, missing_mask, outlier_mask = generate_sample_data(500, 3)
    
    print(f"Original data shape: {X.shape}")
    print(f"Missing values: {np.sum(missing_mask)}")
    print(f"Outliers: {np.sum(outlier_mask)}")
    
    # Create preprocessing pipeline
    from channelpy.pipeline.preprocessors import CompositePreprocessor
    
    preprocessor = CompositePreprocessor([
        MissingDataHandler(strategy='median'),
        OutlierDetector(method='iqr', action='clip'),
        StandardScaler()
    ])
    
    # Process data
    X_clean = preprocessor.fit_transform(X)
    
    print(f"Cleaned data shape: {X_clean.shape}")
    print(f"Missing values: {np.sum(np.isnan(X_clean))}")
    print(f"Data range: [{X_clean.min():.3f}, {X_clean.max():.3f}]")
    print(f"Data mean: {X_clean.mean(axis=0)}")
    print(f"Data std: {X_clean.std(axis=0)}")
    print()


def demonstrate_channel_encoding():
    """Demonstrate channel encoding after preprocessing"""
    print("=== Channel Encoding After Preprocessing ===")
    
    # Generate and preprocess data
    X, _, _ = generate_sample_data(200, 2)
    
    # Preprocess
    X_clean = normalize(X)  # Quick normalize
    
    # Create channel pipeline
    pipeline = ChannelPipeline()
    pipeline.add_preprocessor(lambda x: normalize(x))  # Normalize
    pipeline.add_encoder(ThresholdEncoder(threshold_i=0.5, threshold_q=0.75))
    
    # Simple interpreter that counts states
    def count_states(states):
        if isinstance(states, list) and len(states) > 0:
            state_array = states[0]
            if hasattr(state_array, 'count_by_state'):
                return state_array.count_by_state()
        return {}
    
    pipeline.add_interpreter(count_states)
    
    # Fit and transform
    pipeline.fit(X)
    decisions, states = pipeline.transform(X)
    
    print("Channel states after preprocessing:")
    if decisions and len(decisions) > 0:
        state_counts = decisions[0]
        for state, count in state_counts.items():
            print(f"  {state}: {count}")
    
    # Visualize states
    if states and len(states) > 0:
        state_array = states[0]
        if hasattr(state_array, 'count_by_state'):
            fig, ax = plot_state_distribution(state_array, "State Distribution After Preprocessing")
            plt.tight_layout()
            plt.savefig('preprocessing_state_distribution.png', dpi=150, bbox_inches='tight')
            print("Saved: preprocessing_state_distribution.png")
    print()


def main():
    """Main demonstration function"""
    print("ChannelPy Preprocessing Example")
    print("=" * 40)
    
    demonstrate_scalers()
    demonstrate_missing_data_handling()
    demonstrate_outlier_detection()
    demonstrate_time_series_features()
    demonstrate_statistical_features()
    demonstrate_complete_pipeline()
    demonstrate_channel_encoding()
    
    print("Preprocessing example completed successfully!")


if __name__ == "__main__":
    main()







