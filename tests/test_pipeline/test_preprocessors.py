"""
Tests for pipeline.preprocessors module
"""
import pytest
import numpy as np
import pandas as pd
from channelpy.pipeline.preprocessors import (
    StandardScaler, RobustScaler, MissingDataHandler, OutlierDetector,
    TimeSeriesFeatureExtractor, StatisticalFeatureExtractor, CompositePreprocessor,
    normalize, standardize
)


def test_standard_scaler():
    """Test StandardScaler"""
    scaler = StandardScaler()
    
    # Test data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Fit and transform
    X_scaled = scaler.fit_transform(X)
    
    # Check that mean is ~0 and std is ~1
    assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)
    
    # Test inverse transform
    X_original = scaler.inverse_transform(X_scaled)
    assert np.allclose(X_original, X)


def test_robust_scaler():
    """Test RobustScaler"""
    scaler = RobustScaler()
    
    # Test data with outliers
    X = np.array([[1, 2], [3, 4], [5, 6], [100, 200]])  # Outliers at end
    
    X_scaled = scaler.fit_transform(X)
    
    # Should be more robust to outliers than StandardScaler
    assert X_scaled.shape == X.shape


def test_missing_data_handler():
    """Test MissingDataHandler"""
    handler = MissingDataHandler(strategy='median')
    
    # Test data with missing values
    X = np.array([[1, 2, np.nan], [3, np.nan, 6], [5, 8, 9]])
    
    X_filled, missing_mask = handler.fit_transform(X)
    
    # Check that missing values are filled
    assert not np.any(np.isnan(X_filled))
    assert missing_mask.shape == X.shape
    assert np.sum(missing_mask) == 2  # Two missing values


def test_outlier_detector():
    """Test OutlierDetector"""
    detector = OutlierDetector(method='iqr', action='clip')
    
    # Test data with outliers
    X = np.array([[1, 2], [3, 4], [5, 6], [100, 200]])  # Outliers at end
    
    X_clean, outlier_mask = detector.fit_transform(X)
    
    # Check that outliers are clipped
    assert X_clean.shape == X.shape
    assert np.sum(outlier_mask) > 0  # Should detect outliers


def test_time_series_feature_extractor():
    """Test TimeSeriesFeatureExtractor"""
    extractor = TimeSeriesFeatureExtractor(window_size=5, features=['mean', 'std'])
    
    # Test time series data
    X = np.random.randn(100)
    
    features = extractor.fit_transform(X)
    
    # Check that features are extracted
    assert 'mean' in features
    assert 'std' in features
    assert len(features['mean']) == len(X)


def test_statistical_feature_extractor():
    """Test StatisticalFeatureExtractor"""
    extractor = StatisticalFeatureExtractor(features=['mean', 'std', 'skew'])
    
    # Test data
    X = np.random.randn(100)
    
    features = extractor.fit_transform(X)
    
    # Check that features are extracted
    assert 'mean' in features
    assert 'std' in features
    assert 'skew' in features
    assert isinstance(features['mean'], (int, float))


def test_composite_preprocessor():
    """Test CompositePreprocessor"""
    preprocessors = [
        MissingDataHandler(strategy='median'),
        OutlierDetector(method='iqr', action='clip'),
        StandardScaler()
    ]
    
    composite = CompositePreprocessor(preprocessors)
    
    # Test data with missing values and outliers
    X = np.array([[1, 2, np.nan], [3, 4, 6], [5, 8, 9], [100, 200, 300]])
    
    X_processed = composite.fit_transform(X)
    
    # Check that data is processed
    assert X_processed.shape[0] == X.shape[0]  # Same number of rows
    assert not np.any(np.isnan(X_processed))  # No missing values


def test_convenience_functions():
    """Test convenience functions"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Test normalize
    X_norm = normalize(X)
    assert np.allclose(np.min(X_norm, axis=0), 0)
    assert np.allclose(np.max(X_norm, axis=0), 1)
    
    # Test standardize
    X_std = standardize(X)
    assert np.allclose(np.mean(X_std, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(X_std, axis=0), 1, atol=1e-10)


def test_preprocessor_callable():
    """Test that preprocessors are callable"""
    scaler = StandardScaler()
    X = np.array([[1, 2], [3, 4]])
    
    # Should work without explicit fit_transform
    X_scaled = scaler(X)
    assert X_scaled.shape == X.shape







