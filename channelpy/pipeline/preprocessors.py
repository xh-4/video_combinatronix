"""
Data preprocessing utilities for channel pipelines

This module provides preprocessors for Stage 1 of the pipeline (Raw data → Features)
"""
from typing import Optional, Union, Dict, Any, Callable
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessors
    
    Preprocessors transform raw data into features suitable for encoding
    """
    
    def __init__(self):
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y=None):
        """
        Learn preprocessing parameters from data
        
        Parameters
        ----------
        X : array-like
            Input data
        y : array-like, optional
            Target values (for supervised preprocessing)
            
        Returns
        -------
        self : BasePreprocessor
            Fitted preprocessor
        """
        pass
    
    @abstractmethod
    def transform(self, X):
        """
        Apply preprocessing transformation
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        X_transformed : array-like
            Preprocessed data
        """
        pass
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def __call__(self, X):
        """Make preprocessor callable"""
        if not self.is_fitted:
            return self.fit_transform(X)
        return self.transform(X)


class StandardScaler(BasePreprocessor):
    """
    Standardize features by removing mean and scaling to unit variance
    
    z = (x - μ) / σ
    
    Examples
    --------
    >>> scaler = StandardScaler()
    >>> X_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """
        Parameters
        ----------
        with_mean : bool
            If True, center data before scaling
        with_std : bool
            If True, scale data to unit variance
        """
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X, y=None):
        """Learn mean and standard deviation"""
        X = np.asarray(X)
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = 0.0
        
        if self.with_std:
            self.std_ = np.std(X, axis=0)
            # Avoid division by zero
            self.std_[self.std_ == 0] = 1.0
        else:
            self.std_ = 1.0
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Standardize data"""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X)
        X_scaled = (X - self.mean_) / self.std_
        return X_scaled
    
    def inverse_transform(self, X_scaled):
        """Reverse standardization"""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X_scaled = np.asarray(X_scaled)
        X = X_scaled * self.std_ + self.mean_
        return X


class RobustScaler(BasePreprocessor):
    """
    Scale features using statistics robust to outliers
    
    Uses median and IQR instead of mean and std
    z = (x - median) / IQR
    
    Better for data with outliers
    
    Examples
    --------
    >>> scaler = RobustScaler()
    >>> X_scaled = scaler.fit_transform(X_train)
    """
    
    def __init__(self, quantile_range: tuple = (25.0, 75.0)):
        """
        Parameters
        ----------
        quantile_range : tuple
            (q_min, q_max) percentiles for IQR computation
        """
        super().__init__()
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None
    
    def fit(self, X, y=None):
        """Learn median and IQR"""
        X = np.asarray(X)
        
        # Compute median
        self.center_ = np.median(X, axis=0)
        
        # Compute IQR
        q_min, q_max = self.quantile_range
        quantiles = np.percentile(X, [q_min, q_max], axis=0)
        self.scale_ = quantiles[1] - quantiles[0]
        
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Scale data robustly"""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X)
        X_scaled = (X - self.center_) / self.scale_
        return X_scaled


class MissingDataHandler(BasePreprocessor):
    """
    Handle missing data with various strategies
    
    Strategies:
    - 'mean': Fill with mean
    - 'median': Fill with median  
    - 'mode': Fill with most frequent
    - 'constant': Fill with constant value
    - 'forward': Forward fill
    - 'backward': Backward fill
    
    Also tracks which values were missing
    
    Examples
    --------
    >>> handler = MissingDataHandler(strategy='median')
    >>> X_filled, missing_mask = handler.fit_transform(X_with_nans)
    """
    
    def __init__(self, strategy: str = 'median', fill_value: float = 0.0):
        """
        Parameters
        ----------
        strategy : str
            Imputation strategy
        fill_value : float
            Value to use for 'constant' strategy
        """
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
    
    def fit(self, X, y=None):
        """Learn imputation statistics"""
        X = np.asarray(X)
        
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'mode':
            # Most frequent value per column
            self.statistics_ = self._compute_mode(X)
        elif self.strategy == 'constant':
            self.statistics_ = self.fill_value
        elif self.strategy in ['forward', 'backward']:
            self.statistics_ = None  # No statistics needed
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Fill missing values
        
        Returns
        -------
        X_filled : array
            Data with missing values filled
        missing_mask : array
            Boolean mask indicating which values were missing
        """
        if not self.is_fitted:
            raise RuntimeError("Handler not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=float)
        missing_mask = np.isnan(X)
        
        X_filled = X.copy()
        
        if self.strategy in ['mean', 'median', 'mode', 'constant']:
            # Replace NaN with statistics
            for i in range(X.shape[1] if X.ndim > 1 else 1):
                if X.ndim > 1:
                    mask = missing_mask[:, i]
                    X_filled[mask, i] = self.statistics_[i]
                else:
                    X_filled[missing_mask] = self.statistics_
        
        elif self.strategy == 'forward':
            # Forward fill
            X_filled = pd.DataFrame(X).fillna(method='ffill').values
        
        elif self.strategy == 'backward':
            # Backward fill
            X_filled = pd.DataFrame(X).fillna(method='bfill').values
        
        return X_filled, missing_mask
    
    def _compute_mode(self, X):
        """Compute mode for each column"""
        # TODO: Implement mode computation
        # For now, use median as fallback
        return np.nanmedian(X, axis=0)


class OutlierDetector(BasePreprocessor):
    """
    Detect and handle outliers
    
    Methods:
    - 'iqr': Interquartile range method
    - 'zscore': Z-score method
    - 'isolation_forest': Isolation forest (sklearn)
    
    Actions:
    - 'flag': Only flag outliers (return mask)
    - 'clip': Clip to bounds
    - 'remove': Remove outliers (return filtered data)
    
    Examples
    --------
    >>> detector = OutlierDetector(method='iqr', action='clip')
    >>> X_clean, outlier_mask = detector.fit_transform(X)
    """
    
    def __init__(self, method: str = 'iqr', action: str = 'clip', 
                 threshold: float = 1.5):
        """
        Parameters
        ----------
        method : str
            Outlier detection method
        action : str
            What to do with outliers
        threshold : float
            Threshold parameter (meaning depends on method)
        """
        super().__init__()
        self.method = method
        self.action = action
        self.threshold = threshold
        self.lower_bound_ = None
        self.upper_bound_ = None
    
    def fit(self, X, y=None):
        """Learn outlier bounds"""
        X = np.asarray(X)
        
        if self.method == 'iqr':
            q25, q75 = np.percentile(X, [25, 75], axis=0)
            iqr = q75 - q25
            self.lower_bound_ = q25 - self.threshold * iqr
            self.upper_bound_ = q75 + self.threshold * iqr
        
        elif self.method == 'zscore':
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            self.lower_bound_ = mean - self.threshold * std
            self.upper_bound_ = mean + self.threshold * std
        
        elif self.method == 'isolation_forest':
            # TODO: Implement isolation forest
            # For now, fallback to IQR
            q25, q75 = np.percentile(X, [25, 75], axis=0)
            iqr = q75 - q25
            self.lower_bound_ = q25 - self.threshold * iqr
            self.upper_bound_ = q75 + self.threshold * iqr
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Detect and handle outliers
        
        Returns
        -------
        X_processed : array
            Processed data
        outlier_mask : array
            Boolean mask indicating outliers
        """
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        X = np.asarray(X)
        
        # Detect outliers
        outlier_mask = (X < self.lower_bound_) | (X > self.upper_bound_)
        
        X_processed = X.copy()
        
        if self.action == 'flag':
            # Just return data with mask
            pass
        
        elif self.action == 'clip':
            # Clip to bounds
            X_processed = np.clip(X, self.lower_bound_, self.upper_bound_)
        
        elif self.action == 'remove':
            # Remove rows with any outliers
            if X.ndim > 1:
                keep_mask = ~np.any(outlier_mask, axis=1)
                X_processed = X[keep_mask]
                outlier_mask = outlier_mask[keep_mask]
            else:
                X_processed = X[~outlier_mask]
                outlier_mask = outlier_mask[~outlier_mask]
        
        return X_processed, outlier_mask


class TimeSeriesFeatureExtractor(BasePreprocessor):
    """
    Extract features from time series data
    
    Features:
    - Rolling statistics (mean, std, min, max)
    - Momentum (rate of change)
    - Trend (linear regression slope)
    - Volatility (rolling std of returns)
    
    Examples
    --------
    >>> extractor = TimeSeriesFeatureExtractor(window_size=20)
    >>> features = extractor.fit_transform(price_series)
    >>> # Returns dict: {'mean': ..., 'trend': ..., 'volatility': ...}
    """
    
    def __init__(self, window_size: int = 20, 
                 features: list = ['mean', 'std', 'trend', 'momentum']):
        """
        Parameters
        ----------
        window_size : int
            Size of rolling window
        features : list
            List of features to extract
        """
        super().__init__()
        self.window_size = window_size
        self.features = features
    
    def fit(self, X, y=None):
        """No fitting needed for time series features"""
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Extract time series features
        
        Returns
        -------
        features : dict
            Dictionary of extracted features
        """
        X = np.asarray(X)
        features = {}
        
        if 'mean' in self.features:
            features['mean'] = self._rolling_mean(X)
        
        if 'std' in self.features:
            features['std'] = self._rolling_std(X)
        
        if 'min' in self.features:
            features['min'] = self._rolling_min(X)
        
        if 'max' in self.features:
            features['max'] = self._rolling_max(X)
        
        if 'trend' in self.features:
            features['trend'] = self._compute_trend(X)
        
        if 'momentum' in self.features:
            features['momentum'] = self._compute_momentum(X)
        
        if 'volatility' in self.features:
            features['volatility'] = self._compute_volatility(X)
        
        return features
    
    def _rolling_mean(self, X):
        """Compute rolling mean"""
        # TODO: Use pandas rolling for efficiency
        return pd.Series(X).rolling(self.window_size).mean().values
    
    def _rolling_std(self, X):
        """Compute rolling standard deviation"""
        return pd.Series(X).rolling(self.window_size).std().values
    
    def _rolling_min(self, X):
        """Compute rolling minimum"""
        return pd.Series(X).rolling(self.window_size).min().values
    
    def _rolling_max(self, X):
        """Compute rolling maximum"""
        return pd.Series(X).rolling(self.window_size).max().values
    
    def _compute_trend(self, X):
        """
        Compute rolling linear regression slope
        
        TODO: Implement efficient rolling regression
        """
        # Placeholder: use simple difference
        return np.gradient(self._rolling_mean(X))
    
    def _compute_momentum(self, X):
        """Compute momentum (rate of change)"""
        return np.diff(X, prepend=X[0]) / self.window_size
    
    def _compute_volatility(self, X):
        """Compute volatility (std of returns)"""
        returns = np.diff(np.log(X + 1e-10), prepend=0)
        return pd.Series(returns).rolling(self.window_size).std().values


class StatisticalFeatureExtractor(BasePreprocessor):
    """
    Extract statistical features from data
    
    Features:
    - Mean, median, mode
    - Standard deviation, variance
    - Skewness, kurtosis
    - Percentiles
    
    Examples
    --------
    >>> extractor = StatisticalFeatureExtractor()
    >>> features = extractor.fit_transform(data)
    """
    
    def __init__(self, features: list = ['mean', 'std', 'skew', 'kurtosis']):
        """
        Parameters
        ----------
        features : list
            List of statistical features to extract
        """
        super().__init__()
        self.features = features
    
    def fit(self, X, y=None):
        """No fitting needed"""
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Extract statistical features
        
        Returns
        -------
        features : dict
            Dictionary of statistical features
        """
        X = np.asarray(X)
        features = {}
        
        if 'mean' in self.features:
            features['mean'] = np.mean(X)
        
        if 'median' in self.features:
            features['median'] = np.median(X)
        
        if 'std' in self.features:
            features['std'] = np.std(X)
        
        if 'var' in self.features:
            features['var'] = np.var(X)
        
        if 'skew' in self.features:
            from scipy.stats import skew
            features['skew'] = skew(X)
        
        if 'kurtosis' in self.features:
            from scipy.stats import kurtosis
            features['kurtosis'] = kurtosis(X)
        
        if 'percentile_25' in self.features:
            features['percentile_25'] = np.percentile(X, 25)
        
        if 'percentile_75' in self.features:
            features['percentile_75'] = np.percentile(X, 75)
        
        return features


class CompositePreprocessor(BasePreprocessor):
    """
    Chain multiple preprocessors together
    
    Examples
    --------
    >>> preprocessor = CompositePreprocessor([
    ...     MissingDataHandler(strategy='median'),
    ...     OutlierDetector(method='iqr', action='clip'),
    ...     StandardScaler()
    ... ])
    >>> X_clean = preprocessor.fit_transform(X_raw)
    """
    
    def __init__(self, preprocessors: list):
        """
        Parameters
        ----------
        preprocessors : list
            List of preprocessor instances
        """
        super().__init__()
        self.preprocessors = preprocessors
    
    def fit(self, X, y=None):
        """Fit all preprocessors in sequence"""
        X_transformed = X
        for preprocessor in self.preprocessors:
            preprocessor.fit(X_transformed, y)
            result = preprocessor.transform(X_transformed)
            # Handle preprocessors that return tuples (data, mask)
            if isinstance(result, tuple):
                X_transformed = result[0]
            else:
                X_transformed = result
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply all preprocessors in sequence"""
        X_transformed = X
        for preprocessor in self.preprocessors:
            result = preprocessor.transform(X_transformed)
            if isinstance(result, tuple):
                X_transformed = result[0]
            else:
                X_transformed = result
        
        return X_transformed


# Convenience functions

def normalize(X):
    """Quick normalize to [0, 1]"""
    X = np.asarray(X)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min + 1e-10)


def standardize(X):
    """Quick standardize (z-score)"""
    scaler = StandardScaler()
    return scaler.fit_transform(X)







