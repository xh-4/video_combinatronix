"""
Feature → State encoders
"""
from typing import Optional, Tuple, Callable
import numpy as np
from ..core.state import State, StateArray


class ThresholdEncoder:
    """
    Simple threshold-based encoder
    
    Examples
    --------
    >>> encoder = ThresholdEncoder(threshold_i=0.5, threshold_q=0.75)
    >>> encoder.encode(0.8)
    ψ
    >>> encoder.encode(0.3)
    ∅
    """
    
    def __init__(self, threshold_i: float = 0.5, threshold_q: float = 0.75):
        self.threshold_i = threshold_i
        self.threshold_q = threshold_q
    
    def encode(self, value: float) -> State:
        """Encode single value"""
        return State(
            i=int(value > self.threshold_i),
            q=int(value > self.threshold_q)
        )
    
    def encode_array(self, values: np.ndarray) -> StateArray:
        """Encode array of values"""
        return StateArray(
            i=(values > self.threshold_i).astype(np.int8),
            q=(values > self.threshold_q).astype(np.int8)
        )
    
    def __call__(self, values):
        """Make callable"""
        if isinstance(values, (int, float)):
            return self.encode(values)
        else:
            return self.encode_array(np.asarray(values))


class LearnedThresholdEncoder:
    """
    Learn optimal thresholds from data
    
    Examples
    --------
    >>> encoder = LearnedThresholdEncoder()
    >>> encoder.fit(train_features, train_labels)
    >>> states = encoder(test_features)
    """
    
    def __init__(self, method: str = 'statistical'):
        """
        Parameters
        ----------
        method : str
            'statistical': Use percentiles
            'supervised': Optimize for classification
        """
        self.method = method
        self.threshold_i = None
        self.threshold_q = None
    
    def fit(self, X, y=None):
        """Learn thresholds"""
        X = np.asarray(X)
        
        if self.method == 'statistical':
            self.threshold_i = np.median(X)
            self.threshold_q = np.percentile(X, 75)
        
        elif self.method == 'supervised':
            if y is None:
                raise ValueError("Method 'supervised' requires labels")
            
            # Find thresholds that maximize separation
            self.threshold_i = self._find_optimal_threshold(X, y)
            self.threshold_q = self._find_optimal_threshold(
                X[X > self.threshold_i], 
                y[X > self.threshold_i]
            ) if np.any(X > self.threshold_i) else self.threshold_i * 1.5
        
        return self
    
    def _find_optimal_threshold(self, X, y):
        """Find threshold maximizing class separation"""
        candidates = np.percentile(X, np.linspace(10, 90, 9))
        best_threshold = candidates[0]
        best_score = -np.inf
        
        for threshold in candidates:
            bit_values = (X > threshold).astype(int)
            score = np.corrcoef(bit_values, y)[0, 1]
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def __call__(self, X):
        """Encode features"""
        if self.threshold_i is None:
            raise RuntimeError("Encoder not fitted")
        
        X = np.asarray(X)
        return StateArray(
            i=(X > self.threshold_i).astype(np.int8),
            q=(X > self.threshold_q).astype(np.int8)
        )


class DualFeatureEncoder:
    """
    Encode from two separate features (one for i, one for q)
    
    Examples
    --------
    >>> encoder = DualFeatureEncoder()
    >>> encoder.fit(train_features_i, train_features_q)
    >>> states = encoder(test_features_i, test_features_q)
    """
    
    def __init__(self):
        self.encoder_i = LearnedThresholdEncoder()
        self.encoder_q = LearnedThresholdEncoder()
    
    def fit(self, X_i, X_q, y=None):
        """Fit both encoders"""
        self.encoder_i.fit(X_i, y)
        self.encoder_q.fit(X_q, y)
        return self
    
    def __call__(self, X_i, X_q):
        """Encode from two features"""
        bits_i = (np.asarray(X_i) > self.encoder_i.threshold_i).astype(np.int8)
        bits_q = (np.asarray(X_q) > self.encoder_q.threshold_q).astype(np.int8)
        return StateArray(i=bits_i, q=bits_q)







