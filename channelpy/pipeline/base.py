"""
Base pipeline architecture
"""
from typing import Any, Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np
from ..core.state import State, StateArray


class BasePipeline(ABC):
    """
    Base class for channel pipelines
    
    Three-stage architecture:
    1. Preprocess: Raw data → Features
    2. Encode: Features → States
    3. Interpret: States → Decisions
    """
    
    def __init__(self):
        self.preprocessors = []
        self.encoders = []
        self.interpreters = []
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit pipeline on data
        
        Parameters
        ----------
        X : array-like
            Input data
        y : array-like, optional
            Target labels
        """
        pass
    
    @abstractmethod
    def transform(self, X):
        """
        Transform data through pipeline
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        decisions : array-like
            Pipeline output
        states : array-like
            Intermediate states (for debugging)
        """
        pass
    
    def fit_transform(self, X, y=None):
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def add_preprocessor(self, preprocessor: Callable):
        """Add preprocessing step"""
        self.preprocessors.append(preprocessor)
    
    def add_encoder(self, encoder: Callable):
        """Add encoding step"""
        self.encoders.append(encoder)
    
    def add_interpreter(self, interpreter: Callable):
        """Add interpretation step"""
        self.interpreters.append(interpreter)
    
    def _preprocess(self, X):
        """Apply all preprocessors"""
        result = X
        for prep in self.preprocessors:
            result = prep(result)
        return result
    
    def _encode(self, features):
        """Apply all encoders"""
        states = []
        for encoder in self.encoders:
            state = encoder(features)
            states.append(state)
        return states
    
    def _interpret(self, states):
        """Apply all interpreters"""
        decisions = []
        for interpreter in self.interpreters:
            decision = interpreter(states)
            decisions.append(decision)
        return decisions


class ChannelPipeline(BasePipeline):
    """
    Concrete implementation of channel pipeline
    
    Examples
    --------
    >>> pipeline = ChannelPipeline()
    >>> pipeline.add_preprocessor(normalize)
    >>> pipeline.add_encoder(threshold_encoder)
    >>> pipeline.add_interpreter(rule_based_interpreter)
    >>> pipeline.fit(train_data, train_labels)
    >>> decisions, states = pipeline.transform(test_data)
    """
    
    def fit(self, X, y=None):
        """Fit pipeline"""
        # Fit preprocessors
        features = X
        for prep in self.preprocessors:
            if hasattr(prep, 'fit'):
                prep.fit(features, y)
            features = prep.transform(features) if hasattr(prep, 'transform') else prep(features)
        
        # Fit encoders
        for encoder in self.encoders:
            if hasattr(encoder, 'fit'):
                encoder.fit(features, y)
        
        # Fit interpreters
        if y is not None:
            states = self._encode(features)
            for interpreter in self.interpreters:
                if hasattr(interpreter, 'fit'):
                    interpreter.fit(states, y)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform through pipeline"""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        # Stage 1: Preprocess
        features = self._preprocess(X)
        
        # Stage 2: Encode
        states = self._encode(features)
        
        # Stage 3: Interpret
        decisions = self._interpret(states)
        
        return decisions, states







