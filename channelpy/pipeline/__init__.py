"""
Pipeline module for channel algebra data processing.

Contains pipeline architecture, preprocessing, and encoding strategies.
"""

from .base import BasePipeline, ChannelPipeline
from .preprocessors import (
    BasePreprocessor, StandardScaler, RobustScaler, MissingDataHandler,
    OutlierDetector, TimeSeriesFeatureExtractor, StatisticalFeatureExtractor,
    CompositePreprocessor, normalize, standardize
)
from .encoders import (
    ThresholdEncoder, LearnedThresholdEncoder, DualFeatureEncoder
)
from .interpreters import (
    BaseInterpreter, RuleBasedInterpreter, LookupTableInterpreter, FSMInterpreter,
    PatternMatcher, NestedStateInterpreter, ParallelChannelInterpreter,
    ThresholdBasedInterpreter, quick_interpret
)

__all__ = [
    'BasePipeline', 'ChannelPipeline',
    'BasePreprocessor', 'StandardScaler', 'RobustScaler', 'MissingDataHandler',
    'OutlierDetector', 'TimeSeriesFeatureExtractor', 'StatisticalFeatureExtractor',
    'CompositePreprocessor', 'normalize', 'standardize',
    'ThresholdEncoder', 'LearnedThresholdEncoder', 'DualFeatureEncoder',
    'BaseInterpreter', 'RuleBasedInterpreter', 'LookupTableInterpreter', 'FSMInterpreter',
    'PatternMatcher', 'NestedStateInterpreter', 'ParallelChannelInterpreter',
    'ThresholdBasedInterpreter', 'quick_interpret',
]
