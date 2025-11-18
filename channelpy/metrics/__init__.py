"""
Metrics module for channel algebra.

Contains quality metrics and evaluation tools for channel encodings.
"""

from .quality import (
    encoding_accuracy, state_distribution_quality, discrimination_power,
    threshold_stability, information_content, encoding_consistency,
    state_transition_quality, channel_correlation, encoding_robustness,
    comprehensive_quality_report
)

__all__ = [
    'encoding_accuracy', 'state_distribution_quality', 'discrimination_power',
    'threshold_stability', 'information_content', 'encoding_consistency',
    'state_transition_quality', 'channel_correlation', 'encoding_robustness',
    'comprehensive_quality_report',
]







