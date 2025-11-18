"""
Quality metrics for channel encodings

Comprehensive evaluation of channel encoding quality and performance
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI


def encoding_accuracy(predicted_states: StateArray, 
                     true_states: StateArray) -> float:
    """
    Accuracy of state encoding
    
    Proportion of exactly matching states
    
    Parameters
    ----------
    predicted_states : StateArray
        Predicted states
    true_states : StateArray
        True states
        
    Returns
    -------
    accuracy : float
        Accuracy between 0 and 1
    """
    if len(predicted_states) != len(true_states):
        raise ValueError("State arrays must have same length")
    
    matches = sum(1 for p, t in zip(predicted_states, true_states) if p == t)
    return matches / len(predicted_states)


def state_distribution_quality(states: StateArray) -> Dict:
    """
    Measure quality of state distribution
    
    Good encoding should have:
    - Balanced distribution (not all one state)
    - Meaningful PSI states (not too many)
    
    Parameters
    ----------
    states : StateArray
        States to evaluate
        
    Returns
    -------
    metrics : dict
        Quality metrics
    """
    counts = states.count_by_state()
    total = len(states)
    
    if total == 0:
        return {
            'balance_score': 0.0,
            'psi_ratio': 0.0,
            'psi_quality': 0.0,
            'overall_quality': 0.0,
            'distribution': counts
        }
    
    # Distribution entropy
    probs = np.array([counts[s] / total for s in [EMPTY, DELTA, PHI, PSI]])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(4)  # For 4 states
    
    # Balance score (0 = all one state, 1 = perfectly balanced)
    balance = entropy / max_entropy
    
    # PSI ratio (should be moderate, not too high or low)
    psi_ratio = counts[PSI] / total
    psi_quality = 1.0 - abs(psi_ratio - 0.25) * 4  # Ideal is 0.25
    
    return {
        'balance_score': balance,
        'psi_ratio': psi_ratio,
        'psi_quality': psi_quality,
        'overall_quality': (balance + psi_quality) / 2,
        'distribution': counts
    }


def discrimination_power(states: StateArray, labels: np.ndarray) -> float:
    """
    How well do states discriminate between classes?
    
    Uses mutual information between states and labels
    
    Parameters
    ----------
    states : StateArray
        Encoded states
    labels : np.ndarray
        True class labels
        
    Returns
    -------
    discrimination : float
        Discrimination power between 0 and 1
    """
    try:
        from sklearn.metrics import mutual_info_score
    except ImportError:
        print("scikit-learn not installed. Install with: pip install scikit-learn")
        return 0.0
    
    if len(states) != len(labels):
        raise ValueError("States and labels must have same length")
    
    state_ints = states.to_ints()
    mi = mutual_info_score(labels, state_ints)
    
    # Normalize by max possible MI
    max_mi = np.log(min(4, len(np.unique(labels))))
    
    return mi / max_mi if max_mi > 0 else 0.0


def threshold_stability(threshold_history: List[float]) -> Dict:
    """
    Measure stability of adaptive thresholds
    
    Low variance = stable, High variance = unstable
    
    Parameters
    ----------
    threshold_history : List[float]
        History of threshold values
        
    Returns
    -------
    stability : dict
        Stability metrics
    """
    if not threshold_history:
        return {
            'mean': 0.0,
            'std': 0.0,
            'coefficient_of_variation': 0.0,
            'stability_score': 0.0
        }
    
    thresholds = np.array(threshold_history)
    mean_thresh = np.mean(thresholds)
    
    return {
        'mean': mean_thresh,
        'std': np.std(thresholds),
        'coefficient_of_variation': np.std(thresholds) / (mean_thresh + 1e-10),
        'stability_score': 1.0 / (1.0 + np.std(thresholds))
    }


def information_content(states: StateArray) -> float:
    """
    Shannon entropy of state distribution
    
    Higher entropy = more information content
    
    Parameters
    ----------
    states : StateArray
        States to evaluate
        
    Returns
    -------
    entropy : float
        Information content (entropy)
    """
    counts = states.count_by_state()
    total = len(states)
    
    if total == 0:
        return 0.0
    
    probs = np.array([counts[s] / total for s in [EMPTY, DELTA, PHI, PSI]])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return entropy


def encoding_consistency(states: StateArray, 
                        feature_values: np.ndarray,
                        threshold: float) -> Dict:
    """
    Measure consistency of encoding with respect to threshold
    
    Parameters
    ----------
    states : StateArray
        Encoded states
    feature_values : np.ndarray
        Original feature values
    threshold : float
        Threshold used for encoding
        
    Returns
    -------
    consistency : dict
        Consistency metrics
    """
    if len(states) != len(feature_values):
        raise ValueError("States and features must have same length")
    
    # Check consistency of i-bit with threshold
    predicted_i = (feature_values > threshold).astype(int)
    actual_i = states.i
    
    i_consistency = np.mean(predicted_i == actual_i)
    
    # Check consistency of q-bit (assuming higher threshold)
    q_threshold = threshold * 1.5  # Example higher threshold
    predicted_q = (feature_values > q_threshold).astype(int)
    actual_q = states.q
    
    q_consistency = np.mean(predicted_q == actual_q)
    
    return {
        'i_bit_consistency': i_consistency,
        'q_bit_consistency': q_consistency,
        'overall_consistency': (i_consistency + q_consistency) / 2
    }


def state_transition_quality(states: StateArray) -> Dict:
    """
    Analyze quality of state transitions
    
    Good transitions should be:
    - Not too random (some structure)
    - Not too rigid (some flexibility)
    
    Parameters
    ----------
    states : StateArray
        Sequence of states
        
    Returns
    -------
    transition_quality : dict
        Transition quality metrics
    """
    if len(states) < 2:
        return {
            'transition_entropy': 0.0,
            'transition_regularity': 0.0,
            'overall_quality': 0.0
        }
    
    # Count transitions
    transitions = {}
    for i in range(len(states) - 1):
        from_state = states[i]
        to_state = states[i + 1]
        transition = (from_state, to_state)
        transitions[transition] = transitions.get(transition, 0) + 1
    
    # Transition entropy
    total_transitions = len(states) - 1
    transition_probs = np.array(list(transitions.values())) / total_transitions
    transition_entropy = -np.sum(transition_probs * np.log(transition_probs + 1e-10))
    
    # Transition regularity (inverse of entropy, normalized)
    max_entropy = np.log(len(transitions)) if len(transitions) > 1 else 0
    regularity = 1.0 - (transition_entropy / max_entropy) if max_entropy > 0 else 0.0
    
    return {
        'transition_entropy': transition_entropy,
        'transition_regularity': regularity,
        'num_transitions': len(transitions),
        'overall_quality': regularity
    }


def channel_correlation(channels: Dict[str, StateArray]) -> Dict:
    """
    Measure correlation between different channels
    
    Parameters
    ----------
    channels : Dict[str, StateArray]
        Dictionary of channel_name -> StateArray
        
    Returns
    -------
    correlations : dict
        Correlation metrics
    """
    if len(channels) < 2:
        return {'average_correlation': 0.0, 'correlations': {}}
    
    channel_names = list(channels.keys())
    correlations = {}
    
    for i, name1 in enumerate(channel_names):
        for j, name2 in enumerate(channel_names[i+1:], i+1):
            states1 = channels[name1]
            states2 = channels[name2]
            
            if len(states1) != len(states2):
                continue
            
            # Convert to integers for correlation
            ints1 = states1.to_ints()
            ints2 = states2.to_ints()
            
            # Compute correlation
            correlation = np.corrcoef(ints1, ints2)[0, 1]
            if not np.isnan(correlation):
                correlations[f"{name1}_{name2}"] = correlation
    
    avg_correlation = np.mean(list(correlations.values())) if correlations else 0.0
    
    return {
        'average_correlation': avg_correlation,
        'correlations': correlations
    }


def encoding_robustness(states: StateArray, 
                       feature_values: np.ndarray,
                       noise_levels: List[float]) -> Dict:
    """
    Test robustness of encoding to noise
    
    Parameters
    ----------
    states : StateArray
        Original encoded states
    feature_values : np.ndarray
        Original feature values
    noise_levels : List[float]
        Noise levels to test (0.0 to 1.0)
        
    Returns
    -------
    robustness : dict
        Robustness metrics
    """
    robustness_scores = []
    
    for noise_level in noise_levels:
        # Add noise to features
        noise = np.random.normal(0, noise_level, feature_values.shape)
        noisy_features = feature_values + noise
        
        # Re-encode with same threshold (simplified)
        threshold = np.median(feature_values)  # Example threshold
        noisy_states = StateArray.from_bits(
            i=(noisy_features > threshold).astype(int),
            q=(noisy_features > threshold * 1.5).astype(int)
        )
        
        # Compute accuracy
        accuracy = encoding_accuracy(states, noisy_states)
        robustness_scores.append(accuracy)
    
    return {
        'noise_levels': noise_levels,
        'robustness_scores': robustness_scores,
        'average_robustness': np.mean(robustness_scores),
        'robustness_trend': np.polyfit(noise_levels, robustness_scores, 1)[0]
    }


def comprehensive_quality_report(states: StateArray,
                               labels: Optional[np.ndarray] = None,
                               feature_values: Optional[np.ndarray] = None,
                               threshold: Optional[float] = None,
                               threshold_history: Optional[List[float]] = None) -> Dict:
    """
    Generate comprehensive quality report for channel encoding
    
    Parameters
    ----------
    states : StateArray
        Encoded states to evaluate
    labels : np.ndarray, optional
        True class labels for discrimination analysis
    feature_values : np.ndarray, optional
        Original feature values for consistency analysis
    threshold : float, optional
        Threshold used for encoding
    threshold_history : List[float], optional
        History of threshold values for stability analysis
        
    Returns
    -------
    report : dict
        Comprehensive quality report
    """
    report = {
        'basic_metrics': {},
        'distribution_quality': {},
        'discrimination_analysis': {},
        'consistency_analysis': {},
        'stability_analysis': {},
        'overall_score': 0.0
    }
    
    # Basic metrics
    report['basic_metrics'] = {
        'total_states': len(states),
        'information_content': information_content(states),
        'state_distribution': states.count_by_state()
    }
    
    # Distribution quality
    report['distribution_quality'] = state_distribution_quality(states)
    
    # Discrimination analysis
    if labels is not None:
        report['discrimination_analysis'] = {
            'discrimination_power': discrimination_power(states, labels)
        }
    
    # Consistency analysis
    if feature_values is not None and threshold is not None:
        report['consistency_analysis'] = encoding_consistency(
            states, feature_values, threshold
        )
    
    # Stability analysis
    if threshold_history is not None:
        report['stability_analysis'] = threshold_stability(threshold_history)
    
    # Transition quality
    report['transition_quality'] = state_transition_quality(states)
    
    # Overall score
    scores = []
    
    # Distribution quality score
    scores.append(report['distribution_quality'].get('overall_quality', 0.0))
    
    # Discrimination score
    if 'discrimination_analysis' in report:
        scores.append(report['discrimination_analysis'].get('discrimination_power', 0.0))
    
    # Consistency score
    if 'consistency_analysis' in report:
        scores.append(report['consistency_analysis'].get('overall_consistency', 0.0))
    
    # Stability score
    if 'stability_analysis' in report:
        scores.append(report['stability_analysis'].get('stability_score', 0.0))
    
    # Transition quality score
    scores.append(report['transition_quality'].get('overall_quality', 0.0))
    
    report['overall_score'] = np.mean(scores) if scores else 0.0
    
    return report


def compare_encodings(encodings: Dict[str, StateArray],
                     labels: Optional[np.ndarray] = None) -> Dict:
    """
    Compare multiple encodings and rank them
    
    Parameters
    ----------
    encodings : Dict[str, StateArray]
        Dictionary of encoding_name -> StateArray
    labels : np.ndarray, optional
        True class labels
        
    Returns
    -------
    comparison : dict
        Comparison results with rankings
    """
    results = {}
    
    for name, states in encodings.items():
        # Basic quality metrics
        distribution_quality = state_distribution_quality(states)
        transition_quality = state_transition_quality(states)
        
        result = {
            'distribution_quality': distribution_quality['overall_quality'],
            'transition_quality': transition_quality['overall_quality'],
            'information_content': information_content(states)
        }
        
        # Discrimination power if labels available
        if labels is not None:
            result['discrimination_power'] = discrimination_power(states, labels)
        
        # Overall score
        scores = [result['distribution_quality'], result['transition_quality']]
        if 'discrimination_power' in result:
            scores.append(result['discrimination_power'])
        
        result['overall_score'] = np.mean(scores)
        results[name] = result
    
    # Rank encodings
    ranked = sorted(results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    
    return {
        'results': results,
        'ranking': [name for name, _ in ranked],
        'best_encoding': ranked[0][0] if ranked else None
    }







