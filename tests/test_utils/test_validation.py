"""
Tests for utils.validation module
"""
import pytest
import numpy as np
from channelpy.utils.validation import (
    ValidationError, validate_state_bits, validate_threshold, validate_threshold_pair,
    validate_array_shape, validate_probability, validate_window_size, validate_positive,
    validate_non_negative, validate_fitted, validate_same_length, validate_input,
    require_fitted, validate_state_input, validate_array_input, ValidationContext,
    validate_pipeline_data, validate_encoder_thresholds
)


def test_validate_state_bits():
    """Test state bit validation"""
    # Valid bits
    validate_state_bits(0, 0)
    validate_state_bits(1, 1)
    validate_state_bits(0, 1)
    validate_state_bits(1, 0)
    
    # Invalid bits
    with pytest.raises(ValidationError):
        validate_state_bits(2, 0)
    with pytest.raises(ValidationError):
        validate_state_bits(0, -1)
    with pytest.raises(ValidationError):
        validate_state_bits(1.5, 0)


def test_validate_threshold():
    """Test threshold validation"""
    # Valid thresholds
    validate_threshold(0.5)
    validate_threshold(1.0)
    validate_threshold(-1.0)
    validate_threshold(0)
    
    # Invalid thresholds
    with pytest.raises(ValidationError):
        validate_threshold("0.5")
    with pytest.raises(ValidationError):
        validate_threshold(np.inf)
    with pytest.raises(ValidationError):
        validate_threshold(np.nan)


def test_validate_threshold_pair():
    """Test threshold pair validation"""
    # Valid pairs
    validate_threshold_pair(0.3, 0.7)
    validate_threshold_pair(0.5, 0.5)
    validate_threshold_pair(-1.0, 1.0)
    
    # Invalid pairs
    with pytest.raises(ValidationError):
        validate_threshold_pair(0.7, 0.3)  # i > q
    with pytest.raises(ValidationError):
        validate_threshold_pair("0.5", 0.7)  # Non-numeric


def test_validate_array_shape():
    """Test array shape validation"""
    # Valid arrays
    arr1d = np.array([1, 2, 3])
    arr2d = np.array([[1, 2], [3, 4]])
    
    validate_array_shape(arr1d, min_dims=1)
    validate_array_shape(arr2d, expected_shape=(2, 2))
    validate_array_shape(arr2d, min_dims=2, max_dims=2)
    
    # Invalid arrays
    with pytest.raises(ValidationError):
        validate_array_shape([1, 2, 3])  # Not numpy array
    with pytest.raises(ValidationError):
        validate_array_shape(arr1d, expected_shape=(2, 2))  # Wrong dimensions
    with pytest.raises(ValidationError):
        validate_array_shape(arr1d, min_dims=2)  # Too few dimensions


def test_validate_probability():
    """Test probability validation"""
    # Valid probabilities
    validate_probability(0.0)
    validate_probability(1.0)
    validate_probability(0.5)
    
    # Invalid probabilities
    with pytest.raises(ValidationError):
        validate_probability(-0.1)
    with pytest.raises(ValidationError):
        validate_probability(1.1)
    with pytest.raises(ValidationError):
        validate_probability("0.5")


def test_validate_window_size():
    """Test window size validation"""
    # Valid window sizes
    validate_window_size(1)
    validate_window_size(100)
    
    # Invalid window sizes
    with pytest.raises(ValidationError):
        validate_window_size(0)
    with pytest.raises(ValidationError):
        validate_window_size(-1)
    with pytest.raises(ValidationError):
        validate_window_size(1.5)


def test_validate_positive():
    """Test positive value validation"""
    # Valid positive values
    validate_positive(1)
    validate_positive(0.1)
    validate_positive(100)
    
    # Invalid values
    with pytest.raises(ValidationError):
        validate_positive(0)
    with pytest.raises(ValidationError):
        validate_positive(-1)
    with pytest.raises(ValidationError):
        validate_positive("1")


def test_validate_non_negative():
    """Test non-negative value validation"""
    # Valid non-negative values
    validate_non_negative(0)
    validate_non_negative(1)
    validate_non_negative(0.5)
    
    # Invalid values
    with pytest.raises(ValidationError):
        validate_non_negative(-0.1)
    with pytest.raises(ValidationError):
        validate_non_negative(-1)


def test_validate_fitted():
    """Test fitted object validation"""
    class MockFitted:
        def __init__(self, fitted=True):
            self.is_fitted = fitted
    
    # Fitted object
    fitted_obj = MockFitted(True)
    validate_fitted(fitted_obj)
    
    # Not fitted
    unfitted_obj = MockFitted(False)
    with pytest.raises(ValidationError):
        validate_fitted(unfitted_obj)
    
    # No is_fitted attribute
    no_attr_obj = object()
    with pytest.raises(ValidationError):
        validate_fitted(no_attr_obj)


def test_validate_same_length():
    """Test same length validation"""
    # Same length arrays
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]
    validate_same_length(arr1, arr2)
    
    # Different length arrays
    arr3 = [1, 2]
    with pytest.raises(ValidationError):
        validate_same_length(arr1, arr3)
    
    # With names
    with pytest.raises(ValidationError):
        validate_same_length(arr1, arr3, names=['array1', 'array3'])


def test_validate_input_decorator():
    """Test validate_input decorator"""
    @validate_input(threshold=validate_threshold, window_size=validate_window_size)
    def test_func(threshold, window_size, other_param):
        return threshold + window_size
    
    # Valid inputs
    result = test_func(0.5, 10, "other")
    assert result == 10.5
    
    # Invalid inputs
    with pytest.raises(ValidationError):
        test_func("invalid", 10, "other")
    
    with pytest.raises(ValidationError):
        test_func(0.5, -1, "other")


def test_require_fitted_decorator():
    """Test require_fitted decorator"""
    class MockModel:
        def __init__(self, fitted=True):
            self.is_fitted = fitted
        
        @require_fitted
        def predict(self, X):
            return X
    
    # Fitted model
    fitted_model = MockModel(True)
    result = fitted_model.predict([1, 2, 3])
    assert result == [1, 2, 3]
    
    # Not fitted model
    unfitted_model = MockModel(False)
    with pytest.raises(ValidationError):
        unfitted_model.predict([1, 2, 3])


def test_validate_state_input_decorator():
    """Test validate_state_input decorator"""
    from channelpy.core.state import State
    
    @validate_state_input
    def test_func(state):
        return state.i + state.q
    
    # Valid state
    state = State(1, 1)
    result = test_func(state)
    assert result == 2
    
    # This would be caught by State constructor, but decorator provides extra safety
    # (State constructor already validates bits)


def test_validate_array_input_decorator():
    """Test validate_array_input decorator"""
    @validate_array_input(min_dims=1, max_dims=2)
    def test_func(arr):
        return arr.shape
    
    # Valid array
    arr1d = np.array([1, 2, 3])
    result = test_func(arr1d)
    assert result == (3,)
    
    # Invalid array
    arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    with pytest.raises(ValidationError):
        test_func(arr3d)


def test_validation_context():
    """Test ValidationContext"""
    with ValidationContext("Test context"):
        validate_threshold(0.5)  # Should work
    
    with pytest.raises(ValidationError) as exc_info:
        with ValidationContext("Test context"):
            validate_threshold("invalid")
    
    assert "Test context" in str(exc_info.value)


def test_validate_pipeline_data():
    """Test pipeline data validation"""
    # Valid data
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    validate_pipeline_data(X, y)
    
    # Invalid X
    X_invalid = np.array([[1, 2], [np.nan, 4]])
    with pytest.raises(ValidationError):
        validate_pipeline_data(X_invalid)
    
    # Mismatched lengths
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0])
    with pytest.raises(ValidationError):
        validate_pipeline_data(X, y)


def test_validate_encoder_thresholds():
    """Test encoder threshold validation"""
    # Valid thresholds
    validate_encoder_thresholds(0.3, 0.7)
    
    # Invalid thresholds
    with pytest.raises(ValidationError):
        validate_encoder_thresholds(0.7, 0.3)







