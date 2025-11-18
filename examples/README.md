# ChannelPy Examples

This directory contains comprehensive examples and tutorials for the ChannelPy library.

## Quick Start Examples

### Basic Usage
- `basic_usage.py` - Simple state operations and basic pipeline
- `quick_start.py` - Minimal example to get started quickly

### Complete Tutorials
- `complete_tutorial.py` - Comprehensive tutorial covering all features
- `trading_example.py` - Real-world trading system example
- `preprocessing_example.py` - Detailed preprocessing examples
- `interpreters_example.py` - All interpreter types demonstrated

## Data Generation

### Synthetic Datasets
- `datasets.py` - Data generation functions for testing and tutorials
  - `make_classification_data()` - Classification datasets
  - `make_trading_data()` - OHLCV trading data
  - `make_medical_data()` - Medical diagnosis data
  - `make_time_series_data()` - Time series with trends and seasonality
  - `make_state_sequence()` - Markov state sequences
  - `generate_streaming_data()` - Infinite streaming data generator

### Pre-defined Datasets
- `load_example_dataset()` - Load common datasets:
  - `'iris_simple'` - Simplified iris dataset
  - `'trading_sample'` - Sample trading data
  - `'medical_sample'` - Sample medical data

## Domain-Specific Examples

### Financial/Trading
- `trading_example.py` - Complete trading system
- `risk_management_example.py` - Risk assessment using channel algebra
- `portfolio_optimization_example.py` - Multi-asset portfolio analysis

### Medical/Healthcare
- `medical_diagnosis_example.py` - Medical diagnosis system
- `patient_monitoring_example.py` - Real-time patient monitoring
- `drug_effectiveness_example.py` - Drug trial analysis

### Industrial/Manufacturing
- `quality_control_example.py` - Manufacturing quality control
- `predictive_maintenance_example.py` - Equipment maintenance prediction
- `anomaly_detection_example.py` - Industrial anomaly detection

### Scientific Research
- `experiment_analysis_example.py` - Scientific experiment analysis
- `hypothesis_testing_example.py` - Statistical hypothesis testing
- `data_validation_example.py` - Research data validation

## Advanced Examples

### Custom Components
- `custom_encoder_example.py` - Building custom encoders
- `custom_interpreter_example.py` - Building custom interpreters
- `custom_preprocessor_example.py` - Building custom preprocessors

### Performance Optimization
- `performance_benchmark.py` - Performance comparison and optimization
- `memory_efficient_example.py` - Large-scale data processing
- `parallel_processing_example.py` - Multi-core processing

### Integration Examples
- `sklearn_integration.py` - Integration with scikit-learn
- `pandas_integration.py` - Integration with pandas DataFrames
- `matplotlib_integration.py` - Advanced plotting and visualization

## Running Examples

### Prerequisites
```bash
pip install channelpy
```

### Basic Example
```python
from channelpy import State, PSI, DELTA, PHI, EMPTY
from channelpy.examples.datasets import make_classification_data

# Generate data
X, y = make_classification_data(n_samples=100)

# Create states
states = [PSI if label == 1 else EMPTY for label in y]
print(f"Generated {len(states)} states")
```

### Complete Tutorial
```python
# Run the complete tutorial
python examples/complete_tutorial.py
```

### Specific Examples
```python
# Run trading example
python examples/trading_example.py

# Run medical example
python examples/medical_diagnosis_example.py
```

## Example Outputs

Most examples generate:
- **Console output** - Progress and results
- **Plots** - Visualizations saved as PNG files
- **Data files** - Generated datasets and results

## Contributing Examples

To add new examples:

1. Create a new Python file in this directory
2. Follow the naming convention: `domain_example.py`
3. Include comprehensive docstrings
4. Add to this README
5. Test with different datasets and parameters

## Example Structure

Each example should include:

```python
"""
Example: Brief description

Detailed explanation of what this example demonstrates
and how to use it.
"""

import numpy as np
from channelpy import ...

def main():
    """Main example function"""
    # 1. Data generation/preparation
    # 2. Pipeline setup
    # 3. Processing
    # 4. Analysis
    # 5. Visualization
    # 6. Results

if __name__ == "__main__":
    main()
```

## Getting Help

- Check the [main documentation](../README.md)
- Look at the [API reference](../docs/api.md)
- Run examples with `--help` for additional options
- Open issues for questions or suggestions







