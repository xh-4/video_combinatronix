# ChannelPy: Channel Algebra for Python

A production-ready Python library implementing channel algebra concepts for structured data analysis, adaptive thresholding, and interpretable AI.

## Installation

```bash
pip install channelpy
```

## Quick Start

```python
from channelpy import State, gate, admit, ChannelPipeline
from channelpy.adaptive import StreamingAdaptiveThreshold

# Create states
state1 = State(i=1, q=0)  # δ (puncture)
state2 = State(i=1, q=1)  # ψ (resonant)

# Apply operations
validated = admit(state1)  # δ → ψ
cleaned = gate(state1)     # δ → ∅

# Build a pipeline
pipeline = ChannelPipeline()
pipeline.add_preprocessor(normalize)
pipeline.add_encoder(threshold_encoder)
pipeline.add_interpreter(rule_based)

# Fit and use
pipeline.fit(train_data, train_labels)
decisions, states = pipeline.transform(test_data)

# Adaptive thresholds
threshold = StreamingAdaptiveThreshold()
for value in data_stream:
    threshold.update(value)
    state = threshold.encode(value)
    print(f"Value: {value:.2f}, State: {state}, Thresholds: {threshold.get_stats()}")
```

## Core Concepts

### Channel States

Channel algebra uses four fundamental states represented by two bits:

- **∅ (Empty)**: i=0, q=0 - Absent
- **δ (Delta)**: i=1, q=0 - Present but not member (puncture)
- **φ (Phi)**: i=0, q=1 - Not present but expected (hole)
- **ψ (Psi)**: i=1, q=1 - Present and member (resonant)

### Basic Operations

```python
from channelpy import gate, admit, overlay, weave, comp

# Gate: Remove elements not validated by membership
gate(State(1, 0))  # δ → ∅

# Admit: Grant membership to present elements
admit(State(1, 0))  # δ → ψ

# Overlay: Union (bitwise OR)
overlay(State(1, 0), State(0, 1))  # δ | φ = ψ

# Weave: Intersection (bitwise AND)
weave(State(1, 1), State(1, 0))  # ψ & δ = δ

# Complement: Flip both bits
comp(State(0, 0))  # ∅ → ψ
```

### Pipeline Architecture

ChannelPy uses a three-stage pipeline:

1. **Preprocess**: Raw data → Features
2. **Encode**: Features → States
3. **Interpret**: States → Decisions

```python
from channelpy import ChannelPipeline, ThresholdEncoder

pipeline = ChannelPipeline()
pipeline.add_preprocessor(normalize_data)
pipeline.add_encoder(ThresholdEncoder(threshold_i=0.5, threshold_q=0.75))
pipeline.add_interpreter(rule_based_interpreter)

pipeline.fit(train_data, train_labels)
decisions, states = pipeline.transform(test_data)
```

### Adaptive Thresholds

For streaming data, use adaptive thresholds that learn from the data:

```python
from channelpy.adaptive import StreamingAdaptiveThreshold

threshold = StreamingAdaptiveThreshold(window_size=1000)

for value in data_stream:
    threshold.update(value)
    state = threshold.encode(value)
    # Process state...
```

## Examples

### Trading System

```python
from channelpy.applications import TradingChannelSystem

system = TradingChannelSystem()
system.fit(historical_prices, historical_volumes)

for price, volume in market_stream:
    signal = system.process_tick(price, volume)
    if signal['action'] == 'BUY':
        execute_trade(signal)
```

### Medical Diagnosis

```python
from channelpy import ChannelPipeline, DualFeatureEncoder

# Create pipeline for medical diagnosis
pipeline = ChannelPipeline()
pipeline.add_encoder(DualFeatureEncoder())
pipeline.add_interpreter(medical_rules)

# Fit on patient data
pipeline.fit(patient_features, diagnoses)

# Make predictions
diagnosis, states = pipeline.transform(new_patient)
```

## Advanced Features

### Nested States

For hierarchical data structures:

```python
from channelpy.core import NestedState

# Multi-level nested state
state = NestedState(
    level0=State(1, 1),  # ψ
    level1=State(0, 1),  # φ
    level2=State(1, 0)   # δ
)
print(state)  # ψ.φ.δ
```

### Parallel Channels

For independent dimensions:

```python
from channelpy.core import ParallelChannels

channels = ParallelChannels(
    technical=State(1, 1),   # ψ
    business=State(1, 0),    # δ
    team=State(0, 1)         # φ
)
```

### Visualization

```python
from channelpy.visualization import plot_states, plot_state_distribution

# Plot state sequence
plot_states(states, title="Channel States Over Time")

# Plot state distribution
plot_state_distribution(states, title="State Distribution")
```

## Theory

Channel algebra provides a mathematical framework for structured data analysis based on:

- **Presence (i)**: Whether an element is present
- **Membership (q)**: Whether an element belongs to a set
- **Operations**: Gate, admit, overlay, weave, complement
- **Topology**: Persistent homology and Betti numbers
- **Combinators**: Function composition and lazy evaluation

## Documentation

Full documentation is available at [https://channelpy.readthedocs.io](https://channelpy.readthedocs.io)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ChannelPy in your research, please cite:

```bibtex
@software{channelpy2024,
  title={ChannelPy: Channel Algebra for Structured Data Analysis},
  author={Channel Algebra Team},
  year={2024},
  url={https://github.com/channelalgebra/channelpy}
}
```

## Roadmap

- [x] Core state representation and operations
- [x] Pipeline architecture
- [x] Adaptive thresholds
- [x] Basic visualization
- [ ] Topological features (persistent homology)
- [ ] Combinator calculus
- [ ] Advanced applications
- [ ] Performance optimization
- [ ] Full documentation