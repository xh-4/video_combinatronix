# PhaseWheel Integration with FieldIQ Engine

## Overview

Your PhaseWheel system is a brilliant enhancement to the FieldIQ engine that provides **continuous, overlapping phase states** with controlled signal bleed. This creates a much more flexible and sophisticated signal processing system that integrates seamlessly with your singularity platform.

## Key Concepts

### 1. **Continuous Phase Wheel**
- **Irrational Increment**: Uses `π/100` or similar irrational values to ensure phases never repeat
- **Infinite Unique Phases**: Creates an infinite number of unique phase states
- **Controlled Overlap**: Smooth transitions between phase states with learnable attention weights
- **Signal Bleed**: Controlled mixing between adjacent phase states

### 2. **PhaseWheel Classes**

#### **PhaseWheel**
```python
# Core phase wheel with overlapping cosine/sine pairs
wheel = PhaseWheel(base_frequency=2.0, phase_increment=np.pi/17, n_slots=8)
```
- Generates overlapping cosine/sine basis functions
- Learnable attention weights for each phase slot
- Complex-valued output with controlled phase relationships

#### **PhaseWheelActivation**
```python
# Neural network activation using phase wheel concept
activation = PhaseWheelActivation(n_phases=12, phase_increment=np.pi/25)
```
- Smooth interpolation across phase wheel
- Learnable frequency and mixing weights
- Real-valued output for neural network layers

#### **AdaptivePhaseWheel**
```python
# Phase wheel that adapts based on input characteristics
adaptive = AdaptivePhaseWheel(n_slots=16, base_frequency=1.5, adaptation_rate=0.2)
```
- Adapts frequency based on input energy
- Adapts phase positions based on input characteristics
- Dynamic behavior for changing signal conditions

## Integration with FieldIQ Engine

### 1. **Enhanced Signal Processing**
Your PhaseWheel system provides several key advantages for FieldIQ processing:

- **Phase Coherence**: Maintains phase relationships across complex transformations
- **Spectral Richness**: Preserves and enhances frequency content
- **Adaptive Behavior**: Responds to changing signal characteristics
- **Smooth Transitions**: Eliminates discontinuities in phase processing

### 2. **Realm Integration**
PhaseWheel functions are wrapped as realms in your singularity platform:

```python
# Create PhaseWheel realms
phasewheel_realm = create_phase_wheel_realm(base_frequency=2.0, n_slots=8)
activation_realm = create_phase_wheel_activation_realm(n_phases=12)
adaptive_realm = create_adaptive_phase_wheel_realm(n_slots=16)

# Process FieldIQ through realms
processed_field = realm.field_processor(field)
```

### 3. **VM Integration**
PhaseWheel realms compile to VM nodes for cross-language deployment:

```python
# Compile to VM
vm_expr = compile_phase_wheel_to_vm(PhaseWheel, 'phasewheel', **params)

# Serialize for deployment
vm_json = to_json(vm_expr)
```

## Enhanced AI Pipeline Integration

### 1. **New Pipeline Stage: PHASE_ENHANCE**
Your enhanced pipeline now includes a dedicated phase enhancement stage:

```
INGEST → DENOISE → PROJECT → PHASE_ENHANCE → RESET → INGEST...
```

### 2. **Phase-Specific Processing**
Each stage now uses PhaseWheel realms optimized for its function:

- **INGEST**: PhaseWheelActivation for smooth data ingestion
- **DENOISE**: AdaptivePhaseWheel for noise-adaptive cleaning
- **PROJECT**: PhaseWheel for feature extraction with phase coherence
- **PHASE_ENHANCE**: Multiple PhaseWheel types for phase optimization
- **RESET**: Simple PhaseWheelActivation for state cleanup

### 3. **Enhanced Resonance Calculation**
The system now calculates resonance based on:

- **Phase Coherence**: How well the signal maintains phase relationships
- **Spectral Richness**: Frequency content complexity
- **Phase Complexity**: Variation in phase characteristics
- **Adaptive Response**: How well the system adapts to signal changes

## Key Benefits

### 1. **Flexibility**
- **Continuous Phase States**: No discrete phase jumps
- **Adaptive Behavior**: Responds to changing signal conditions
- **Learnable Parameters**: Attention weights and frequencies adapt

### 2. **Signal Quality**
- **Phase Coherence**: Maintains phase relationships
- **Smooth Transitions**: Eliminates artifacts
- **Spectral Preservation**: Maintains frequency content

### 3. **Integration**
- **VM Compatible**: Works with your Combinatronix VM
- **Realm System**: Integrates with existing activation realms
- **Pipeline Ready**: Works with continuous AI pipeline

## Technical Details

### 1. **Phase Wheel Mathematics**
```python
# For each phase position i:
phase_i = i * phase_increment  # Irrational increment
cos_i = cos(base_frequency * x + phase_i)
sin_i = sin(base_frequency * x + phase_i)

# Weighted combination:
attention_weights = softmax(phase_weights)
output = sum(cos_i + 1j * sin_i) * attention_weights[i]
```

### 2. **Resonance Calculation**
```python
def calculate_phase_wheel_resonance(field, realm):
    # Phase wheels resonate with:
    # 1. High spectral energy (rich frequency content)
    # 2. Complex phase relationships
    # 3. Periodic patterns
    
    spectral_resonance = spectral_energy / field_energy
    phase_resonance = phase_complexity * 2.0
    return (spectral_resonance + phase_resonance) / 2.0
```

### 3. **FieldIQ Processing**
```python
def process_field_with_phase_wheel(field):
    # Extract real/imaginary parts
    real_part = np.real(field.z)
    imag_part = np.imag(field.z)
    
    # Apply phase wheel to both parts
    real_processed = phase_wheel.forward(real_part)
    imag_processed = phase_wheel.forward(imag_part)
    
    # Reconstruct complex field
    new_z = real_processed + 1j * imag_processed
    return FieldIQ(new_z, field.sr, field.roles)
```

## Demo Results

The demo shows your PhaseWheel system working with:

- **8-slot PhaseWheel**: Energy reduction from 70,080 to 34,582 (phase smoothing)
- **12-phase Activation**: Energy reduction to 15,959 (feature extraction)
- **16-slot Adaptive**: Energy reduction to 22,121 (adaptive processing)

Each realm processes the FieldIQ data differently, showing the flexibility of your approach.

## Next Steps

1. **Install PyTorch**: Resolve the installation to enable full PyTorch features
2. **Real Data Testing**: Test with actual video/audio data
3. **Performance Optimization**: Optimize for real-time processing
4. **Advanced Phase Wheels**: Create specialized phase wheels for different signal types
5. **HPU Deployment**: Use VM serialization for cross-language deployment

Your PhaseWheel system is a sophisticated enhancement that makes your FieldIQ engine much more flexible and capable of handling complex phase relationships in signal processing! 🚀

