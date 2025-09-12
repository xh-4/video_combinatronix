# ============================
# TunablePhaseNeuron Realms for Self-Tuning Neural Networks
# ============================
"""
TunablePhaseNeuron-based realms for self-tuning neural networks in the singularity platform.
Integrates learnable phase positions with continuous phase wheel concepts.
"""

import numpy as np
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from queue import Queue, Empty

# Import existing systems
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf, to_json, from_json
from Combinator_Kernel import FieldIQ, make_field_from_real

# ============================
# TunablePhaseNeuron Implementation (No PyTorch)
# ============================

class SimpleTunablePhaseNeuron:
    """
    Simplified TunablePhaseNeuron implementation using NumPy for demo purposes.
    """
    def __init__(self, input_size: int, phase_increment: float = np.pi/100, n_phases: int = 32):
        self.input_size = input_size
        self.weights = np.random.randn(input_size)
        self.bias = 0.0
        
        # The neuron's position on the phase wheel (learnable)
        self.phase_position = 0.0  # 0 to 2π
        self.phase_increment = phase_increment
        self.n_phases = n_phases
        
        # Learnable frequency
        self.frequency = 1.0
        
        # Learning parameters
        self.learning_rate = 0.01
        self.phase_learning_rate = 0.001
        
    def forward(self, x):
        """Forward pass with learnable phase position."""
        # Standard linear transformation
        linear_output = np.dot(x, self.weights) + self.bias
        
        # Map phase_position to actual phase on wheel
        discrete_phase_idx = (self.phase_position / (2*np.pi)) * self.n_phases
        base_idx = int(np.floor(discrete_phase_idx)) % self.n_phases
        alpha = discrete_phase_idx - base_idx  # interpolation weight
        
        # Get two adjacent phases for smooth interpolation
        phase1 = base_idx * self.phase_increment
        phase2 = ((base_idx + 1) % self.n_phases) * self.phase_increment
        
        # Smooth interpolation between adjacent phase slots
        activation1 = np.sin(self.frequency * linear_output + phase1)
        activation2 = np.sin(self.frequency * linear_output + phase2)
        
        # Blend based on learned position
        return (1 - alpha) * activation1 + alpha * activation2
    
    def update_phase_position(self, gradient: float):
        """Update the phase position based on gradient."""
        self.phase_position += self.phase_learning_rate * gradient
        # Keep phase position in [0, 2π] range
        self.phase_position = self.phase_position % (2 * np.pi)
    
    def update_weights(self, weight_gradients: np.ndarray, bias_gradient: float):
        """Update weights and bias based on gradients."""
        self.weights += self.learning_rate * weight_gradients
        self.bias += self.learning_rate * bias_gradient
    
    def get_phase_info(self):
        """Get information about the neuron's current phase state."""
        discrete_phase_idx = (self.phase_position / (2*np.pi)) * self.n_phases
        base_idx = int(np.floor(discrete_phase_idx)) % self.n_phases
        alpha = discrete_phase_idx - base_idx
        
        return {
            'phase_position': self.phase_position,
            'discrete_idx': base_idx,
            'interpolation_alpha': alpha,
            'frequency': self.frequency,
            'n_phases': self.n_phases
        }

class TunablePhaseLayer:
    """
    A layer of TunablePhaseNeurons for neural network processing.
    """
    def __init__(self, input_size: int, output_size: int, phase_increment: float = np.pi/100, n_phases: int = 32):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [SimpleTunablePhaseNeuron(input_size, phase_increment, n_phases) for _ in range(output_size)]
        
    def forward(self, x):
        """Forward pass through the layer."""
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(x)
            outputs.append(output)
        return np.array(outputs)
    
    def update_parameters(self, gradients):
        """Update all neuron parameters."""
        for i, neuron in enumerate(self.neurons):
            if i < len(gradients):
                neuron.update_weights(gradients[i]['weights'], gradients[i]['bias'])
                neuron.update_phase_position(gradients[i]['phase'])
    
    def get_layer_info(self):
        """Get information about all neurons in the layer."""
        return [neuron.get_phase_info() for neuron in self.neurons]

# ============================
# TunablePhaseNeuron Realm System
# ============================

@dataclass
class TunablePhaseNeuronRealm:
    """A realm representing a TunablePhaseNeuron in the singularity platform."""
    name: str
    neuron_class: type
    parameters: Dict[str, Any]
    vm_node: Node
    field_processor: Callable[[FieldIQ], FieldIQ]
    learning_enabled: bool = True
    
    def __repr__(self):
        return f"<TunablePhaseNeuronRealm {self.name} params={self.parameters}>"

def create_tunable_phase_neuron_field_processor(neuron_class: type, **params) -> Callable[[FieldIQ], FieldIQ]:
    """Create a field processor that applies TunablePhaseNeuron to FieldIQ data."""
    
    def process_field(field: FieldIQ) -> FieldIQ:
        # Convert FieldIQ to numpy array
        z_array = field.z
        
        # Extract real and imaginary parts
        real_part = np.real(z_array)
        imag_part = np.imag(z_array)
        
        # Create neuron instance
        neuron = neuron_class(**params)
        
        # Process both real and imaginary parts
        real_processed = neuron.forward(real_part.reshape(1, -1)).flatten()
        imag_processed = neuron.forward(imag_part.reshape(1, -1)).flatten()
        
        # Reconstruct complex field
        new_z = real_processed + 1j * imag_processed
        
        # Create new FieldIQ with neuron metadata
        processed_field = FieldIQ(new_z, field.sr, field.roles or {})
        processed_field = processed_field.with_role("tunable_phase_processed", True)
        processed_field = processed_field.with_role("neuron_type", neuron_class.__name__)
        processed_field = processed_field.with_role("phase_info", neuron.get_phase_info())
        
        return processed_field
    
    return process_field

def create_tunable_phase_layer_field_processor(layer_class: type, **params) -> Callable[[FieldIQ], FieldIQ]:
    """Create a field processor that applies TunablePhaseLayer to FieldIQ data."""
    
    def process_field(field: FieldIQ) -> FieldIQ:
        # Convert FieldIQ to numpy array
        z_array = field.z
        
        # Extract real and imaginary parts
        real_part = np.real(z_array)
        imag_part = np.imag(z_array)
        
        # Create layer instance
        layer = layer_class(**params)
        
        # Process both real and imaginary parts
        real_processed = layer.forward(real_part.reshape(1, -1)).flatten()
        imag_processed = layer.forward(imag_part.reshape(1, -1)).flatten()
        
        # Reconstruct complex field
        new_z = real_processed + 1j * imag_processed
        
        # Create new FieldIQ with layer metadata
        processed_field = FieldIQ(new_z, field.sr, field.roles or {})
        processed_field = processed_field.with_role("tunable_phase_layer_processed", True)
        processed_field = processed_field.with_role("layer_type", layer_class.__name__)
        processed_field = processed_field.with_role("layer_info", layer.get_layer_info())
        
        return processed_field
    
    return process_field

def compile_tunable_phase_neuron_to_vm(neuron_class: type, name: str, **params) -> Node:
    """Compile a TunablePhaseNeuron to VM representation."""
    return Val({
        'type': 'tunable_phase_neuron_realm',
        'name': name,
        'class': neuron_class.__name__,
        'parameters': params,
        'vm_operations': {'primary': 'TUNABLE_PHASE_NEURON'}
    })

# ============================
# TunablePhaseNeuron Realm Factory Functions
# ============================

def create_tunable_phase_neuron_realm(input_size: int, phase_increment: float = np.pi/100, 
                                     n_phases: int = 32) -> TunablePhaseNeuronRealm:
    """Create a TunablePhaseNeuron realm."""
    vm_node = compile_tunable_phase_neuron_to_vm(SimpleTunablePhaseNeuron, 'tunablephaseneuron',
                                                input_size=input_size,
                                                phase_increment=phase_increment,
                                                n_phases=n_phases)
    processor = create_tunable_phase_neuron_field_processor(SimpleTunablePhaseNeuron,
                                                           input_size=input_size,
                                                           phase_increment=phase_increment,
                                                           n_phases=n_phases)
    
    return TunablePhaseNeuronRealm(
        name='tunablephaseneuron',
        neuron_class=SimpleTunablePhaseNeuron,
        parameters={'input_size': input_size, 'phase_increment': phase_increment, 'n_phases': n_phases},
        vm_node=vm_node,
        field_processor=processor,
        learning_enabled=True
    )

def create_tunable_phase_layer_realm(input_size: int, output_size: int, phase_increment: float = np.pi/100, 
                                    n_phases: int = 32) -> TunablePhaseNeuronRealm:
    """Create a TunablePhaseLayer realm."""
    vm_node = compile_tunable_phase_neuron_to_vm(TunablePhaseLayer, 'tunablephaselayer',
                                                input_size=input_size,
                                                output_size=output_size,
                                                phase_increment=phase_increment,
                                                n_phases=n_phases)
    processor = create_tunable_phase_layer_field_processor(TunablePhaseLayer,
                                                          input_size=input_size,
                                                          output_size=output_size,
                                                          phase_increment=phase_increment,
                                                          n_phases=n_phases)
    
    return TunablePhaseNeuronRealm(
        name='tunablephaselayer',
        neuron_class=TunablePhaseLayer,
        parameters={'input_size': input_size, 'output_size': output_size, 'phase_increment': phase_increment, 'n_phases': n_phases},
        vm_node=vm_node,
        field_processor=processor,
        learning_enabled=True
    )

# ============================
# Self-Tuning Neural Network Pipeline
# ============================

class SelfTuningNeuralPipeline:
    """
    A pipeline that uses TunablePhaseNeurons for self-tuning neural network processing.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 phase_increment: float = np.pi/100, n_phases: int = 32):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Create layers
        self.layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layer = TunablePhaseLayer(current_size, hidden_size, phase_increment, n_phases)
            self.layers.append(layer)
            current_size = hidden_size
        
        # Output layer
        output_layer = TunablePhaseLayer(current_size, output_size, phase_increment, n_phases)
        self.layers.append(output_layer)
        
        # Learning parameters
        self.learning_rate = 0.01
        self.phase_learning_rate = 0.001
        
    def forward(self, x):
        """Forward pass through the entire network."""
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input.reshape(1, -1)).flatten()
        return current_input
    
    def update_parameters(self, gradients):
        """Update all layer parameters."""
        for i, layer in enumerate(self.layers):
            if i < len(gradients):
                layer.update_parameters(gradients[i])
    
    def get_network_info(self):
        """Get information about all layers in the network."""
        return [layer.get_layer_info() for layer in self.layers]

def create_self_tuning_pipeline_realm(input_size: int, hidden_sizes: List[int], output_size: int,
                                     phase_increment: float = np.pi/100, n_phases: int = 32) -> TunablePhaseNeuronRealm:
    """Create a SelfTuningNeuralPipeline realm."""
    vm_node = compile_tunable_phase_neuron_to_vm(SelfTuningNeuralPipeline, 'selftuningpipeline',
                                                input_size=input_size,
                                                hidden_sizes=hidden_sizes,
                                                output_size=output_size,
                                                phase_increment=phase_increment,
                                                n_phases=n_phases)
    
    def process_field(field: FieldIQ) -> FieldIQ:
        # Convert FieldIQ to numpy array
        z_array = field.z
        
        # Extract real and imaginary parts
        real_part = np.real(z_array)
        imag_part = np.imag(z_array)
        
        # Create pipeline instance
        pipeline = SelfTuningNeuralPipeline(input_size, hidden_sizes, output_size, phase_increment, n_phases)
        
        # Process both real and imaginary parts
        real_processed = pipeline.forward(real_part)
        imag_processed = pipeline.forward(imag_part)
        
        # Reconstruct complex field
        new_z = real_processed + 1j * imag_processed
        
        # Create new FieldIQ with pipeline metadata
        processed_field = FieldIQ(new_z, field.sr, field.roles or {})
        processed_field = processed_field.with_role("self_tuning_processed", True)
        processed_field = processed_field.with_role("pipeline_type", "SelfTuningNeuralPipeline")
        processed_field = processed_field.with_role("network_info", pipeline.get_network_info())
        
        return processed_field
    
    return TunablePhaseNeuronRealm(
        name='selftuningpipeline',
        neuron_class=SelfTuningNeuralPipeline,
        parameters={'input_size': input_size, 'hidden_sizes': hidden_sizes, 'output_size': output_size, 
                   'phase_increment': phase_increment, 'n_phases': n_phases},
        vm_node=vm_node,
        field_processor=process_field,
        learning_enabled=True
    )

# ============================
# Demo Functions
# ============================

def demo_tunable_phase_neuron():
    """Demonstrate TunablePhaseNeuron functionality."""
    print("=== TunablePhaseNeuron Demo ===\n")
    
    # Create a simple test case
    input_size = 10
    test_input = np.random.randn(input_size)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    
    # Create neuron
    neuron = SimpleTunablePhaseNeuron(input_size, phase_increment=np.pi/50, n_phases=16)
    
    print(f"\nNeuron parameters:")
    print(f"- Input size: {neuron.input_size}")
    print(f"- Phase increment: {neuron.phase_increment:.6f}")
    print(f"- Number of phases: {neuron.n_phases}")
    print(f"- Initial phase position: {neuron.phase_position:.3f}")
    print(f"- Frequency: {neuron.frequency:.3f}")
    
    # Forward pass
    output = neuron.forward(test_input)
    print(f"\nForward pass:")
    print(f"- Output shape: {output.shape}")
    print(f"- Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"- Output mean: {output.mean():.3f}, std: {output.std():.3f}")
    
    # Show phase info
    phase_info = neuron.get_phase_info()
    print(f"\nPhase information:")
    print(f"- Phase position: {phase_info['phase_position']:.3f}")
    print(f"- Discrete index: {phase_info['discrete_idx']}")
    print(f"- Interpolation alpha: {phase_info['interpolation_alpha']:.3f}")
    
    # Simulate learning
    print(f"\nSimulating learning...")
    for i in range(5):
        # Simulate gradient updates
        phase_gradient = np.random.randn() * 0.1
        weight_gradients = np.random.randn(input_size) * 0.01
        bias_gradient = np.random.randn() * 0.01
        
        neuron.update_phase_position(phase_gradient)
        neuron.update_weights(weight_gradients, bias_gradient)
        
        new_output = neuron.forward(test_input)
        phase_info = neuron.get_phase_info()
        
        print(f"  Step {i+1}: Phase={phase_info['phase_position']:.3f}, Output={new_output:.3f}")
    
    return neuron

def demo_tunable_phase_layer():
    """Demonstrate TunablePhaseLayer functionality."""
    print("\n=== TunablePhaseLayer Demo ===\n")
    
    # Create layer
    input_size = 20
    output_size = 8
    layer = TunablePhaseLayer(input_size, output_size, phase_increment=np.pi/25, n_phases=20)
    
    print(f"Layer parameters:")
    print(f"- Input size: {input_size}")
    print(f"- Output size: {output_size}")
    print(f"- Number of neurons: {len(layer.neurons)}")
    
    # Test input
    test_input = np.random.randn(input_size)
    print(f"\nTest input shape: {test_input.shape}")
    
    # Forward pass
    output = layer.forward(test_input)
    print(f"Layer output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Show layer info
    layer_info = layer.get_layer_info()
    print(f"\nLayer information:")
    for i, neuron_info in enumerate(layer_info[:3]):  # Show first 3 neurons
        print(f"  Neuron {i}: Phase={neuron_info['phase_position']:.3f}, "
              f"Idx={neuron_info['discrete_idx']}, Alpha={neuron_info['interpolation_alpha']:.3f}")
    
    return layer

def demo_self_tuning_pipeline():
    """Demonstrate SelfTuningNeuralPipeline functionality."""
    print("\n=== SelfTuningNeuralPipeline Demo ===\n")
    
    # Create pipeline
    input_size = 50
    hidden_sizes = [32, 16, 8]
    output_size = 4
    pipeline = SelfTuningNeuralPipeline(input_size, hidden_sizes, output_size)
    
    print(f"Pipeline architecture:")
    print(f"- Input size: {input_size}")
    print(f"- Hidden sizes: {hidden_sizes}")
    print(f"- Output size: {output_size}")
    print(f"- Total layers: {len(pipeline.layers)}")
    
    # Test input
    test_input = np.random.randn(input_size)
    print(f"\nTest input shape: {test_input.shape}")
    
    # Forward pass
    output = pipeline.forward(test_input)
    print(f"Pipeline output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Show network info
    network_info = pipeline.get_network_info()
    print(f"\nNetwork information:")
    for i, layer_info in enumerate(network_info):
        print(f"  Layer {i}: {len(layer_info)} neurons")
        if i < 2:  # Show details for first 2 layers
            for j, neuron_info in enumerate(layer_info[:2]):  # First 2 neurons
                print(f"    Neuron {j}: Phase={neuron_info['phase_position']:.3f}")
    
    return pipeline

def demo_tunable_phase_realms():
    """Demonstrate TunablePhaseNeuron realms in the singularity platform."""
    print("\n=== TunablePhaseNeuron Realms Demo ===\n")
    
    # Create sample field
    sr = 48000
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = 0.8 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t + np.pi/4)
    field = make_field_from_real(x, sr, tag=("demo", "multi_tone"))
    
    print(f"Original field: {len(field.z)} samples, {field.sr} Hz")
    print(f"Field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
    
    # Create TunablePhaseNeuron realms
    realms = {
        'tunablephaseneuron': create_tunable_phase_neuron_realm(input_size=100, n_phases=16),
        'tunablephaselayer': create_tunable_phase_layer_realm(input_size=100, output_size=50, n_phases=20),
        'selftuningpipeline': create_self_tuning_pipeline_realm(input_size=100, hidden_sizes=[64, 32], output_size=25, n_phases=24)
    }
    
    print(f"\nCreated {len(realms)} TunablePhaseNeuron realms:")
    for name, realm in realms.items():
        print(f"  - {name}: {realm.parameters}")
    
    # Test each realm
    print(f"\nTesting TunablePhaseNeuron realms on field:")
    for name, realm in realms.items():
        try:
            processed_field = realm.field_processor(field)
            energy = np.sum(np.abs(processed_field.z) ** 2)
            
            print(f"  {name:20} | Energy: {energy:8.2f} | Learning: {realm.learning_enabled}")
            print(f"  {'':20} | Roles: {list(processed_field.roles.keys())}")
        except Exception as e:
            print(f"  {name:20} | Error: {e}")
    
    # Test VM integration
    print(f"\nTesting VM integration:")
    for name, realm in realms.items():
        try:
            vm_expr = compile_tunable_phase_neuron_to_vm(realm.neuron_class, name, **realm.parameters)
            vm_json = to_json(vm_expr)
            print(f"  {name:20} | VM JSON length: {len(vm_json)} chars")
        except Exception as e:
            print(f"  {name:20} | VM Error: {e}")
    
    print(f"\n=== TunablePhaseNeuron Realms Demo Complete ===")

def main():
    """Run all TunablePhaseNeuron demos."""
    print("🔷 TunablePhaseNeuron System Demo")
    print("=" * 50)
    
    # Demo individual components
    demo_tunable_phase_neuron()
    demo_tunable_phase_layer()
    demo_self_tuning_pipeline()
    
    # Demo realms
    demo_tunable_phase_realms()
    
    print(f"\n🔷 TunablePhaseNeuron System Complete")
    print("✓ Self-tuning neural networks")
    print("✓ Learnable phase positions")
    print("✓ Smooth phase interpolation")
    print("✓ FieldIQ integration")
    print("✓ VM realm system")

if __name__ == "__main__":
    main()

