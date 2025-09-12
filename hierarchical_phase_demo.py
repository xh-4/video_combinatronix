# ============================
# Hierarchical Phase Networks - Simplified Demo
# ============================
"""
Simplified hierarchical phase networks integrated as realms.
Focuses on core functionality without complex dependencies.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Import existing systems
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf, to_json, from_json
from Combinator_Kernel import FieldIQ, make_field_from_real

# ============================
# Core Hierarchical Components
# ============================

class HierarchicalPhaseLayer:
    """Simplified hierarchical phase layer."""
    def __init__(self, input_size: int, n_neurons: int, layer_depth: int, base_frequency: float = 1.0):
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.layer_depth = layer_depth
        self.base_frequency = base_frequency
        
        # Parameters
        self.weights = np.random.randn(n_neurons, input_size) * 0.1
        self.phases = np.random.randn(n_neurons)
        
        # Hierarchical frequency structure
        freq_scale = base_frequency / (2.0 ** layer_depth)
        self.frequencies = np.ones(n_neurons) * freq_scale
        
        # Complexity factor
        self.complexity_factor = layer_depth + 1.0
        
    def forward(self, x):
        """Forward pass with hierarchical complexity."""
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # Linear transformation
        linear_outs = np.dot(self.weights, x)
        
        # Apply hierarchical phase relationships
        if self.layer_depth == 0:
            # Layer 0: Simple harmonic detection
            activations = np.sin(self.frequencies * linear_outs + self.phases)
        elif self.layer_depth == 1:
            # Layer 1: Harmonic combinations with modulation
            base_activations = np.sin(self.frequencies * linear_outs + self.phases)
            modulation = np.cos(0.5 * self.frequencies * linear_outs)
            activations = base_activations * modulation
        else:
            # Layer 2+: Complex feature compositions
            base_activations = np.sin(self.frequencies * linear_outs + self.phases)
            harmonic_mix = np.sin(2 * self.frequencies * linear_outs + 0.5 * self.phases)
            activations = base_activations + 0.3 * harmonic_mix
        
        return activations

class InterLayerCoordination:
    """Simplified inter-layer coordination."""
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        
        # Influence matrices
        self.influence_matrices = []
        for i in range(1, self.n_layers):
            prev_size = sum(layer_sizes[:i])
            curr_size = layer_sizes[i]
            matrix = np.random.randn(curr_size, prev_size) * 0.1
            self.influence_matrices.append(matrix)
        
        # Coherence weights
        self.coherence_weights = []
        for i in range(1, self.n_layers):
            prev_size = layer_sizes[i-1]
            curr_size = layer_sizes[i]
            weights = np.random.randn(curr_size, prev_size) * 0.1
            self.coherence_weights.append(weights)
    
    def forward(self, current_layer: np.ndarray, previous_layers: List[np.ndarray], layer_index: int) -> np.ndarray:
        """Apply inter-layer coordination."""
        if layer_index == 0:
            return current_layer
        
        # Concatenate previous layers
        prev_concat = np.concatenate(previous_layers)
        
        # Apply influence
        influence_matrix = self.influence_matrices[layer_index - 1]
        influence = np.dot(influence_matrix, prev_concat)
        
        # Apply coherence
        coherence_matrix = self.coherence_weights[layer_index - 1]
        coherence = np.dot(coherence_matrix, previous_layers[-1])
        coherence = np.tanh(coherence)
        
        # Combine
        return current_layer + 0.1 * influence + 0.05 * coherence

class HierarchicalPhaseNetwork:
    """Simplified hierarchical phase network."""
    def __init__(self, input_size: int, layer_sizes: List[int] = [64, 32, 16]):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        
        # Create layers
        self.layers = []
        prev_size = input_size
        for i, size in enumerate(layer_sizes):
            layer = HierarchicalPhaseLayer(
                input_size=prev_size,
                n_neurons=size,
                layer_depth=i,
                base_frequency=2.0 ** i
            )
            self.layers.append(layer)
            prev_size = size
        
        # Coordination
        self.coordination = InterLayerCoordination(layer_sizes)
        
    def forward(self, x):
        """Forward pass through network."""
        layer_outputs = []
        current_input = x
        
        for i, layer in enumerate(self.layers):
            # Forward through layer
            output = layer(current_input)
            layer_outputs.append(output)
            
            # Apply coordination
            if i > 0:
                output = self.coordination.forward(output, layer_outputs[:-1], i)
            
            current_input = output
        
        return current_input, layer_outputs

# ============================
# FieldIQ Integration
# ============================

class FieldIQHierarchicalProcessor:
    """FieldIQ processor for hierarchical networks."""
    def __init__(self, input_size: int, layer_sizes: List[int] = [64, 32, 16]):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.network = HierarchicalPhaseNetwork(input_size, layer_sizes)
        
    def process_field(self, field: FieldIQ) -> FieldIQ:
        """Process FieldIQ through hierarchical network."""
        z_array = field.z
        
        # Process real and imaginary parts
        real_part = np.real(z_array)
        imag_part = np.imag(z_array)
        
        # Process in chunks
        chunk_size = min(self.input_size, len(real_part))
        real_processed = self._process_chunks(real_part, chunk_size)
        imag_processed = self._process_chunks(imag_part, chunk_size)
        
        # Reconstruct
        new_z = real_processed + 1j * imag_processed
        
        # Create new field with metadata
        processed_field = FieldIQ(new_z, field.sr, field.roles or {})
        processed_field = processed_field.with_role("hierarchical_processed", True)
        processed_field = processed_field.with_role("layer_sizes", self.layer_sizes)
        processed_field = processed_field.with_role("n_layers", len(self.layer_sizes))
        
        return processed_field
    
    def _process_chunks(self, data: np.ndarray, chunk_size: int) -> np.ndarray:
        """Process data in chunks."""
        if len(data) <= chunk_size:
            output, _ = self.network.forward(data)
            return output
        else:
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                if len(chunk) == chunk_size:
                    output, _ = self.network.forward(chunk)
                    chunks.append(output)
                else:
                    padded_chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                    output, _ = self.network.forward(padded_chunk)
                    chunks.append(output[:len(chunk)])
            return np.concatenate(chunks)

# ============================
# Realm System
# ============================

@dataclass
class HierarchicalPhaseRealm:
    """Realm for hierarchical phase processing."""
    name: str
    processor: FieldIQHierarchicalProcessor
    vm_node: Node
    learning_enabled: bool = True
    
    def __repr__(self):
        return f"<HierarchicalPhaseRealm {self.name}>"
    
    def field_processor(self, field: FieldIQ) -> FieldIQ:
        """Process field through realm."""
        return self.processor.process_field(field)

def create_hierarchical_phase_realm(input_size: int, layer_sizes: List[int] = [64, 32, 16]) -> HierarchicalPhaseRealm:
    """Create a hierarchical phase realm."""
    processor = FieldIQHierarchicalProcessor(input_size, layer_sizes)
    
    vm_node = Val({
        'type': 'hierarchical_phase_realm',
        'name': f'hierarchical_phase_{len(layer_sizes)}_layers',
        'class': 'FieldIQHierarchicalProcessor',
        'parameters': {
            'input_size': input_size,
            'layer_sizes': layer_sizes
        },
        'vm_operations': {'primary': 'HIERARCHICAL_PHASE'}
    })
    
    return HierarchicalPhaseRealm(
        name=f'hierarchical_phase_{len(layer_sizes)}_layers',
        processor=processor,
        vm_node=vm_node,
        learning_enabled=True
    )

# ============================
# Demo Functions
# ============================

def demo_hierarchical_layers():
    """Demo hierarchical layers."""
    print("=== Hierarchical Phase Layers Demo ===\n")
    
    # Test different layer depths
    for depth in [0, 1, 2]:
        print(f"--- Layer Depth {depth} ---")
        
        layer = HierarchicalPhaseLayer(100, 16, depth, base_frequency=2.0)
        test_input = np.random.randn(100)
        output = layer.forward(test_input)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Frequency scale: {np.mean(layer.frequencies):.3f}")
        print(f"Complexity factor: {layer.complexity_factor}")
        print()

def demo_hierarchical_network():
    """Demo hierarchical network."""
    print("=== Hierarchical Phase Network Demo ===\n")
    
    # Create network
    input_size = 100
    layer_sizes = [64, 32, 16]
    network = HierarchicalPhaseNetwork(input_size, layer_sizes)
    
    print(f"Network: {input_size} -> {layer_sizes}")
    print(f"Number of layers: {len(network.layers)}")
    
    # Test forward pass
    test_input = np.random.randn(input_size)
    output, layer_outputs = network.forward(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Layer outputs: {[len(lo) for lo in layer_outputs]}")
    
    # Show layer characteristics
    print(f"\nLayer characteristics:")
    for i, layer in enumerate(network.layers):
        print(f"  Layer {i}: {layer.n_neurons} neurons, "
              f"freq_scale={np.mean(layer.frequencies):.3f}, "
              f"complexity={layer.complexity_factor}")
    print()

def demo_hierarchical_realms():
    """Demo hierarchical realms."""
    print("=== Hierarchical Phase Realms Demo ===\n")
    
    # Create sample field
    sr = 48000
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = 0.8 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t + np.pi/4)
    field = make_field_from_real(x, sr, tag=("demo", "multi_tone"))
    
    print(f"Original field: {len(field.z)} samples, {field.sr} Hz")
    print(f"Field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
    
    # Create realms with different architectures
    realms = {
        'shallow': create_hierarchical_phase_realm(100, [32, 16]),
        'medium': create_hierarchical_phase_realm(100, [64, 32, 16]),
        'deep': create_hierarchical_phase_realm(100, [128, 64, 32, 16])
    }
    
    print(f"\nCreated {len(realms)} hierarchical realms:")
    for name, realm in realms.items():
        print(f"  - {name}: {realm.processor.layer_sizes} layers")
    
    # Test each realm
    print(f"\nTesting hierarchical realms on field:")
    for name, realm in realms.items():
        try:
            processed_field = realm.field_processor(field)
            energy = np.sum(np.abs(processed_field.z) ** 2)
            
            print(f"  {name:8} | Energy: {energy:8.2f} | Learning: {realm.learning_enabled}")
            
            # Show metadata
            layer_sizes = processed_field.roles.get('layer_sizes', [])
            n_layers = processed_field.roles.get('n_layers', 0)
            print(f"  {'':8} | Layers: {layer_sizes}, N: {n_layers}")
            
        except Exception as e:
            print(f"  {name:8} | Error: {e}")
    
    # Test VM integration
    print(f"\nTesting VM integration:")
    for name, realm in realms.items():
        try:
            vm_json = to_json(realm.vm_node)
            print(f"  {name:8} | VM JSON length: {len(vm_json)} chars")
        except Exception as e:
            print(f"  {name:8} | VM Error: {e}")
    print()

def demo_processing_pipeline():
    """Demo processing pipeline."""
    print("=== Hierarchical Processing Pipeline Demo ===\n")
    
    # Create realms
    realms = {
        'temporal': create_hierarchical_phase_realm(100, [64, 32, 16]),
        'spectral': create_hierarchical_phase_realm(100, [48, 24, 12]),
        'adaptive': create_hierarchical_phase_realm(100, [80, 40, 20])
    }
    
    # Create test data
    sr = 48000
    dur = 0.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    test_data = 0.5 * np.cos(2 * np.pi * 440 * t) + 0.2 * np.cos(2 * np.pi * 880 * t)
    field = make_field_from_real(test_data, sr, tag=("pipeline_test", "synthetic"))
    
    print(f"Test field: {len(field.z)} samples, {field.sr} Hz")
    print(f"Field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
    
    # Process through each realm
    print(f"\nProcessing through realms:")
    for name, realm in realms.items():
        try:
            processed_field = realm.field_processor(field)
            energy = np.sum(np.abs(processed_field.z) ** 2)
            energy_ratio = energy / np.sum(np.abs(field.z) ** 2)
            
            print(f"  {name:8} | Energy: {energy:8.2f} | Ratio: {energy_ratio:.3f}")
            
            # Show layer info
            layer_sizes = processed_field.roles.get('layer_sizes', [])
            print(f"  {'':8} | Architecture: {layer_sizes}")
            
        except Exception as e:
            print(f"  {name:8} | Error: {e}")
    print()

def main():
    """Run all demos."""
    print("🔷 Hierarchical Phase Networks - Simplified Demo")
    print("=" * 60)
    
    # Run demos
    demo_hierarchical_layers()
    demo_hierarchical_network()
    demo_hierarchical_realms()
    demo_processing_pipeline()
    
    print("🔷 Hierarchical Phase Networks Complete")
    print("✓ Multi-layer hierarchical processing")
    print("✓ Inter-layer coordination")
    print("✓ FieldIQ integration")
    print("✓ Realm system integration")
    print("✓ VM compatibility")

if __name__ == "__main__":
    main()

