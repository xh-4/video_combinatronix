# ============================
# Hierarchical Phase Networks for Multi-Layer AI Processing
# ============================
"""
Hierarchical phase networks that learn features at multiple levels of abstraction,
integrated as realms in the singularity platform.
"""

import numpy as np
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Tuple
from queue import Queue, Empty

# Import existing systems
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf, to_json, from_json
from Combinator_Kernel import FieldIQ, make_field_from_real

# ============================
# Hierarchical Phase Layer (No PyTorch)
# ============================

class HierarchicalPhaseLayer:
    """
    Simplified HierarchicalPhaseLayer using NumPy for demo purposes.
    """
    def __init__(self, input_size: int, n_neurons: int, layer_depth: int, base_frequency: float = 1.0):
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.layer_depth = layer_depth
        self.base_frequency = base_frequency
        
        # Standard parameters
        self.weights = np.random.randn(n_neurons, input_size) * 0.1
        self.phases = np.random.randn(n_neurons)
        
        # Hierarchical frequency structure
        # Layer 0: fine details (high freq)
        # Layer 1: medium patterns (mid freq)  
        # Layer 2: coarse structure (low freq)
        freq_scale = base_frequency / (2.0 ** layer_depth)
        self.frequencies = np.ones(n_neurons) * freq_scale
        
        # Feature complexity increases with depth
        self.complexity_factor = layer_depth + 1.0
        
        # Learning parameters
        self.learning_rate = 0.01
        self.phase_learning_rate = 0.001
        self.frequency_learning_rate = 0.001
        
        # Layer-specific tracking
        self.feature_history = []
        self.complexity_metrics = {}
        
    def forward(self, x):
        """Forward pass with hierarchical complexity."""
        # Ensure x is 1D
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # Each neuron's linear response
        linear_outs = np.dot(self.weights, x)  # (n_neurons,)
        
        # Higher layers use more complex phase relationships
        if self.layer_depth == 0:
            # Layer 0: Simple harmonic detection
            activations = np.sin(self.frequencies * linear_outs + self.phases)
        elif self.layer_depth == 1:
            # Layer 1: Harmonic combinations (beating, modulation)
            base_activations = np.sin(self.frequencies * linear_outs + self.phases)
            modulation = np.cos(0.5 * self.frequencies * linear_outs)
            activations = base_activations * modulation
        else:
            # Layer 2+: Complex feature compositions
            base_activations = np.sin(self.frequencies * linear_outs + self.phases)
            harmonic_mix = np.sin(2 * self.frequencies * linear_outs + 0.5 * self.phases)
            activations = base_activations + 0.3 * harmonic_mix
        
        # Track feature complexity
        self._update_complexity_metrics(activations)
        
        return activations
    
    def _update_complexity_metrics(self, activations):
        """Update complexity metrics for this layer."""
        self.complexity_metrics = {
            'layer_depth': self.layer_depth,
            'activation_variance': np.var(activations),
            'activation_range': np.max(activations) - np.min(activations),
            'frequency_scale': np.mean(self.frequencies),
            'complexity_factor': self.complexity_factor,
            'n_neurons': self.n_neurons
        }
        
        # Store in history
        self.feature_history.append({
            'activations': activations.copy(),
            'metrics': self.complexity_metrics.copy(),
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.feature_history) > 100:
            self.feature_history = self.feature_history[-100:]

class InterLayerCoordination:
    """
    Simplified InterLayerCoordination using NumPy for demo purposes.
    """
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        
        # Learn how layers should influence each other
        self.influence_matrices = []
        for i in range(1, self.n_layers):
            prev_size = sum(layer_sizes[:i])
            curr_size = layer_sizes[i]
            matrix = np.random.randn(curr_size, prev_size) * 0.1
            self.influence_matrices.append(matrix)
        
        # Phase coherence tracking
        self.coherence_weights = []
        for i in range(1, self.n_layers):
            prev_size = layer_sizes[i-1]
            curr_size = layer_sizes[i]
            weights = np.random.randn(curr_size, prev_size) * 0.1
            self.coherence_weights.append(weights)
        
        # Coordination tracking
        self.coordination_history = []
        
    def forward(self, current_layer: np.ndarray, previous_layers: List[np.ndarray], layer_index: int) -> np.ndarray:
        """Apply inter-layer coordination."""
        if layer_index == 0:
            return current_layer
        
        # Concatenate all previous layer outputs
        prev_concat = np.concatenate(previous_layers)
        
        # Learn inter-layer influences
        influence_matrix = self.influence_matrices[layer_index - 1]
        influence = np.dot(influence_matrix, prev_concat)
        
        # Phase coherence: encourage harmonic relationships between layers
        coherence_term = self._compute_phase_coherence(
            current_layer, previous_layers[-1], layer_index
        )
        
        # Combine current layer with influences from previous layers
        coordinated_output = current_layer + 0.1 * influence + 0.05 * coherence_term
        
        # Track coordination
        self._update_coordination_tracking(current_layer, coordinated_output, layer_index)
        
        return coordinated_output
    
    def _compute_phase_coherence(self, current: np.ndarray, previous: np.ndarray, layer_idx: int) -> np.ndarray:
        """Encourage harmonic relationships between adjacent layers."""
        # Previous layer frequencies should relate to current layer
        # e.g., if prev layer found 440Hz, current might find 880Hz or 220Hz
        
        coherence_matrix = self.coherence_weights[layer_idx - 1]
        coherence = np.dot(coherence_matrix, previous)
        
        return np.tanh(coherence)  # bounded influence
    
    def _update_coordination_tracking(self, original: np.ndarray, coordinated: np.ndarray, layer_idx: int):
        """Track coordination effects."""
        coordination_info = {
            'layer_index': layer_idx,
            'original_variance': np.var(original),
            'coordinated_variance': np.var(coordinated),
            'coordination_effect': np.var(coordinated) - np.var(original),
            'timestamp': time.time()
        }
        
        self.coordination_history.append(coordination_info)
        
        # Keep only recent history
        if len(self.coordination_history) > 100:
            self.coordination_history = self.coordination_history[-100:]

class DynamicFeatureRouting:
    """
    Simplified DynamicFeatureRouting using NumPy for demo purposes.
    """
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        
        # Learn which features should flow to which higher-level features
        self.routing_gates = []
        for i in range(len(layer_sizes) - 1):
            curr_size = layer_sizes[i]
            next_size = layer_sizes[i + 1]
            gate = np.random.randn(next_size, curr_size) * 0.1
            self.routing_gates.append(gate)
        
        # Routing tracking
        self.routing_history = []
        
    def route_features(self, layer_features: np.ndarray, layer_idx: int) -> np.ndarray:
        """Dynamically route features based on current input."""
        if layer_idx >= len(self.routing_gates):
            return layer_features
        
        # Compute routing weights
        routing_gate = self.routing_gates[layer_idx]
        routing_scores = np.dot(routing_gate, layer_features)
        routing_weights = self._softmax(routing_scores)
        
        # Weighted feature routing
        routed_features = layer_features * routing_weights
        
        # Track routing
        self._update_routing_tracking(layer_features, routed_features, layer_idx)
        
        return routed_features
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function for routing weights."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    def _update_routing_tracking(self, original: np.ndarray, routed: np.ndarray, layer_idx: int):
        """Track routing effects."""
        routing_info = {
            'layer_index': layer_idx,
            'original_variance': np.var(original),
            'routed_variance': np.var(routed),
            'routing_effect': np.var(routed) - np.var(original),
            'timestamp': time.time()
        }
        
        self.routing_history.append(routing_info)
        
        # Keep only recent history
        if len(self.routing_history) > 100:
            self.routing_history = self.routing_history[-100:]

class HierarchicalPhaseNetwork:
    """
    Simplified HierarchicalPhaseNetwork using NumPy for demo purposes.
    """
    def __init__(self, input_size: int, layer_sizes: List[int] = [64, 32, 16]):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        
        # Create hierarchical layers
        self.layers = []
        prev_size = input_size
        for i, size in enumerate(layer_sizes):
            layer = HierarchicalPhaseLayer(
                input_size=prev_size,
                n_neurons=size,
                layer_depth=i,
                base_frequency=2.0 ** i  # frequency scaling across layers
            )
            self.layers.append(layer)
            prev_size = size
        
        # Cross-layer coordination mechanisms
        self.inter_layer_coordination = InterLayerCoordination(layer_sizes)
        self.dynamic_feature_routing = DynamicFeatureRouting(layer_sizes)
        
        # Network-level tracking
        self.network_history = []
        
    def forward(self, x):
        """Forward pass through hierarchical network."""
        layer_outputs = []
        current_input = x
        
        for i, layer in enumerate(self.layers):
            # Forward through layer
            output = layer(current_input)
            layer_outputs.append(output)
            
            # Apply inter-layer coordination
            if i > 0:
                coordinated_output = self.inter_layer_coordination(
                    current_layer=output,
                    previous_layers=layer_outputs[:-1],
                    layer_index=i
                )
                output = coordinated_output
            
            # Apply dynamic feature routing
            routed_output = self.dynamic_feature_routing.route_features(output, i)
            output = routed_output
            
            current_input = output
        
        # Track network-level processing
        self._update_network_tracking(layer_outputs)
        
        return current_input, layer_outputs
    
    def _update_network_tracking(self, layer_outputs: List[np.ndarray]):
        """Track network-level processing."""
        network_info = {
            'n_layers': len(layer_outputs),
            'layer_sizes': [len(output) for output in layer_outputs],
            'total_activations': sum(len(output) for output in layer_outputs),
            'layer_variances': [np.var(output) for output in layer_outputs],
            'timestamp': time.time()
        }
        
        self.network_history.append(network_info)
        
        # Keep only recent history
        if len(self.network_history) > 100:
            self.network_history = self.network_history[-100:]

# ============================
# FieldIQ Integration
# ============================

class FieldIQHierarchicalProcessor:
    """
    A processor that applies HierarchicalPhaseNetwork to FieldIQ data.
    """
    def __init__(self, input_size: int, layer_sizes: List[int] = [64, 32, 16],
                 processing_mode: str = 'temporal'):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.processing_mode = processing_mode
        
        # Create hierarchical network
        self.network = HierarchicalPhaseNetwork(input_size, layer_sizes)
        
    def process_field(self, field: FieldIQ) -> FieldIQ:
        """Process FieldIQ data through HierarchicalPhaseNetwork."""
        z_array = field.z
        
        if self.processing_mode == 'temporal':
            # Process temporal sequence
            real_part = np.real(z_array)
            imag_part = np.imag(z_array)
            
            # Process in chunks to handle large arrays
            chunk_size = min(self.input_size, len(real_part))
            real_processed = self._process_chunks(real_part, chunk_size)
            imag_processed = self._process_chunks(imag_part, chunk_size)
            
        elif self.processing_mode == 'spectral':
            # Process spectral representation
            fft_data = np.fft.fft(z_array)
            fft_real = np.real(fft_data)
            fft_imag = np.imag(fft_data)
            
            chunk_size = min(self.input_size, len(fft_real))
            real_processed = self._process_chunks(fft_real, chunk_size)
            imag_processed = self._process_chunks(fft_imag, chunk_size)
            
            # Convert back to time domain
            processed_fft = real_processed + 1j * imag_processed
            time_domain = np.fft.ifft(processed_fft)
            real_processed = np.real(time_domain)
            imag_processed = np.imag(time_domain)
            
        else:  # adaptive
            # Choose processing mode based on field characteristics
            field_energy = np.sum(np.abs(z_array) ** 2)
            if field_energy > 1000:
                # High energy - use spectral processing
                fft_data = np.fft.fft(z_array)
                fft_real = np.real(fft_data)
                fft_imag = np.imag(fft_data)
                
                chunk_size = min(self.input_size, len(fft_real))
                real_processed = self._process_chunks(fft_real, chunk_size)
                imag_processed = self._process_chunks(fft_imag, chunk_size)
                
                # Convert back to time domain
                processed_fft = real_processed + 1j * imag_processed
                time_domain = np.fft.ifft(processed_fft)
                real_processed = np.real(time_domain)
                imag_processed = np.imag(time_domain)
            else:
                # Low energy - use temporal processing
                real_part = np.real(z_array)
                imag_part = np.imag(z_array)
                
                chunk_size = min(self.input_size, len(real_part))
                real_processed = self._process_chunks(real_part, chunk_size)
                imag_processed = self._process_chunks(imag_part, chunk_size)
        
        # Reconstruct complex field
        new_z = real_processed + 1j * imag_processed
        
        # Create new FieldIQ with processing metadata
        processed_field = FieldIQ(new_z, field.sr, field.roles or {})
        processed_field = processed_field.with_role("hierarchical_processed", True)
        processed_field = processed_field.with_role("processing_mode", self.processing_mode)
        processed_field = processed_field.with_role("layer_sizes", self.layer_sizes)
        processed_field = processed_field.with_role("network_metrics", self._get_network_metrics())
        
        return processed_field
    
    def _process_chunks(self, data: np.ndarray, chunk_size: int) -> np.ndarray:
        """Process data in chunks to handle large arrays."""
        if len(data) <= chunk_size:
            # Process entire array
            output, _ = self.network.forward(data)
            return output
        else:
            # Process in chunks and concatenate
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                if len(chunk) == chunk_size:
                    output, _ = self.network.forward(chunk)
                    chunks.append(output)
                else:
                    # Pad last chunk if necessary
                    padded_chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                    output, _ = self.network.forward(padded_chunk)
                    chunks.append(output[:len(chunk)])
            
            return np.concatenate(chunks)
    
    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get network-level metrics."""
        metrics = {
            'n_layers': len(self.network.layers),
            'layer_sizes': self.layer_sizes,
            'processing_mode': self.processing_mode
        }
        
        # Add layer-specific metrics
        for i, layer in enumerate(self.network.layers):
            metrics[f'layer_{i}_complexity'] = layer.complexity_metrics
            metrics[f'layer_{i}_frequency_scale'] = np.mean(layer.frequencies)
        
        # Add coordination metrics
        if self.network.inter_layer_coordination.coordination_history:
            latest_coordination = self.network.inter_layer_coordination.coordination_history[-1]
            metrics['coordination_effect'] = latest_coordination.get('coordination_effect', 0)
        
        # Add routing metrics
        if self.network.dynamic_feature_routing.routing_history:
            latest_routing = self.network.dynamic_feature_routing.routing_history[-1]
            metrics['routing_effect'] = latest_routing.get('routing_effect', 0)
        
        return metrics

# ============================
# Hierarchical Phase Realm System
# ============================

@dataclass
class HierarchicalPhaseRealm:
    """Realm for HierarchicalPhaseNetwork processing."""
    name: str
    processor: FieldIQHierarchicalProcessor
    vm_node: Node
    learning_enabled: bool = True
    
    def __repr__(self):
        return f"<HierarchicalPhaseRealm {self.name}>"
    
    def field_processor(self, field: FieldIQ) -> FieldIQ:
        """Process field through the realm."""
        return self.processor.process_field(field)

def create_hierarchical_phase_realm(input_size: int, layer_sizes: List[int] = [64, 32, 16],
                                   processing_mode: str = 'temporal') -> HierarchicalPhaseRealm:
    """Create a HierarchicalPhaseNetwork realm."""
    
    # Create processor
    processor = FieldIQHierarchicalProcessor(input_size, layer_sizes, processing_mode)
    
    # Create VM node
    vm_node = Val({
        'type': 'hierarchical_phase_realm',
        'name': f'hierarchical_phase_{processing_mode}',
        'class': 'FieldIQHierarchicalProcessor',
        'parameters': {
            'input_size': input_size, 
            'layer_sizes': layer_sizes,
            'processing_mode': processing_mode
        },
        'vm_operations': {'primary': 'HIERARCHICAL_PHASE'}
    })
    
    return HierarchicalPhaseRealm(
        name=f'hierarchical_phase_{processing_mode}',
        processor=processor,
        vm_node=vm_node,
        learning_enabled=True
    )

# ============================
# Hierarchical Processing Pipeline
# ============================

class HierarchicalProcessingPipeline:
    """
    Hierarchical AI pipeline using HierarchicalPhaseNetworks.
    """
    
    def __init__(self, num_carriers: int = 4):
        self.num_carriers = num_carriers
        self.carriers = []
        self.data_queue = Queue()
        self.running = False
        
        # Create HierarchicalPhaseNetwork realms
        self.processing_realms = {
            'temporal': create_hierarchical_phase_realm(100, [64, 32, 16], 'temporal'),
            'spectral': create_hierarchical_phase_realm(100, [48, 24, 12], 'spectral'),
            'adaptive': create_hierarchical_phase_realm(100, [80, 40, 20], 'adaptive')
        }
        
        # Initialize carriers
        self._initialize_carriers()
    
    def _initialize_carriers(self):
        """Initialize carriers."""
        for i in range(self.num_carriers):
            carrier = {
                'carrier_id': f'hierarchical_carrier_{i}',
                'capacity': 1.0,
                'current_load': 0.0,
                'active_realm': None,
                'learning_rate': 0.01 + i * 0.005
            }
            self.carriers.append(carrier)
    
    def ingest_data(self, data: Any) -> FieldIQ:
        """Ingest data and convert to FieldIQ."""
        if isinstance(data, np.ndarray):
            sr = 48000
            if len(data.shape) == 1:
                field = make_field_from_real(data, sr, tag=("hierarchical_ingest", "raw"))
            else:
                field = make_field_from_real(data.flatten(), sr, tag=("hierarchical_ingest", "raw"))
        else:
            # Create synthetic data
            sr = 48000
            dur = 1.0
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            base_freq = 440
            phase_mod = 0.5 * np.sin(2 * np.pi * 0.5 * t)
            synthetic = (0.5 * np.cos(2 * np.pi * base_freq * t + phase_mod) + 
                       0.3 * np.cos(2 * np.pi * base_freq * 2 * t + phase_mod * 2) +
                       0.1 * np.random.randn(len(t)))
            field = make_field_from_real(synthetic, sr, tag=("hierarchical_ingest", "synthetic"))
        
        return field
    
    def select_processing_realm(self, field: FieldIQ) -> str:
        """Select the best processing realm based on field characteristics."""
        field_energy = np.sum(np.abs(field.z) ** 2)
        spectral_energy = np.sum(np.abs(np.fft.fft(field.z)) ** 2)
        phase_complexity = np.std(np.angle(field.z))
        
        # Select realm based on field characteristics
        if field_energy > 50000 and spectral_energy > 100000:
            return 'adaptive'  # High energy, complex signal
        elif phase_complexity > 1.0:
            return 'spectral'  # Complex phase relationships
        else:
            return 'temporal'  # Simple temporal processing
    
    def process_field_with_realm(self, field: FieldIQ, realm_name: str) -> FieldIQ:
        """Process field with the specified realm."""
        if realm_name in self.processing_realms:
            realm = self.processing_realms[realm_name]
            return realm.field_processor(field)
        else:
            return field
    
    async def continuous_hierarchical_pipeline(self):
        """Continuous pipeline with hierarchical processing."""
        print("🚀 Starting Hierarchical Phase AI Pipeline...")
        print("   Using HierarchicalPhaseNetworks for multi-layer feature learning")
        self.running = True
        
        cycle_count = 0
        while self.running:
            cycle_count += 1
            print(f"\n--- Hierarchical Cycle {cycle_count} ---")
            
            # Process available data
            try:
                data = self.data_queue.get_nowait()
                field = self.ingest_data(data)
                print(f"📥 Ingested data: {len(field.z)} samples")
            except Empty:
                # Create synthetic data for demo
                sr = 48000
                dur = 0.5
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                
                # Vary signal complexity
                freq = 440 + 50 * np.sin(cycle_count * 0.1)
                phase_mod = 0.3 * np.sin(cycle_count * 0.2)
                noise_level = 0.1 + 0.05 * np.sin(cycle_count * 0.3)
                
                synthetic = (0.5 * np.cos(2 * np.pi * freq * t + phase_mod) + 
                           0.2 * np.cos(2 * np.pi * freq * 1.5 * t + phase_mod * 1.5) +
                           noise_level * np.random.randn(len(t)))
                
                field = make_field_from_real(synthetic, sr, tag=("hierarchical_synthetic", f"cycle_{cycle_count}"))
                print(f"📥 Generated synthetic data: {freq:.1f}Hz, phase_mod={phase_mod:.3f}")
            
            # Select processing realm
            realm_name = self.select_processing_realm(field)
            print(f"🧠 Selected processing realm: {realm_name}")
            
            # Find available carrier
            available_carrier = min(self.carriers, key=lambda c: c['current_load'])
            
            if available_carrier['current_load'] < 0.8:
                # Process field
                processed_field = self.process_field_with_realm(field, realm_name)
                
                # Update carrier
                available_carrier['current_load'] = min(1.0, available_carrier['current_load'] + 0.2)
                available_carrier['active_realm'] = realm_name
                
                # Show processing results
                original_energy = np.sum(np.abs(field.z) ** 2)
                processed_energy = np.sum(np.abs(processed_field.z) ** 2)
                energy_ratio = processed_energy / original_energy if original_energy > 0 else 1.0
                
                print(f"⚡ Processed on {available_carrier['carrier_id']}")
                print(f"   Energy: {original_energy:.0f} → {processed_energy:.0f} (ratio: {energy_ratio:.3f})")
                
                # Show hierarchical metrics
                network_metrics = processed_field.roles.get('network_metrics', {})
                if network_metrics:
                    print(f"   Layer sizes: {network_metrics.get('layer_sizes', [])}")
                    print(f"   Coordination effect: {network_metrics.get('coordination_effect', 0):.6f}")
                    print(f"   Routing effect: {network_metrics.get('routing_effect', 0):.6f}")
                    
                    # Show layer-specific metrics
                    for i in range(network_metrics.get('n_layers', 0)):
                        layer_complexity = network_metrics.get(f'layer_{i}_complexity', {})
                        if layer_complexity:
                            print(f"   Layer {i}: var={layer_complexity.get('activation_variance', 0):.3f}, "
                                  f"freq_scale={layer_complexity.get('frequency_scale', 0):.3f}")
                
                # Simulate learning (update hierarchical parameters)
                realm = self.processing_realms[realm_name]
                if realm.learning_enabled:
                    # Simulate hierarchical learning
                    learning_gradient = np.random.randn() * 0.1
                    # Update network parameters
                    for layer in realm.processor.network.layers:
                        layer.phases += learning_gradient * 0.01
                        layer.phases = layer.phases % (2 * np.pi)
                    print(f"   Learning: Updated hierarchical phases by {learning_gradient:.3f}")
            
            # Show carrier status
            print(f"📊 Carrier status:")
            for carrier in self.carriers:
                realm_str = carrier['active_realm'] if carrier['active_realm'] else "idle"
                print(f"   {carrier['carrier_id']}: {realm_str} (load: {carrier['current_load']:.2f})")
            
            # Small delay
            await asyncio.sleep(0.5)
    
    def add_data(self, data: Any):
        """Add data to the pipeline queue."""
        self.data_queue.put(data)
    
    def stop(self):
        """Stop the pipeline."""
        self.running = False
        print("🛑 Stopping hierarchical phase AI pipeline...")

# ============================
# Demo Functions
# ============================

def demo_hierarchical_phase_layer():
    """Demonstrate HierarchicalPhaseLayer functionality."""
    print("=== HierarchicalPhaseLayer Demo ===\n")
    
    # Test different layer depths
    layer_depths = [0, 1, 2]
    
    for depth in layer_depths:
        print(f"\n--- Layer Depth {depth} ---")
        
        # Create layer
        input_size = 100
        n_neurons = 16
        layer = HierarchicalPhaseLayer(input_size, n_neurons, depth, base_frequency=2.0)
        
        print(f"Layer parameters:")
        print(f"- Input size: {input_size}")
        print(f"- Number of neurons: {n_neurons}")
        print(f"- Layer depth: {depth}")
        print(f"- Base frequency: {2.0}")
        print(f"- Frequency scale: {np.mean(layer.frequencies):.3f}")
        
        # Test input
        test_input = np.random.randn(input_size)
        print(f"Test input shape: {test_input.shape}")
        
        # Forward pass
        output = layer.forward(test_input)
        print(f"Layer output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Show complexity metrics
        print(f"Complexity metrics:")
        print(f"- Activation variance: {layer.complexity_metrics.get('activation_variance', 0):.6f}")
        print(f"- Activation range: {layer.complexity_metrics.get('activation_range', 0):.6f}")
        print(f"- Complexity factor: {layer.complexity_metrics.get('complexity_factor', 0):.3f}")

def demo_hierarchical_phase_network():
    """Demonstrate HierarchicalPhaseNetwork functionality."""
    print("\n=== HierarchicalPhaseNetwork Demo ===\n")
    
    # Create network
    input_size = 100
    layer_sizes = [64, 32, 16]
    network = HierarchicalPhaseNetwork(input_size, layer_sizes)
    
    print(f"Network parameters:")
    print(f"- Input size: {input_size}")
    print(f"- Layer sizes: {layer_sizes}")
    print(f"- Number of layers: {len(network.layers)}")
    
    # Test input
    test_input = np.random.randn(input_size)
    print(f"Test input shape: {test_input.shape}")
    
    # Forward pass
    output, layer_outputs = network.forward(test_input)
    print(f"Network output shape: {output.shape}")
    print(f"Layer outputs: {[len(lo) for lo in layer_outputs]}")
    
    # Show layer-specific metrics
    print(f"\nLayer-specific metrics:")
    for i, layer in enumerate(network.layers):
        print(f"  Layer {i}: {layer.complexity_metrics}")
    
    # Show coordination metrics
    if network.inter_layer_coordination.coordination_history:
        latest_coordination = network.inter_layer_coordination.coordination_history[-1]
        print(f"\nCoordination metrics:")
        print(f"  Latest coordination effect: {latest_coordination.get('coordination_effect', 0):.6f}")
    
    # Show routing metrics
    if network.dynamic_feature_routing.routing_history:
        latest_routing = network.dynamic_feature_routing.routing_history[-1]
        print(f"\nRouting metrics:")
        print(f"  Latest routing effect: {latest_routing.get('routing_effect', 0):.6f}")

def demo_hierarchical_phase_realms():
    """Demonstrate HierarchicalPhaseNetwork realms."""
    print("\n=== HierarchicalPhaseNetwork Realms Demo ===\n")
    
    # Create sample field
    sr = 48000
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = 0.8 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t + np.pi/4)
    field = make_field_from_real(x, sr, tag=("demo", "multi_tone"))
    
    print(f"Original field: {len(field.z)} samples, {field.sr} Hz")
    print(f"Field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
    
    # Create realms
    realms = {
        'temporal': create_hierarchical_phase_realm(100, [64, 32, 16], 'temporal'),
        'spectral': create_hierarchical_phase_realm(100, [48, 24, 12], 'spectral'),
        'adaptive': create_hierarchical_phase_realm(100, [80, 40, 20], 'adaptive')
    }
    
    print(f"\nCreated {len(realms)} HierarchicalPhaseNetwork realms:")
    for name, realm in realms.items():
        print(f"  - {name}: {realm.processor.layer_sizes} layers, "
              f"mode={realm.processor.processing_mode}")
    
    # Test each realm
    print(f"\nTesting HierarchicalPhaseNetwork realms on field:")
    for name, realm in realms.items():
        try:
            processed_field = realm.field_processor(field)
            energy = np.sum(np.abs(processed_field.z) ** 2)
            
            print(f"  {name:10} | Energy: {energy:8.2f} | Learning: {realm.learning_enabled}")
            
            # Show network metrics
            network_metrics = processed_field.roles.get('network_metrics', {})
            if network_metrics:
                print(f"  {'':10} | Layers: {network_metrics.get('layer_sizes', [])}, "
                      f"Coord: {network_metrics.get('coordination_effect', 0):.6f}")
            
        except Exception as e:
            print(f"  {name:10} | Error: {e}")
    
    # Test VM integration
    print(f"\nTesting VM integration:")
    for name, realm in realms.items():
        try:
            vm_json = to_json(realm.vm_node)
            print(f"  {name:10} | VM JSON length: {len(vm_json)} chars")
        except Exception as e:
            print(f"  {name:10} | VM Error: {e}")

async def demo_hierarchical_pipeline():
    """Demonstrate the hierarchical phase AI pipeline."""
    print("\n=== Hierarchical Phase AI Pipeline Demo ===\n")
    
    # Create pipeline
    pipeline = HierarchicalProcessingPipeline(num_carriers=3)
    
    print(f"Created hierarchical pipeline with {len(pipeline.carriers)} carriers")
    print(f"Processing realms: {list(pipeline.processing_realms.keys())}")
    
    # Add test data
    test_data = np.random.randn(1000) * 0.5
    pipeline.add_data(test_data)
    
    # Run for a few cycles
    try:
        await asyncio.wait_for(pipeline.continuous_hierarchical_pipeline(), timeout=10.0)
    except asyncio.TimeoutError:
        print("\n⏰ Hierarchical pipeline demo completed after 10 seconds")
    
    pipeline.stop()

def main():
    """Run all hierarchical phase network demos."""
    print("🔷 Hierarchical Phase Networks Demo")
    print("=" * 60)
    
    # Demo individual components
    demo_hierarchical_phase_layer()
    demo_hierarchical_phase_network()
    
    # Demo realms
    demo_hierarchical_phase_realms()
    
    # Demo hierarchical pipeline
    asyncio.run(demo_hierarchical_pipeline())
    
    print(f"\n🔷 Hierarchical Phase Networks Complete")
    print("✓ Multi-layer hierarchical processing")
    print("✓ Inter-layer coordination mechanisms")
    print("✓ Dynamic feature routing")
    print("✓ FieldIQ integration with hierarchical metrics")
    print("✓ Hierarchical processing pipeline")

if __name__ == "__main__":
    main()

