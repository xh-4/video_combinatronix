# ============================
# Feature Discovery Phase Networks for Adaptive Neural Networks
# ============================
"""
Advanced phase wheel networks that automatically discover features through
harmonic relationships, quadrature pairs, and adaptive learning mechanisms.
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
# Feature Discovery Phase Layer (No PyTorch)
# ============================

class FeatureDiscoveryPhaseLayer:
    """
    Simplified FeatureDiscoveryPhaseLayer using NumPy for demo purposes.
    """
    def __init__(self, input_size: int, n_neurons: int, discovery_mode: str = 'harmonic'):
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.discovery_mode = discovery_mode
        
        # Standard parameters
        self.weights = np.random.randn(n_neurons, input_size) * 0.1
        self.phases = np.random.randn(n_neurons)
        self.frequencies = np.ones(n_neurons)
        
        # Feature discovery mechanisms
        if discovery_mode == 'harmonic':
            self.harmonic_ratios = np.ones(n_neurons)
        elif discovery_mode == 'quadrature':
            self.quadrature_pairs = np.random.randn(n_neurons // 2, 2) * 0.1
        elif discovery_mode == 'adaptive':
            self.adaptation_rate = 0.1
        
        # Learning parameters
        self.learning_rate = 0.01
        self.phase_learning_rate = 0.001
        self.frequency_learning_rate = 0.001
        
        # Feature tracking
        self.feature_correlations = np.zeros((n_neurons, n_neurons))
        self.discovery_history = []
        
    def forward(self, x):
        """Forward pass with feature discovery."""
        # Ensure x is 1D
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # Each neuron's linear response
        linear_outs = np.dot(self.weights, x)  # (n_neurons,)
        
        if self.discovery_mode == 'harmonic':
            return self._harmonic_discovery(linear_outs)
        elif self.discovery_mode == 'quadrature':
            return self._quadrature_discovery(linear_outs)
        else:
            return self._adaptive_discovery(linear_outs)
    
    def _harmonic_discovery(self, linear_outs):
        """Neurons discover harmonic relationships."""
        # Neurons learn to be integer multiples of each other
        base_freq = self.frequencies[0]  # first neuron sets fundamental
        harmonic_freqs = base_freq * np.round(self.harmonic_ratios)
        
        activations = np.sin(harmonic_freqs * linear_outs + self.phases)
        return activations
    
    def _quadrature_discovery(self, linear_outs):
        """Neurons discover sin/cos pairs automatically."""
        activations = []
        for i in range(0, self.n_neurons, 2):
            if i+1 < self.n_neurons:
                # Force neurons i and i+1 to be quadrature pairs
                freq = self.frequencies[i]
                base_phase = self.phases[i]
                
                # First neuron: sin(ωx + φ)
                act1 = np.sin(freq * linear_outs[i] + base_phase)
                # Second neuron: cos(ωx + φ) = sin(ωx + φ + π/2)
                act2 = np.sin(freq * linear_outs[i+1] + base_phase + np.pi/2)
                
                activations.extend([act1, act2])
            else:
                # Odd neuron count - add single neuron
                freq = self.frequencies[i]
                base_phase = self.phases[i]
                act = np.sin(freq * linear_outs[i] + base_phase)
                activations.append(act)
        
        return np.array(activations)
    
    def _adaptive_discovery(self, linear_outs):
        """Adaptive discovery based on data characteristics."""
        # Simple adaptive mechanism - adjust frequencies based on input energy
        input_energy = np.sum(linear_outs ** 2)
        energy_factor = 1.0 + 0.1 * np.tanh(input_energy - 1.0)
        
        # Apply energy-based frequency scaling
        scaled_freqs = self.frequencies * energy_factor
        
        activations = np.sin(scaled_freqs * linear_outs + self.phases)
        return activations
    
    def discover_features(self, x_batch):
        """Let neurons discover what features are in the data."""
        activations = []
        
        for i in range(len(x_batch)):
            act = self.forward(x_batch[i])
            activations.append(act)
        
        # Stack activations: (batch, n_neurons)
        activation_matrix = np.array(activations)
        
        # Compute correlation matrix
        correlations = np.corrcoef(activation_matrix.T)
        
        # Update feature correlation tracker
        self.feature_correlations = 0.9 * self.feature_correlations + 0.1 * correlations
        
        # Track discovery progress
        discovery_info = {
            'mode': self.discovery_mode,
            'correlation_variance': np.var(correlations),
            'feature_diversity': np.var(np.diag(correlations)),
            'max_correlation': np.max(correlations - np.eye(len(correlations)))
        }
        self.discovery_history.append(discovery_info)
        
        return activation_matrix
    
    def get_discovered_features(self):
        """Extract what features the network has discovered."""
        # High correlation = redundant neurons
        # Low correlation = diverse feature detection
        
        eigenvals, eigenvecs = np.linalg.eigh(self.feature_correlations)
        
        # Principal components show discovered feature directions
        return {
            'feature_diversity': np.var(eigenvals),
            'primary_features': eigenvals[-3:] if len(eigenvals) >= 3 else eigenvals,
            'feature_directions': eigenvecs[:, -3:] if len(eigenvecs) >= 3 else eigenvecs,
            'correlation_matrix': self.feature_correlations,
            'discovery_mode': self.discovery_mode,
            'n_neurons': self.n_neurons
        }
    
    def update_parameters(self, x, target, loss_fn):
        """Update parameters based on feature discovery."""
        output = self.forward(x)
        loss = loss_fn(output, target)
        
        # Compute gradients (simplified)
        loss_derivative = 2 * (output - target)  # MSE derivative
        
        # Update weights
        self.weights += self.learning_rate * loss_derivative[:, np.newaxis] * x[np.newaxis, :]
        
        # Update phases
        self.phases += self.phase_learning_rate * loss_derivative
        
        # Update frequencies
        self.frequencies += self.frequency_learning_rate * loss_derivative
        
        # Update discovery-specific parameters
        if self.discovery_mode == 'harmonic':
            self.harmonic_ratios += 0.01 * loss_derivative
        elif self.discovery_mode == 'quadrature':
            self.quadrature_pairs += 0.01 * loss_derivative[:len(self.quadrature_pairs)]
        
        # Keep phases in [0, 2π] range
        self.phases = self.phases % (2 * np.pi)
        
        # Keep frequencies positive
        self.frequencies = np.maximum(0.1, self.frequencies)
        
        return loss

class AdaptiveFeatureDiscovery:
    """
    Simplified AdaptiveFeatureDiscovery using NumPy for demo purposes.
    """
    def __init__(self, input_size: int, n_neurons: int):
        self.input_size = input_size
        self.n_neurons = n_neurons
        
        # Create individual neurons (simplified TunablePhaseNeuron)
        self.neurons = []
        for i in range(n_neurons):
            neuron = {
                'weights': np.random.randn(input_size) * 0.1,
                'phase_position': np.random.rand() * 2 * np.pi,
                'frequency': 1.0 + i * 0.1,  # Different frequencies
                'phase_increment': np.pi / 100,
                'n_phases': 32
            }
            self.neurons.append(neuron)
        
        # Mutual information tracker
        self.feature_correlations = np.zeros((n_neurons, n_neurons))
        self.discovery_history = []
        
    def forward_neuron(self, neuron, x):
        """Forward pass for a single neuron."""
        # Linear transformation
        linear_output = np.dot(x, neuron['weights'])
        
        # Phase wheel activation
        discrete_phase_idx = (neuron['phase_position'] / (2*np.pi)) * neuron['n_phases']
        base_idx = int(discrete_phase_idx) % neuron['n_phases']
        alpha = discrete_phase_idx - base_idx
        
        phase1 = base_idx * neuron['phase_increment']
        phase2 = ((base_idx + 1) % neuron['n_phases']) * neuron['phase_increment']
        
        activation1 = np.sin(neuron['frequency'] * linear_output + phase1)
        activation2 = np.sin(neuron['frequency'] * linear_output + phase2)
        
        return (1 - alpha) * activation1 + alpha * activation2
    
    def discover_features(self, x_batch):
        """Let neurons discover what features are in the data."""
        activations = []
        
        for x in x_batch:
            neuron_activations = []
            for neuron in self.neurons:
                act = self.forward_neuron(neuron, x)
                neuron_activations.append(act)
            activations.append(neuron_activations)
        
        # Stack activations: (batch, n_neurons)
        activation_matrix = np.array(activations)
        
        # Compute correlation matrix
        correlations = np.corrcoef(activation_matrix.T)
        
        # Update feature correlation tracker
        self.feature_correlations = 0.9 * self.feature_correlations + 0.1 * correlations
        
        # Track discovery progress
        discovery_info = {
            'correlation_variance': np.var(correlations),
            'feature_diversity': np.var(np.diag(correlations)),
            'max_correlation': np.max(correlations - np.eye(len(correlations))),
            'n_neurons': self.n_neurons
        }
        self.discovery_history.append(discovery_info)
        
        return activation_matrix
    
    def get_discovered_features(self):
        """Extract what features the network has discovered."""
        # High correlation = redundant neurons
        # Low correlation = diverse feature detection
        
        eigenvals, eigenvecs = np.linalg.eigh(self.feature_correlations)
        
        # Principal components show discovered feature directions
        return {
            'feature_diversity': np.var(eigenvals),
            'primary_features': eigenvals[-3:] if len(eigenvals) >= 3 else eigenvals,
            'feature_directions': eigenvecs[:, -3:] if len(eigenvecs) >= 3 else eigenvecs,
            'correlation_matrix': self.feature_correlations,
            'n_neurons': self.n_neurons,
            'discovery_history': self.discovery_history[-10:]  # Last 10 discoveries
        }

# ============================
# FieldIQ Integration
# ============================

class FieldIQFeatureDiscoveryProcessor:
    """
    A processor that applies FeatureDiscoveryPhaseLayer to FieldIQ data.
    """
    def __init__(self, input_size: int, n_neurons: int, discovery_mode: str = 'harmonic',
                 processing_mode: str = 'temporal'):
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.discovery_mode = discovery_mode
        self.processing_mode = processing_mode
        
        # Create feature discovery layer
        self.layer = FeatureDiscoveryPhaseLayer(input_size, n_neurons, discovery_mode)
        
    def process_field(self, field: FieldIQ) -> FieldIQ:
        """Process FieldIQ data through FeatureDiscoveryPhaseLayer."""
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
        processed_field = processed_field.with_role("feature_discovery_processed", True)
        processed_field = processed_field.with_role("discovery_mode", self.discovery_mode)
        processed_field = processed_field.with_role("processing_mode", self.processing_mode)
        processed_field = processed_field.with_role("n_neurons", self.n_neurons)
        processed_field = processed_field.with_role("discovered_features", self.layer.get_discovered_features())
        
        return processed_field
    
    def _process_chunks(self, data: np.ndarray, chunk_size: int) -> np.ndarray:
        """Process data in chunks to handle large arrays."""
        if len(data) <= chunk_size:
            # Process entire array
            return self.layer.forward(data)
        else:
            # Process in chunks and concatenate
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                if len(chunk) == chunk_size:
                    processed_chunk = self.layer.forward(chunk)
                    chunks.append(processed_chunk)
                else:
                    # Pad last chunk if necessary
                    padded_chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                    processed_chunk = self.layer.forward(padded_chunk)
                    chunks.append(processed_chunk[:len(chunk)])
            
            return np.concatenate(chunks)

# ============================
# Feature Discovery Realm System
# ============================

@dataclass
class FeatureDiscoveryRealm:
    """Realm for FeatureDiscoveryPhaseLayer processing."""
    name: str
    processor: FieldIQFeatureDiscoveryProcessor
    vm_node: Node
    learning_enabled: bool = True
    
    def __repr__(self):
        return f"<FeatureDiscoveryRealm {self.name}>"
    
    def field_processor(self, field: FieldIQ) -> FieldIQ:
        """Process field through the realm."""
        return self.processor.process_field(field)

def create_feature_discovery_realm(input_size: int, n_neurons: int, 
                                  discovery_mode: str = 'harmonic',
                                  processing_mode: str = 'temporal') -> FeatureDiscoveryRealm:
    """Create a FeatureDiscoveryPhaseLayer realm."""
    
    # Create processor
    processor = FieldIQFeatureDiscoveryProcessor(input_size, n_neurons, discovery_mode, processing_mode)
    
    # Create VM node
    vm_node = Val({
        'type': 'feature_discovery_realm',
        'name': f'feature_discovery_{discovery_mode}',
        'class': 'FieldIQFeatureDiscoveryProcessor',
        'parameters': {
            'input_size': input_size, 
            'n_neurons': n_neurons, 
            'discovery_mode': discovery_mode,
            'processing_mode': processing_mode
        },
        'vm_operations': {'primary': 'FEATURE_DISCOVERY'}
    })
    
    return FeatureDiscoveryRealm(
        name=f'feature_discovery_{discovery_mode}',
        processor=processor,
        vm_node=vm_node,
        learning_enabled=True
    )

# ============================
# Adaptive Feature Discovery Pipeline
# ============================

class AdaptiveFeatureDiscoveryPipeline:
    """
    Adaptive AI pipeline using FeatureDiscoveryPhaseLayers.
    """
    
    def __init__(self, num_carriers: int = 4):
        self.num_carriers = num_carriers
        self.carriers = []
        self.data_queue = Queue()
        self.running = False
        
        # Create FeatureDiscoveryPhaseLayer realms
        self.processing_realms = {
            'harmonic': create_feature_discovery_realm(100, 16, 'harmonic', 'temporal'),
            'quadrature': create_feature_discovery_realm(100, 24, 'quadrature', 'spectral'),
            'adaptive': create_feature_discovery_realm(100, 32, 'adaptive', 'adaptive')
        }
        
        # Initialize carriers
        self._initialize_carriers()
    
    def _initialize_carriers(self):
        """Initialize carriers."""
        for i in range(self.num_carriers):
            carrier = {
                'carrier_id': f'feature_discovery_carrier_{i}',
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
                field = make_field_from_real(data, sr, tag=("feature_discovery_ingest", "raw"))
            else:
                field = make_field_from_real(data.flatten(), sr, tag=("feature_discovery_ingest", "raw"))
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
            field = make_field_from_real(synthetic, sr, tag=("feature_discovery_ingest", "synthetic"))
        
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
            return 'quadrature'  # Complex phase relationships
        else:
            return 'harmonic'  # Simple harmonic processing
    
    def process_field_with_realm(self, field: FieldIQ, realm_name: str) -> FieldIQ:
        """Process field with the specified realm."""
        if realm_name in self.processing_realms:
            realm = self.processing_realms[realm_name]
            return realm.field_processor(field)
        else:
            return field
    
    async def continuous_feature_discovery_pipeline(self):
        """Continuous pipeline with feature discovery learning."""
        print("🚀 Starting Adaptive Feature Discovery AI Pipeline...")
        print("   Using FeatureDiscoveryPhaseLayers for automatic feature extraction")
        self.running = True
        
        cycle_count = 0
        while self.running:
            cycle_count += 1
            print(f"\n--- Feature Discovery Cycle {cycle_count} ---")
            
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
                
                field = make_field_from_real(synthetic, sr, tag=("feature_discovery_synthetic", f"cycle_{cycle_count}"))
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
                
                # Show discovered features
                discovered_features = processed_field.roles.get('discovered_features', {})
                if discovered_features:
                    print(f"   Discovery mode: {discovered_features.get('discovery_mode', 'unknown')}")
                    print(f"   Feature diversity: {discovered_features.get('feature_diversity', 0):.6f}")
                    print(f"   Primary features: {discovered_features.get('primary_features', [])[:3]}")
                    print(f"   N neurons: {discovered_features.get('n_neurons', 0)}")
                
                # Simulate learning (update feature discovery)
                realm = self.processing_realms[realm_name]
                if realm.learning_enabled:
                    # Simulate feature discovery learning
                    learning_gradient = np.random.randn() * 0.1
                    # Update discovery parameters
                    if realm_name == 'harmonic':
                        realm.processor.layer.harmonic_ratios += learning_gradient * 0.01
                    elif realm_name == 'quadrature':
                        realm.processor.layer.quadrature_pairs += learning_gradient * 0.01
                    print(f"   Learning: Updated {realm_name} discovery by {learning_gradient:.3f}")
            
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
        print("🛑 Stopping adaptive feature discovery AI pipeline...")

# ============================
# Demo Functions
# ============================

def demo_feature_discovery_layer():
    """Demonstrate FeatureDiscoveryPhaseLayer functionality."""
    print("=== FeatureDiscoveryPhaseLayer Demo ===\n")
    
    # Test different discovery modes
    discovery_modes = ['harmonic', 'quadrature', 'adaptive']
    
    for mode in discovery_modes:
        print(f"\n--- {mode.upper()} Discovery Mode ---")
        
        # Create layer
        input_size = 100
        n_neurons = 8
        layer = FeatureDiscoveryPhaseLayer(input_size, n_neurons, mode)
        
        print(f"Layer parameters:")
        print(f"- Input size: {input_size}")
        print(f"- Number of neurons: {n_neurons}")
        print(f"- Discovery mode: {mode}")
        
        # Test input
        test_input = np.random.randn(input_size)
        print(f"Test input shape: {test_input.shape}")
        
        # Forward pass
        output = layer.forward(test_input)
        print(f"Layer output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Show discovery-specific parameters
        if mode == 'harmonic':
            print(f"Harmonic ratios: {layer.harmonic_ratios[:5]}")
        elif mode == 'quadrature':
            print(f"Quadrature pairs: {layer.quadrature_pairs[:3]}")
        elif mode == 'adaptive':
            print(f"Adaptation rate: {layer.adaptation_rate}")
        
        # Test feature discovery
        x_batch = np.random.randn(5, input_size)
        activation_matrix = layer.discover_features(x_batch)
        print(f"Feature discovery matrix shape: {activation_matrix.shape}")
        
        # Show discovered features
        features = layer.get_discovered_features()
        print(f"Discovered features:")
        print(f"- Feature diversity: {features['feature_diversity']:.6f}")
        print(f"- Primary features: {features['primary_features'][:3]}")
        print(f"- Discovery history length: {len(features.get('discovery_history', []))}")

def demo_adaptive_feature_discovery():
    """Demonstrate AdaptiveFeatureDiscovery functionality."""
    print("\n=== AdaptiveFeatureDiscovery Demo ===\n")
    
    # Create adaptive discovery
    input_size = 100
    n_neurons = 12
    discovery = AdaptiveFeatureDiscovery(input_size, n_neurons)
    
    print(f"Adaptive discovery parameters:")
    print(f"- Input size: {input_size}")
    print(f"- Number of neurons: {n_neurons}")
    
    # Test with batch data
    x_batch = np.random.randn(10, input_size)
    print(f"Test batch shape: {x_batch.shape}")
    
    # Discover features
    activation_matrix = discovery.discover_features(x_batch)
    print(f"Activation matrix shape: {activation_matrix.shape}")
    
    # Show discovered features
    features = discovery.get_discovered_features()
    print(f"Discovered features:")
    print(f"- Feature diversity: {features['feature_diversity']:.6f}")
    print(f"- Primary features: {features['primary_features'][:3]}")
    print(f"- Discovery history length: {len(features.get('discovery_history', []))}")
    
    # Show neuron characteristics
    print(f"\nNeuron characteristics:")
    for i, neuron in enumerate(discovery.neurons[:5]):  # Show first 5
        print(f"  Neuron {i}: Phase={neuron['phase_position']:.3f}, "
              f"Freq={neuron['frequency']:.3f}")

def demo_feature_discovery_realms():
    """Demonstrate FeatureDiscoveryPhaseLayer realms."""
    print("\n=== FeatureDiscoveryPhaseLayer Realms Demo ===\n")
    
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
        'harmonic': create_feature_discovery_realm(100, 16, 'harmonic', 'temporal'),
        'quadrature': create_feature_discovery_realm(100, 24, 'quadrature', 'spectral'),
        'adaptive': create_feature_discovery_realm(100, 32, 'adaptive', 'adaptive')
    }
    
    print(f"\nCreated {len(realms)} FeatureDiscoveryPhaseLayer realms:")
    for name, realm in realms.items():
        print(f"  - {name}: {realm.processor.n_neurons} neurons, "
              f"mode={realm.processor.discovery_mode}")
    
    # Test each realm
    print(f"\nTesting FeatureDiscoveryPhaseLayer realms on field:")
    for name, realm in realms.items():
        try:
            processed_field = realm.field_processor(field)
            energy = np.sum(np.abs(processed_field.z) ** 2)
            
            print(f"  {name:10} | Energy: {energy:8.2f} | Learning: {realm.learning_enabled}")
            
            # Show discovered features
            discovered_features = processed_field.roles.get('discovered_features', {})
            if discovered_features:
                print(f"  {'':10} | Mode: {discovered_features.get('discovery_mode', 'unknown')}, "
                      f"Diversity: {discovered_features.get('feature_diversity', 0):.6f}")
            
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

async def demo_adaptive_pipeline():
    """Demonstrate the adaptive feature discovery AI pipeline."""
    print("\n=== Adaptive Feature Discovery AI Pipeline Demo ===\n")
    
    # Create pipeline
    pipeline = AdaptiveFeatureDiscoveryPipeline(num_carriers=3)
    
    print(f"Created adaptive pipeline with {len(pipeline.carriers)} carriers")
    print(f"Processing realms: {list(pipeline.processing_realms.keys())}")
    
    # Add test data
    test_data = np.random.randn(1000) * 0.5
    pipeline.add_data(test_data)
    
    # Run for a few cycles
    try:
        await asyncio.wait_for(pipeline.continuous_feature_discovery_pipeline(), timeout=10.0)
    except asyncio.TimeoutError:
        print("\n⏰ Adaptive pipeline demo completed after 10 seconds")
    
    pipeline.stop()

def main():
    """Run all feature discovery network demos."""
    print("🔷 Feature Discovery Phase Networks Demo")
    print("=" * 60)
    
    # Demo individual components
    demo_feature_discovery_layer()
    demo_adaptive_feature_discovery()
    
    # Demo realms
    demo_feature_discovery_realms()
    
    # Demo adaptive pipeline
    asyncio.run(demo_adaptive_pipeline())
    
    print(f"\n🔷 Feature Discovery Phase Networks Complete")
    print("✓ Harmonic discovery for frequency decomposition")
    print("✓ Quadrature discovery for sin/cos pairs")
    print("✓ Adaptive discovery for data-driven learning")
    print("✓ FieldIQ integration with feature tracking")
    print("✓ Adaptive feature discovery pipeline")

if __name__ == "__main__":
    main()

