# ============================
# Coordinated Phase Networks for Self-Tuning Neural Networks
# ============================
"""
Enhanced phase wheel networks with gradient analysis and coordination mechanisms
for sophisticated self-tuning neural networks in the singularity platform.
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
# Gradient Analysis Implementation (No PyTorch)
# ============================

class PhaseWheelGradientAnalysis:
    """
    Simplified PhaseWheelGradientAnalysis using NumPy for demo purposes.
    """
    def __init__(self, input_size: int, n_phases: int = 16):
        self.input_size = input_size
        self.n_phases = n_phases
        
        # Learnable parameters
        self.weights = np.random.randn(input_size) * 0.1
        self.phase_position = 1.5  # learnable phase
        self.frequency = 2.0
        
        # Learning parameters
        self.learning_rate = 0.01
        self.phase_learning_rate = 0.001
        self.frequency_learning_rate = 0.001
        
        # Gradient tracking
        self.phase_gradient = 0.0
        self.frequency_gradient = 0.0
        self.weight_gradients = np.zeros(input_size)
        
    def forward(self, x):
        """Forward pass with phase-shifted activation."""
        # Ensure x is 1D
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # Linear transformation
        linear_out = np.dot(x, self.weights)
        
        # Phase-shifted activation
        activation = np.sin(self.frequency * linear_out + self.phase_position)
        
        return activation
    
    def analyze_gradients(self, x, target, loss_fn):
        """Analyze how gradients flow through phase parameters."""
        output = self.forward(x)
        loss = loss_fn(output, target)
        
        # Compute gradients numerically (simplified)
        # ∂L/∂φ = ∂L/∂activation · ∂activation/∂φ
        # = ∂L/∂activation · cos(ωx + φ)
        
        # Compute activation derivative
        linear_out = np.dot(x, self.weights)
        activation_derivative = np.cos(self.frequency * linear_out + self.phase_position)
        
        # Compute loss derivative (simplified)
        loss_derivative = 2 * (output - target)  # MSE derivative
        
        # Phase gradient: ∂L/∂φ = ∂L/∂activation · cos(ωx + φ)
        self.phase_gradient = loss_derivative * activation_derivative
        
        # Frequency gradient: ∂L/∂ω = ∂L/∂activation · x · cos(ωx + φ)
        self.frequency_gradient = loss_derivative * linear_out * activation_derivative
        
        # Weight gradients: ∂L/∂w = ∂L/∂activation · ω · cos(ωx + φ) · x
        self.weight_gradients = loss_derivative * self.frequency * activation_derivative * x
        
        print(f"Phase position: {self.phase_position:.4f}")
        print(f"Phase gradient: {self.phase_gradient:.6f}")
        print(f"Frequency gradient: {self.frequency_gradient:.6f}")
        print(f"Weight gradient norm: {np.linalg.norm(self.weight_gradients):.6f}")
        
        return loss
    
    def update_parameters(self):
        """Update parameters based on computed gradients."""
        self.phase_position += self.phase_learning_rate * self.phase_gradient
        self.frequency += self.frequency_learning_rate * self.frequency_gradient
        self.weights += self.learning_rate * self.weight_gradients
        
        # Keep phase in [0, 2π] range
        self.phase_position = self.phase_position % (2 * np.pi)
        
        # Keep frequency positive
        self.frequency = max(0.1, self.frequency)

class CoordinatedPhaseLayer:
    """
    Simplified CoordinatedPhaseLayer using NumPy for demo purposes.
    """
    def __init__(self, input_size: int, n_neurons: int, coordination_strength: float = 0.1):
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.coordination_strength = coordination_strength
        
        # Each neuron has its own weights and phase
        self.weights = np.random.randn(n_neurons, input_size) * 0.1
        self.phases = np.random.randn(n_neurons)  # one phase per neuron
        self.frequencies = np.ones(n_neurons)
        
        # Coordination terms
        self.target_phase_spacing = np.linspace(0, 2*np.pi, n_neurons + 1)[:-1]
        
        # Learning parameters
        self.learning_rate = 0.01
        self.phase_learning_rate = 0.001
        self.frequency_learning_rate = 0.001
        
    def forward(self, x):
        """Forward pass through coordinated layer."""
        # Ensure x is 1D
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # Each neuron processes input
        linear_outs = np.dot(self.weights, x)  # (n_neurons,)
        
        # Apply phase-shifted activations
        activations = np.sin(self.frequencies * linear_outs + self.phases)
        
        return activations
    
    def coordination_loss(self):
        """Encourage neurons to spread out across phase wheel."""
        # Sort phases to find spacing
        sorted_phases = np.sort(self.phases)
        
        # Compute actual spacing vs ideal spacing
        actual_spacing = np.diff(np.append(sorted_phases, sorted_phases[0] + 2*np.pi))
        ideal_spacing = 2*np.pi / self.n_neurons
        
        # Penalize uneven spacing
        spacing_loss = np.var(actual_spacing)
        
        return self.coordination_strength * spacing_loss
    
    def phase_repulsion_loss(self):
        """Penalize phases that are too close together."""
        phase_diffs = self.phases[:, np.newaxis] - self.phases[np.newaxis, :]
        
        # Wrap to [-π, π]
        phase_diffs = np.arctan2(np.sin(phase_diffs), np.cos(phase_diffs))
        
        # Penalize phases that are too close
        repulsion = np.exp(-0.5 * phase_diffs**2 / 0.1**2)
        
        # Subtract self-repulsion (diagonal terms)
        np.fill_diagonal(repulsion, 0)
        
        return np.sum(repulsion)
    
    def phase_diversity_reward(self):
        """Measure how well neurons tile the phase space."""
        # Discretize wheel into 64 bins
        phase_coverage = np.zeros(64)
        for phase in self.phases:
            idx = int((phase / (2*np.pi)) * 64) % 64
            phase_coverage[idx] += 1
        
        # Reward uniform coverage
        return -np.var(phase_coverage)
    
    def get_coordination_metrics(self):
        """Get metrics about phase coordination."""
        sorted_phases = np.sort(self.phases)
        actual_spacing = np.diff(np.append(sorted_phases, sorted_phases[0] + 2*np.pi))
        ideal_spacing = 2*np.pi / self.n_neurons
        
        return {
            'coordination_loss': self.coordination_loss(),
            'repulsion_loss': self.phase_repulsion_loss(),
            'diversity_reward': self.phase_diversity_reward(),
            'spacing_variance': np.var(actual_spacing),
            'ideal_spacing': ideal_spacing,
            'phase_spread': np.max(self.phases) - np.min(self.phases)
        }

# ============================
# FieldIQ Integration
# ============================

class FieldIQCoordinatedProcessor:
    """
    A processor that applies CoordinatedPhaseLayer to FieldIQ data.
    """
    def __init__(self, input_size: int, n_neurons: int, coordination_strength: float = 0.1,
                 processing_mode: str = 'temporal'):
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.coordination_strength = coordination_strength
        self.processing_mode = processing_mode
        
        # Create coordinated layer
        self.layer = CoordinatedPhaseLayer(input_size, n_neurons, coordination_strength)
        
    def process_field(self, field: FieldIQ) -> FieldIQ:
        """Process FieldIQ data through CoordinatedPhaseLayer."""
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
        processed_field = processed_field.with_role("coordinated_phase_processed", True)
        processed_field = processed_field.with_role("processing_mode", self.processing_mode)
        processed_field = processed_field.with_role("n_neurons", self.n_neurons)
        processed_field = processed_field.with_role("coordination_metrics", self.layer.get_coordination_metrics())
        
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
# Coordinated Phase Realm System
# ============================

@dataclass
class CoordinatedPhaseRealm:
    """Realm for CoordinatedPhaseLayer processing."""
    name: str
    processor: FieldIQCoordinatedProcessor
    vm_node: Node
    learning_enabled: bool = True
    
    def __repr__(self):
        return f"<CoordinatedPhaseRealm {self.name}>"
    
    def field_processor(self, field: FieldIQ) -> FieldIQ:
        """Process field through the realm."""
        return self.processor.process_field(field)

def create_coordinated_phase_realm(input_size: int, n_neurons: int, 
                                  coordination_strength: float = 0.1,
                                  processing_mode: str = 'temporal') -> CoordinatedPhaseRealm:
    """Create a CoordinatedPhaseLayer realm."""
    
    # Create processor
    processor = FieldIQCoordinatedProcessor(input_size, n_neurons, coordination_strength, processing_mode)
    
    # Create VM node
    vm_node = Val({
        'type': 'coordinated_phase_realm',
        'name': f'coordinated_phase_{processing_mode}',
        'class': 'FieldIQCoordinatedProcessor',
        'parameters': {
            'input_size': input_size, 
            'n_neurons': n_neurons, 
            'coordination_strength': coordination_strength,
            'processing_mode': processing_mode
        },
        'vm_operations': {'primary': 'COORDINATED_PHASE'}
    })
    
    return CoordinatedPhaseRealm(
        name=f'coordinated_phase_{processing_mode}',
        processor=processor,
        vm_node=vm_node,
        learning_enabled=True
    )

# ============================
# Self-Tuning Coordinated Pipeline
# ============================

class SelfTuningCoordinatedPipeline:
    """
    Self-tuning AI pipeline using CoordinatedPhaseLayers.
    """
    
    def __init__(self, num_carriers: int = 4):
        self.num_carriers = num_carriers
        self.carriers = []
        self.data_queue = Queue()
        self.running = False
        
        # Create CoordinatedPhaseLayer realms
        self.processing_realms = {
            'temporal': create_coordinated_phase_realm(100, 16, 0.1, 'temporal'),
            'spectral': create_coordinated_phase_realm(100, 24, 0.2, 'spectral'),
            'adaptive': create_coordinated_phase_realm(100, 32, 0.15, 'adaptive')
        }
        
        # Initialize carriers
        self._initialize_carriers()
    
    def _initialize_carriers(self):
        """Initialize carriers."""
        for i in range(self.num_carriers):
            carrier = {
                'carrier_id': f'coordinated_carrier_{i}',
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
                field = make_field_from_real(data, sr, tag=("coordinated_ingest", "raw"))
            else:
                field = make_field_from_real(data.flatten(), sr, tag=("coordinated_ingest", "raw"))
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
            field = make_field_from_real(synthetic, sr, tag=("coordinated_ingest", "synthetic"))
        
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
    
    async def continuous_coordinated_pipeline(self):
        """Continuous pipeline with coordinated phase learning."""
        print("🚀 Starting Coordinated Phase AI Pipeline...")
        print("   Using CoordinatedPhaseLayers for sophisticated self-tuning")
        self.running = True
        
        cycle_count = 0
        while self.running:
            cycle_count += 1
            print(f"\n--- Coordinated Cycle {cycle_count} ---")
            
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
                
                field = make_field_from_real(synthetic, sr, tag=("coordinated_synthetic", f"cycle_{cycle_count}"))
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
                
                # Show coordination metrics
                coordination_metrics = processed_field.roles.get('coordination_metrics', {})
                if coordination_metrics:
                    print(f"   Coordination loss: {coordination_metrics.get('coordination_loss', 0):.6f}")
                    print(f"   Repulsion loss: {coordination_metrics.get('repulsion_loss', 0):.6f}")
                    print(f"   Diversity reward: {coordination_metrics.get('diversity_reward', 0):.6f}")
                    print(f"   Phase spread: {coordination_metrics.get('phase_spread', 0):.3f}")
                
                # Simulate learning (update coordination)
                realm = self.processing_realms[realm_name]
                if realm.learning_enabled:
                    # Simulate gradient-based learning
                    learning_gradient = np.random.randn() * 0.1
                    # Update phases to improve coordination
                    realm.processor.layer.phases += learning_gradient * 0.01
                    realm.processor.layer.phases = realm.processor.layer.phases % (2 * np.pi)
                    print(f"   Learning: Updated phase coordination by {learning_gradient:.3f}")
            
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
        print("🛑 Stopping coordinated phase AI pipeline...")

# ============================
# Demo Functions
# ============================

def demo_gradient_analysis():
    """Demonstrate PhaseWheelGradientAnalysis functionality."""
    print("=== PhaseWheelGradientAnalysis Demo ===\n")
    
    # Create test case
    input_size = 50
    test_input = np.random.randn(input_size)
    target = np.random.randn()
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Target: {target:.3f}")
    
    # Create gradient analyzer
    analyzer = PhaseWheelGradientAnalysis(input_size, n_phases=16)
    
    print(f"\nAnalyzer parameters:")
    print(f"- Input size: {analyzer.input_size}")
    print(f"- Number of phases: {analyzer.n_phases}")
    print(f"- Initial phase position: {analyzer.phase_position:.3f}")
    print(f"- Initial frequency: {analyzer.frequency:.3f}")
    
    # Define loss function (MSE)
    def mse_loss(output, target):
        return (output - target) ** 2
    
    # Analyze gradients
    print(f"\nAnalyzing gradients...")
    loss = analyzer.analyze_gradients(test_input, target, mse_loss)
    print(f"Loss: {loss:.6f}")
    
    # Simulate learning steps
    print(f"\nSimulating learning...")
    for i in range(5):
        loss = analyzer.analyze_gradients(test_input, target, mse_loss)
        analyzer.update_parameters()
        
        print(f"  Step {i+1}: Loss={loss:.6f}, Phase={analyzer.phase_position:.3f}, "
              f"Freq={analyzer.frequency:.3f}")
    
    return analyzer

def demo_coordinated_phase_layer():
    """Demonstrate CoordinatedPhaseLayer functionality."""
    print("\n=== CoordinatedPhaseLayer Demo ===\n")
    
    # Create layer
    input_size = 100
    n_neurons = 8
    layer = CoordinatedPhaseLayer(input_size, n_neurons, coordination_strength=0.1)
    
    print(f"Layer parameters:")
    print(f"- Input size: {input_size}")
    print(f"- Number of neurons: {n_neurons}")
    print(f"- Coordination strength: {layer.coordination_strength}")
    
    # Test input
    test_input = np.random.randn(input_size)
    print(f"\nTest input shape: {test_input.shape}")
    
    # Forward pass
    output = layer.forward(test_input)
    print(f"Layer output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Show coordination metrics
    metrics = layer.get_coordination_metrics()
    print(f"\nCoordination metrics:")
    print(f"- Coordination loss: {metrics['coordination_loss']:.6f}")
    print(f"- Repulsion loss: {metrics['repulsion_loss']:.6f}")
    print(f"- Diversity reward: {metrics['diversity_reward']:.6f}")
    print(f"- Spacing variance: {metrics['spacing_variance']:.6f}")
    print(f"- Phase spread: {metrics['phase_spread']:.3f}")
    
    # Show phase distribution
    print(f"\nPhase distribution:")
    for i, phase in enumerate(layer.phases):
        print(f"  Neuron {i}: Phase={phase:.3f}, Freq={layer.frequencies[i]:.3f}")
    
    return layer

def demo_coordinated_phase_realms():
    """Demonstrate CoordinatedPhaseLayer realms."""
    print("\n=== CoordinatedPhaseLayer Realms Demo ===\n")
    
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
        'temporal': create_coordinated_phase_realm(100, 16, 0.1, 'temporal'),
        'spectral': create_coordinated_phase_realm(100, 24, 0.2, 'spectral'),
        'adaptive': create_coordinated_phase_realm(100, 32, 0.15, 'adaptive')
    }
    
    print(f"\nCreated {len(realms)} CoordinatedPhaseLayer realms:")
    for name, realm in realms.items():
        print(f"  - {name}: {realm.processor.n_neurons} neurons, "
              f"coordination={realm.processor.coordination_strength}")
    
    # Test each realm
    print(f"\nTesting CoordinatedPhaseLayer realms on field:")
    for name, realm in realms.items():
        try:
            processed_field = realm.field_processor(field)
            energy = np.sum(np.abs(processed_field.z) ** 2)
            
            print(f"  {name:10} | Energy: {energy:8.2f} | Learning: {realm.learning_enabled}")
            
            # Show coordination metrics
            coordination_metrics = processed_field.roles.get('coordination_metrics', {})
            if coordination_metrics:
                print(f"  {'':10} | Coord loss: {coordination_metrics.get('coordination_loss', 0):.6f}, "
                      f"Repulsion: {coordination_metrics.get('repulsion_loss', 0):.6f}")
            
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

async def demo_coordinated_pipeline():
    """Demonstrate the coordinated phase AI pipeline."""
    print("\n=== Coordinated Phase AI Pipeline Demo ===\n")
    
    # Create pipeline
    pipeline = SelfTuningCoordinatedPipeline(num_carriers=3)
    
    print(f"Created coordinated pipeline with {len(pipeline.carriers)} carriers")
    print(f"Processing realms: {list(pipeline.processing_realms.keys())}")
    
    # Add test data
    test_data = np.random.randn(1000) * 0.5
    pipeline.add_data(test_data)
    
    # Run for a few cycles
    try:
        await asyncio.wait_for(pipeline.continuous_coordinated_pipeline(), timeout=10.0)
    except asyncio.TimeoutError:
        print("\n⏰ Coordinated pipeline demo completed after 10 seconds")
    
    pipeline.stop()

def main():
    """Run all coordinated phase network demos."""
    print("🔷 Coordinated Phase Networks Demo")
    print("=" * 60)
    
    # Demo individual components
    demo_gradient_analysis()
    demo_coordinated_phase_layer()
    
    # Demo realms
    demo_coordinated_phase_realms()
    
    # Demo coordinated pipeline
    asyncio.run(demo_coordinated_pipeline())
    
    print(f"\n🔷 Coordinated Phase Networks Complete")
    print("✓ Gradient analysis for phase parameters")
    print("✓ Coordinated phase layers with spacing control")
    print("✓ Phase repulsion and diversity mechanisms")
    print("✓ FieldIQ integration with coordination metrics")
    print("✓ Self-tuning coordinated pipeline")

if __name__ == "__main__":
    main()

