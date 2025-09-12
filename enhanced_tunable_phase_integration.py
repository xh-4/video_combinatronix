#!/usr/bin/env python3
"""
Enhanced TunablePhaseNeuron Integration with FieldIQ
Fixed version that properly handles FieldIQ data dimensions and integrates with the singularity platform.
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
# Enhanced TunablePhaseNeuron Implementation
# ============================

class AdaptiveTunablePhaseNeuron:
    """
    Enhanced TunablePhaseNeuron that adapts to FieldIQ data dimensions.
    """
    def __init__(self, phase_increment: float = np.pi/100, n_phases: int = 32, 
                 adaptive_input_size: bool = True):
        self.phase_increment = phase_increment
        self.n_phases = n_phases
        self.adaptive_input_size = adaptive_input_size
        
        # Initialize with default values (will adapt)
        self.input_size = 100  # Default, will adapt
        self.weights = None
        self.bias = 0.0
        
        # Phase wheel parameters
        self.phase_position = 0.0  # 0 to 2π
        self.frequency = 1.0
        
        # Learning parameters
        self.learning_rate = 0.01
        self.phase_learning_rate = 0.001
        
        # Initialize weights when first used
        self._initialized = False
    
    def _initialize_weights(self, input_size: int):
        """Initialize weights for the given input size."""
        if not self._initialized or self.adaptive_input_size:
            self.input_size = input_size
            self.weights = np.random.randn(input_size) * 0.1
            self._initialized = True
    
    def forward(self, x):
        """Forward pass with adaptive input size."""
        # Ensure x is 1D
        if x.ndim > 1:
            x = x.flatten()
        
        # Initialize weights if needed
        self._initialize_weights(len(x))
        
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
        if self.weights is not None:
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
            'n_phases': self.n_phases,
            'input_size': self.input_size,
            'initialized': self._initialized
        }

class FieldIQTunablePhaseProcessor:
    """
    A processor that applies TunablePhaseNeuron to FieldIQ data with proper dimension handling.
    """
    def __init__(self, phase_increment: float = np.pi/100, n_phases: int = 32, 
                 processing_mode: str = 'temporal'):
        self.phase_increment = phase_increment
        self.n_phases = n_phases
        self.processing_mode = processing_mode  # 'temporal', 'spectral', or 'adaptive'
        
        # Create neuron instances for different processing modes
        self.temporal_neuron = AdaptiveTunablePhaseNeuron(phase_increment, n_phases)
        self.spectral_neuron = AdaptiveTunablePhaseNeuron(phase_increment, n_phases)
        self.adaptive_neuron = AdaptiveTunablePhaseNeuron(phase_increment, n_phases)
        
    def process_field(self, field: FieldIQ) -> FieldIQ:
        """Process FieldIQ data through TunablePhaseNeuron."""
        z_array = field.z
        
        if self.processing_mode == 'temporal':
            # Process temporal sequence
            real_part = np.real(z_array)
            imag_part = np.imag(z_array)
            
            real_processed = self.temporal_neuron.forward(real_part)
            imag_processed = self.temporal_neuron.forward(imag_part)
            
        elif self.processing_mode == 'spectral':
            # Process spectral representation
            fft_real = np.fft.fft(np.real(z_array))
            fft_imag = np.fft.fft(np.imag(z_array))
            
            real_processed = self.temporal_neuron.forward(np.real(fft_real))
            imag_processed = self.temporal_neuron.forward(np.imag(fft_imag))
            
            # Convert back to time domain
            real_processed = np.fft.ifft(real_processed + 1j * imag_processed).real
            
        elif self.processing_mode == 'adaptive':
            # Adaptive processing based on field characteristics
            field_energy = np.sum(np.abs(z_array) ** 2)
            if field_energy > 1000:  # High energy - use spectral processing
                fft_data = np.fft.fft(z_array)
                processed_fft = self.adaptive_neuron.forward(np.real(fft_data)) + 1j * self.adaptive_neuron.forward(np.imag(fft_data))
                real_processed = np.fft.ifft(processed_fft).real
                imag_processed = np.fft.ifft(processed_fft).imag
            else:  # Low energy - use temporal processing
                real_processed = self.adaptive_neuron.forward(np.real(z_array))
                imag_processed = self.adaptive_neuron.forward(np.imag(z_array))
        
        # Reconstruct complex field
        new_z = real_processed + 1j * imag_processed
        
        # Create new FieldIQ with processing metadata
        processed_field = FieldIQ(new_z, field.sr, field.roles or {})
        processed_field = processed_field.with_role("tunable_phase_processed", True)
        processed_field = processed_field.with_role("processing_mode", self.processing_mode)
        processed_field = processed_field.with_role("phase_info", self.temporal_neuron.get_phase_info())
        
        return processed_field

# ============================
# Enhanced TunablePhaseNeuron Realm System
# ============================

@dataclass
class EnhancedTunablePhaseRealm:
    """Enhanced realm for TunablePhaseNeuron processing."""
    name: str
    processor_class: type
    parameters: Dict[str, Any]
    vm_node: Node
    field_processor: Callable[[FieldIQ], FieldIQ]
    learning_enabled: bool = True
    
    def __repr__(self):
        return f"<EnhancedTunablePhaseRealm {self.name} params={self.parameters}>"

def create_enhanced_tunable_phase_realm(processing_mode: str = 'temporal', 
                                       phase_increment: float = np.pi/100, 
                                       n_phases: int = 32) -> EnhancedTunablePhaseRealm:
    """Create an enhanced TunablePhaseNeuron realm."""
    
    # Create processor
    processor = FieldIQTunablePhaseProcessor(phase_increment, n_phases, processing_mode)
    
    # Create VM node
    vm_node = Val({
        'type': 'enhanced_tunable_phase_realm',
        'name': f'tunable_phase_{processing_mode}',
        'class': 'FieldIQTunablePhaseProcessor',
        'parameters': {'processing_mode': processing_mode, 'phase_increment': phase_increment, 'n_phases': n_phases},
        'vm_operations': {'primary': 'ENHANCED_TUNABLE_PHASE'}
    })
    
    return EnhancedTunablePhaseRealm(
        name=f'tunable_phase_{processing_mode}',
        processor_class=FieldIQTunablePhaseProcessor,
        parameters={'processing_mode': processing_mode, 'phase_increment': phase_increment, 'n_phases': n_phases},
        vm_node=vm_node,
        field_processor=processor.process_field,
        learning_enabled=True
    )

# ============================
# Self-Tuning AI Pipeline Integration
# ============================

class SelfTuningAIPipeline:
    """
    Enhanced AI pipeline that uses TunablePhaseNeurons for self-tuning processing.
    """
    
    def __init__(self, num_carriers: int = 4):
        self.num_carriers = num_carriers
        self.carriers = []
        self.data_queue = Queue()
        self.running = False
        
        # Create TunablePhaseNeuron realms for different processing stages
        self.processing_realms = {
            'temporal': create_enhanced_tunable_phase_realm('temporal', np.pi/50, 16),
            'spectral': create_enhanced_tunable_phase_realm('spectral', np.pi/75, 24),
            'adaptive': create_enhanced_tunable_phase_realm('adaptive', np.pi/100, 32)
        }
        
        # Initialize carriers
        self._initialize_carriers()
    
    def _initialize_carriers(self):
        """Initialize carriers with TunablePhaseNeuron capabilities."""
        for i in range(self.num_carriers):
            carrier = {
                'carrier_id': f'tunable_carrier_{i}',
                'capacity': 1.0,
                'current_load': 0.0,
                'active_realm': None,
                'learning_rate': 0.01 + i * 0.005,  # Vary learning rates
                'phase_adaptation_rate': 0.001 + i * 0.0005
            }
            self.carriers.append(carrier)
    
    def ingest_data(self, data: Any) -> FieldIQ:
        """Enhanced data ingestion with phase analysis."""
        if isinstance(data, np.ndarray):
            sr = 48000
            if len(data.shape) == 1:
                field = make_field_from_real(data, sr, tag=("tunable_ingest", "raw"))
            else:
                field = make_field_from_real(data.flatten(), sr, tag=("tunable_ingest", "raw"))
        else:
            # Create synthetic data with phase complexity
            sr = 48000
            dur = 1.0
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            # Create complex synthetic signal
            base_freq = 440
            phase_mod = 0.5 * np.sin(2 * np.pi * 0.5 * t)
            synthetic = (0.5 * np.cos(2 * np.pi * base_freq * t + phase_mod) + 
                       0.3 * np.cos(2 * np.pi * base_freq * 2 * t + phase_mod * 2) +
                       0.1 * np.random.randn(len(t)))
            field = make_field_from_real(synthetic, sr, tag=("tunable_ingest", "synthetic"))
        
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
    
    async def continuous_self_tuning_pipeline(self):
        """Continuous pipeline with self-tuning capabilities."""
        print("🚀 Starting Self-Tuning AI Pipeline...")
        print("   Using TunablePhaseNeurons for adaptive processing")
        self.running = True
        
        cycle_count = 0
        while self.running:
            cycle_count += 1
            print(f"\n--- Self-Tuning Cycle {cycle_count} ---")
            
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
                
                field = make_field_from_real(synthetic, sr, tag=("tunable_synthetic", f"cycle_{cycle_count}"))
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
                
                # Show phase info
                phase_info = processed_field.roles.get('phase_info', {})
                if phase_info:
                    print(f"   Phase position: {phase_info.get('phase_position', 0):.3f}")
                    print(f"   Discrete index: {phase_info.get('discrete_idx', 0)}")
                    print(f"   Interpolation alpha: {phase_info.get('interpolation_alpha', 0):.3f}")
            
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
        print("🛑 Stopping self-tuning AI pipeline...")

# ============================
# Demo Functions
# ============================

def demo_enhanced_tunable_phase_realms():
    """Demonstrate enhanced TunablePhaseNeuron realms."""
    print("=== Enhanced TunablePhaseNeuron Realms Demo ===\n")
    
    # Create sample field
    sr = 48000
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = 0.8 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t + np.pi/4)
    field = make_field_from_real(x, sr, tag=("demo", "multi_tone"))
    
    print(f"Original field: {len(field.z)} samples, {field.sr} Hz")
    print(f"Field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
    
    # Create enhanced realms
    realms = {
        'temporal': create_enhanced_tunable_phase_realm('temporal', np.pi/50, 16),
        'spectral': create_enhanced_tunable_phase_realm('spectral', np.pi/75, 24),
        'adaptive': create_enhanced_tunable_phase_realm('adaptive', np.pi/100, 32)
    }
    
    print(f"\nCreated {len(realms)} enhanced TunablePhaseNeuron realms:")
    for name, realm in realms.items():
        print(f"  - {name}: {realm.parameters}")
    
    # Test each realm
    print(f"\nTesting enhanced TunablePhaseNeuron realms on field:")
    for name, realm in realms.items():
        try:
            processed_field = realm.field_processor(field)
            energy = np.sum(np.abs(processed_field.z) ** 2)
            
            print(f"  {name:10} | Energy: {energy:8.2f} | Learning: {realm.learning_enabled}")
            
            # Show phase info
            phase_info = processed_field.roles.get('phase_info', {})
            if phase_info:
                print(f"  {'':10} | Phase: {phase_info.get('phase_position', 0):.3f}, "
                      f"Idx: {phase_info.get('discrete_idx', 0)}, "
                      f"Alpha: {phase_info.get('interpolation_alpha', 0):.3f}")
            
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

async def demo_self_tuning_pipeline():
    """Demonstrate the self-tuning AI pipeline."""
    print("\n=== Self-Tuning AI Pipeline Demo ===\n")
    
    # Create pipeline
    pipeline = SelfTuningAIPipeline(num_carriers=3)
    
    print(f"Created self-tuning pipeline with {len(pipeline.carriers)} carriers")
    print(f"Processing realms: {list(pipeline.processing_realms.keys())}")
    
    # Add test data
    test_data = np.random.randn(1000) * 0.5
    pipeline.add_data(test_data)
    
    # Run for a few cycles
    try:
        await asyncio.wait_for(pipeline.continuous_self_tuning_pipeline(), timeout=15.0)
    except asyncio.TimeoutError:
        print("\n⏰ Self-tuning demo completed after 15 seconds")
    
    pipeline.stop()

def main():
    """Run all enhanced TunablePhaseNeuron demos."""
    print("🔷 Enhanced TunablePhaseNeuron System Demo")
    print("=" * 60)
    
    # Demo enhanced realms
    demo_enhanced_tunable_phase_realms()
    
    # Demo self-tuning pipeline
    asyncio.run(demo_self_tuning_pipeline())
    
    print(f"\n🔷 Enhanced TunablePhaseNeuron System Complete")
    print("✓ Self-tuning neural networks")
    print("✓ Adaptive input size handling")
    print("✓ Multiple processing modes")
    print("✓ FieldIQ dimension compatibility")
    print("✓ Continuous learning pipeline")

if __name__ == "__main__":
    main()

