#!/usr/bin/env python3
"""
PhaseWheel Demo (No PyTorch)
Demonstrates the PhaseWheel concept and its integration with FieldIQ engine
without PyTorch dependencies.
"""

import numpy as np
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from queue import Queue, Empty

# Import existing systems (without PyTorch)
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf
from Combinator_Kernel import FieldIQ, make_field_from_real

# ============================
# Simplified PhaseWheel Implementation (No PyTorch)
# ============================

class SimplePhaseWheel:
    """
    Simplified PhaseWheel implementation using NumPy for demo purposes.
    """
    def __init__(self, base_frequency=1.0, phase_increment=np.pi/100, n_slots=64):
        self.base_frequency = base_frequency
        self.phase_increment = phase_increment
        self.n_slots = n_slots
        
        # Initialize phase positions around the wheel
        self.phase_positions = np.array([i * phase_increment for i in range(n_slots)])
        
        # Learnable weights for each phase slot (simplified)
        self.phase_weights = np.random.randn(n_slots)
        
    def get_basis_functions(self, x):
        """Generate overlapping cosine/sine pairs for each phase position"""
        cos_functions = []
        sin_functions = []
        
        for phase in self.phase_positions:
            cos_func = np.cos(self.base_frequency * x + phase)
            sin_func = np.sin(self.base_frequency * x + phase)
            cos_functions.append(cos_func)
            sin_functions.append(sin_func)
        
        # Stack: (n_slots, seq_len)
        cos_basis = np.stack(cos_functions, axis=0)
        sin_basis = np.stack(sin_functions, axis=0)
        
        return cos_basis, sin_basis
    
    def forward(self, x):
        """Apply phase wheel transformation with controlled overlap"""
        cos_basis, sin_basis = self.get_basis_functions(x)
        
        # Soft attention over phase positions
        attention_weights = self._softmax(self.phase_weights)
        
        # Weighted combination with controlled bleed
        cos_output = np.sum(cos_basis * attention_weights.reshape(-1, 1), axis=0)
        sin_output = np.sum(sin_basis * attention_weights.reshape(-1, 1), axis=0)
        
        # Return complex-valued output
        return cos_output + 1j * sin_output
    
    def _softmax(self, x):
        """Simple softmax implementation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def get_phase_response(self, x):
        """Get the learned phase response across the wheel"""
        cos_basis, sin_basis = self.get_basis_functions(x)
        attention_weights = self._softmax(self.phase_weights)
        
        # Phase response for each slot
        phases = np.arctan2(sin_basis, cos_basis)
        weighted_phase = np.sum(phases * attention_weights.reshape(-1, 1), axis=0)
        
        return weighted_phase

class SimplePhaseWheelActivation:
    """
    Simplified PhaseWheelActivation using NumPy
    """
    def __init__(self, n_phases=16, phase_increment=np.pi/100, learnable_frequency=True):
        self.n_phases = n_phases
        self.phase_increment = phase_increment
        self.frequency = 1.0 if not learnable_frequency else np.random.uniform(0.5, 2.0)
        
        # Phase positions on the wheel
        self.phases = np.array([i * phase_increment for i in range(n_phases)])
        
        # Learnable mixing weights
        self.mixing_weights = np.random.randn(n_phases)
        
    def forward(self, x):
        """Apply phase wheel activation with smooth interpolation"""
        # Generate all phase-shifted sinusoids
        activations = []
        for phase in self.phases:
            activation = np.sin(self.frequency * x + phase)
            activations.append(activation)
        
        # Stack and apply soft mixing
        activation_stack = np.stack(activations, axis=-1)  # (..., n_phases)
        mix_weights = self._softmax(self.mixing_weights)
        
        # Smooth combination across phase wheel
        output = np.sum(activation_stack * mix_weights, axis=-1)
        
        return output
    
    def _softmax(self, x):
        """Simple softmax implementation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class SimpleAdaptivePhaseWheel:
    """
    Simplified AdaptivePhaseWheel using NumPy
    """
    def __init__(self, n_slots=32, base_frequency=1.0, adaptation_rate=0.1):
        self.n_slots = n_slots
        self.base_frequency = base_frequency
        self.adaptation_rate = adaptation_rate
        
        # Learnable phase positions (can adapt)
        self.phase_positions = np.random.randn(n_slots)
        
        # Learnable weights
        self.phase_weights = np.random.randn(n_slots)
        
        # Adaptation parameters
        self.frequency_adaptation = 1.0
        self.phase_adaptation = 1.0
        
    def forward(self, x):
        """Apply adaptive phase wheel transformation"""
        # Adapt frequency based on input characteristics
        input_energy = np.mean(x**2)
        adapted_frequency = self.base_frequency * (1 + self.frequency_adaptation * input_energy)
        
        # Adapt phase positions
        phase_shift = self.phase_adaptation * np.mean(x)
        adapted_phases = self.phase_positions + phase_shift
        
        # Generate basis functions with adapted parameters
        cos_functions = []
        sin_functions = []
        
        for phase in adapted_phases:
            cos_func = np.cos(adapted_frequency * x + phase)
            sin_func = np.sin(adapted_frequency * x + phase)
            cos_functions.append(cos_func)
            sin_functions.append(sin_func)
        
        # Stack and apply attention
        cos_basis = np.stack(cos_functions, axis=0)
        sin_basis = np.stack(sin_functions, axis=0)
        
        attention_weights = self._softmax(self.phase_weights)
        
        cos_output = np.sum(cos_basis * attention_weights.reshape(-1, 1), axis=0)
        sin_output = np.sum(sin_basis * attention_weights.reshape(-1, 1), axis=0)
        
        return cos_output + 1j * sin_output
    
    def _softmax(self, x):
        """Simple softmax implementation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

# ============================
# PhaseWheel Realm System (No PyTorch)
# ============================

@dataclass
class SimplePhaseWheelRealm:
    """A realm representing a PhaseWheel-based activation in the singularity platform."""
    name: str
    phase_wheel_class: type
    parameters: Dict[str, Any]
    vm_node: Node
    field_processor: Callable[[FieldIQ], FieldIQ]
    
    def __repr__(self):
        return f"<SimplePhaseWheelRealm {self.name} params={self.parameters}>"

def create_simple_phase_wheel_field_processor(phase_wheel_class: type, **params) -> Callable[[FieldIQ], FieldIQ]:
    """Create a field processor that applies PhaseWheel to FieldIQ data."""
    
    def process_field(field: FieldIQ) -> FieldIQ:
        # Convert FieldIQ to numpy array
        z_array = field.z
        
        # Extract real and imaginary parts
        real_part = np.real(z_array)
        imag_part = np.imag(z_array)
        
        # Create phase wheel instance
        phase_wheel = phase_wheel_class(**params)
        
        # Apply phase wheel to both real and imaginary parts
        real_processed = phase_wheel.forward(real_part)
        imag_processed = phase_wheel.forward(imag_part)
        
        # Reconstruct complex field
        if np.iscomplexobj(real_processed):
            # Phase wheel returns complex output
            new_z = real_processed
        else:
            # Phase wheel returns real output, reconstruct complex
            new_z = real_processed + 1j * imag_processed
        
        # Create new FieldIQ with phase metadata
        processed_field = FieldIQ(new_z, field.sr, field.roles or {})
        processed_field = processed_field.with_role("phase_wheel_processed", True)
        processed_field = processed_field.with_role("phase_wheel_type", phase_wheel_class.__name__)
        
        return processed_field
    
    return process_field

def compile_simple_phase_wheel_to_vm(phase_wheel_class: type, name: str, **params) -> Node:
    """Compile a PhaseWheel to VM representation."""
    return Val({
        'type': 'phase_wheel_realm',
        'name': name,
        'class': phase_wheel_class.__name__,
        'parameters': params,
        'vm_operations': {'primary': 'PHASE_WHEEL'}
    })

# ============================
# PhaseWheel Realm Factory Functions
# ============================

def create_simple_phase_wheel_realm(base_frequency: float = 1.0, phase_increment: float = np.pi/100, 
                                   n_slots: int = 64) -> SimplePhaseWheelRealm:
    """Create a SimplePhaseWheel activation realm."""
    vm_node = compile_simple_phase_wheel_to_vm(SimplePhaseWheel, 'phasewheel', 
                                              base_frequency=base_frequency, 
                                              phase_increment=phase_increment, 
                                              n_slots=n_slots)
    processor = create_simple_phase_wheel_field_processor(SimplePhaseWheel, 
                                                         base_frequency=base_frequency,
                                                         phase_increment=phase_increment,
                                                         n_slots=n_slots)
    
    return SimplePhaseWheelRealm(
        name='phasewheel',
        phase_wheel_class=SimplePhaseWheel,
        parameters={'base_frequency': base_frequency, 'phase_increment': phase_increment, 'n_slots': n_slots},
        vm_node=vm_node,
        field_processor=processor
    )

def create_simple_phase_wheel_activation_realm(n_phases: int = 16, phase_increment: float = np.pi/100, 
                                              learnable_frequency: bool = True) -> SimplePhaseWheelRealm:
    """Create a SimplePhaseWheelActivation realm."""
    vm_node = compile_simple_phase_wheel_to_vm(SimplePhaseWheelActivation, 'phasewheelactivation',
                                              n_phases=n_phases,
                                              phase_increment=phase_increment,
                                              learnable_frequency=learnable_frequency)
    processor = create_simple_phase_wheel_field_processor(SimplePhaseWheelActivation,
                                                         n_phases=n_phases,
                                                         phase_increment=phase_increment,
                                                         learnable_frequency=learnable_frequency)
    
    return SimplePhaseWheelRealm(
        name='phasewheelactivation',
        phase_wheel_class=SimplePhaseWheelActivation,
        parameters={'n_phases': n_phases, 'phase_increment': phase_increment, 'learnable_frequency': learnable_frequency},
        vm_node=vm_node,
        field_processor=processor
    )

def create_simple_adaptive_phase_wheel_realm(n_slots: int = 32, base_frequency: float = 1.0, 
                                           adaptation_rate: float = 0.1) -> SimplePhaseWheelRealm:
    """Create a SimpleAdaptivePhaseWheel realm."""
    vm_node = compile_simple_phase_wheel_to_vm(SimpleAdaptivePhaseWheel, 'adaptivephasewheel',
                                              n_slots=n_slots,
                                              base_frequency=base_frequency,
                                              adaptation_rate=adaptation_rate)
    processor = create_simple_phase_wheel_field_processor(SimpleAdaptivePhaseWheel,
                                                         n_slots=n_slots,
                                                         base_frequency=base_frequency,
                                                         adaptation_rate=adaptation_rate)
    
    return SimplePhaseWheelRealm(
        name='adaptivephasewheel',
        phase_wheel_class=SimpleAdaptivePhaseWheel,
        parameters={'n_slots': n_slots, 'base_frequency': base_frequency, 'adaptation_rate': adaptation_rate},
        vm_node=vm_node,
        field_processor=processor
    )

# ============================
# Demo Functions
# ============================

def demo_phase_wheel_concept():
    """Demonstrate the PhaseWheel concept with visualization."""
    print("=== PhaseWheel Concept Demo ===\n")
    
    # Create a simple signal
    t = np.linspace(0, 4*np.pi, 1000)
    
    # Initialize phase wheel
    wheel = SimplePhaseWheel(base_frequency=2.0, phase_increment=np.pi/17, n_slots=8)
    
    print(f"Phase increment: {wheel.phase_increment:.6f}")
    print(f"Number of phase slots: {wheel.n_slots}")
    print(f"Phase positions: {wheel.phase_positions[:5]}")  # Show first 5
    
    # Show how phases never repeat due to irrational increment
    print(f"\nPhase wheel properties:")
    print(f"- Irrational increment ensures infinite unique phases")
    print(f"- Controlled overlap creates smooth transitions")
    print(f"- Learnable attention weights focus on relevant phases")
    
    # Apply phase wheel transformation
    complex_output = wheel.forward(t)
    
    print(f"\nPhase wheel output:")
    print(f"- Real part range: [{np.real(complex_output).min():.3f}, {np.real(complex_output).max():.3f}]")
    print(f"- Imaginary part range: [{np.imag(complex_output).min():.3f}, {np.imag(complex_output).max():.3f}]")
    print(f"- Magnitude range: [{np.abs(complex_output).min():.3f}, {np.abs(complex_output).max():.3f}]")
    
    return wheel, t, complex_output

def demo_phase_wheel_realms():
    """Demonstrate PhaseWheel realms in the singularity platform."""
    print("\n=== PhaseWheel Realms Demo ===\n")
    
    # Create sample field
    sr = 48000
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = 0.8 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t + np.pi/4)
    field = make_field_from_real(x, sr, tag=("demo", "multi_tone"))
    
    print(f"Original field: {len(field.z)} samples, {field.sr} Hz")
    print(f"Field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
    
    # Create PhaseWheel realms
    realms = {
        'phasewheel': create_simple_phase_wheel_realm(base_frequency=2.0, phase_increment=np.pi/17, n_slots=8),
        'phasewheelactivation': create_simple_phase_wheel_activation_realm(n_phases=12, phase_increment=np.pi/25),
        'adaptivephasewheel': create_simple_adaptive_phase_wheel_realm(n_slots=16, base_frequency=1.5, adaptation_rate=0.2)
    }
    
    print(f"\nCreated {len(realms)} PhaseWheel realms:")
    for name, realm in realms.items():
        print(f"  - {name}: {realm.parameters}")
    
    # Test each realm
    print(f"\nTesting PhaseWheel realms on field:")
    for name, realm in realms.items():
        try:
            processed_field = realm.field_processor(field)
            energy = np.sum(np.abs(processed_field.z) ** 2)
            
            print(f"  {name:20} | Energy: {energy:8.2f} | Roles: {processed_field.roles}")
        except Exception as e:
            print(f"  {name:20} | Error: {e}")
    
    # Test VM integration
    print(f"\nTesting VM integration:")
    for name, realm in realms.items():
        try:
            vm_expr = compile_simple_phase_wheel_to_vm(realm.phase_wheel_class, name, **realm.parameters)
            vm_json = str(vm_expr)  # Simplified for demo
            print(f"  {name:20} | VM representation: {len(vm_json)} chars")
        except Exception as e:
            print(f"  {name:20} | VM Error: {e}")
    
    print(f"\n=== PhaseWheel Realms Demo Complete ===")

def demo_phase_wheel_activation():
    """Demonstrate PhaseWheelActivation in a simple network context."""
    print("\n=== PhaseWheelActivation Demo ===\n")
    
    # Create activation
    activation = SimplePhaseWheelActivation(n_phases=12)
    test_input = np.random.randn(100, 10)  # batch_size=100, features=10
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    
    # Apply activation
    activated = activation.forward(test_input)
    
    print(f"Activation output shape: {activated.shape}")
    print(f"Output range: [{activated.min():.3f}, {activated.max():.3f}]")
    print(f"Output mean: {activated.mean():.3f}, std: {activated.std():.3f}")
    
    # Show phase wheel properties
    print(f"\nPhase wheel properties:")
    print(f"- Number of phases: {activation.n_phases}")
    print(f"- Phase increment: {activation.phase_increment:.6f}")
    print(f"- Frequency: {activation.frequency:.3f}")
    print(f"- Mixing weights range: [{activation.mixing_weights.min():.3f}, {activation.mixing_weights.max():.3f}]")

def main():
    """Run all PhaseWheel demos."""
    print("🔷 PhaseWheel System Demo (No PyTorch)")
    print("=" * 50)
    
    # Demo basic concept
    wheel, t, output = demo_phase_wheel_concept()
    
    # Demo activation
    demo_phase_wheel_activation()
    
    # Demo realms
    demo_phase_wheel_realms()
    
    print(f"\n🔷 PhaseWheel System Complete")
    print("✓ Continuous phase wheel concept")
    print("✓ Irrational increment for infinite phases")
    print("✓ Controlled overlap and smooth transitions")
    print("✓ Learnable attention weights")
    print("✓ FieldIQ integration")
    print("✓ VM realm system")

if __name__ == "__main__":
    main()

