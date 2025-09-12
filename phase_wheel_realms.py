# ============================
# PhaseWheel Realms for FieldIQ Engine
# ============================
"""
PhaseWheel-based activation realms for enhanced FieldIQ signal processing.
Integrates continuous phase wheel concepts with the singularity platform.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Tuple
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf, to_json, from_json
from Combinator_Kernel import FieldIQ, make_field_from_real

# ============================
# Enhanced PhaseWheel Classes
# ============================

class PhaseWheel(nn.Module):
    """
    Continuous phase wheel with irrational increment creating infinite, 
    overlapping phase states with controlled signal bleed.
    """
    def __init__(self, base_frequency=1.0, phase_increment=np.pi/100, n_slots=64):
        super().__init__()
        self.base_frequency = nn.Parameter(torch.tensor(base_frequency))
        self.phase_increment = phase_increment  # irrational increment
        self.n_slots = n_slots
        
        # Initialize phase positions around the wheel
        # Each slot is offset by the irrational increment
        phase_positions = torch.tensor([i * phase_increment for i in range(n_slots)])
        self.register_buffer('phase_positions', phase_positions)
        
        # Learnable weights for each phase slot
        self.phase_weights = nn.Parameter(torch.randn(n_slots))
        
    def get_basis_functions(self, x):
        """Generate overlapping cosine/sine pairs for each phase position"""
        # x shape: (..., seq_len) or (..., features)
        
        cos_functions = []
        sin_functions = []
        
        for i, phase in enumerate(self.phase_positions):
            cos_func = torch.cos(self.base_frequency * x + phase)
            sin_func = torch.sin(self.base_frequency * x + phase)
            cos_functions.append(cos_func)
            sin_functions.append(sin_func)
            
        # Stack: (..., n_slots, seq_len/features)
        cos_basis = torch.stack(cos_functions, dim=-2)
        sin_basis = torch.stack(sin_functions, dim=-2)
        
        return cos_basis, sin_basis
    
    def forward(self, x):
        """
        Apply phase wheel transformation with controlled overlap
        """
        cos_basis, sin_basis = self.get_basis_functions(x)
        
        # Soft attention over phase positions
        attention_weights = torch.softmax(self.phase_weights, dim=0)
        
        # Weighted combination with controlled bleed
        cos_output = torch.sum(cos_basis * attention_weights.view(1, -1, 1), dim=-2)
        sin_output = torch.sum(sin_basis * attention_weights.view(1, -1, 1), dim=-2)
        
        # Return complex-valued output or magnitude
        return cos_output + 1j * sin_output
    
    def get_phase_response(self, x):
        """Get the learned phase response across the wheel"""
        cos_basis, sin_basis = self.get_basis_functions(x)
        attention_weights = torch.softmax(self.phase_weights, dim=0)
        
        # Phase response for each slot
        phases = torch.atan2(sin_basis, cos_basis)
        weighted_phase = torch.sum(phases * attention_weights.view(1, -1, 1), dim=-2)
        
        return weighted_phase

class PhaseWheelActivation(nn.Module):
    """
    Neural network activation using the phase wheel concept
    """
    def __init__(self, n_phases=16, phase_increment=np.pi/100, learnable_frequency=True):
        super().__init__()
        self.n_phases = n_phases
        self.phase_increment = phase_increment
        
        if learnable_frequency:
            self.frequency = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('frequency', torch.tensor(1.0))
            
        # Phase positions on the wheel
        phases = torch.tensor([i * phase_increment for i in range(n_phases)])
        self.register_buffer('phases', phases)
        
        # Learnable mixing weights
        self.mixing_weights = nn.Parameter(torch.randn(n_phases))
        
    def forward(self, x):
        """
        Apply phase wheel activation with smooth interpolation
        """
        # Generate all phase-shifted sinusoids
        activations = []
        for phase in self.phases:
            activation = torch.sin(self.frequency * x + phase)
            activations.append(activation)
        
        # Stack and apply soft mixing
        activation_stack = torch.stack(activations, dim=-1)  # (..., n_phases)
        mix_weights = torch.softmax(self.mixing_weights, dim=0)
        
        # Smooth combination across phase wheel
        output = torch.sum(activation_stack * mix_weights, dim=-1)
        
        return output

class AdaptivePhaseWheel(nn.Module):
    """
    Phase wheel that adapts its frequency and phase distribution based on input
    """
    def __init__(self, n_slots=32, base_frequency=1.0, adaptation_rate=0.1):
        super().__init__()
        self.n_slots = n_slots
        self.base_frequency = nn.Parameter(torch.tensor(base_frequency))
        self.adaptation_rate = adaptation_rate
        
        # Learnable phase positions (can adapt)
        self.phase_positions = nn.Parameter(torch.randn(n_slots))
        
        # Learnable weights
        self.phase_weights = nn.Parameter(torch.randn(n_slots))
        
        # Adaptation parameters
        self.frequency_adaptation = nn.Parameter(torch.tensor(1.0))
        self.phase_adaptation = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        """
        Apply adaptive phase wheel transformation
        """
        # Adapt frequency based on input characteristics
        input_energy = torch.mean(x**2)
        adapted_frequency = self.base_frequency * (1 + self.frequency_adaptation * input_energy)
        
        # Adapt phase positions
        phase_shift = self.phase_adaptation * torch.mean(x)
        adapted_phases = self.phase_positions + phase_shift
        
        # Generate basis functions with adapted parameters
        cos_functions = []
        sin_functions = []
        
        for phase in adapted_phases:
            cos_func = torch.cos(adapted_frequency * x + phase)
            sin_func = torch.sin(adapted_frequency * x + phase)
            cos_functions.append(cos_func)
            sin_functions.append(sin_func)
        
        # Stack and apply attention
        cos_basis = torch.stack(cos_functions, dim=-2)
        sin_basis = torch.stack(sin_functions, dim=-2)
        
        attention_weights = torch.softmax(self.phase_weights, dim=0)
        
        cos_output = torch.sum(cos_basis * attention_weights.view(1, -1, 1), dim=-2)
        sin_output = torch.sum(sin_basis * attention_weights.view(1, -1, 1), dim=-2)
        
        return cos_output + 1j * sin_output

# ============================
# PhaseWheel Realm System
# ============================

@dataclass
class PhaseWheelRealm:
    """A realm representing a PhaseWheel-based activation in the singularity platform."""
    name: str
    phase_wheel_class: type
    parameters: Dict[str, Any]
    vm_node: Node
    field_processor: Callable[[FieldIQ], FieldIQ]
    
    def __repr__(self):
        return f"<PhaseWheelRealm {self.name} params={self.parameters}>"

def create_phase_wheel_field_processor(phase_wheel_class: type, **params) -> Callable[[FieldIQ], FieldIQ]:
    """Create a field processor that applies PhaseWheel to FieldIQ data."""
    
    def process_field(field: FieldIQ) -> FieldIQ:
        # Convert FieldIQ to PyTorch tensor
        z_tensor = torch.tensor(field.z, dtype=torch.complex64)
        
        # Extract real and imaginary parts
        real_part = torch.real(z_tensor)
        imag_part = torch.imag(z_tensor)
        
        # Create phase wheel instance
        phase_wheel = phase_wheel_class(**params)
        
        # Apply phase wheel to both real and imaginary parts
        real_processed = phase_wheel(real_part)
        imag_processed = phase_wheel(imag_part)
        
        # Reconstruct complex tensor
        if torch.is_complex(real_processed):
            # Phase wheel returns complex output
            new_z = real_processed
        else:
            # Phase wheel returns real output, reconstruct complex
            new_z = torch.complex(real_processed, imag_processed)
        
        # Convert back to numpy and create new FieldIQ
        new_z_np = new_z.detach().numpy()
        return FieldIQ(new_z_np, field.sr, field.roles)
    
    return process_field

def compile_phase_wheel_to_vm(phase_wheel_class: type, name: str, **params) -> Node:
    """Compile a PhaseWheel to VM representation."""
    return Val({
        'type': 'phase_wheel_realm',
        'name': name,
        'class': phase_wheel_class.__name__,
        'parameters': params,
        'vm_operations': _get_phase_wheel_vm_ops(name)
    })

def _get_phase_wheel_vm_ops(name: str) -> Dict[str, str]:
    """Get VM operation mappings for phase wheel functions."""
    ops_map = {
        'phasewheel': 'PHASE_WHEEL',
        'phasewheelactivation': 'PHASE_WHEEL_ACTIVATION',
        'adaptivephasewheel': 'ADAPTIVE_PHASE_WHEEL'
    }
    return {'primary': ops_map.get(name.lower(), 'UNKNOWN')}

# ============================
# PhaseWheel Realm Factory Functions
# ============================

def create_phase_wheel_realm(base_frequency: float = 1.0, phase_increment: float = np.pi/100, 
                            n_slots: int = 64) -> PhaseWheelRealm:
    """Create a PhaseWheel activation realm."""
    vm_node = compile_phase_wheel_to_vm(PhaseWheel, 'phasewheel', 
                                       base_frequency=base_frequency, 
                                       phase_increment=phase_increment, 
                                       n_slots=n_slots)
    processor = create_phase_wheel_field_processor(PhaseWheel, 
                                                  base_frequency=base_frequency,
                                                  phase_increment=phase_increment,
                                                  n_slots=n_slots)
    
    return PhaseWheelRealm(
        name='phasewheel',
        phase_wheel_class=PhaseWheel,
        parameters={'base_frequency': base_frequency, 'phase_increment': phase_increment, 'n_slots': n_slots},
        vm_node=vm_node,
        field_processor=processor
    )

def create_phase_wheel_activation_realm(n_phases: int = 16, phase_increment: float = np.pi/100, 
                                       learnable_frequency: bool = True) -> PhaseWheelRealm:
    """Create a PhaseWheelActivation realm."""
    vm_node = compile_phase_wheel_to_vm(PhaseWheelActivation, 'phasewheelactivation',
                                       n_phases=n_phases,
                                       phase_increment=phase_increment,
                                       learnable_frequency=learnable_frequency)
    processor = create_phase_wheel_field_processor(PhaseWheelActivation,
                                                  n_phases=n_phases,
                                                  phase_increment=phase_increment,
                                                  learnable_frequency=learnable_frequency)
    
    return PhaseWheelRealm(
        name='phasewheelactivation',
        phase_wheel_class=PhaseWheelActivation,
        parameters={'n_phases': n_phases, 'phase_increment': phase_increment, 'learnable_frequency': learnable_frequency},
        vm_node=vm_node,
        field_processor=processor
    )

def create_adaptive_phase_wheel_realm(n_slots: int = 32, base_frequency: float = 1.0, 
                                     adaptation_rate: float = 0.1) -> PhaseWheelRealm:
    """Create an AdaptivePhaseWheel realm."""
    vm_node = compile_phase_wheel_to_vm(AdaptivePhaseWheel, 'adaptivephasewheel',
                                       n_slots=n_slots,
                                       base_frequency=base_frequency,
                                       adaptation_rate=adaptation_rate)
    processor = create_phase_wheel_field_processor(AdaptivePhaseWheel,
                                                  n_slots=n_slots,
                                                  base_frequency=base_frequency,
                                                  adaptation_rate=adaptation_rate)
    
    return PhaseWheelRealm(
        name='adaptivephasewheel',
        phase_wheel_class=AdaptivePhaseWheel,
        parameters={'n_slots': n_slots, 'base_frequency': base_frequency, 'adaptation_rate': adaptation_rate},
        vm_node=vm_node,
        field_processor=processor
    )

# ============================
# Enhanced FieldIQ Processing
# ============================

def create_phase_wheel_pipeline(realms: List[PhaseWheelRealm]) -> PhaseWheelRealm:
    """Create a pipeline of PhaseWheel realms."""
    if not realms:
        raise ValueError("At least one realm required for pipeline")
    
    # Start with first realm
    pipeline = realms[0]
    
    # Compose with remaining realms
    for realm in realms[1:]:
        pipeline = compose_phase_wheel_realms(pipeline, realm)
    
    return pipeline

def compose_phase_wheel_realms(realm1: PhaseWheelRealm, realm2: PhaseWheelRealm) -> PhaseWheelRealm:
    """Compose two PhaseWheel realms into a single realm."""
    def composed_processor(field: FieldIQ) -> FieldIQ:
        return realm2.field_processor(realm1.field_processor(field))
    
    vm_node = app(realm1.vm_node, realm2.vm_node)
    
    return PhaseWheelRealm(
        name=f"{realm1.name}_composed_{realm2.name}",
        phase_wheel_class=type('ComposedPhaseWheel', (), {}),
        parameters={**realm1.parameters, **realm2.parameters},
        vm_node=vm_node,
        field_processor=composed_processor
    )

def calculate_phase_wheel_resonance(field: FieldIQ, realm: PhaseWheelRealm) -> float:
    """Calculate resonance between field and PhaseWheel realm."""
    # Phase wheels resonate with periodic and phase-rich content
    field_energy = np.sum(np.abs(field.z) ** 2)
    
    # Calculate spectral characteristics
    fft = np.fft.fft(field.z)
    spectral_energy = np.sum(np.abs(fft) ** 2)
    spectral_centroid = np.sum(np.abs(fft) * np.arange(len(fft))) / np.sum(np.abs(fft))
    
    # Phase wheels resonate with:
    # 1. High spectral energy (rich frequency content)
    # 2. Complex phase relationships
    # 3. Periodic patterns
    
    spectral_resonance = min(1.0, spectral_energy / (field_energy + 1e-8))
    phase_complexity = np.std(np.angle(field.z))
    phase_resonance = min(1.0, phase_complexity * 2.0)
    
    # Combine resonances
    total_resonance = (spectral_resonance + phase_resonance) / 2.0
    
    return min(1.0, total_resonance)

# ============================
# Demo and Testing
# ============================

def demo_phase_wheel_realms():
    """Demonstrate PhaseWheel realms in the singularity platform."""
    print("=== PhaseWheel Realms Demo ===\n")
    
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
        'phasewheel': create_phase_wheel_realm(base_frequency=2.0, phase_increment=np.pi/17, n_slots=8),
        'phasewheelactivation': create_phase_wheel_activation_realm(n_phases=12, phase_increment=np.pi/25),
        'adaptivephasewheel': create_adaptive_phase_wheel_realm(n_slots=16, base_frequency=1.5, adaptation_rate=0.2)
    }
    
    print(f"\nCreated {len(realms)} PhaseWheel realms:")
    for name, realm in realms.items():
        print(f"  - {name}: {realm.parameters}")
    
    # Test each realm
    print(f"\nTesting PhaseWheel realms on field:")
    for name, realm in realms.items():
        try:
            processed_field = realm.field_processor(field)
            resonance = calculate_phase_wheel_resonance(field, realm)
            energy = np.sum(np.abs(processed_field.z) ** 2)
            
            print(f"  {name:20} | Energy: {energy:8.2f} | Resonance: {resonance:.3f}")
        except Exception as e:
            print(f"  {name:20} | Error: {e}")
    
    # Test realm composition
    print(f"\nTesting PhaseWheel realm composition:")
    try:
        composed = compose_phase_wheel_realms(realms['phasewheel'], realms['phasewheelactivation'])
        processed_composed = composed.field_processor(field)
        energy = np.sum(np.abs(processed_composed.z) ** 2)
        print(f"  PhaseWheel + Activation | Energy: {energy:8.2f}")
    except Exception as e:
        print(f"  PhaseWheel + Activation | Error: {e}")
    
    # Test pipeline
    print(f"\nTesting PhaseWheel pipeline:")
    try:
        pipeline_realms = [realms['phasewheel'], realms['adaptivephasewheel']]
        pipeline = create_phase_wheel_pipeline(pipeline_realms)
        processed_pipeline = pipeline.field_processor(field)
        energy = np.sum(np.abs(processed_pipeline.z) ** 2)
        print(f"  Pipeline (2 realms) | Energy: {energy:8.2f}")
    except Exception as e:
        print(f"  Pipeline (2 realms) | Error: {e}")
    
    # Test VM integration
    print(f"\nTesting VM integration:")
    for name, realm in list(realms.items())[:2]:  # Test first 2
        try:
            vm_expr = compile_phase_wheel_to_vm(realm.phase_wheel_class, name, **realm.parameters)
            vm_json = to_json(vm_expr)
            print(f"  {name:20} | VM JSON length: {len(vm_json)} chars")
        except Exception as e:
            print(f"  {name:20} | VM Error: {e}")
    
    print(f"\n=== PhaseWheel Realms Demo Complete ===")

if __name__ == "__main__":
    demo_phase_wheel_realms()

