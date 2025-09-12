# ============================
# PyTorch Activation Functions Realm
# ============================
"""
PyTorch Activation Functions as Realms in the Singularity Platform
Integrates PyTorch activation functions with the Combinatronix VM system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf, to_json, from_json
from Combinator_Kernel import FieldIQ, make_field_from_real

# Import the activation functions
import sys
import os
sys.path.append(r'c:\Users\The School\Desktop\Code\AI')
from activations import (
    SoftStep, LogRectifier, GatedTanh, Sinusoid, 
    GaussianBump, DampedSinusoid, BentIdentity, RationalLinear, QWSActivation,
    DampedSin, MirrorFold, MirrorFoldLeaky, MirrorFoldSoft, SelectiveReflection,
    MultiWell, SpinorActivate, QuadratureUnit, SingularityEdge
)

# ============================
# Activation Realm Data Structures
# ============================

@dataclass
class ActivationRealm:
    """A realm representing a PyTorch activation function in the singularity platform."""
    name: str
    activation_class: type
    parameters: Dict[str, Any]
    vm_node: Node
    field_processor: Callable[[FieldIQ], FieldIQ]
    
    def __repr__(self):
        return f"<ActivationRealm {self.name} params={self.parameters}>"

@dataclass
class ActivationField:
    """A field processed through activation functions with resonance properties."""
    field: FieldIQ
    activation_history: List[str]
    resonance_strength: float
    temporal_phase: float
    
    def __post_init__(self):
        if self.activation_history is None:
            self.activation_history = []
        if self.resonance_strength is None:
            self.resonance_strength = 1.0
        if self.temporal_phase is None:
            self.temporal_phase = 0.0

# ============================
# Activation Function Compilation to VM
# ============================

def compile_activation_to_vm(activation_class: type, name: str, **params) -> Node:
    """Compile a PyTorch activation function to VM representation."""
    return Val({
        'type': 'activation_realm',
        'name': name,
        'class': activation_class.__name__,
        'parameters': params,
        'vm_operations': _get_activation_vm_ops(name)
    })

def _get_activation_vm_ops(name: str) -> Dict[str, str]:
    """Get VM operation mappings for activation functions."""
    ops_map = {
        'softstep': 'SIGMOID_GATE',
        'logrectifier': 'LOG_RELU',
        'gatedtanh': 'GATED_TANH',
        'sinusoid': 'SINUSOIDAL',
        'gaussianbump': 'GAUSSIAN_BUMP',
        'dampedsinusoid': 'DAMPED_SINUSOID',
        'bentidentity': 'BENT_IDENTITY',
        'rationallinear': 'RATIONAL_LINEAR'
    }
    return {'primary': ops_map.get(name.lower(), 'UNKNOWN')}

# ============================
# Field Processing with Activations
# ============================

def create_activation_field_processor(activation_class: type, **params) -> Callable[[FieldIQ], FieldIQ]:
    """Create a field processor that applies PyTorch activation to FieldIQ data."""
    
    def process_field(field: FieldIQ) -> FieldIQ:
        # Convert FieldIQ to PyTorch tensor
        z_tensor = torch.tensor(field.z, dtype=torch.complex64)
        
        # Extract real and imaginary parts
        real_part = torch.real(z_tensor)
        imag_part = torch.imag(z_tensor)
        
        # Create activation instance
        activation = activation_class(**params)
        
        # Apply activation to both real and imaginary parts
        real_activated = activation(real_part)
        imag_activated = activation(imag_part)
        
        # Reconstruct complex tensor
        activated_z = torch.complex(real_activated, imag_activated)
        
        # Convert back to numpy and create new FieldIQ
        new_z = activated_z.detach().numpy()
        return FieldIQ(new_z, field.sr, field.roles)
    
    return process_field

# ============================
# Realm Factory Functions
# ============================

def create_softstep_realm(tau: float = 0.8, bias: float = 0.0) -> ActivationRealm:
    """Create a SoftStep activation realm."""
    vm_node = compile_activation_to_vm(SoftStep, 'softstep', tau=tau, bias=bias)
    processor = create_activation_field_processor(SoftStep, tau=tau, bias=bias)
    
    return ActivationRealm(
        name='softstep',
        activation_class=SoftStep,
        parameters={'tau': tau, 'bias': bias},
        vm_node=vm_node,
        field_processor=processor
    )

def create_logrectifier_realm(alpha: float = 3.0) -> ActivationRealm:
    """Create a LogRectifier activation realm."""
    vm_node = compile_activation_to_vm(LogRectifier, 'logrectifier', alpha=alpha)
    processor = create_activation_field_processor(LogRectifier, alpha=alpha)
    
    return ActivationRealm(
        name='logrectifier',
        activation_class=LogRectifier,
        parameters={'alpha': alpha},
        vm_node=vm_node,
        field_processor=processor
    )

def create_gatedtanh_realm(beta: float = 1.5) -> ActivationRealm:
    """Create a GatedTanh activation realm."""
    vm_node = compile_activation_to_vm(GatedTanh, 'gatedtanh', beta=beta)
    processor = create_activation_field_processor(GatedTanh, beta=beta)
    
    return ActivationRealm(
        name='gatedtanh',
        activation_class=GatedTanh,
        parameters={'beta': beta},
        vm_node=vm_node,
        field_processor=processor
    )

def create_sinusoid_realm(omega: float = 1.5, phi: float = 0.0) -> ActivationRealm:
    """Create a Sinusoid activation realm."""
    vm_node = compile_activation_to_vm(Sinusoid, 'sinusoid', omega=omega, phi=phi)
    processor = create_activation_field_processor(Sinusoid, omega=omega, phi=phi)
    
    return ActivationRealm(
        name='sinusoid',
        activation_class=Sinusoid,
        parameters={'omega': omega, 'phi': phi},
        vm_node=vm_node,
        field_processor=processor
    )

def create_gaussianbump_realm(mu: float = 0.0, sigma: float = 1.0) -> ActivationRealm:
    """Create a GaussianBump activation realm."""
    vm_node = compile_activation_to_vm(GaussianBump, 'gaussianbump', mu=mu, sigma=sigma)
    processor = create_activation_field_processor(GaussianBump, mu=mu, sigma=sigma)
    
    return ActivationRealm(
        name='gaussianbump',
        activation_class=GaussianBump,
        parameters={'mu': mu, 'sigma': sigma},
        vm_node=vm_node,
        field_processor=processor
    )

def create_dampedsinusoid_realm(omega: float = 2.0, gamma: float = 0.15) -> ActivationRealm:
    """Create a DampedSinusoid activation realm."""
    vm_node = compile_activation_to_vm(DampedSinusoid, 'dampedsinusoid', omega=omega, gamma=gamma)
    processor = create_activation_field_processor(DampedSinusoid, omega=omega, gamma=gamma)
    
    return ActivationRealm(
        name='dampedsinusoid',
        activation_class=DampedSinusoid,
        parameters={'omega': omega, 'gamma': gamma},
        vm_node=vm_node,
        field_processor=processor
    )

def create_bentidentity_realm(a: float = 0.7) -> ActivationRealm:
    """Create a BentIdentity activation realm."""
    vm_node = compile_activation_to_vm(BentIdentity, 'bentidentity', a=a)
    processor = create_activation_field_processor(BentIdentity, a=a)
    
    return ActivationRealm(
        name='bentidentity',
        activation_class=BentIdentity,
        parameters={'a': a},
        vm_node=vm_node,
        field_processor=processor
    )

def create_rationallinear_realm() -> ActivationRealm:
    """Create a RationalLinear activation realm."""
    vm_node = compile_activation_to_vm(RationalLinear, 'rationallinear')
    processor = create_activation_field_processor(RationalLinear)
    
    return ActivationRealm(
        name='rationallinear',
        activation_class=RationalLinear,
        parameters={},
        vm_node=vm_node,
        field_processor=processor
    )

def create_qwsactivation_realm(sigma0=0.6, a1=1.2, mu1=-1.2, sigma1=0.35, 
                              a2=1.6, mu2=-2.6, sigma2=0.45, p1=0.9, nu1=1.0, 
                              tau1=0.25, p2=0.9, nu2=2.2, tau2=0.25, d=0.15, 
                              omega=2.4, gamma=0.08) -> ActivationRealm:
    """Create a QWSActivation realm with complex multi-component activation."""
    vm_node = compile_activation_to_vm(QWSActivation, 'qwsactivation', 
                                      sigma0=sigma0, a1=a1, mu1=mu1, sigma1=sigma1,
                                      a2=a2, mu2=mu2, sigma2=sigma2, p1=p1, nu1=nu1,
                                      tau1=tau1, p2=p2, nu2=nu2, tau2=tau2, d=d,
                                      omega=omega, gamma=gamma)
    processor = create_activation_field_processor(QWSActivation, 
                                                sigma0=sigma0, a1=a1, mu1=mu1, sigma1=sigma1,
                                                a2=a2, mu2=mu2, sigma2=sigma2, p1=p1, nu1=nu1,
                                                tau1=tau1, p2=p2, nu2=nu2, tau2=tau2, d=d,
                                                omega=omega, gamma=gamma)
    
    return ActivationRealm(
        name='qwsactivation',
        activation_class=QWSActivation,
        parameters={
            'sigma0': sigma0, 'a1': a1, 'mu1': mu1, 'sigma1': sigma1,
            'a2': a2, 'mu2': mu2, 'sigma2': sigma2, 'p1': p1, 'nu1': nu1,
            'tau1': tau1, 'p2': p2, 'nu2': nu2, 'tau2': tau2, 'd': d,
            'omega': omega, 'gamma': gamma
        },
        vm_node=vm_node,
        field_processor=processor
    )

# ============================
# Advanced Signal Processing Realms
# ============================

def create_dampedsin_realm(alpha=0.25, beta=2.0) -> ActivationRealm:
    """Create a DampedSin activation realm."""
    vm_node = compile_activation_to_vm(DampedSin, 'dampedsin', alpha=alpha, beta=beta)
    processor = create_activation_field_processor(DampedSin, alpha=alpha, beta=beta)
    
    return ActivationRealm(
        name='dampedsin',
        activation_class=DampedSin,
        parameters={'alpha': alpha, 'beta': beta},
        vm_node=vm_node,
        field_processor=processor
    )

def create_mirrorfold_realm() -> ActivationRealm:
    """Create a MirrorFold activation realm."""
    vm_node = compile_activation_to_vm(MirrorFold, 'mirrorfold')
    processor = create_activation_field_processor(MirrorFold)
    
    return ActivationRealm(
        name='mirrorfold',
        activation_class=MirrorFold,
        parameters={},
        vm_node=vm_node,
        field_processor=processor
    )

def create_mirrorfoldleaky_realm(leak=0.01) -> ActivationRealm:
    """Create a MirrorFoldLeaky activation realm."""
    vm_node = compile_activation_to_vm(MirrorFoldLeaky, 'mirrorfoldleaky', leak=leak)
    processor = create_activation_field_processor(MirrorFoldLeaky, leak=leak)
    
    return ActivationRealm(
        name='mirrorfoldleaky',
        activation_class=MirrorFoldLeaky,
        parameters={'leak': leak},
        vm_node=vm_node,
        field_processor=processor
    )

def create_mirrorfoldsoft_realm(eps=1e-3) -> ActivationRealm:
    """Create a MirrorFoldSoft activation realm."""
    vm_node = compile_activation_to_vm(MirrorFoldSoft, 'mirrorfoldsoft', eps=eps)
    processor = create_activation_field_processor(MirrorFoldSoft, eps=eps)
    
    return ActivationRealm(
        name='mirrorfoldsoft',
        activation_class=MirrorFoldSoft,
        parameters={'eps': eps},
        vm_node=vm_node,
        field_processor=processor
    )

def create_selectivereflection_realm(w=2.0, k=3.0) -> ActivationRealm:
    """Create a SelectiveReflection activation realm."""
    vm_node = compile_activation_to_vm(SelectiveReflection, 'selectivereflection', w=w, k=k)
    processor = create_activation_field_processor(SelectiveReflection, w=w, k=k)
    
    return ActivationRealm(
        name='selectivereflection',
        activation_class=SelectiveReflection,
        parameters={'w': w, 'k': k},
        vm_node=vm_node,
        field_processor=processor
    )

def create_multiwell_realm(gamma=0.5, omega=2.0) -> ActivationRealm:
    """Create a MultiWell activation realm."""
    vm_node = compile_activation_to_vm(MultiWell, 'multiwell', gamma=gamma, omega=omega)
    processor = create_activation_field_processor(MultiWell, gamma=gamma, omega=omega)
    
    return ActivationRealm(
        name='multiwell',
        activation_class=MultiWell,
        parameters={'gamma': gamma, 'omega': omega},
        vm_node=vm_node,
        field_processor=processor
    )

def create_spinoractivate_realm(phi=0.0) -> ActivationRealm:
    """Create a SpinorActivate activation realm."""
    vm_node = compile_activation_to_vm(SpinorActivate, 'spinoractivate', phi=phi)
    processor = create_activation_field_processor(SpinorActivate, phi=phi)
    
    return ActivationRealm(
        name='spinoractivate',
        activation_class=SpinorActivate,
        parameters={'phi': phi},
        vm_node=vm_node,
        field_processor=processor
    )

def create_quadratureunit_realm(psi=0.0, return_tuple=True) -> ActivationRealm:
    """Create a QuadratureUnit activation realm."""
    vm_node = compile_activation_to_vm(QuadratureUnit, 'quadratureunit', psi=psi, return_tuple=return_tuple)
    processor = create_activation_field_processor(QuadratureUnit, psi=psi, return_tuple=return_tuple)
    
    return ActivationRealm(
        name='quadratureunit',
        activation_class=QuadratureUnit,
        parameters={'psi': psi, 'return_tuple': return_tuple},
        vm_node=vm_node,
        field_processor=processor
    )

def create_singularityedge_realm(tau=0.5, k=10.0) -> ActivationRealm:
    """Create a SingularityEdge activation realm."""
    vm_node = compile_activation_to_vm(SingularityEdge, 'singularityedge', tau=tau, k=k)
    processor = create_activation_field_processor(SingularityEdge, tau=tau, k=k)
    
    return ActivationRealm(
        name='singularityedge',
        activation_class=SingularityEdge,
        parameters={'tau': tau, 'k': k},
        vm_node=vm_node,
        field_processor=processor
    )

# ============================
# Realm Combinators
# ============================

def compose_realms(realm1: ActivationRealm, realm2: ActivationRealm) -> ActivationRealm:
    """Compose two activation realms into a single realm."""
    def composed_processor(field: FieldIQ) -> FieldIQ:
        return realm2.field_processor(realm1.field_processor(field))
    
    vm_node = app(realm1.vm_node, realm2.vm_node)
    
    return ActivationRealm(
        name=f"{realm1.name}_composed_{realm2.name}",
        activation_class=type('ComposedActivation', (), {}),
        parameters={**realm1.parameters, **realm2.parameters},
        vm_node=vm_node,
        field_processor=composed_processor
    )

def create_activation_pipeline(realms: List[ActivationRealm]) -> ActivationRealm:
    """Create a pipeline of activation realms."""
    if not realms:
        raise ValueError("At least one realm required for pipeline")
    
    # Start with first realm
    pipeline = realms[0]
    
    # Compose with remaining realms
    for realm in realms[1:]:
        pipeline = compose_realms(pipeline, realm)
    
    return pipeline

# ============================
# Resonance Integration
# ============================

def create_resonance_activation_field(field: FieldIQ, realm: ActivationRealm, 
                                    resonance_strength: float = 1.0) -> ActivationField:
    """Create an activation field with resonance properties."""
    processed_field = realm.field_processor(field)
    
    return ActivationField(
        field=processed_field,
        activation_history=[realm.name],
        resonance_strength=resonance_strength,
        temporal_phase=0.0
    )

def calculate_activation_resonance(field: FieldIQ, realm: ActivationRealm) -> float:
    """Calculate resonance strength between field and activation realm."""
    # Simple resonance calculation based on field energy and activation characteristics
    field_energy = np.sum(np.abs(field.z) ** 2)
    
    # Different activations have different resonance characteristics
    if realm.name == 'sinusoid':
        # Sinusoid resonates with periodic content
        fft = np.fft.fft(field.z)
        spectral_energy = np.sum(np.abs(fft) ** 2)
        return min(1.0, spectral_energy / (field_energy + 1e-8))
    
    elif realm.name == 'gaussianbump':
        # Gaussian bump resonates with localized content
        local_energy = np.sum(np.abs(field.z) ** 2) / len(field.z)
        return min(1.0, local_energy)
    
    else:
        # Default resonance based on field energy
        return min(1.0, field_energy / 1000.0)

# ============================
# VM Integration
# ============================

def compile_realm_to_vm_expression(realm: ActivationRealm) -> Node:
    """Compile an activation realm to a VM expression."""
    return realm.vm_node

def execute_realm_vm_expression(vm_expr: Node, field: FieldIQ) -> FieldIQ:
    """Execute a VM expression representing an activation realm on a field."""
    # Reduce the VM expression
    reduced = reduce_whnf(vm_expr)
    
    # Extract realm information
    if isinstance(reduced, Val) and isinstance(reduced.v, dict):
        realm_data = reduced.v
        if realm_data.get('type') == 'activation_realm':
            # Find the corresponding realm and apply it
            realm_name = realm_data['name']
            # This would need to be connected to a realm registry in practice
            return field  # Placeholder
    
    return field

# ============================
# Demo and Testing
# ============================

def demo_activation_realms():
    """Demonstrate activation realms in the singularity platform."""
    print("=== PyTorch Activation Realms Demo ===\n")
    
    # Create sample field
    sr = 48000
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = 0.8 * np.cos(2 * np.pi * 440 * t)  # 440 Hz tone
    field = make_field_from_real(x, sr, tag=("demo", "tone"))
    
    print(f"Original field: {len(field.z)} samples, {field.sr} Hz")
    print(f"Field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
    
    # Create various activation realms
    realms = {
        'softstep': create_softstep_realm(tau=0.5, bias=0.1),
        'logrectifier': create_logrectifier_realm(alpha=2.0),
        'gatedtanh': create_gatedtanh_realm(beta=1.2),
        'sinusoid': create_sinusoid_realm(omega=2.0, phi=np.pi/4),
        'gaussianbump': create_gaussianbump_realm(mu=0.0, sigma=0.5),
        'dampedsinusoid': create_dampedsinusoid_realm(omega=3.0, gamma=0.2),
        'bentidentity': create_bentidentity_realm(a=0.8),
        'rationallinear': create_rationallinear_realm()
    }
    
    print(f"\nCreated {len(realms)} activation realms:")
    for name, realm in realms.items():
        print(f"  - {name}: {realm.parameters}")
    
    # Test each realm
    print(f"\nTesting realms on field:")
    for name, realm in realms.items():
        processed_field = realm.field_processor(field)
        resonance = calculate_activation_resonance(field, realm)
        
        print(f"  {name:15} | Energy: {np.sum(np.abs(processed_field.z) ** 2):8.2f} | Resonance: {resonance:.3f}")
    
    # Test realm composition
    print(f"\nTesting realm composition:")
    composed = compose_realms(realms['sinusoid'], realms['gaussianbump'])
    processed_composed = composed.field_processor(field)
    print(f"  Sinusoid + GaussianBump | Energy: {np.sum(np.abs(processed_composed.z) ** 2):8.2f}")
    
    # Test pipeline
    print(f"\nTesting activation pipeline:")
    pipeline_realms = [realms['softstep'], realms['logrectifier'], realms['gatedtanh']]
    pipeline = create_activation_pipeline(pipeline_realms)
    processed_pipeline = pipeline.field_processor(field)
    print(f"  Pipeline (3 realms) | Energy: {np.sum(np.abs(processed_pipeline.z) ** 2):8.2f}")
    
    # Test VM integration
    print(f"\nTesting VM integration:")
    for name, realm in list(realms.items())[:3]:  # Test first 3
        vm_expr = compile_realm_to_vm_expression(realm)
        vm_json = to_json(vm_expr)
        print(f"  {name:15} | VM JSON length: {len(vm_json)} chars")
    
    print(f"\n=== Activation Realms Demo Complete ===")

if __name__ == "__main__":
    demo_activation_realms()
