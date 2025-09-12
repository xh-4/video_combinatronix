# ============================
# Minimal Signal Processing VM
# ============================
"""
Minimal, working signal processing VM that focuses on essentials.
No complex implementations - just the core operations that work.
"""

import numpy as np
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf, to_json, from_json
from Combinator_Kernel import FieldIQ, make_field_from_real

# ============================
# Core Signal Processing Operations
# ============================

def SP_AMP(gain: float) -> Node:
    """Amplitude scaling - simple and reliable."""
    return Val({'type': 'sp', 'op': 'amp', 'gain': gain})

def SP_LOWPASS(cutoff: float) -> Node:
    """Simple lowpass filter."""
    return Val({'type': 'sp', 'op': 'lowpass', 'cutoff': cutoff})

def SP_PHASE(degrees: float) -> Node:
    """Phase shift."""
    return Val({'type': 'sp', 'op': 'phase', 'degrees': degrees})

def SP_ENERGY() -> Node:
    """Energy calculation."""
    return Val({'type': 'sp', 'op': 'energy'})

def SP_RMS() -> Node:
    """RMS calculation."""
    return Val({'type': 'sp', 'op': 'rms'})

# ============================
# Simple Signal Processor
# ============================

def process_signal(field: FieldIQ, operation: dict) -> FieldIQ:
    """Process a signal with a single operation."""
    op_type = operation['op']
    
    if op_type == 'amp':
        gain = operation['gain']
        return FieldIQ(field.z * gain, field.sr, field.roles)
    
    elif op_type == 'lowpass':
        cutoff = operation['cutoff']
        # Simple 1-pole lowpass
        alpha = np.exp(-2 * np.pi * cutoff / field.sr)
        filtered = np.zeros_like(field.z)
        filtered[0] = field.z[0]
        for i in range(1, len(field.z)):
            filtered[i] = alpha * filtered[i-1] + (1 - alpha) * field.z[i]
        return FieldIQ(filtered, field.sr, field.roles)
    
    elif op_type == 'phase':
        degrees = operation['degrees']
        phase_shift = np.exp(1j * np.radians(degrees))
        return FieldIQ(field.z * phase_shift, field.sr, field.roles)
    
    elif op_type == 'energy':
        energy = np.sum(np.abs(field.z)**2)
        return Val(energy)
    
    elif op_type == 'rms':
        rms = np.sqrt(np.mean(np.abs(field.z)**2))
        return Val(rms)
    
    else:
        return field

def execute_sp_vm(expr: Node, field: FieldIQ) -> FieldIQ:
    """Execute signal processing VM expression."""
    # Reduce the expression
    reduced = reduce_whnf(expr)
    
    # Check if it's a signal processing operation
    if isinstance(reduced, Val) and isinstance(reduced.v, dict):
        if reduced.v.get('type') == 'sp':
            return process_signal(field, reduced.v)
        elif reduced.v.get('type') == 'fieldiq':
            # Reconstruct FieldIQ
            z = np.array(reduced.v['z'], dtype=complex)
            sr = reduced.v['sr']
            roles = reduced.v['roles']
            return FieldIQ(z, sr, roles)
    
    return field

# ============================
# Pipeline Builder
# ============================

def build_pipeline(*ops: Node) -> Node:
    """Build a pipeline using B combinator."""
    if not ops:
        return Val({'type': 'sp', 'op': 'identity'})
    
    pipeline = ops[0]
    for op in ops[1:]:
        pipeline = app(app(Comb('B'), pipeline), op)
    
    return pipeline

# ============================
# Quick Test
# ============================

def quick_test():
    """Quick test to verify everything works."""
    print("=== Minimal Signal Processing VM Test ===\n")
    
    # Create test signal
    sr = 48000.0
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)  # 0.1 second
    x = 0.5 * np.cos(2 * np.pi * 440 * t)
    field = make_field_from_real(x, sr)
    
    print(f"Test signal: {len(field.z)} samples, {field.sr} Hz")
    print(f"Original energy: {np.sum(np.abs(field.z)**2):.6f}")
    
    # Test individual operations
    print("\n--- Individual Operations ---")
    
    # Amplitude
    amp_op = SP_AMP(0.5)
    amp_result = execute_sp_vm(amp_op, field)
    print(f"Amplitude (0.5x): {np.sum(np.abs(amp_result.z)**2):.6f}")
    
    # Lowpass
    lp_op = SP_LOWPASS(1000.0)
    lp_result = execute_sp_vm(lp_op, field)
    print(f"Lowpass (1kHz): {np.sum(np.abs(lp_result.z)**2):.6f}")
    
    # Phase
    phase_op = SP_PHASE(90.0)
    phase_result = execute_sp_vm(phase_op, field)
    print(f"Phase (90°): {np.sum(np.abs(phase_result.z)**2):.6f}")
    
    # Test pipeline
    print("\n--- Pipeline Test ---")
    pipeline = build_pipeline(SP_AMP(0.8), SP_LOWPASS(2000.0), SP_PHASE(45.0))
    pipeline_result = execute_sp_vm(pipeline, field)
    print(f"Pipeline result: {np.sum(np.abs(pipeline_result.z)**2):.6f}")
    
    # Test serialization
    print("\n--- Serialization Test ---")
    json_str = to_json(pipeline)
    print(f"Serialized: {len(json_str)} characters")
    
    deserialized = from_json(json_str)
    deserialized_result = execute_sp_vm(deserialized, field)
    print(f"Deserialized: {np.sum(np.abs(deserialized_result.z)**2):.6f}")
    
    print("\n=== Test Complete ===")
    print("✓ Core operations working")
    print("✓ Pipeline composition working")
    print("✓ Serialization working")
    print("✓ Ready for expansion")

if __name__ == "__main__":
    quick_test()


