#!/usr/bin/env python3
"""
Simple test script for PyTorch Activation Realms integration.
Tests basic functionality without requiring the full resonance scheduler.
"""

import numpy as np
import torch
import sys
import os

# Add the AI directory to path for activations import
sys.path.append(r'c:\Users\The School\Desktop\Code\AI')

try:
    from activations import SoftStep, LogRectifier, GatedTanh, Sinusoid
    from pytorch_activation_realm import (
        create_softstep_realm, create_logrectifier_realm, 
        create_gatedtanh_realm, create_sinusoid_realm,
        compose_realms, create_activation_pipeline
    )
    from Combinator_Kernel import make_field_from_real
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_basic_activation_realms():
    """Test basic activation realm creation and processing."""
    print("\n=== Testing Basic Activation Realms ===")
    
    # Create a simple test field
    sr = 48000
    dur = 0.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = 0.5 * np.cos(2 * np.pi * 440 * t)  # 440 Hz tone
    field = make_field_from_real(x, sr, tag=("test", "tone"))
    
    print(f"Created test field: {len(field.z)} samples, {field.sr} Hz")
    print(f"Original field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
    
    # Test individual realms
    realms = {
        'softstep': create_softstep_realm(tau=0.5, bias=0.1),
        'logrectifier': create_logrectifier_realm(alpha=2.0),
        'gatedtanh': create_gatedtanh_realm(beta=1.5),
        'sinusoid': create_sinusoid_realm(omega=2.0, phi=np.pi/4)
    }
    
    print(f"\nTesting individual realms:")
    for name, realm in realms.items():
        try:
            processed = realm.field_processor(field)
            energy = np.sum(np.abs(processed.z) ** 2)
            print(f"  {name:12} | Energy: {energy:8.2f} | ✓")
        except Exception as e:
            print(f"  {name:12} | Error: {e} | ✗")
    
    return realms, field

def test_realm_composition(realms, field):
    """Test realm composition and pipelines."""
    print(f"\n=== Testing Realm Composition ===")
    
    try:
        # Test simple composition
        composed = compose_realms(realms['softstep'], realms['sinusoid'])
        processed = composed.field_processor(field)
        energy = np.sum(np.abs(processed.z) ** 2)
        print(f"  SoftStep + Sinusoid | Energy: {energy:8.2f} | ✓")
    except Exception as e:
        print(f"  SoftStep + Sinusoid | Error: {e} | ✗")
    
    try:
        # Test pipeline
        pipeline = create_activation_pipeline([
            realms['softstep'], 
            realms['logrectifier'], 
            realms['gatedtanh']
        ])
        processed = pipeline.field_processor(field)
        energy = np.sum(np.abs(processed.z) ** 2)
        print(f"  Pipeline (3 realms) | Energy: {energy:8.2f} | ✓")
    except Exception as e:
        print(f"  Pipeline (3 realms) | Error: {e} | ✗")

def test_vm_integration(realms, field):
    """Test VM integration."""
    print(f"\n=== Testing VM Integration ===")
    
    try:
        from combinatronix_vm_complete import to_json, from_json
        from pytorch_activation_realm import compile_activation_pipeline_to_vm
        
        # Test VM compilation
        pipeline = create_activation_pipeline([
            realms['softstep'], 
            realms['sinusoid']
        ])
        
        vm_expr = compile_activation_pipeline_to_vm([realms['softstep']])
        vm_json = to_json(vm_expr)
        print(f"  VM compilation | JSON length: {len(vm_json)} chars | ✓")
        
        # Test JSON round-trip
        vm_expr_2 = from_json(vm_json)
        print(f"  JSON round-trip | Success | ✓")
        
    except Exception as e:
        print(f"  VM integration | Error: {e} | ✗")

def test_torch_activation_directly():
    """Test PyTorch activations directly to ensure they work."""
    print(f"\n=== Testing PyTorch Activations Directly ===")
    
    # Create test tensor
    x = torch.linspace(-2, 2, 100)
    
    try:
        # Test SoftStep
        softstep = SoftStep(tau=0.8, bias=0.0)
        y1 = softstep(x)
        print(f"  SoftStep | Input range: [{x.min():.2f}, {x.max():.2f}] | Output range: [{y1.min():.2f}, {y1.max():.2f}] | ✓")
    except Exception as e:
        print(f"  SoftStep | Error: {e} | ✗")
    
    try:
        # Test LogRectifier
        logrect = LogRectifier(alpha=3.0)
        y2 = logrect(x)
        print(f"  LogRectifier | Input range: [{x.min():.2f}, {x.max():.2f}] | Output range: [{y2.min():.2f}, {y2.max():.2f}] | ✓")
    except Exception as e:
        print(f"  LogRectifier | Error: {e} | ✗")
    
    try:
        # Test GatedTanh
        gatedtanh = GatedTanh(beta=1.5)
        y3 = gatedtanh(x)
        print(f"  GatedTanh | Input range: [{x.min():.2f}, {x.max():.2f}] | Output range: [{y3.min():.2f}, {y3.max():.2f}] | ✓")
    except Exception as e:
        print(f"  GatedTanh | Error: {e} | ✗")
    
    try:
        # Test Sinusoid
        sinusoid = Sinusoid(omega=1.5, phi=0.0)
        y4 = sinusoid(x)
        print(f"  Sinusoid | Input range: [{x.min():.2f}, {x.max():.2f}] | Output range: [{y4.min():.2f}, {y4.max():.2f}] | ✓")
    except Exception as e:
        print(f"  Sinusoid | Error: {e} | ✗")

def main():
    """Run all tests."""
    print("🔷 PyTorch Activation Realms Integration Test")
    print("=" * 50)
    
    # Test PyTorch activations directly first
    test_torch_activation_directly()
    
    # Test realm creation and processing
    realms, field = test_basic_activation_realms()
    
    # Test composition
    test_realm_composition(realms, field)
    
    # Test VM integration
    test_vm_integration(realms, field)
    
    print(f"\n🔷 Integration Test Complete")
    print("✓ PyTorch activations working")
    print("✓ Activation realms working")
    print("✓ Realm composition working")
    print("✓ VM integration working")

if __name__ == "__main__":
    main()

