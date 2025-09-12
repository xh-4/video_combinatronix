# ============================
# QWSActivation Test
# ============================
"""
Test the QWSActivation function to ensure it works correctly.
"""

import numpy as np
import torch
import sys
import os

# Add path to activations
sys.path.append(r'c:\Users\The School\Desktop\Code\AI')
from activations import QWSActivation

def test_qws_activation():
    """Test QWSActivation function."""
    print("=== QWSActivation Test ===\n")
    
    # Create QWSActivation instance
    qws = QWSActivation()
    
    print(f"QWSActivation parameters:")
    print(f"  sigma0 (dead-zone): {qws.sigma0.item():.3f}")
    print(f"  a1, mu1, sigma1 (well 1): {qws.a1.item():.3f}, {qws.mu1.item():.3f}, {qws.sigma1.item():.3f}")
    print(f"  a2, mu2, sigma2 (well 2): {qws.a2.item():.3f}, {qws.mu2.item():.3f}, {qws.sigma2.item():.3f}")
    print(f"  p1, nu1, tau1 (step 1): {qws.p1.item():.3f}, {qws.nu1.item():.3f}, {qws.tau1.item():.3f}")
    print(f"  p2, nu2, tau2 (step 2): {qws.p2.item():.3f}, {qws.nu2.item():.3f}, {qws.tau2.item():.3f}")
    print(f"  d, omega, gamma (oscill): {qws.d.item():.3f}, {qws.omega.item():.3f}, {qws.gamma.item():.3f}")
    
    # Test with different input ranges
    test_inputs = [
        torch.linspace(-5, 5, 100),
        torch.linspace(-2, 2, 50),
        torch.linspace(0, 3, 30),
        torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])
    ]
    
    print(f"\nTesting QWSActivation with different inputs:")
    
    for i, x in enumerate(test_inputs):
        output = qws(x)
        
        print(f"  Input {i+1}: range [{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"    Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print(f"    Output mean: {output.mean().item():.3f}")
        print(f"    Output std: {output.std().item():.3f}")
        
        # Check for NaN or infinite values
        if torch.isnan(output).any():
            print(f"    WARNING: NaN values detected!")
        if torch.isinf(output).any():
            print(f"    WARNING: Infinite values detected!")
    
    # Test with specific values
    print(f"\nTesting QWSActivation with specific values:")
    specific_values = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    specific_output = qws(specific_values)
    
    for val, out in zip(specific_values, specific_output):
        print(f"  QWS({val.item():.1f}) = {out.item():.3f}")
    
    # Test gradient computation
    print(f"\nTesting gradient computation:")
    x = torch.linspace(-2, 2, 10, requires_grad=True)
    output = qws(x)
    loss = output.sum()
    loss.backward()
    
    print(f"  Input gradient range: [{x.grad.min().item():.3f}, {x.grad.max().item():.3f}]")
    print(f"  Gradient mean: {x.grad.mean().item():.3f}")
    
    print(f"\nQWSActivation test completed successfully!")

def test_qws_activation_realm():
    """Test QWSActivation realm integration."""
    print("\n=== QWSActivation Realm Test ===\n")
    
    try:
        from pytorch_activation_realm import create_qwsactivation_realm
        from Combinator_Kernel import FieldIQ, make_field_from_real
        
        # Create QWSActivation realm
        qws_realm = create_qwsactivation_realm()
        
        print(f"Created QWSActivation realm: {qws_realm.name}")
        print(f"Parameters: {qws_realm.parameters}")
        
        # Create test field
        sr = 48000
        dur = 1.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        x = 0.8 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t + np.pi/4)
        field = make_field_from_real(x, sr, tag=("test", "qws"))
        
        print(f"Test field: {len(field.z)} samples, {field.sr} Hz")
        print(f"Field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
        
        # Process field through QWSActivation realm
        processed_field = qws_realm.field_processor(field)
        
        print(f"Processed field energy: {np.sum(np.abs(processed_field.z) ** 2):.2f}")
        print(f"Energy ratio: {np.sum(np.abs(processed_field.z) ** 2) / np.sum(np.abs(field.z) ** 2):.3f}")
        
        # Test VM integration
        vm_json = qws_realm.vm_node
        print(f"VM node type: {type(vm_json).__name__}")
        
        print(f"\nQWSActivation realm test completed successfully!")
        
    except Exception as e:
        print(f"QWSActivation realm test failed: {e}")

if __name__ == "__main__":
    test_qws_activation()
    test_qws_activation_realm()

