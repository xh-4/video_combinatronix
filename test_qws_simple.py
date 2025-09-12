# ============================
# QWSActivation Simple Test (No PyTorch)
# ============================
"""
Simple test of QWSActivation concept without PyTorch dependencies.
"""

import numpy as np
import sys
import os

def mock_qws_activation(x, sigma0=0.6, a1=1.2, mu1=-1.2, sigma1=0.35, 
                       a2=1.6, mu2=-2.6, sigma2=0.45, p1=0.9, nu1=1.0, 
                       tau1=0.25, p2=0.9, nu2=2.2, tau2=0.25, d=0.15, 
                       omega=2.4, gamma=0.08):
    """Mock QWSActivation function using NumPy."""
    
    def _gauss(x, mu, sigma):
        sigma = max(sigma, 1e-8)
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def _sigmoid(z, tau):
        tau = max(tau, 1e-8)
        return 1.0 / (1.0 + np.exp(-z / tau))
    
    # dead-zone notch (≈0 near 0, ≈1 away)
    notch = 1.0 - np.exp(-0.5 * (x / max(sigma0, 1e-8)) ** 2)
    
    # two negative wells
    well1 = -a1 * _gauss(x, mu1, sigma1)
    well2 = -a2 * _gauss(x, mu2, sigma2)
    
    # two stepped excitations (tiered plateaus)
    step1 = p1 * _sigmoid(x - nu1, tau1)
    step2 = p2 * _sigmoid(x - nu2, tau2)
    
    # subtle damped oscillation
    oscill = d * np.exp(-gamma * x**2) * np.sin(omega * x)
    
    return notch * (well1 + well2 + step1 + step2 + oscill)

def test_qws_activation():
    """Test QWSActivation function."""
    print("=== QWSActivation Test (NumPy) ===\n")
    
    # Test with different input ranges
    test_inputs = [
        np.linspace(-5, 5, 100),
        np.linspace(-2, 2, 50),
        np.linspace(0, 3, 30),
        np.array([0.0, 1.0, -1.0, 2.0, -2.0])
    ]
    
    print(f"Testing QWSActivation with different inputs:")
    
    for i, x in enumerate(test_inputs):
        output = mock_qws_activation(x)
        
        print(f"  Input {i+1}: range [{x.min():.3f}, {x.max():.3f}]")
        print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"    Output mean: {output.mean():.3f}")
        print(f"    Output std: {output.std():.3f}")
        
        # Check for NaN or infinite values
        if np.isnan(output).any():
            print(f"    WARNING: NaN values detected!")
        if np.isinf(output).any():
            print(f"    WARNING: Infinite values detected!")
    
    # Test with specific values
    print(f"\nTesting QWSActivation with specific values:")
    specific_values = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    specific_output = mock_qws_activation(specific_values)
    
    for val, out in zip(specific_values, specific_output):
        print(f"  QWS({val:.1f}) = {out:.3f}")
    
    print(f"\nQWSActivation test completed successfully!")

def test_qws_components():
    """Test individual components of QWSActivation."""
    print("\n=== QWSActivation Components Test ===\n")
    
    x = np.linspace(-3, 3, 100)
    
    # Test individual components
    sigma0 = 0.6
    a1, mu1, sigma1 = 1.2, -1.2, 0.35
    a2, mu2, sigma2 = 1.6, -2.6, 0.45
    p1, nu1, tau1 = 0.9, 1.0, 0.25
    p2, nu2, tau2 = 0.9, 2.2, 0.25
    d, omega, gamma = 0.15, 2.4, 0.08
    
    # Dead-zone notch
    notch = 1.0 - np.exp(-0.5 * (x / max(sigma0, 1e-8)) ** 2)
    print(f"Dead-zone notch range: [{notch.min():.3f}, {notch.max():.3f}]")
    
    # Negative wells
    def _gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / max(sigma, 1e-8)) ** 2)
    
    well1 = -a1 * _gauss(x, mu1, sigma1)
    well2 = -a2 * _gauss(x, mu2, sigma2)
    print(f"Well 1 range: [{well1.min():.3f}, {well1.max():.3f}]")
    print(f"Well 2 range: [{well2.min():.3f}, {well2.max():.3f}]")
    
    # Stepped excitations
    def _sigmoid(z, tau):
        return 1.0 / (1.0 + np.exp(-z / max(tau, 1e-8)))
    
    step1 = p1 * _sigmoid(x - nu1, tau1)
    step2 = p2 * _sigmoid(x - nu2, tau2)
    print(f"Step 1 range: [{step1.min():.3f}, {step1.max():.3f}]")
    print(f"Step 2 range: [{step2.min():.3f}, {step2.max():.3f}]")
    
    # Damped oscillation
    oscill = d * np.exp(-gamma * x**2) * np.sin(omega * x)
    print(f"Oscillation range: [{oscill.min():.3f}, {oscill.max():.3f}]")
    
    # Combined output
    combined = notch * (well1 + well2 + step1 + step2 + oscill)
    print(f"Combined output range: [{combined.min():.3f}, {combined.max():.3f}]")
    
    print(f"\nQWSActivation components test completed!")

if __name__ == "__main__":
    test_qws_activation()
    test_qws_components()

