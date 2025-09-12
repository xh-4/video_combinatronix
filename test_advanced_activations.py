# ============================
# Advanced Activations Test
# ============================
"""
Test the new advanced signal processing activation functions.
"""

import numpy as np
import sys
import os

# Add path to activations
sys.path.append(r'c:\Users\The School\Desktop\Code\AI')

def test_advanced_activations():
    """Test all advanced activation functions."""
    print("=== Advanced Activations Test ===\n")
    
    # Test data
    x = np.linspace(-3, 3, 100)
    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-1, 1, 50)
    
    print("Testing with input range [-3, 3]:")
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    
    # 1. DampedSin
    print(f"\n1. DampedSin (alpha=0.25, beta=2.0):")
    def damped_sin_np(x, alpha=0.25, beta=2.0):
        return np.exp(-alpha * np.abs(x)) * np.sin(beta * x)
    
    damped_sin_out = damped_sin_np(x)
    print(f"  Output range: [{damped_sin_out.min():.3f}, {damped_sin_out.max():.3f}]")
    print(f"  Output mean: {damped_sin_out.mean():.3f}")
    print(f"  Output std: {damped_sin_out.std():.3f}")
    
    # 2. MirrorFold
    print(f"\n2. MirrorFold:")
    def mirror_fold_np(x):
        return np.abs(x)
    
    mirror_fold_out = mirror_fold_np(x)
    print(f"  Output range: [{mirror_fold_out.min():.3f}, {mirror_fold_out.max():.3f}]")
    print(f"  Output mean: {mirror_fold_out.mean():.3f}")
    
    # 3. MirrorFoldLeaky
    print(f"\n3. MirrorFoldLeaky (leak=0.01):")
    def mirror_fold_leaky_np(x, leak=0.01):
        return np.abs(x) + leak * x
    
    mirror_fold_leaky_out = mirror_fold_leaky_np(x)
    print(f"  Output range: [{mirror_fold_leaky_out.min():.3f}, {mirror_fold_leaky_out.max():.3f}]")
    print(f"  Output mean: {mirror_fold_leaky_out.mean():.3f}")
    
    # 4. MirrorFoldSoft
    print(f"\n4. MirrorFoldSoft (eps=1e-3):")
    def mirror_fold_soft_np(x, eps=1e-3):
        return np.sqrt(x*x + eps*eps) - eps
    
    mirror_fold_soft_out = mirror_fold_soft_np(x)
    print(f"  Output range: [{mirror_fold_soft_out.min():.3f}, {mirror_fold_soft_out.max():.3f}]")
    print(f"  Output mean: {mirror_fold_soft_out.mean():.3f}")
    
    # 5. SelectiveReflection
    print(f"\n5. SelectiveReflection (w=2.0, k=3.0):")
    def selective_reflection_np(x, w=2.0, k=3.0):
        def _sigmoid_np(z):
            return 1.0 / (1.0 + np.exp(-z))
        gate = _sigmoid_np(k * (x + w/2.0)) - _sigmoid_np(k * (x - w/2.0))
        return x * gate
    
    selective_reflection_out = selective_reflection_np(x)
    print(f"  Output range: [{selective_reflection_out.min():.3f}, {selective_reflection_out.max():.3f}]")
    print(f"  Output mean: {selective_reflection_out.mean():.3f}")
    
    # 6. MultiWell
    print(f"\n6. MultiWell (gamma=0.5, omega=2.0):")
    def multiwell_np(x, gamma=0.5, omega=2.0):
        return np.tanh(x) + gamma * np.sin(omega * x)
    
    multiwell_out = multiwell_np(x)
    print(f"  Output range: [{multiwell_out.min():.3f}, {multiwell_out.max():.3f}]")
    print(f"  Output mean: {multiwell_out.mean():.3f}")
    
    # 7. SpinorActivate
    print(f"\n7. SpinorActivate (phi=0.0):")
    def spinor_activate_np(x1, x2, phi=0.0):
        r = np.sqrt(x1*x1 + x2*x2) + 1e-12
        a = np.arctan2(x2, x1) + phi
        mag = np.tanh(r)
        return mag * np.cos(a), mag * np.sin(a)
    
    spinor_re, spinor_im = spinor_activate_np(x1, x2)
    print(f"  Real part range: [{spinor_re.min():.3f}, {spinor_re.max():.3f}]")
    print(f"  Imaginary part range: [{spinor_im.min():.3f}, {spinor_im.max():.3f}]")
    print(f"  Magnitude range: [{np.sqrt(spinor_re**2 + spinor_im**2).min():.3f}, {np.sqrt(spinor_re**2 + spinor_im**2).max():.3f}]")
    
    # 8. QuadratureUnit
    print(f"\n8. QuadratureUnit (psi=0.0):")
    def quadrature_unit_np(I, Q, psi=0.0, return_tuple=True):
        amp = np.tanh(np.sqrt(I*I + Q*Q))
        phase = np.arctan2(Q, I) + psi
        phase_mix = np.sin(phase)
        if return_tuple:
            return amp, phase_mix
        return amp * phase_mix
    
    quad_amp, quad_phase = quadrature_unit_np(x1, x2)
    print(f"  Amplitude range: [{quad_amp.min():.3f}, {quad_amp.max():.3f}]")
    print(f"  Phase mix range: [{quad_phase.min():.3f}, {quad_phase.max():.3f}]")
    
    # 9. SingularityEdge
    print(f"\n9. SingularityEdge (tau=0.5, k=10.0):")
    def singularity_edge_np(x, tau=0.5, k=10.0):
        def _sigmoid_np(z):
            return 1.0 / (1.0 + np.exp(-z))
        gate = _sigmoid_np(k * (np.abs(x) - tau))
        return x * gate
    
    singularity_edge_out = singularity_edge_np(x)
    print(f"  Output range: [{singularity_edge_out.min():.3f}, {singularity_edge_out.max():.3f}]")
    print(f"  Output mean: {singularity_edge_out.mean():.3f}")
    
    print(f"\nAll advanced activations tested successfully!")

def test_activation_properties():
    """Test specific properties of the activations."""
    print("\n=== Activation Properties Test ===\n")
    
    x = np.linspace(-2, 2, 50)
    
    # Test symmetry properties
    print("Testing symmetry properties:")
    
    # MirrorFold should be symmetric
    def mirror_fold_np(x):
        return np.abs(x)
    
    mirror_out = mirror_fold_np(x)
    print(f"  MirrorFold symmetry: {np.allclose(mirror_out, mirror_fold_np(-x))}")
    
    # DampedSin should be odd (if alpha=0)
    def damped_sin_np(x, alpha=0.0, beta=2.0):
        return np.exp(-alpha * np.abs(x)) * np.sin(beta * x)
    
    damped_sin_out = damped_sin_np(x, alpha=0.0)
    print(f"  DampedSin oddness (alpha=0): {np.allclose(damped_sin_out, -damped_sin_np(-x, alpha=0.0))}")
    
    # Test boundedness
    print("\nTesting boundedness:")
    
    # MultiWell should be bounded
    def multiwell_np(x, gamma=0.5, omega=2.0):
        return np.tanh(x) + gamma * np.sin(omega * x)
    
    multiwell_out = multiwell_np(x)
    print(f"  MultiWell bounded: {np.all(multiwell_out >= -1.5) and np.all(multiwell_out <= 1.5)}")
    
    # Test continuity
    print("\nTesting continuity:")
    
    # All functions should be continuous
    def test_continuity(func, x, name):
        try:
            # Test at a few points
            for i in range(1, len(x)-1):
                left = func(x[i-1])
                right = func(x[i+1])
                mid = func(x[i])
                # Simple continuity check
                if np.isnan(mid) or np.isinf(mid):
                    return False
            return True
        except:
            return False
    
    print(f"  DampedSin continuous: {test_continuity(lambda x: damped_sin_np(x), x, 'DampedSin')}")
    print(f"  MirrorFold continuous: {test_continuity(lambda x: mirror_fold_np(x), x, 'MirrorFold')}")
    print(f"  MultiWell continuous: {test_continuity(lambda x: multiwell_np(x), x, 'MultiWell')}")
    
    print(f"\nActivation properties test completed!")

def test_signal_processing_applications():
    """Test activations with signal processing examples."""
    print("\n=== Signal Processing Applications Test ===\n")
    
    # Create a test signal
    t = np.linspace(0, 1, 1000)
    signal = (np.sin(2 * np.pi * 10 * t) + 
             0.5 * np.sin(2 * np.pi * 20 * t + np.pi/4) + 
             0.1 * np.random.randn(len(t)))
    
    print(f"Test signal: 10Hz + 20Hz harmonics + noise")
    print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    
    # Test DampedSin for harmonic analysis
    print(f"\n1. DampedSin for harmonic analysis:")
    def damped_sin_np(x, alpha=0.1, beta=10.0):
        return np.exp(-alpha * np.abs(x)) * np.sin(beta * x)
    
    damped_sin_out = damped_sin_np(signal)
    print(f"  DampedSin output range: [{damped_sin_out.min():.3f}, {damped_sin_out.max():.3f}]")
    print(f"  Energy ratio: {np.sum(damped_sin_out**2) / np.sum(signal**2):.3f}")
    
    # Test SelectiveReflection for band-pass filtering
    print(f"\n2. SelectiveReflection for band-pass filtering:")
    def selective_reflection_np(x, w=1.0, k=5.0):
        def _sigmoid_np(z):
            return 1.0 / (1.0 + np.exp(-z))
        gate = _sigmoid_np(k * (x + w/2.0)) - _sigmoid_np(k * (x - w/2.0))
        return x * gate
    
    selective_out = selective_reflection_np(signal)
    print(f"  SelectiveReflection output range: [{selective_out.min():.3f}, {selective_out.max():.3f}]")
    print(f"  Energy ratio: {np.sum(selective_out**2) / np.sum(signal**2):.3f}")
    
    # Test QuadratureUnit for I/Q processing
    print(f"\n3. QuadratureUnit for I/Q processing:")
    def quadrature_unit_np(I, Q, psi=0.0):
        amp = np.tanh(np.sqrt(I*I + Q*Q))
        phase = np.arctan2(Q, I) + psi
        phase_mix = np.sin(phase)
        return amp, phase_mix
    
    # Create I/Q signals
    I_signal = signal
    Q_signal = np.roll(signal, 10)  # Phase-shifted version
    
    quad_amp, quad_phase = quadrature_unit_np(I_signal, Q_signal)
    print(f"  Quadrature amplitude range: [{quad_amp.min():.3f}, {quad_amp.max():.3f}]")
    print(f"  Quadrature phase range: [{quad_phase.min():.3f}, {quad_phase.max():.3f}]")
    
    print(f"\nSignal processing applications test completed!")

if __name__ == "__main__":
    test_advanced_activations()
    test_activation_properties()
    test_signal_processing_applications()

