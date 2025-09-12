# ============================
# Simple Combinator Kernel SIREN Field
# ============================
"""
Simplified SIREN field implementation using Combinator Kernel methodology.
Focuses on core concepts without complex training loops.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass

from Combinator_Kernel import (
    FieldIQ, make_field_from_real, 
    Unary, Binary, Field,
    K, S, B, C, W, PLUS, TIMES,
    split_add, split_mul,
    phase_deg, freq_shift, amp, lowpass_hz
)

# ============================
# Coordinate Field System
# ============================

@dataclass
class CoordinateField:
    """Coordinate-based field using Combinator Kernel patterns."""
    x: np.ndarray  # x coordinates normalized to [-1,1]
    y: np.ndarray  # y coordinates normalized to [-1,1]
    t: np.ndarray  # t coordinates normalized to [-1,1]
    sr: float      # sample rate
    roles: Dict[str, Any] = None
    
    def to_spatial_field(self) -> FieldIQ:
        """Convert spatial coordinates to FieldIQ."""
        z = self.x + 1j * self.y
        return FieldIQ(z, self.sr, {**(self.roles or {}), 'type': 'spatial'})
    
    def to_temporal_field(self) -> FieldIQ:
        """Convert temporal coordinates to FieldIQ."""
        return make_field_from_real(self.t, self.sr, tag=("temporal", "coordinate"))

def create_coordinate_field(T: int, H: int, W: int, sr: float = 1.0) -> CoordinateField:
    """Create normalized coordinate field for video processing."""
    # Create coordinate grids (normalized to [-1,1])
    y_coords = np.linspace(-1, 1, H)
    x_coords = np.linspace(-1, 1, W)
    t_coords = np.linspace(-1, 1, T)
    
    # Create meshgrids
    Y, X, T = np.meshgrid(y_coords, x_coords, t_coords, indexing='ij')
    
    return CoordinateField(
        x=X.flatten(),
        y=Y.flatten(),
        t=T.flatten(),
        sr=sr,
        roles={'field_type': 'coordinate', 'dimensions': (T, H, W)}
    )

# ============================
# SIREN Combinator Operations
# ============================

def siren_activation(w0: float = 30.0) -> Unary:
    """Create SIREN-style sine activation using Combinator patterns."""
    def activation(field: FieldIQ) -> FieldIQ:
        # Apply sine activation to complex field
        activated_z = np.sin(w0 * field.z)
        return FieldIQ(
            activated_z, 
            field.sr, 
            {**field.roles, 'siren_activated': True, 'w0': w0}
        )
    return activation

def fourier_features_combinator(num_bands: int = 4, max_freq: float = 10.0) -> Unary:
    """Create Fourier features using Combinator patterns."""
    def fourier_processor(field: FieldIQ) -> FieldIQ:
        if num_bands <= 0:
            return field
        
        # Generate frequency bands
        ks = np.arange(num_bands)
        freqs = (2.0**ks) * np.pi * max_freq / (2.0**(num_bands-1))
        
        # Apply multi-scale frequency processing
        processed_components = []
        
        for k, freq in enumerate(freqs):
            # Quadrature decomposition
            sin_component = np.sin(freq * field.z)
            cos_component = np.cos(freq * field.z)
            
            processed_components.extend([sin_component, cos_component])
        
        # Combine all components
        combined_z = field.z
        for comp in processed_components:
            combined_z = np.concatenate([combined_z, comp])
        
        return FieldIQ(
            combined_z,
            field.sr,
            {**field.roles, 'fourier_features': True, 'num_bands': num_bands}
        )
    
    return fourier_processor

def coordinate_encoding_combinator() -> Unary:
    """Create coordinate encoding using Combinator patterns."""
    def encoder(field: FieldIQ) -> FieldIQ:
        # Extract coordinate components
        x_coords = np.real(field.z)
        y_coords = np.imag(field.z)
        
        # Create encoded representation
        encoded_x = np.sin(2 * np.pi * x_coords)
        encoded_y = np.cos(2 * np.pi * y_coords)
        
        encoded_z = encoded_x + 1j * encoded_y
        
        return FieldIQ(
            encoded_z,
            field.sr,
            {**field.roles, 'coordinate_encoded': True}
        )
    
    return encoder

# ============================
# SIREN Field Combinator Pipeline
# ============================

def create_siren_field_combinator(
    w0: float = 30.0,
    w0_hidden: float = 30.0,
    fourier_bands: int = 4,
    max_freq: float = 10.0,
    depth: int = 3
) -> Unary:
    """Create SIREN field processor using Combinator patterns."""
    def siren_field_processor(field: FieldIQ) -> FieldIQ:
        # Start with coordinate encoding
        encoded = coordinate_encoding_combinator()(field)
        
        # Apply Fourier features if enabled
        if fourier_bands > 0:
            fourier_processed = fourier_features_combinator(fourier_bands, max_freq)(encoded)
        else:
            fourier_processed = encoded
        
        # Apply SIREN activations in sequence
        current = fourier_processed
        
        for i in range(depth):
            if i == 0:
                # First layer with w0
                current = siren_activation(w0)(current)
            else:
                # Hidden layers with w0_hidden
                current = siren_activation(w0_hidden)(current)
            
            # Add amplitude scaling
            current = amp(0.8)(current)
        
        return current
    
    return siren_field_processor

# ============================
# Demo Functions
# ============================

def demo_siren_activations():
    """Demo SIREN activations with Combinator patterns."""
    print("=== SIREN Activations Demo ===\n")
    
    # Create test field
    sr = 48000.0
    t = np.linspace(0, 1.0, int(sr), endpoint=False)
    x = 0.5 * np.cos(2 * np.pi * 440 * t)
    field = make_field_from_real(x, sr, tag=("test", "sine_wave"))
    
    print(f"Original field: {len(field.z)} samples")
    print(f"Original energy: {np.sum(np.abs(field.z)**2):.6f}")
    
    # Apply SIREN activation
    siren_proc = siren_activation(30.0)
    processed = siren_proc(field)
    
    print(f"SIREN processed: {len(processed.z)} samples")
    print(f"SIREN energy: {np.sum(np.abs(processed.z)**2):.6f}")
    print(f"Valid quadrature: {processed.is_valid_quadrature()}")
    
    return siren_proc, processed

def demo_fourier_features():
    """Demo Fourier features with Combinator patterns."""
    print("\n=== Fourier Features Demo ===\n")
    
    # Create test field
    sr = 48000.0
    t = np.linspace(0, 1.0, int(sr), endpoint=False)
    x = 0.5 * np.cos(2 * np.pi * 440 * t)
    field = make_field_from_real(x, sr, tag=("test", "sine_wave"))
    
    print(f"Original field: {len(field.z)} samples")
    print(f"Original energy: {np.sum(np.abs(field.z)**2):.6f}")
    
    # Apply Fourier features
    fourier_proc = fourier_features_combinator(num_bands=4, max_freq=10.0)
    processed = fourier_proc(field)
    
    print(f"Fourier processed: {len(processed.z)} samples")
    print(f"Fourier energy: {np.sum(np.abs(processed.z)**2):.6f}")
    print(f"Expansion factor: {len(processed.z) / len(field.z):.1f}x")
    
    return fourier_proc, processed

def demo_coordinate_processing():
    """Demo coordinate processing with Combinator patterns."""
    print("\n=== Coordinate Processing Demo ===\n")
    
    # Create coordinate field
    coord_field = create_coordinate_field(T=4, H=16, W=16, sr=1.0)
    
    print(f"Coordinate field: {len(coord_field.x)} points")
    print(f"X range: [{coord_field.x.min():.3f}, {coord_field.x.max():.3f}]")
    print(f"Y range: [{coord_field.y.min():.3f}, {coord_field.y.max():.3f}]")
    print(f"T range: [{coord_field.t.min():.3f}, {coord_field.t.max():.3f}]")
    
    # Convert to FieldIQ
    spatial_field = coord_field.to_spatial_field()
    temporal_field = coord_field.to_temporal_field()
    
    print(f"\nFieldIQ conversion:")
    print(f"  Spatial field: {len(spatial_field.z)} samples")
    print(f"  Temporal field: {len(temporal_field.z)} samples")
    print(f"  Valid quadrature: {spatial_field.is_valid_quadrature()}")
    
    # Apply SIREN processing
    siren_processor = create_siren_field_combinator(w0=30.0, depth=2)
    
    processed_spatial = siren_processor(spatial_field)
    processed_temporal = siren_processor(temporal_field)
    
    print(f"\nSIREN processing:")
    print(f"  Processed spatial energy: {np.sum(np.abs(processed_spatial.z)**2):.6f}")
    print(f"  Processed temporal energy: {np.sum(np.abs(processed_temporal.z)**2):.6f}")
    
    return {
        'coord_field': coord_field,
        'spatial_field': spatial_field,
        'temporal_field': temporal_field,
        'processed_spatial': processed_spatial,
        'processed_temporal': processed_temporal
    }

def demo_siren_field_pipeline():
    """Demo complete SIREN field pipeline."""
    print("\n=== SIREN Field Pipeline Demo ===\n")
    
    # Create coordinate field
    coord_field = create_coordinate_field(T=2, H=8, W=8, sr=1.0)
    
    # Create SIREN field combinator
    siren_processor = create_siren_field_combinator(
        w0=30.0,
        w0_hidden=30.0,
        fourier_bands=2,
        max_freq=5.0,
        depth=2
    )
    
    # Process spatial field
    spatial_field = coord_field.to_spatial_field()
    processed = siren_processor(spatial_field)
    
    print(f"Input field: {len(spatial_field.z)} samples")
    print(f"Processed field: {len(processed.z)} samples")
    print(f"Input energy: {np.sum(np.abs(spatial_field.z)**2):.6f}")
    print(f"Processed energy: {np.sum(np.abs(processed.z)**2):.6f}")
    print(f"Valid quadrature: {processed.is_valid_quadrature()}")
    
    # Test with different w0 values
    print(f"\nTesting different w0 values:")
    for w0 in [10.0, 30.0, 50.0]:
        test_processor = siren_activation(w0)
        test_processed = test_processor(spatial_field)
        energy = np.sum(np.abs(test_processed.z)**2)
        print(f"  w0={w0}: Energy = {energy:.6f}")
    
    return siren_processor, processed

def demo_combinator_patterns():
    """Demo various Combinator patterns with SIREN fields."""
    print("\n=== Combinator Patterns Demo ===\n")
    
    # Create test field
    sr = 48000.0
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    x = 0.5 * np.cos(2 * np.pi * 440 * t)
    field = make_field_from_real(x, sr, tag=("test", "sine_wave"))
    
    print(f"Original field energy: {np.sum(np.abs(field.z)**2):.6f}")
    
    # Test different Combinator patterns
    patterns = {
        'SIREN only': siren_activation(30.0),
        'Amplified': amp(0.8),
        'Phase shifted': phase_deg(45.0, 440.0),
        'Low-passed': lowpass_hz(1000.0),
        'Split-add': split_add(amp(0.5)),
        'Split-mul': split_mul(amp(0.8))
    }
    
    for name, pattern in patterns.items():
        try:
            processed = pattern(field)
            energy = np.sum(np.abs(processed.z)**2)
            print(f"  {name}: Energy = {energy:.6f}")
        except Exception as e:
            print(f"  {name}: Error - {str(e)}")
    
    return patterns

def main():
    """Main demo function."""
    print("=== Simple Combinator Kernel SIREN Field ===\n")
    
    # Run demos
    siren_proc, siren_processed = demo_siren_activations()
    fourier_proc, fourier_processed = demo_fourier_features()
    coord_results = demo_coordinate_processing()
    siren_processor, pipeline_processed = demo_siren_field_pipeline()
    combinator_patterns = demo_combinator_patterns()
    
    print("\n=== Demo Summary ===")
    print("✓ SIREN activations with Combinator patterns")
    print("✓ Fourier features using quadrature operations")
    print("✓ Coordinate-based field representation")
    print("✓ SIREN field pipeline processing")
    print("✓ Combinator pattern integration")
    
    print("\n=== Key Features ===")
    print("- Coordinate-to-FieldIQ transformation")
    print("- SIREN activations as sine-based operations")
    print("- Multi-scale Fourier feature processing")
    print("- Functional composition of field operations")
    print("- Integration with Combinator Kernel patterns")
    
    print("\n=== Combinator Kernel Benefits ===")
    print("- Functional composition of field operations")
    print("- Quadrature signal processing integration")
    print("- Composable and reusable field processors")
    print("- Integration with existing Combinator patterns")
    print("- Pipeline-based processing workflows")

if __name__ == "__main__":
    main()


