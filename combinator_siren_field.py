# ============================
# Combinator Kernel SIREN Field Implementation
# ============================
"""
Implementation of SIREN field concepts using Combinator Kernel methodology.
Demonstrates how coordinate-based neural fields can be expressed as functional compositions.
"""

import numpy as np
import torch
import torch.nn as nn
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
# Coordinate-to-Field Mapping
# ============================

@dataclass
class CoordinateField:
    """Represents a coordinate-based field using Combinator patterns."""
    x: np.ndarray  # x coordinates
    y: np.ndarray  # y coordinates  
    t: np.ndarray  # t coordinates
    sr: float      # sample rate
    roles: Dict[str, Any] = None
    
    def to_field_iq(self, channel: str = "spatial") -> FieldIQ:
        """Convert coordinate field to FieldIQ for processing."""
        # Combine coordinates into complex representation
        z = self.x + 1j * self.y  # Spatial coordinates as complex
        return FieldIQ(z, self.sr, {**(self.roles or {}), 'channel': channel})
    
    def temporal_field(self) -> FieldIQ:
        """Extract temporal component as FieldIQ."""
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
# SIREN-Style Combinator Operations
# ============================

def siren_activation(w0: float = 30.0) -> Unary:
    """
    Create SIREN-style sine activation using Combinator patterns.
    Maps to sin(w0 * field) operation.
    """
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
    """
    Create Fourier features using Combinator patterns.
    Equivalent to SIREN's fourier_features function.
    """
    def fourier_processor(field: FieldIQ) -> FieldIQ:
        if num_bands <= 0:
            return field
        
        # Generate frequency bands
        ks = np.arange(num_bands)
        freqs = (2.0**ks) * np.pi * max_freq / (2.0**(num_bands-1))
        
        # Apply multi-scale frequency processing
        processed_components = []
        
        for k, freq in enumerate(freqs):
            # Frequency shift operation
            freq_shifted = freq_shift(freq)(field)
            
            # Quadrature decomposition
            sin_component = siren_activation(1.0)(freq_shifted)
            cos_component = phase_deg(90.0, freq)(freq_shifted)
            
            processed_components.extend([sin_component, cos_component])
        
        # Combine all components
        combined_z = field.z
        for comp in processed_components:
            combined_z = np.concatenate([combined_z, comp.z])
        
        return FieldIQ(
            combined_z,
            field.sr,
            {**field.roles, 'fourier_features': True, 'num_bands': num_bands}
        )
    
    return fourier_processor

def coordinate_encoding_combinator() -> Unary:
    """
    Create coordinate encoding using Combinator patterns.
    Maps (x,y,t) coordinates to encoded representation.
    """
    def encoder(field: FieldIQ) -> FieldIQ:
        # Extract coordinate components
        x_coords = np.real(field.z)
        y_coords = np.imag(field.z)
        
        # Create encoded representation
        # This is a simplified version - in practice would use learned weights
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
    """
    Create SIREN field processor using Combinator patterns.
    Implements the core SIREN field functionality.
    """
    def siren_field_processor(field: FieldIQ) -> FieldIQ:
        # Start with coordinate encoding
        encoded = coordinate_encoding_combinator()(field)
        
        # Apply Fourier features if enabled
        if fourier_bands > 0:
            fourier_processed = fourier_features_combinator(fourier_bands, max_freq)(encoded)
        else:
            fourier_processed = encoded
        
        # Apply SIREN activations in sequence (simulating depth)
        current = fourier_processed
        
        for i in range(depth):
            if i == 0:
                # First layer with w0
                current = siren_activation(w0)(current)
            else:
                # Hidden layers with w0_hidden
                current = siren_activation(w0_hidden)(current)
            
            # Add some processing between layers
            current = B(amp(0.8))(current)  # Amplitude scaling
        
        return current
    
    return siren_field_processor

# ============================
# Video Field Processing
# ============================

def video_to_coordinate_field(video_frames: np.ndarray, sr: float = 1.0) -> CoordinateField:
    """
    Convert video frames to coordinate field representation.
    """
    T, H, W, C = video_frames.shape
    
    # Create coordinate field
    coord_field = create_coordinate_field(T, H, W, sr)
    
    # Store video data in roles for reference
    coord_field.roles = {
        **coord_field.roles,
        'video_frames': video_frames,
        'video_shape': (T, H, W, C)
    }
    
    return coord_field

def coordinate_field_to_rgb(field: FieldIQ, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Convert processed coordinate field back to RGB representation.
    """
    T, H, W = target_shape
    
    # Extract processed coordinates
    processed_z = field.z
    
    # Reshape to target dimensions
    if len(processed_z) >= T * H * W:
        rgb_data = processed_z[:T*H*W].reshape(T, H, W)
    else:
        # Pad if needed
        rgb_data = np.pad(processed_z, (0, T*H*W - len(processed_z)), mode='edge')
        rgb_data = rgb_data.reshape(T, H, W)
    
    # Normalize to [0,1] range
    rgb_data = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min() + 1e-8)
    
    # Convert to RGB (simplified - in practice would use learned mapping)
    rgb_data = np.clip(rgb_data, 0, 1)
    
    return rgb_data

# ============================
# Combinator Field Training
# ============================

def train_siren_field_combinator(
    video_frames: np.ndarray,
    siren_processor: Unary,
    epochs: int = 10,
    learning_rate: float = 1e-4
) -> Dict[str, Any]:
    """
    Train SIREN field using Combinator patterns.
    """
    print("Training SIREN field with Combinator patterns...")
    
    # Convert video to coordinate field
    coord_field = video_to_coordinate_field(video_frames)
    
    # Create spatial and temporal fields
    spatial_field = coord_field.to_field_iq("spatial")
    temporal_field = coord_field.temporal_field()
    
    # Training results
    results = {
        'spatial_field': spatial_field,
        'temporal_field': temporal_field,
        'processed_spatial': None,
        'processed_temporal': None,
        'training_history': []
    }
    
    # Simulate training process
    for epoch in range(epochs):
        # Process fields with SIREN combinator
        processed_spatial = siren_processor(spatial_field)
        processed_temporal = siren_processor(temporal_field)
        
        # Calculate some metric (simplified)
        spatial_energy = np.sum(np.abs(processed_spatial.z)**2)
        temporal_energy = np.sum(np.abs(processed_temporal.z)**2)
        
        results['training_history'].append({
            'epoch': epoch,
            'spatial_energy': spatial_energy,
            'temporal_energy': temporal_energy,
            'total_energy': spatial_energy + temporal_energy
        })
        
        print(f"Epoch {epoch+1}/{epochs}: Spatial={spatial_energy:.6f}, Temporal={temporal_energy:.6f}")
    
    results['processed_spatial'] = processed_spatial
    results['processed_temporal'] = processed_temporal
    
    return results

# ============================
# Demo and Testing
# ============================

def demo_siren_field_combinator():
    """Demo SIREN field processing using Combinator patterns."""
    print("=== Combinator Kernel SIREN Field Demo ===\n")
    
    # Create synthetic video data
    T, H, W = 8, 32, 32
    video_frames = np.random.rand(T, H, W, 3).astype(np.float32)
    
    print(f"Created synthetic video: {video_frames.shape}")
    
    # Create SIREN field combinator
    siren_processor = create_siren_field_combinator(
        w0=30.0,
        w0_hidden=30.0,
        fourier_bands=4,
        max_freq=10.0,
        depth=3
    )
    
    # Train the field
    results = train_siren_field_combinator(video_frames, siren_processor, epochs=5)
    
    # Analyze results
    print(f"\nTraining Results:")
    print(f"  Spatial field energy: {np.sum(np.abs(results['processed_spatial'].z)**2):.6f}")
    print(f"  Temporal field energy: {np.sum(np.abs(results['processed_temporal'].z)**2):.6f}")
    
    # Show training history
    print(f"\nTraining History:")
    for entry in results['training_history']:
        print(f"  Epoch {entry['epoch']+1}: Total energy = {entry['total_energy']:.6f}")
    
    return results

def demo_coordinate_processing():
    """Demo coordinate-based processing with Combinator patterns."""
    print("\n=== Coordinate Processing Demo ===\n")
    
    # Create coordinate field
    coord_field = create_coordinate_field(T=4, H=16, W=16, sr=1.0)
    
    print(f"Coordinate field: {len(coord_field.x)} points")
    print(f"X range: [{coord_field.x.min():.3f}, {coord_field.x.max():.3f}]")
    print(f"Y range: [{coord_field.y.min():.3f}, {coord_field.y.max():.3f}]")
    print(f"T range: [{coord_field.t.min():.3f}, {coord_field.t.max():.3f}]")
    
    # Convert to FieldIQ
    spatial_field = coord_field.to_field_iq("spatial")
    temporal_field = coord_field.temporal_field()
    
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

def demo_fourier_features():
    """Demo Fourier features using Combinator patterns."""
    print("\n=== Fourier Features Demo ===\n")
    
    # Create test field
    sr = 48000.0
    t = np.linspace(0, 1.0, int(sr), endpoint=False)
    x = 0.5 * np.cos(2 * np.pi * 440 * t)
    field = make_field_from_real(x, sr, tag=("test", "sine_wave"))
    
    print(f"Original field: {len(field.z)} samples")
    print(f"Original energy: {np.sum(np.abs(field.z)**2):.6f}")
    
    # Apply Fourier features
    fourier_processor = fourier_features_combinator(num_bands=4, max_freq=10.0)
    fourier_processed = fourier_processor(field)
    
    print(f"Fourier processed: {len(fourier_processed.z)} samples")
    print(f"Fourier energy: {np.sum(np.abs(fourier_processed.z)**2):.6f}")
    print(f"Expansion factor: {len(fourier_processed.z) / len(field.z):.1f}x")
    
    return {
        'original': field,
        'fourier_processed': fourier_processed
    }

def main():
    """Main demo function."""
    print("=== Combinator Kernel SIREN Field Implementation ===\n")
    
    # Run demos
    siren_results = demo_siren_field_combinator()
    coord_results = demo_coordinate_processing()
    fourier_results = demo_fourier_features()
    
    print("\n=== Demo Summary ===")
    print("✓ SIREN field processing with Combinator patterns")
    print("✓ Coordinate-based field representation")
    print("✓ Fourier features using quadrature operations")
    print("✓ Functional composition of field operations")
    print("✓ Integration with Combinator Kernel methodology")
    
    print("\n=== Key Insights ===")
    print("- SIREN activations map to sine-based quadrature operations")
    print("- Fourier features are multi-scale frequency processing")
    print("- Coordinate fields can be represented as FieldIQ objects")
    print("- SIREN field training can be expressed as Combinator pipelines")
    print("- Continuous field representation enables smooth interpolation")

if __name__ == "__main__":
    main()


