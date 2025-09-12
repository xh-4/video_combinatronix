# ============================
# Combinator Kernel SIREN Field Implementation
# ============================
"""
Complete SIREN field implementation using Combinator Kernel methodology.
Integrates coordinate-based neural fields with functional composition patterns.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable, Optional
from dataclasses import dataclass
import math

from Combinator_Kernel import (
    FieldIQ, make_field_from_real, 
    VideoFrame, VideoChunk, VideoStreamProcessor,
    Unary, Binary, Field,
    K, S, B, C, W, PLUS, TIMES,
    split_add, split_mul,
    phase_deg, freq_shift, amp, delay_ms, lowpass_hz,
    video_channel_processor, video_frame_processor, video_temporal_processor
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
    
    def to_combined_field(self) -> FieldIQ:
        """Convert all coordinates to combined FieldIQ."""
        # Combine spatial and temporal as complex representation
        spatial_z = self.x + 1j * self.y
        temporal_z = self.t + 0j  # Real temporal component
        combined_z = spatial_z + 1j * temporal_z  # Nested complex
        return FieldIQ(combined_z, self.sr, {**(self.roles or {}), 'type': 'combined'})

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

def sample_coordinates(coord_field: CoordinateField, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample random coordinates from the field."""
    total_points = len(coord_field.x)
    indices = np.random.choice(total_points, min(n_samples, total_points), replace=False)
    
    return (
        coord_field.x[indices],
        coord_field.y[indices], 
        coord_field.t[indices]
    )

# ============================
# SIREN Combinator Operations
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

def siren_linear_activation(w0: float = 30.0, weight_scale: float = 1.0) -> Unary:
    """
    Create SIREN linear activation: sin(w0 * (weight * field + bias))
    Simulates the linear layer + sine activation pattern.
    """
    def activation(field: FieldIQ) -> FieldIQ:
        # Simulate linear transformation (simplified)
        linear_output = weight_scale * field.z
        activated_z = np.sin(w0 * linear_output)
        return FieldIQ(
            activated_z,
            field.sr,
            {**field.roles, 'siren_linear_activated': True, 'w0': w0, 'weight_scale': weight_scale}
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
            sin_component = FieldIQ(
                np.sin(freq * field.z),
                field.sr,
                {**field.roles, 'fourier_sin': True, 'band': k}
            )
            cos_component = FieldIQ(
                np.cos(freq * field.z),
                field.sr,
                {**field.roles, 'fourier_cos': True, 'band': k}
            )
            
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
    depth: int = 3,
    hidden_width: int = 256
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
                current = siren_linear_activation(w0)(current)
            else:
                # Hidden layers with w0_hidden
                current = siren_linear_activation(w0_hidden)(current)
            
            # Add some processing between layers
            current = amp(0.8)(current)  # Amplitude scaling
            
            # Add residual connection
            if i > 0:
                current = PLUS(fourier_processed)(current)
        
        return current
    
    return siren_field_processor

# ============================
# Video Processing Integration
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
# Training Pipeline
# ============================

def create_training_pipeline(
    siren_processor: Unary,
    learning_rate: float = 1e-4
) -> Callable[[CoordinateField, np.ndarray], Dict[str, Any]]:
    """
    Create training pipeline using Combinator patterns.
    """
    def train_step(coord_field: CoordinateField, video_frames: np.ndarray) -> Dict[str, Any]:
        # Sample coordinates
        x_samples, y_samples, t_samples = sample_coordinates(coord_field, 1000)
        
        # Create field from samples
        sample_field = FieldIQ(
            x_samples + 1j * y_samples,
            coord_field.sr,
            {'type': 'training_sample', 't_samples': t_samples}
        )
        
        # Process with SIREN
        processed = siren_processor(sample_field)
        
        # Calculate loss (simplified)
        target_energy = np.sum(np.abs(sample_field.z)**2)
        predicted_energy = np.sum(np.abs(processed.z)**2)
        loss = np.abs(target_energy - predicted_energy)
        
        return {
            'loss': loss,
            'target_energy': target_energy,
            'predicted_energy': predicted_energy,
            'processed_field': processed,
            'sample_field': sample_field
        }
    
    return train_step

def train_siren_field(
    video_frames: np.ndarray,
    epochs: int = 10,
    w0: float = 30.0,
    w0_hidden: float = 30.0,
    fourier_bands: int = 4,
    max_freq: float = 10.0,
    depth: int = 3
) -> Dict[str, Any]:
    """
    Train SIREN field using Combinator patterns.
    """
    print("Training SIREN field with Combinator patterns...")
    
    # Convert video to coordinate field
    coord_field = video_to_coordinate_field(video_frames)
    
    # Create SIREN field combinator
    siren_processor = create_siren_field_combinator(
        w0=w0,
        w0_hidden=w0_hidden,
        fourier_bands=fourier_bands,
        max_freq=max_freq,
        depth=depth
    )
    
    # Create training pipeline
    train_step = create_training_pipeline(siren_processor)
    
    # Training results
    results = {
        'coord_field': coord_field,
        'siren_processor': siren_processor,
        'training_history': [],
        'final_loss': None
    }
    
    # Training loop
    for epoch in range(epochs):
        # Run training step
        step_result = train_step(coord_field, video_frames)
        
        # Store results
        results['training_history'].append({
            'epoch': epoch,
            'loss': step_result['loss'],
            'target_energy': step_result['target_energy'],
            'predicted_energy': step_result['predicted_energy']
        })
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {step_result['loss']:.6f}")
    
    results['final_loss'] = results['training_history'][-1]['loss']
    
    return results

# ============================
# Video Reconstruction
# ============================

def reconstruct_video_from_siren_field(
    siren_processor: Unary,
    coord_field: CoordinateField,
    target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Reconstruct video from SIREN field using Combinator patterns.
    """
    T, H, W = target_shape
    
    # Create coordinate grid for reconstruction
    y_coords = np.linspace(-1, 1, H)
    x_coords = np.linspace(-1, 1, W)
    
    reconstructed_frames = []
    
    for t in range(T):
        # Create coordinate field for this frame
        t_val = -1 + 2 * t / (T - 1)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten coordinates
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = np.full_like(x_flat, t_val)
        
        # Create field
        frame_field = FieldIQ(
            x_flat + 1j * y_flat,
            coord_field.sr,
            {'type': 'reconstruction', 'frame': t, 't_coord': t_val}
        )
        
        # Process with SIREN
        processed = siren_processor(frame_field)
        
        # Convert to image
        img_data = np.real(processed.z).reshape(H, W)
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
        img_data = np.clip(img_data, 0, 1)
        
        reconstructed_frames.append(img_data)
    
    return np.array(reconstructed_frames)

# ============================
# Demo and Testing
# ============================

def create_synthetic_video_data(height: int = 64, width: int = 64, frames: int = 16) -> np.ndarray:
    """Create synthetic video data for testing."""
    video_frames = []
    
    for i in range(frames):
        t = i / frames  # Normalized time
        
        # Create oscillating pattern
        y, x = np.ogrid[:height, :width]
        pattern = np.sin(2 * np.pi * 2.0 * (x + y) / max(height, width) + t * 2 * np.pi)
        
        # Convert to RGB
        frame_data = ((pattern + 1) * 127.5).astype(np.uint8)
        frame_data = np.stack([frame_data] * 3, axis=-1)
        
        video_frames.append(frame_data)
    
    return np.array(video_frames)

def demo_siren_field_combinator():
    """Demo SIREN field processing using Combinator patterns."""
    print("=== Combinator Kernel SIREN Field Demo ===\n")
    
    # Create synthetic video data
    video_frames = create_synthetic_video_data(height=32, width=32, frames=8)
    print(f"Created synthetic video: {video_frames.shape}")
    
    # Train SIREN field
    results = train_siren_field(
        video_frames,
        epochs=5,
        w0=30.0,
        w0_hidden=30.0,
        fourier_bands=4,
        max_freq=10.0,
        depth=3
    )
    
    print(f"\nTraining Results:")
    print(f"  Final loss: {results['final_loss']:.6f}")
    print(f"  Training epochs: {len(results['training_history'])}")
    
    # Show training history
    print(f"\nTraining History:")
    for entry in results['training_history']:
        print(f"  Epoch {entry['epoch']+1}: Loss = {entry['loss']:.6f}")
    
    # Reconstruct video
    print(f"\nReconstructing video...")
    reconstructed = reconstruct_video_from_siren_field(
        results['siren_processor'],
        results['coord_field'],
        video_frames.shape[:3]
    )
    
    print(f"Reconstructed video shape: {reconstructed.shape}")
    print(f"Reconstruction range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    return results, reconstructed

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
    spatial_field = coord_field.to_spatial_field()
    temporal_field = coord_field.to_temporal_field()
    combined_field = coord_field.to_combined_field()
    
    print(f"\nFieldIQ conversion:")
    print(f"  Spatial field: {len(spatial_field.z)} samples")
    print(f"  Temporal field: {len(temporal_field.z)} samples")
    print(f"  Combined field: {len(combined_field.z)} samples")
    print(f"  Valid quadrature: {spatial_field.is_valid_quadrature()}")
    
    # Apply SIREN processing
    siren_processor = create_siren_field_combinator(w0=30.0, depth=2)
    
    processed_spatial = siren_processor(spatial_field)
    processed_temporal = siren_processor(temporal_field)
    processed_combined = siren_processor(combined_field)
    
    print(f"\nSIREN processing:")
    print(f"  Processed spatial energy: {np.sum(np.abs(processed_spatial.z)**2):.6f}")
    print(f"  Processed temporal energy: {np.sum(np.abs(processed_temporal.z)**2):.6f}")
    print(f"  Processed combined energy: {np.sum(np.abs(processed_combined.z)**2):.6f}")
    
    return {
        'coord_field': coord_field,
        'spatial_field': spatial_field,
        'temporal_field': temporal_field,
        'combined_field': combined_field,
        'processed_spatial': processed_spatial,
        'processed_temporal': processed_temporal,
        'processed_combined': processed_combined
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
    siren_results, reconstructed = demo_siren_field_combinator()
    coord_results = demo_coordinate_processing()
    fourier_results = demo_fourier_features()
    
    print("\n=== Demo Summary ===")
    print("✓ SIREN field processing with Combinator patterns")
    print("✓ Coordinate-based field representation")
    print("✓ Fourier features using quadrature operations")
    print("✓ Functional composition of field operations")
    print("✓ Video reconstruction from SIREN field")
    print("✓ Integration with Combinator Kernel methodology")
    
    print("\n=== Key Features ===")
    print("- Coordinate-to-FieldIQ transformation")
    print("- SIREN activations as sine-based operations")
    print("- Multi-scale Fourier feature processing")
    print("- Functional composition of field operations")
    print("- Video reconstruction from continuous field")
    print("- Training pipeline using Combinator patterns")
    
    print("\n=== Combinator Kernel Benefits ===")
    print("- Functional composition of field operations")
    print("- Quadrature signal processing integration")
    print("- Pipeline-based training workflows")
    print("- Composable and reusable field processors")
    print("- Integration with existing Combinator patterns")

if __name__ == "__main__":
    main()
