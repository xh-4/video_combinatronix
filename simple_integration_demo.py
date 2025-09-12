# ============================
# Simple Integration Demo: Sin/Cos + Combinator Kernel
# ============================
"""
Simple demonstration of integrating the Combinator Kernel Sin/Cos processor
with basic video processing workflows.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any

from combinator_sincos_processor import (
    create_frequency_bank, oscillation_analyzer, temporal_reconstruction_pipeline,
    create_video_oscillation_pipeline, VideoFrame, VideoChunk
)
from Combinator_Kernel import (
    FieldIQ, make_field_from_real, load_video_chunks,
    video_frame_processor, video_temporal_processor,
    amp, lowpass_hz, phase_deg
)

def simple_oscillation_demo():
    """Simple demo of oscillation processing."""
    print("=== Simple Oscillation Processing Demo ===\n")
    
    # Create test signal
    sr = 48000.0
    t = np.linspace(0, 1.0, int(sr), endpoint=False)
    x = 0.5 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t)
    field = make_field_from_real(x, sr, tag=("test", "dual_tone"))
    
    print(f"Created test signal: {len(field.z)} samples at {sr} Hz")
    print(f"Valid quadrature: {field.is_valid_quadrature()}")
    
    # Create frequency bank
    bank = create_frequency_bank(K=4, T=8, sr=sr)
    print(f"Created frequency bank with {len(bank.frequencies)} frequencies")
    
    # Analyze oscillations
    analyzer = oscillation_analyzer(bank)
    analysis = analyzer(field)
    
    print(f"\nOscillation analysis:")
    print(f"  Frequencies: {analysis['frequencies']}")
    print(f"  Amplitude range: {np.min(np.abs(analysis['amplitude'])):.4f} - {np.max(np.abs(analysis['amplitude'])):.4f}")
    print(f"  Valid quadrature: {analysis['is_valid_quadrature']}")
    
    # Test reconstruction
    reconstruction_pipeline = temporal_reconstruction_pipeline(bank)
    reconstructed = reconstruction_pipeline(field)
    
    # Calculate reconstruction error
    reconstruction_error = np.mean(np.abs(field.z - reconstructed.z)**2)
    print(f"  Reconstruction error: {reconstruction_error:.6f}")
    
    return analysis, reconstructed

def video_processing_demo():
    """Demo video processing with oscillation analysis."""
    print("\n=== Video Processing Demo ===\n")
    
    # Create synthetic video data
    frames = create_synthetic_video_data(height=32, width=32, frames=16)
    chunk = VideoChunk(
        frames=frames,
        chunk_id=0,
        start_time=0.0,
        end_time=len(frames) / 30.0
    )
    
    print(f"Created video chunk with {len(frames)} frames")
    
    # Create processing pipeline
    pipeline = create_video_oscillation_pipeline(K=4, T=6, window_size=6, overlap=2)
    
    # Process chunk
    results = pipeline(chunk)
    
    print(f"Processed {len(results)} temporal windows")
    
    # Analyze results
    total_error = 0.0
    valid_count = 0
    
    for i, result in enumerate(results):
        error = result['reconstruction_error']
        is_valid = result['is_valid_quadrature']
        
        total_error += error
        if is_valid:
            valid_count += 1
        
        print(f"  Window {i}: Error = {error:.6f}, Valid = {is_valid}")
    
    avg_error = total_error / len(results)
    print(f"\nAverage reconstruction error: {avg_error:.6f}")
    print(f"Valid quadrature windows: {valid_count}/{len(results)}")
    
    return results

def create_synthetic_video_data(height: int = 32, width: int = 32, frames: int = 16) -> List[VideoFrame]:
    """Create synthetic video data for testing."""
    video_frames = []
    
    for i in range(frames):
        t = i / 30.0  # 30 FPS
        
        # Create oscillating pattern
        y, x = np.ogrid[:height, :width]
        pattern = np.sin(2 * np.pi * 2.0 * (x + y) / max(height, width) + t * 2 * np.pi)
        
        # Convert to RGB
        frame_data = ((pattern + 1) * 127.5).astype(np.uint8)
        frame_data = np.stack([frame_data] * 3, axis=-1)
        
        frame = VideoFrame(
            data=frame_data,
            frame_number=i,
            timestamp=i / 30.0,
            fps=30.0
        )
        video_frames.append(frame)
    
    return video_frames

def combinator_patterns_demo():
    """Demo basic Combinator patterns."""
    print("\n=== Combinator Patterns Demo ===\n")
    
    # Create test field
    sr = 48000.0
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    x = 0.5 * np.cos(2 * np.pi * 440 * t)
    field = make_field_from_real(x, sr, tag=("test", "sine_wave"))
    
    print(f"Original field energy: {np.sum(np.abs(field.z)**2):.6f}")
    
    # Apply different patterns
    amplified = amp(0.8)(field)
    lowpassed = lowpass_hz(1000.0)(field)
    phase_shifted = phase_deg(45.0, 440.0)(field)
    
    print(f"Amplified (0.8x) energy: {np.sum(np.abs(amplified.z)**2):.6f}")
    print(f"Lowpassed (1000Hz) energy: {np.sum(np.abs(lowpassed.z)**2):.6f}")
    print(f"Phase shifted (45°) energy: {np.sum(np.abs(phase_shifted.z)**2):.6f}")
    
    # Test composition
    composed = lowpass_hz(1000.0)(amp(0.8)(field))
    print(f"Composed (LP + amp) energy: {np.sum(np.abs(composed.z)**2):.6f}")
    
    return {
        'original': field,
        'amplified': amplified,
        'lowpassed': lowpassed,
        'phase_shifted': phase_shifted,
        'composed': composed
    }

def real_video_processing_demo():
    """Demo with real video files if available."""
    print("\n=== Real Video Processing Demo ===\n")
    
    # Look for video files
    video_files = list(Path(".").glob("*.mp4")) + list(Path(".").glob("*.avi"))
    
    if not video_files:
        print("No video files found - skipping real video demo")
        return None
    
    video_path = str(video_files[0])
    print(f"Processing video: {video_path}")
    
    # Create processing pipeline
    pipeline = create_video_oscillation_pipeline(K=6, T=8, window_size=8, overlap=4)
    
    # Process video
    results = []
    chunk_count = 0
    
    for chunk in load_video_chunks(video_path, chunk_size=16, overlap=4):
        print(f"Processing chunk {chunk_count} with {len(chunk.frames)} frames")
        
        chunk_results = pipeline(chunk)
        results.extend(chunk_results)
        chunk_count += 1
        
        # Limit to first few chunks for demo
        if chunk_count >= 3:
            break
    
    print(f"Processed {chunk_count} chunks, {len(results)} total windows")
    
    # Analyze results
    if results:
        total_error = sum(r['reconstruction_error'] for r in results)
        valid_count = sum(1 for r in results if r['is_valid_quadrature'])
        
        print(f"Average reconstruction error: {total_error / len(results):.6f}")
        print(f"Valid quadrature windows: {valid_count}/{len(results)}")
    
    return results

def main():
    """Main demo function."""
    print("=== Simple Integration Demo: Sin/Cos + Combinator Kernel ===\n")
    
    # Run demos
    oscillation_analysis, reconstructed = simple_oscillation_demo()
    video_results = video_processing_demo()
    combinator_results = combinator_patterns_demo()
    real_video_results = real_video_processing_demo()
    
    print("\n=== Demo Summary ===")
    print("✓ Oscillation analysis and reconstruction")
    print("✓ Video processing with temporal windows")
    print("✓ Combinator pattern application")
    if real_video_results:
        print("✓ Real video file processing")
    else:
        print("○ Real video processing (no video files found)")
    
    print("\n=== Key Features Demonstrated ===")
    print("- Quadrature signal decomposition using sin/cos frequency banks")
    print("- Temporal windowing for video processing")
    print("- Functional composition of signal processing operations")
    print("- Integration with Combinator Kernel patterns")
    print("- Reconstruction error analysis and validation")

if __name__ == "__main__":
    main()


