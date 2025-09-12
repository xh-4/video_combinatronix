# ============================
# Video Oscillation Processing Demo
# ============================
"""
Demonstration of using the Combinator Kernel Sin/Cos processor with real video files.
Shows how to process video streams and extract temporal oscillations.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt

from combinator_sincos_processor import (
    create_frequency_bank, oscillation_analyzer, temporal_reconstruction_pipeline,
    create_video_oscillation_pipeline, VideoFrame, VideoChunk
)
from Combinator_Kernel import load_video_chunks, process_video_stream

def process_video_file(video_path: str, output_dir: str = "oscillation_output"):
    """
    Process a video file and extract oscillation patterns.
    """
    print(f"Processing video: {video_path}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create processing pipeline
    pipeline = create_video_oscillation_pipeline(
        K=6,           # 6 frequency bands
        T=8,           # 8-frame temporal window
        window_size=8, # 8-frame processing windows
        overlap=4      # 50% overlap
    )
    
    # Process video chunks
    results = []
    chunk_count = 0
    
    for chunk in load_video_chunks(video_path, chunk_size=16, overlap=4):
        print(f"Processing chunk {chunk_count} with {len(chunk.frames)} frames")
        
        # Process chunk
        chunk_results = pipeline(chunk)
        results.extend(chunk_results)
        
        # Save sample frames
        if chunk_count < 3:  # Save first 3 chunks
            save_sample_frames(chunk, chunk_results, output_dir, chunk_count)
        
        chunk_count += 1
    
    print(f"Processed {chunk_count} chunks, {len(results)} total windows")
    
    # Analyze results
    analyze_oscillation_results(results, output_dir)
    
    return results

def save_sample_frames(chunk: VideoChunk, results: List[Dict], output_dir: str, chunk_id: int):
    """Save sample frames showing original vs reconstructed."""
    for i, result in enumerate(results[:3]):  # Save first 3 windows
        # Get original window data
        original_window = result['original_window']
        reconstructed = result['reconstructed']
        
        # Convert back to image format for visualization
        original_img = reconstruct_image_from_field(original_window, chunk.frames[0].height, chunk.frames[0].width)
        recon_img = reconstruct_image_from_field(reconstructed, chunk.frames[0].height, chunk.frames[0].width)
        
        # Create side-by-side comparison
        comparison = np.concatenate([original_img, recon_img], axis=1)
        
        # Save image
        output_path = Path(output_dir) / f"chunk_{chunk_id}_window_{i}_comparison.png"
        cv2.imwrite(str(output_path), comparison)
        
        print(f"Saved comparison: {output_path}")

def reconstruct_image_from_field(field, height: int, width: int) -> np.ndarray:
    """Reconstruct image from FieldIQ data."""
    # Get magnitude of the field
    magnitude = np.abs(field.z)
    
    # Reshape to image dimensions
    if len(magnitude) >= height * width:
        img_data = magnitude[:height * width].reshape(height, width)
    else:
        # Pad if needed
        img_data = np.pad(magnitude, (0, height * width - len(magnitude)), mode='edge')
        img_data = img_data.reshape(height, width)
    
    # Normalize to 0-255 range
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
    img_data = (img_data * 255).astype(np.uint8)
    
    return img_data

def analyze_oscillation_results(results: List[Dict], output_dir: str):
    """Analyze and visualize oscillation results."""
    print("\n=== Oscillation Analysis ===")
    
    # Collect statistics
    reconstruction_errors = [r['reconstruction_error'] for r in results]
    valid_quadrature_count = sum(1 for r in results if r['is_valid_quadrature'])
    
    print(f"Total windows processed: {len(results)}")
    print(f"Valid quadrature windows: {valid_quadrature_count}/{len(results)}")
    print(f"Average reconstruction error: {np.mean(reconstruction_errors):.6f}")
    print(f"Min reconstruction error: {np.min(reconstruction_errors):.6f}")
    print(f"Max reconstruction error: {np.max(reconstruction_errors):.6f}")
    
    # Analyze frequency content
    all_amplitudes = []
    all_frequencies = []
    
    for result in results:
        analysis = result['oscillation_analysis']
        all_amplitudes.extend(analysis['amplitude'])
        all_frequencies.extend(analysis['frequencies'])
    
    print(f"\nFrequency analysis:")
    print(f"  Frequency range: {np.min(all_frequencies):.4f} - {np.max(all_frequencies):.4f} rad/sample")
    print(f"  Amplitude range: {np.min(np.abs(all_amplitudes)):.6f} - {np.max(np.abs(all_amplitudes)):.6f}")
    
    # Create visualization
    create_oscillation_plots(results, output_dir)

def create_oscillation_plots(results: List[Dict], output_dir: str):
    """Create plots showing oscillation patterns."""
    try:
        import matplotlib.pyplot as plt
        
        # Plot reconstruction errors over time
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Reconstruction errors
        plt.subplot(2, 2, 1)
        reconstruction_errors = [r['reconstruction_error'] for r in results]
        plt.plot(reconstruction_errors, 'b-', alpha=0.7)
        plt.title('Reconstruction Error Over Time')
        plt.xlabel('Window Index')
        plt.ylabel('MSE')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Amplitude distribution
        plt.subplot(2, 2, 2)
        all_amplitudes = []
        for result in results:
            analysis = result['oscillation_analysis']
            all_amplitudes.extend(np.abs(analysis['amplitude']))
        
        plt.hist(all_amplitudes, bins=30, alpha=0.7, color='green')
        plt.title('Amplitude Distribution')
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Frequency content
        plt.subplot(2, 2, 3)
        all_frequencies = []
        for result in results:
            analysis = result['oscillation_analysis']
            all_frequencies.extend(analysis['frequencies'])
        
        plt.plot(all_frequencies, 'ro-', alpha=0.7)
        plt.title('Frequency Content')
        plt.xlabel('Frequency Index')
        plt.ylabel('Frequency (rad/sample)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Quadrature validity
        plt.subplot(2, 2, 4)
        valid_counts = [1 if r['is_valid_quadrature'] else 0 for r in results]
        plt.plot(valid_counts, 'go-', alpha=0.7)
        plt.title('Quadrature Validity')
        plt.xlabel('Window Index')
        plt.ylabel('Valid (1) / Invalid (0)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path(output_dir) / "oscillation_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved oscillation analysis plot: {plot_path}")
        
    except ImportError:
        print("Matplotlib not available - skipping plots")

def demo_with_synthetic_video():
    """Demo with synthetic video data."""
    print("=== Synthetic Video Demo ===")
    
    # Create synthetic video
    frames = create_synthetic_video_data(height=64, width=64, frames=30)
    chunk = VideoChunk(
        frames=frames,
        chunk_id=0,
        start_time=0.0,
        end_time=len(frames) / 30.0
    )
    
    # Process with oscillation pipeline
    pipeline = create_video_oscillation_pipeline(K=4, T=6, window_size=6, overlap=2)
    results = pipeline(chunk)
    
    # Analyze results
    analyze_oscillation_results(results, "synthetic_output")
    
    return results

def create_synthetic_video_data(height: int = 64, width: int = 64, frames: int = 30) -> List[VideoFrame]:
    """Create synthetic video data with complex oscillating patterns."""
    video_frames = []
    
    for i in range(frames):
        t = i / 30.0  # 30 FPS
        
        # Create complex oscillating pattern
        y, x = np.ogrid[:height, :width]
        
        # Multiple frequency components
        pattern1 = np.sin(2 * np.pi * 2.0 * (x + y) / max(height, width) + t * 2 * np.pi)
        pattern2 = 0.5 * np.sin(2 * np.pi * 4.0 * (x - y) / max(height, width) + t * 3 * np.pi)
        pattern3 = 0.3 * np.cos(2 * np.pi * 1.0 * (x * y) / (height * width) + t * 1.5 * np.pi)
        
        # Combine patterns
        combined_pattern = pattern1 + pattern2 + pattern3
        
        # Add some noise
        noise = 0.1 * np.random.randn(height, width)
        final_pattern = combined_pattern + noise
        
        # Convert to RGB
        frame_data = ((final_pattern + 2) * 127.5).clip(0, 255).astype(np.uint8)
        frame_data = np.stack([frame_data] * 3, axis=-1)
        
        frame = VideoFrame(
            data=frame_data,
            frame_number=i,
            timestamp=i / 30.0,
            fps=30.0
        )
        video_frames.append(frame)
    
    return video_frames

def main():
    """Main demo function."""
    print("=== Video Oscillation Processing Demo ===\n")
    
    # Check if we have a video file to process
    video_files = list(Path(".").glob("*.mp4")) + list(Path(".").glob("*.avi"))
    
    if video_files:
        print(f"Found video files: {[f.name for f in video_files]}")
        video_path = str(video_files[0])
        results = process_video_file(video_path)
    else:
        print("No video files found, using synthetic data...")
        results = demo_with_synthetic_video()
    
    print("\n=== Demo Complete ===")
    print("Check the output directory for:")
    print("- Comparison images (original vs reconstructed)")
    print("- Oscillation analysis plots")
    print("- Processing statistics")

if __name__ == "__main__":
    main()


