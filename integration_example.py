# ============================
# Integration Example: Sin/Cos Network + Combinator Kernel
# ============================
"""
Example showing how to integrate the Combinator Kernel Sin/Cos processor
with existing video processing workflows.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple

from combinator_sincos_processor import (
    create_frequency_bank, oscillation_analyzer, temporal_reconstruction_pipeline,
    create_video_oscillation_pipeline, VideoFrame, VideoChunk
)
from Combinator_Kernel import (
    FieldIQ, make_field_from_real, load_video_chunks,
    video_frame_processor, video_temporal_processor,
    B, S, split_add, amp, lowpass_hz, phase_deg
)

def create_enhanced_oscillation_processor(K: int = 6, T: int = 8) -> Dict[str, Any]:
    """
    Create an enhanced oscillation processor that combines multiple Combinator patterns.
    """
    # Create frequency bank
    bank = create_frequency_bank(K, T, 48000.0)
    
    # Create base oscillation analyzer
    base_analyzer = oscillation_analyzer(bank)
    
    # Create enhanced processors using Combinator patterns
    enhanced_processor = B(lowpass_hz(1000.0))(  # Low-pass filter
        split_add(  # Split-add pattern
            B(amp(0.8))(  # Amplitude scaling
                phase_deg(45.0, 440.0)  # Phase shift
            )
        )
    )
    
    def enhanced_analysis(field: FieldIQ) -> Dict[str, Any]:
        # Get base oscillation analysis
        base_analysis = base_analyzer(field)
        
        # Apply enhanced processing
        enhanced_field = enhanced_processor(field)
        
        # Combine results
        return {
            'base_analysis': base_analysis,
            'enhanced_field': enhanced_field,
            'processing_applied': 'lowpass + split_add + amp + phase_shift',
            'original_energy': np.sum(np.abs(field.z)**2),
            'enhanced_energy': np.sum(np.abs(enhanced_field.z)**2)
        }
    
    return {
        'analyzer': enhanced_analysis,
        'bank': bank,
        'processor': enhanced_processor
    }

def process_video_with_enhanced_oscillations(video_path: str, output_dir: str = "enhanced_output"):
    """
    Process video with enhanced oscillation analysis using Combinator patterns.
    """
    print(f"Processing video with enhanced oscillations: {video_path}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create enhanced processor
    enhanced_proc = create_enhanced_oscillation_processor(K=6, T=8)
    
    # Create video processing pipeline
    def process_chunk(chunk: VideoChunk) -> List[Dict[str, Any]]:
        results = []
        
        # Convert chunk to temporal FieldIQ
        temporal_processor = video_temporal_processor(
            enhanced_proc['analyzer'],
            channel=0,
            sr=48000.0
        )
        
        # Process temporal sequence
        analysis = temporal_processor(chunk)
        if analysis:
            results.append({
                'chunk_id': chunk.chunk_id,
                'analysis': analysis,
                'frame_count': len(chunk.frames),
                'duration': chunk.duration
            })
        
        return results
    
    # Process video
    all_results = []
    chunk_count = 0
    
    for chunk in load_video_chunks(video_path, chunk_size=16, overlap=4):
        print(f"Processing chunk {chunk_count}")
        chunk_results = process_chunk(chunk)
        all_results.extend(chunk_results)
        chunk_count += 1
    
    print(f"Processed {chunk_count} chunks")
    
    # Analyze and save results
    analyze_enhanced_results(all_results, output_dir)
    
    return all_results

def analyze_enhanced_results(results: List[Dict], output_dir: str):
    """Analyze enhanced processing results."""
    print("\n=== Enhanced Processing Analysis ===")
    
    if not results:
        print("No results to analyze")
        return
    
    # Collect statistics
    original_energies = []
    enhanced_energies = []
    energy_ratios = []
    
    for result in results:
        analysis = result['analysis']
        orig_energy = analysis['original_energy']
        enh_energy = analysis['enhanced_energy']
        
        original_energies.append(orig_energy)
        enhanced_energies.append(enh_energy)
        energy_ratios.append(enh_energy / (orig_energy + 1e-8))
    
    print(f"Processed {len(results)} chunks")
    print(f"Average original energy: {np.mean(original_energies):.6f}")
    print(f"Average enhanced energy: {np.mean(enhanced_energies):.6f}")
    print(f"Average energy ratio: {np.mean(energy_ratios):.4f}")
    
    # Save detailed analysis
    save_analysis_report(results, output_dir)

def save_analysis_report(results: List[Dict], output_dir: str):
    """Save detailed analysis report."""
    report_path = Path(output_dir) / "enhanced_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Enhanced Oscillation Processing Report\n")
        f.write("=" * 50 + "\n\n")
        
        for i, result in enumerate(results):
            f.write(f"Chunk {i}:\n")
            f.write(f"  Frame count: {result['frame_count']}\n")
            f.write(f"  Duration: {result['duration']:.3f}s\n")
            
            analysis = result['analysis']
            base_analysis = analysis['base_analysis']
            
            f.write(f"  Original energy: {analysis['original_energy']:.6f}\n")
            f.write(f"  Enhanced energy: {analysis['enhanced_energy']:.6f}\n")
            f.write(f"  Processing: {analysis['processing_applied']}\n")
            f.write(f"  Valid quadrature: {base_analysis['is_valid_quadrature']}\n")
            f.write(f"  Frequency count: {len(base_analysis['frequencies'])}\n")
            f.write("\n")
    
    print(f"Saved analysis report: {report_path}")

def create_comparison_processor():
    """
    Create a processor that compares different Combinator patterns.
    """
    def comparison_processor(field: FieldIQ) -> Dict[str, Any]:
        # Original field
        original = field
        
        # Different processing chains
        lowpass_only = B(lowpass_hz(1000.0))(field)
        amp_only = amp(0.8)(field)
        phase_only = phase_deg(45.0, 440.0)(field)
        
        # Combined processing
        combined = B(lowpass_hz(1000.0))(
            B(amp(0.8))(
                phase_deg(45.0, 440.0)
            )
        )(field)
        
        # Split-add processing
        split_add_proc = split_add(amp(0.5))(field)
        
        return {
            'original': original,
            'lowpass_only': lowpass_only,
            'amp_only': amp_only,
            'phase_only': phase_only,
            'combined': combined,
            'split_add': split_add_proc,
            'comparison_metrics': {
                'original_energy': np.sum(np.abs(original.z)**2),
                'lowpass_energy': np.sum(np.abs(lowpass_only.z)**2),
                'amp_energy': np.sum(np.abs(amp_only.z)**2),
                'phase_energy': np.sum(np.abs(phase_only.z)**2),
                'combined_energy': np.sum(np.abs(combined.z)**2),
                'split_add_energy': np.sum(np.abs(split_add_proc.z)**2)
            }
        }
    
    return comparison_processor

def demo_combinator_comparison():
    """Demo different Combinator patterns on synthetic data."""
    print("\n=== Combinator Pattern Comparison Demo ===")
    
    # Create test signal
    sr = 48000.0
    t = np.linspace(0, 1.0, int(sr), endpoint=False)
    x = 0.5 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t)
    field = make_field_from_real(x, sr, tag=("test", "dual_tone"))
    
    # Create comparison processor
    comparison_proc = create_comparison_processor()
    results = comparison_proc(field)
    
    # Display results
    print("Processing comparison results:")
    metrics = results['comparison_metrics']
    
    for name, energy in metrics.items():
        print(f"  {name}: {energy:.6f}")
    
    # Calculate energy ratios
    original_energy = metrics['original_energy']
    print(f"\nEnergy ratios (relative to original):")
    for name, energy in metrics.items():
        if name != 'original_energy':
            ratio = energy / original_energy
            print(f"  {name}: {ratio:.4f}")

def main():
    """Main demo function."""
    print("=== Integration Example: Sin/Cos + Combinator Kernel ===\n")
    
    # Demo combinator comparison
    demo_combinator_comparison()
    
    # Check for video files
    video_files = list(Path(".").glob("*.mp4")) + list(Path(".").glob("*.avi"))
    
    if video_files:
        print(f"\nFound video files: {[f.name for f in video_files]}")
        video_path = str(video_files[0])
        
        # Process with enhanced oscillations
        results = process_video_with_enhanced_oscillations(video_path)
        
        print(f"\nProcessed video with enhanced oscillation analysis")
        print(f"Results saved to: enhanced_output/")
    else:
        print("\nNo video files found - run with video files to see full processing")
    
    print("\n=== Integration Demo Complete ===")
    print("This example shows how to:")
    print("- Combine oscillation analysis with Combinator patterns")
    print("- Process video streams with enhanced signal processing")
    print("- Compare different processing approaches")
    print("- Generate detailed analysis reports")

if __name__ == "__main__":
    main()
