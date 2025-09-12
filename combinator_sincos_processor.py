# ============================
# Combinator Kernel Sin/Cos Video Processor
# ============================
"""
Implementation of the sin/cos network core workings using Combinator Kernel methodology.
This provides a functional, composable approach to temporal video analysis and reconstruction.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import cv2

# Import the Combinator Kernel
from Combinator_Kernel import (
    FieldIQ, make_field_from_real, 
    VideoFrame, VideoChunk, VideoStreamProcessor,
    Unary, Binary, Field,
    K, S, B, C, W, PLUS, TIMES,
    split_add, split_mul,
    gate_percent, fold_beats,
    phase_deg, freq_shift, amp, delay_ms, lowpass_hz,
    video_channel_processor, video_frame_processor, video_temporal_processor
)

# ============================
# Core Sin/Cos Combinator Operations
# ============================

@dataclass
class FrequencyBank:
    """Learned frequency bank for quadrature decomposition."""
    frequencies: np.ndarray  # (K,) frequencies in radians per sample
    sample_rate: float
    
    def __post_init__(self):
        self.K = len(self.frequencies)
        # Pre-compute sin/cos basis functions
        t = np.arange(self.K * 2)  # Extended for window processing
        self.sin_basis = np.sin(np.outer(self.frequencies, t))
        self.cos_basis = np.cos(np.outer(self.frequencies, t))

def create_frequency_bank(K: int, T: int, sr: float) -> FrequencyBank:
    """Create frequency bank with K frequencies covering 0.5*T range."""
    # Spread frequencies from slow to fast (similar to original)
    freqs = np.linspace(1.0, 0.5*T, K) * (2*np.pi/T)
    return FrequencyBank(freqs, sr)

def quadrature_decompose(field: FieldIQ, bank: FrequencyBank) -> Tuple[FieldIQ, FieldIQ]:
    """
    Decompose temporal field into quadrature components using frequency bank.
    Returns (I_component, Q_component) as FieldIQ objects.
    """
    z = field.z
    T = len(z)
    
    # Extract temporal window (assuming single channel for now)
    if len(z.shape) > 1:
        z = z.flatten()
    
    # Project onto sin/cos basis for each frequency
    q_components = np.zeros(bank.K, dtype=complex)
    p_components = np.zeros(bank.K, dtype=complex)
    
    for k in range(bank.K):
        # Use the full temporal window for projection
        window_size = min(T, len(bank.sin_basis[k]))
        window_data = z[:window_size]
        
        # Project onto sin/cos basis
        sin_basis = bank.sin_basis[k, :window_size]
        cos_basis = bank.cos_basis[k, :window_size]
        
        q_components[k] = np.sum(window_data * sin_basis)
        p_components[k] = np.sum(window_data * cos_basis)
    
    # Create FieldIQ objects for I and Q components
    I_field = FieldIQ(p_components, field.sr, {**field.roles, 'component': 'I'})
    Q_field = FieldIQ(q_components, field.sr, {**field.roles, 'component': 'Q'})
    
    return I_field, Q_field

def oscillation_analyzer(bank: FrequencyBank) -> Unary:
    """
    Create oscillation analyzer that extracts amplitude and phase from quadrature components.
    Returns a processor that takes a FieldIQ and returns amplitude/phase information.
    """
    def analyzer(field: FieldIQ) -> Dict[str, Any]:
        I_field, Q_field = quadrature_decompose(field, bank)
        
        # Compute amplitude and phase
        amplitude = np.sqrt(I_field.z * I_field.z + Q_field.z * Q_field.z + 1e-8)
        cos_phase = I_field.z / (amplitude + 1e-8)
        sin_phase = Q_field.z / (amplitude + 1e-8)
        
        return {
            'amplitude': amplitude,
            'cos_phase': cos_phase,
            'sin_phase': sin_phase,
            'I_component': I_field,
            'Q_component': Q_field,
            'frequencies': bank.frequencies,
            'is_valid_quadrature': I_field.is_valid_quadrature() and Q_field.is_valid_quadrature()
        }
    
    return analyzer

# ============================
# Temporal Window Processing
# ============================

def temporal_window_processor(window_size: int, overlap: int = 0) -> Callable[[VideoChunk], List[FieldIQ]]:
    """
    Process video chunks with temporal windowing using Combinator patterns.
    """
    def process_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        frames = chunk.frames
        if len(frames) < window_size:
            # Pad with last frame if needed
            last_frame = frames[-1] if frames else None
            while len(frames) < window_size and last_frame:
                frames.append(last_frame)
        
        # Create temporal windows
        windows = []
        for i in range(0, len(frames) - window_size + 1, max(1, window_size - overlap)):
            window_frames = frames[i:i + window_size]
            
            # Convert window to temporal FieldIQ
            temporal_data = []
            for frame in window_frames:
                field = frame.to_field_iq(channel=0, sr=48000.0)
                temporal_data.append(field.z)
            
            # Concatenate temporal data
            if temporal_data:
                z_concat = np.concatenate(temporal_data)
                window_field = FieldIQ(
                    z_concat, 
                    48000.0, 
                    {'window_id': i, 'window_size': window_size, 'video_chunk': chunk.chunk_id}
                )
                windows.append(window_field)
        
        return windows
    
    return process_chunk

def multi_scale_oscillation_processor(frequencies_list: List[List[float]], sr: float = 48000.0) -> Unary:
    """
    Create multi-scale oscillation processor using multiple frequency banks.
    """
    banks = [FrequencyBank(np.array(freqs), sr) for freqs in frequencies_list]
    
    def processor(field: FieldIQ) -> Dict[str, Any]:
        results = {}
        
        for i, bank in enumerate(banks):
            analyzer = oscillation_analyzer(bank)
            analysis = analyzer(field)
            results[f'scale_{i}'] = analysis
        
        # Combine results across scales
        combined_amplitude = np.concatenate([results[f'scale_{i}']['amplitude'] for i in range(len(banks))])
        combined_cos_phase = np.concatenate([results[f'scale_{i}']['cos_phase'] for i in range(len(banks))])
        combined_sin_phase = np.concatenate([results[f'scale_{i}']['sin_phase'] for i in range(len(banks))])
        
        results['combined'] = {
            'amplitude': combined_amplitude,
            'cos_phase': combined_cos_phase,
            'sin_phase': combined_sin_phase,
            'total_frequencies': sum(len(bank.frequencies) for bank in banks)
        }
        
        return results
    
    return processor

# ============================
# Reconstruction Pipeline
# ============================

def quadrature_synthesizer(bank: FrequencyBank, original_length: int) -> Callable[[Dict[str, Any]], FieldIQ]:
    """
    Create synthesizer that reconstructs signal from quadrature components.
    """
    def synthesizer(analysis: Dict[str, Any]) -> FieldIQ:
        amplitude = analysis['amplitude']
        cos_phase = analysis['cos_phase']
        sin_phase = analysis['sin_phase']
        
        # Reconstruct I and Q components
        I_reconstructed = amplitude * cos_phase
        Q_reconstructed = amplitude * sin_phase
        
        # Expand back to original temporal length using inverse projection
        reconstructed_z = np.zeros(original_length, dtype=complex)
        
        for k in range(len(amplitude)):
            # Use sin/cos basis to reconstruct temporal signal
            window_size = min(original_length, len(bank.sin_basis[k]))
            sin_basis = bank.sin_basis[k, :window_size]
            cos_basis = bank.cos_basis[k, :window_size]
            
            # Add contribution from this frequency
            reconstructed_z[:window_size] += I_reconstructed[k] * cos_basis + 1j * Q_reconstructed[k] * sin_basis
        
        reconstructed_field = FieldIQ(
            reconstructed_z, 
            bank.sample_rate, 
            {**analysis.get('roles', {}), 'reconstructed': True}
        )
        
        return reconstructed_field
    
    return synthesizer

def temporal_reconstruction_pipeline(bank: FrequencyBank) -> Unary:
    """
    Create complete reconstruction pipeline using Combinator patterns.
    """
    analyzer = oscillation_analyzer(bank)
    
    def pipeline(field: FieldIQ) -> FieldIQ:
        # Analyze oscillations
        analysis = analyzer(field)
        
        # Create synthesizer with original length
        synthesizer = quadrature_synthesizer(bank, len(field.z))
        
        # Synthesize reconstruction
        reconstructed = synthesizer(analysis)
        
        return reconstructed
    
    return pipeline

# ============================
# Video Processing Pipeline
# ============================

def create_video_oscillation_pipeline(
    K: int = 6, 
    T: int = 8, 
    sr: float = 48000.0,
    window_size: int = 8,
    overlap: int = 2
) -> Callable[[VideoChunk], List[Dict[str, Any]]]:
    """
    Create complete video oscillation processing pipeline.
    """
    # Create frequency bank
    bank = create_frequency_bank(K, T, sr)
    
    # Create processors
    window_processor = temporal_window_processor(window_size, overlap)
    oscillation_processor = oscillation_analyzer(bank)
    reconstruction_pipeline = temporal_reconstruction_pipeline(bank)
    
    def process_video_chunk(chunk: VideoChunk) -> List[Dict[str, Any]]:
        # Process temporal windows
        windows = window_processor(chunk)
        
        results = []
        for window in windows:
            # Analyze oscillations
            analysis = oscillation_processor(window)
            
            # Reconstruct
            reconstructed = reconstruction_pipeline(window)
            
            # Combine results
            result = {
                'original_window': window,
                'reconstructed': reconstructed,
                'oscillation_analysis': analysis,
                'reconstruction_error': np.mean(np.abs(window.z - reconstructed.z)**2),
                'is_valid_quadrature': window.is_valid_quadrature()
            }
            results.append(result)
        
        return results
    
    return process_video_chunk

# ============================
# Demo and Testing
# ============================

def create_synthetic_video_data(height: int = 64, width: int = 64, frames: int = 30) -> List[VideoFrame]:
    """Create synthetic video data for testing."""
    video_frames = []
    
    for i in range(frames):
        # Create oscillating pattern
        t = i / 30.0  # 30 FPS
        frequency = 2.0 + 0.5 * np.sin(t * 0.5)  # Varying frequency
        
        # Create frame with oscillating pattern
        y, x = np.ogrid[:height, :width]
        pattern = np.sin(2 * np.pi * frequency * (x + y) / max(height, width) + t * 2 * np.pi)
        
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

def demo_oscillation_processing():
    """Demo the oscillation processing pipeline."""
    print("=== Combinator Kernel Sin/Cos Video Processor Demo ===\n")
    
    # Create synthetic video data
    print("Creating synthetic video data...")
    frames = create_synthetic_video_data(height=32, width=32, frames=20)
    chunk = VideoChunk(
        frames=frames,
        chunk_id=0,
        start_time=0.0,
        end_time=len(frames) / 30.0
    )
    print(f"Created video chunk with {len(frames)} frames")
    
    # Create processing pipeline
    print("\nSetting up oscillation processing pipeline...")
    pipeline = create_video_oscillation_pipeline(K=4, T=6, window_size=6, overlap=2)
    
    # Process video
    print("Processing video chunk...")
    results = pipeline(chunk)
    
    # Analyze results
    print(f"\nProcessed {len(results)} temporal windows")
    
    total_error = 0.0
    valid_quadrature_count = 0
    
    for i, result in enumerate(results):
        error = result['reconstruction_error']
        is_valid = result['is_valid_quadrature']
        
        total_error += error
        if is_valid:
            valid_quadrature_count += 1
        
        print(f"Window {i:2d}: Reconstruction error = {error:.6f}, Valid quadrature = {is_valid}")
    
    avg_error = total_error / len(results)
    print(f"\nAverage reconstruction error: {avg_error:.6f}")
    print(f"Valid quadrature windows: {valid_quadrature_count}/{len(results)}")
    
    # Show oscillation analysis for first window
    if results:
        analysis = results[0]['oscillation_analysis']
        print(f"\nFirst window oscillation analysis:")
        print(f"  Frequencies: {len(analysis['frequencies'])}")
        print(f"  Amplitude range: {np.min(analysis['amplitude']):.4f} - {np.max(analysis['amplitude']):.4f}")
        print(f"  Valid quadrature: {analysis['is_valid_quadrature']}")
    
    return results

def demo_combinator_patterns():
    """Demo various Combinator patterns applied to video processing."""
    print("\n=== Combinator Patterns Demo ===\n")
    
    # Create a simple test field
    sr = 48000.0
    t = np.linspace(0, 1.0, int(sr), endpoint=False)
    x = 0.5 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t)
    field = make_field_from_real(x, sr, tag=("test", "dual_tone"))
    
    print(f"Original field: {len(field.z)} samples, SR = {field.sr}")
    print(f"Valid quadrature: {field.is_valid_quadrature()}")
    
    # Create frequency bank
    bank = create_frequency_bank(K=4, T=8, sr=sr)
    print(f"Frequency bank: {len(bank.frequencies)} frequencies")
    
    # Apply oscillation analysis
    analyzer = oscillation_analyzer(bank)
    analysis = analyzer(field)
    
    print(f"Oscillation analysis:")
    print(f"  Amplitude: {analysis['amplitude']}")
    print(f"  Valid quadrature: {analysis['is_valid_quadrature']}")
    
    # Apply Combinator patterns
    print(f"\nApplying Combinator patterns:")
    
    # Split-add pattern
    split_add_processor = split_add(amp(0.5))
    processed_field = split_add_processor(field)
    print(f"  Split-add: {len(processed_field.z)} samples")
    
    # Function composition
    composed_processor = B(lowpass_hz(1000.0))(amp(0.8))
    composed_result = composed_processor(field)
    print(f"  Composed (LP + amp): {len(composed_result.z)} samples")
    
    # Temporal gating
    gate_processor = gate_percent(50.0)
    gated_result = gate_processor(field)
    print(f"  Gated (50%): {len(gated_result.z)} samples")

if __name__ == "__main__":
    # Run demos
    demo_oscillation_processing()
    demo_combinator_patterns()
    
    print("\n=== Demo Complete ===")
    print("The Combinator Kernel Sin/Cos processor provides:")
    print("- Functional composition of video processing operations")
    print("- Quadrature signal decomposition and reconstruction")
    print("- Temporal windowing with overlap support")
    print("- Multi-scale oscillation analysis")
    print("- Integration with existing Combinator Kernel patterns")
