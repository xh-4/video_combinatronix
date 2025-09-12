#!/usr/bin/env python3
"""
Audio processing demo using the Combinator Kernel
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Combinator_Kernel import (
    make_field_from_real, FieldIQ,
    lowpass_hz, amp, phase_deg, freq_shift, delay_ms,
    split_add, split_mul, B, W, PLUS, TIMES
)

def demo_audio_processing():
    """Demonstrate audio processing capabilities"""
    print("=== Audio Processing Demo ===\n")
    
    # Create test signals
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Create a complex audio signal
    carrier = 0.8 * np.cos(2 * np.pi * 440 * t)  # 440 Hz tone
    modulator = 0.3 * np.cos(2 * np.pi * 5 * t)  # 5 Hz modulation
    noise = 0.1 * np.random.randn(len(t))        # Noise
    
    # Combine signals
    audio_signal = carrier + modulator + noise
    
    print("1. Basic Signal Processing")
    print("-" * 30)
    
    # Convert to FieldIQ
    field = make_field_from_real(audio_signal, sr, tag=("role", "audio"))
    print(f"Signal length: {len(field.z)} samples")
    print(f"Sample rate: {field.sr} Hz")
    print(f"Valid quadrature: {field.is_valid_quadrature()}")
    print(f"RMS: {np.sqrt(np.mean(field.power)):.4f}")
    
    print("\n2. Filtering and Amplitude")
    print("-" * 30)
    
    # Apply lowpass filter and amplitude scaling
    filtered = B(lowpass_hz(1000.0))(amp(0.7))(field)
    print(f"Filtered RMS: {np.sqrt(np.mean(filtered.power)):.4f}")
    
    print("\n3. Phase and Frequency Operations")
    print("-" * 30)
    
    # Phase shift
    phase_shifted = field.rotate(45.0)
    print(f"Phase shifted by 45°")
    
    # Frequency shift
    freq_shifted = field.freq_shift(100.0)  # Shift up by 100 Hz
    print(f"Frequency shifted by 100 Hz")
    
    print("\n4. Combinator Processing")
    print("-" * 30)
    
    # Create effect chains using combinators
    chorus = split_add(delay_ms(20.0))  # Short delay for chorus
    tremolo = split_mul(freq_shift(5.0))  # Ring modulation for tremolo
    
    # Apply effects
    chorused = chorus(field)
    tremoloed = tremolo(field)
    
    print(f"Chorus effect applied")
    print(f"Tremolo effect applied")
    
    print("\n5. Complex Effect Chains")
    print("-" * 30)
    
    # Create a complex processing chain
    effect_chain = B(lowpass_hz(800.0))(
        B(amp(0.8))(
            split_add(phase_deg(90.0, 440.0))
        )
    )
    
    processed = effect_chain(field)
    print(f"Complex effect chain applied")
    print(f"Processed RMS: {np.sqrt(np.mean(processed.power)):.4f}")
    
    print("\n6. Spectral Analysis")
    print("-" * 30)
    
    # Analyze frequency content
    fft_data = np.fft.fft(field.z)
    freqs = np.fft.fftfreq(len(field.z), 1/sr)
    
    # Find dominant frequency
    dominant_freq = freqs[np.argmax(np.abs(fft_data))]
    print(f"Dominant frequency: {dominant_freq:.2f} Hz")
    
    # Calculate spectral centroid
    spectral_centroid = np.sum(freqs * np.abs(fft_data)) / np.sum(np.abs(fft_data))
    print(f"Spectral centroid: {spectral_centroid:.2f} Hz")
    
    print("\n7. Multiple Signal Processing")
    print("-" * 30)
    
    # Create multiple signals
    signal1 = make_field_from_real(np.cos(2 * np.pi * 220 * t), sr, tag=("freq", 220))
    signal2 = make_field_from_real(np.cos(2 * np.pi * 330 * t), sr, tag=("freq", 330))
    
    # Mix signals using combinators
    mixed = PLUS(signal1)(signal2)
    print(f"Mixed two signals")
    print(f"Mixed RMS: {np.sqrt(np.mean(mixed.power)):.4f}")
    
    # Ring modulation
    ring_mod = TIMES(signal1)(signal2)
    print(f"Ring modulation applied")
    print(f"Ring mod RMS: {np.sqrt(np.mean(ring_mod.power)):.4f}")
    
    print("\n8. Role-based Processing")
    print("-" * 30)
    
    # Tag signals with roles
    guitar = field.with_role("instrument", "guitar")
    guitar = guitar.with_role("effect", "distortion")
    
    bass = signal1.with_role("instrument", "bass")
    bass = bass.with_role("effect", "compression")
    
    print(f"Guitar roles: {guitar.roles}")
    print(f"Bass roles: {bass.roles}")
    
    print("\n🎉 Audio processing demo completed!")

if __name__ == "__main__":
    demo_audio_processing()



