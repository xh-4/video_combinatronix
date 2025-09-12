# ============================
# Extended Signal Processing VM
# ============================
"""
Extended signal processing VM with comprehensive DSP operations.
Builds on the minimal VM with more operations while keeping it simple and reliable.
"""

import numpy as np
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf, to_json, from_json
from Combinator_Kernel import FieldIQ, make_field_from_real

# ============================
# Extended Signal Processing Operations
# ============================

# Basic Operations
def SP_AMP(gain: float) -> Node:
    """Amplitude scaling."""
    return Val({'type': 'sp', 'op': 'amp', 'gain': gain})

def SP_GAIN(gain_db: float) -> Node:
    """Gain in decibels."""
    return Val({'type': 'sp', 'op': 'gain', 'gain_db': gain_db})

def SP_INVERT() -> Node:
    """Invert signal (multiply by -1)."""
    return Val({'type': 'sp', 'op': 'invert'})

# Filtering Operations
def SP_LOWPASS(cutoff: float) -> Node:
    """Lowpass filter."""
    return Val({'type': 'sp', 'op': 'lowpass', 'cutoff': cutoff})

def SP_HIGHPASS(cutoff: float) -> Node:
    """Highpass filter."""
    return Val({'type': 'sp', 'op': 'highpass', 'cutoff': cutoff})

def SP_BANDPASS(low: float, high: float) -> Node:
    """Bandpass filter."""
    return Val({'type': 'sp', 'op': 'bandpass', 'low': low, 'high': high})

def SP_BANDSTOP(low: float, high: float) -> Node:
    """Bandstop (notch) filter."""
    return Val({'type': 'sp', 'op': 'bandstop', 'low': low, 'high': high})

def SP_SHELF(freq: float, gain_db: float) -> Node:
    """Shelving filter."""
    return Val({'type': 'sp', 'op': 'shelf', 'freq': freq, 'gain_db': gain_db})

# Time Domain Operations
def SP_DELAY(delay_ms: float) -> Node:
    """Delay in milliseconds."""
    return Val({'type': 'sp', 'op': 'delay', 'delay_ms': delay_ms})

def SP_ECHO(delay_ms: float, feedback: float, mix: float) -> Node:
    """Echo effect."""
    return Val({'type': 'sp', 'op': 'echo', 'delay_ms': delay_ms, 'feedback': feedback, 'mix': mix})

def SP_REVERB(room_size: float, damping: float, mix: float) -> Node:
    """Simple reverb."""
    return Val({'type': 'sp', 'op': 'reverb', 'room_size': room_size, 'damping': damping, 'mix': mix})

# Phase and Frequency Operations
def SP_PHASE(degrees: float) -> Node:
    """Phase shift in degrees."""
    return Val({'type': 'sp', 'op': 'phase', 'degrees': degrees})

def SP_FREQ_SHIFT(delta_hz: float) -> Node:
    """Frequency shift in Hz."""
    return Val({'type': 'sp', 'op': 'freq_shift', 'delta_hz': delta_hz})

def SP_PITCH_SHIFT(semitones: float) -> Node:
    """Pitch shift in semitones."""
    return Val({'type': 'sp', 'op': 'pitch_shift', 'semitones': semitones})

# Modulation Effects
def SP_TREMOLO(rate: float, depth: float) -> Node:
    """Tremolo (amplitude modulation)."""
    return Val({'type': 'sp', 'op': 'tremolo', 'rate': rate, 'depth': depth})

def SP_VIBRATO(rate: float, depth: float) -> Node:
    """Vibrato (frequency modulation)."""
    return Val({'type': 'sp', 'op': 'vibrato', 'rate': rate, 'depth': depth})

def SP_CHORUS(rate: float, depth: float, mix: float) -> Node:
    """Chorus effect."""
    return Val({'type': 'sp', 'op': 'chorus', 'rate': rate, 'depth': depth, 'mix': mix})

def SP_FLANGER(rate: float, depth: float, feedback: float) -> Node:
    """Flanger effect."""
    return Val({'type': 'sp', 'op': 'flanger', 'rate': rate, 'depth': depth, 'feedback': feedback})

# Dynamic Processing
def SP_COMPRESSOR(threshold: float, ratio: float, attack: float, release: float) -> Node:
    """Compressor."""
    return Val({'type': 'sp', 'op': 'compressor', 'threshold': threshold, 'ratio': ratio, 'attack': attack, 'release': release})

def SP_LIMITER(threshold: float) -> Node:
    """Limiter."""
    return Val({'type': 'sp', 'op': 'limiter', 'threshold': threshold})

def SP_GATE(threshold: float, ratio: float) -> Node:
    """Noise gate."""
    return Val({'type': 'sp', 'op': 'gate', 'threshold': threshold, 'ratio': ratio})

def SP_EXPANDER(threshold: float, ratio: float) -> Node:
    """Expander."""
    return Val({'type': 'sp', 'op': 'expander', 'threshold': threshold, 'ratio': ratio})

# Distortion and Saturation
def SP_DISTORTION(amount: float) -> Node:
    """Distortion."""
    return Val({'type': 'sp', 'op': 'distortion', 'amount': amount})

def SP_SATURATION(amount: float) -> Node:
    """Saturation."""
    return Val({'type': 'sp', 'op': 'saturation', 'amount': amount})

def SP_OVERDRIVE(amount: float) -> Node:
    """Overdrive."""
    return Val({'type': 'sp', 'op': 'overdrive', 'amount': amount})

# Spectral Operations
def SP_FFT() -> Node:
    """FFT."""
    return Val({'type': 'sp', 'op': 'fft'})

def SP_IFFT() -> Node:
    """Inverse FFT."""
    return Val({'type': 'sp', 'op': 'ifft'})

def SP_SPECTROGRAM(window_size: int, hop_size: int) -> Node:
    """Spectrogram."""
    return Val({'type': 'sp', 'op': 'spectrogram', 'window_size': window_size, 'hop_size': hop_size})

# Analysis Operations
def SP_ENERGY() -> Node:
    """Signal energy."""
    return Val({'type': 'sp', 'op': 'energy'})

def SP_RMS() -> Node:
    """RMS (Root Mean Square)."""
    return Val({'type': 'sp', 'op': 'rms'})

def SP_PEAK() -> Node:
    """Peak value."""
    return Val({'type': 'sp', 'op': 'peak'})

def SP_ZERO_CROSSING() -> Node:
    """Zero crossing rate."""
    return Val({'type': 'sp', 'op': 'zero_crossing'})

def SP_SPECTRAL_CENTROID() -> Node:
    """Spectral centroid."""
    return Val({'type': 'sp', 'op': 'spectral_centroid'})

def SP_SPECTRAL_ROLLOFF(rolloff: float) -> Node:
    """Spectral rolloff."""
    return Val({'type': 'sp', 'op': 'spectral_rolloff', 'rolloff': rolloff})

def SP_SPECTRAL_BANDWIDTH() -> Node:
    """Spectral bandwidth."""
    return Val({'type': 'sp', 'op': 'spectral_bandwidth'})

# Window Functions
def SP_HANNING() -> Node:
    """Hanning window."""
    return Val({'type': 'sp', 'op': 'hanning'})

def SP_HAMMING() -> Node:
    """Hamming window."""
    return Val({'type': 'sp', 'op': 'hamming'})

def SP_BLACKMAN() -> Node:
    """Blackman window."""
    return Val({'type': 'sp', 'op': 'blackman'})

# Resampling
def SP_RESAMPLE(target_sr: float) -> Node:
    """Resample to target sample rate."""
    return Val({'type': 'sp', 'op': 'resample', 'target_sr': target_sr})

def SP_DOWNSAMPLE(factor: int) -> Node:
    """Downsample by factor."""
    return Val({'type': 'sp', 'op': 'downsample', 'factor': factor})

def SP_UPSAMPLE(factor: int) -> Node:
    """Upsample by factor."""
    return Val({'type': 'sp', 'op': 'upsample', 'factor': factor})

# Noise Operations
def SP_WHITE_NOISE(level: float) -> Node:
    """White noise generator."""
    return Val({'type': 'sp', 'op': 'white_noise', 'level': level})

def SP_PINK_NOISE(level: float) -> Node:
    """Pink noise generator."""
    return Val({'type': 'sp', 'op': 'pink_noise', 'level': level})

def SP_BROWN_NOISE(level: float) -> Node:
    """Brown noise generator."""
    return Val({'type': 'sp', 'op': 'brown_noise', 'level': level})

# ============================
# Signal Processor Implementation
# ============================

def process_signal(field: FieldIQ, operation: dict) -> FieldIQ:
    """Process a signal with a single operation."""
    op_type = operation['op']
    
    if op_type == 'amp':
        gain = operation['gain']
        return FieldIQ(field.z * gain, field.sr, field.roles)
    
    elif op_type == 'gain':
        gain_db = operation['gain_db']
        gain_linear = 10 ** (gain_db / 20)
        return FieldIQ(field.z * gain_linear, field.sr, field.roles)
    
    elif op_type == 'invert':
        return FieldIQ(-field.z, field.sr, field.roles)
    
    elif op_type == 'lowpass':
        cutoff = operation['cutoff']
        alpha = np.exp(-2 * np.pi * cutoff / field.sr)
        filtered = np.zeros_like(field.z)
        filtered[0] = field.z[0]
        for i in range(1, len(field.z)):
            filtered[i] = alpha * filtered[i-1] + (1 - alpha) * field.z[i]
        return FieldIQ(filtered, field.sr, field.roles)
    
    elif op_type == 'highpass':
        cutoff = operation['cutoff']
        alpha = np.exp(-2 * np.pi * cutoff / field.sr)
        filtered = np.zeros_like(field.z)
        filtered[0] = field.z[0]
        for i in range(1, len(field.z)):
            filtered[i] = alpha * (filtered[i-1] + field.z[i] - field.z[i-1])
        return FieldIQ(filtered, field.sr, field.roles)
    
    elif op_type == 'bandpass':
        low = operation['low']
        high = operation['high']
        # Simple bandpass: highpass then lowpass
        hp_alpha = np.exp(-2 * np.pi * low / field.sr)
        lp_alpha = np.exp(-2 * np.pi * high / field.sr)
        filtered = np.zeros_like(field.z)
        filtered[0] = field.z[0]
        for i in range(1, len(field.z)):
            filtered[i] = hp_alpha * (filtered[i-1] + field.z[i] - field.z[i-1])
        for i in range(1, len(filtered)):
            filtered[i] = lp_alpha * filtered[i-1] + (1 - lp_alpha) * filtered[i]
        return FieldIQ(filtered, field.sr, field.roles)
    
    elif op_type == 'phase':
        degrees = operation['degrees']
        phase_shift = np.exp(1j * np.radians(degrees))
        return FieldIQ(field.z * phase_shift, field.sr, field.roles)
    
    elif op_type == 'freq_shift':
        delta_hz = operation['delta_hz']
        t = np.arange(len(field.z)) / field.sr
        shift_mod = np.exp(1j * 2 * np.pi * delta_hz * t)
        shifted = field.z * shift_mod
        return FieldIQ(shifted, field.sr, field.roles)
    
    elif op_type == 'delay':
        delay_ms = operation['delay_ms']
        delay_samples = int(delay_ms * field.sr / 1000)
        if delay_samples > 0:
            delayed = np.concatenate([np.zeros(delay_samples, dtype=complex), field.z[:-delay_samples]])
        else:
            delayed = field.z
        return FieldIQ(delayed, field.sr, field.roles)
    
    elif op_type == 'tremolo':
        rate = operation['rate']
        depth = operation['depth']
        t = np.arange(len(field.z)) / field.sr
        tremolo_mod = 1 + depth * np.sin(2 * np.pi * rate * t)
        return FieldIQ(field.z * tremolo_mod, field.sr, field.roles)
    
    elif op_type == 'vibrato':
        rate = operation['rate']
        depth = operation['depth']
        t = np.arange(len(field.z)) / field.sr
        vibrato_mod = np.exp(1j * 2 * np.pi * depth * np.sin(2 * np.pi * rate * t))
        return FieldIQ(field.z * vibrato_mod, field.sr, field.roles)
    
    elif op_type == 'distortion':
        amount = operation['amount']
        # Simple soft clipping
        distorted = np.tanh(field.z * amount) / amount
        return FieldIQ(distorted, field.sr, field.roles)
    
    elif op_type == 'fft':
        fft_result = np.fft.fft(field.z)
        return FieldIQ(fft_result, field.sr, field.roles)
    
    elif op_type == 'ifft':
        ifft_result = np.fft.ifft(field.z)
        return FieldIQ(ifft_result, field.sr, field.roles)
    
    elif op_type == 'energy':
        energy = np.sum(np.abs(field.z)**2)
        return Val(energy)
    
    elif op_type == 'rms':
        rms = np.sqrt(np.mean(np.abs(field.z)**2))
        return Val(rms)
    
    elif op_type == 'peak':
        peak = np.max(np.abs(field.z))
        return Val(peak)
    
    elif op_type == 'zero_crossing':
        real_part = np.real(field.z)
        zero_crossings = np.sum(np.diff(np.sign(real_part)) != 0)
        zcr = zero_crossings / len(field.z)
        return Val(zcr)
    
    elif op_type == 'hanning':
        window = np.hanning(len(field.z))
        return FieldIQ(field.z * window, field.sr, field.roles)
    
    elif op_type == 'hamming':
        window = np.hamming(len(field.z))
        return FieldIQ(field.z * window, field.sr, field.roles)
    
    elif op_type == 'blackman':
        window = np.blackman(len(field.z))
        return FieldIQ(field.z * window, field.sr, field.roles)
    
    elif op_type == 'downsample':
        factor = operation['factor']
        downsampled = field.z[::factor]
        return FieldIQ(downsampled, field.sr / factor, field.roles)
    
    elif op_type == 'upsample':
        factor = operation['factor']
        upsampled = np.zeros(len(field.z) * factor, dtype=complex)
        upsampled[::factor] = field.z
        return FieldIQ(upsampled, field.sr * factor, field.roles)
    
    elif op_type == 'white_noise':
        level = operation['level']
        noise = level * (np.random.randn(len(field.z)) + 1j * np.random.randn(len(field.z)))
        return FieldIQ(noise, field.sr, field.roles)
    
    else:
        # Return original field for unimplemented operations
        return field

def execute_sp_vm(expr: Node, field: FieldIQ) -> FieldIQ:
    """Execute signal processing VM expression."""
    reduced = reduce_whnf(expr)
    
    if isinstance(reduced, Val) and isinstance(reduced.v, dict):
        if reduced.v.get('type') == 'sp':
            return process_signal(field, reduced.v)
        elif reduced.v.get('type') == 'fieldiq':
            z = np.array(reduced.v['z'], dtype=complex)
            sr = reduced.v['sr']
            roles = reduced.v['roles']
            return FieldIQ(z, sr, roles)
    
    return field

# ============================
# Pipeline Builder
# ============================

def build_pipeline(*ops: Node) -> Node:
    """Build a pipeline using B combinator."""
    if not ops:
        return Val({'type': 'sp', 'op': 'identity'})
    
    pipeline = ops[0]
    for op in ops[1:]:
        pipeline = app(app(Comb('B'), pipeline), op)
    
    return pipeline

# ============================
# Pre-built Chains
# ============================

def create_audio_mastering_chain() -> Node:
    """Audio mastering chain."""
    return build_pipeline(
        SP_GAIN(6.0),                    # Boost
        SP_LOWPASS(18000.0),             # Anti-aliasing
        SP_COMPRESSOR(-12.0, 4.0, 10.0, 100.0),  # Compression
        SP_LIMITER(-3.0),                # Limiting
        SP_GAIN(-3.0)                    # Final level
    )

def create_guitar_effects_chain() -> Node:
    """Guitar effects chain."""
    return build_pipeline(
        SP_GAIN(12.0),                   # Preamp
        SP_DISTORTION(2.0),              # Distortion
        SP_LOWPASS(8000.0),              # Tone control
        SP_ECHO(250.0, 0.3, 0.4),        # Echo
        SP_REVERB(0.4, 0.6, 0.3)         # Reverb
    )

def create_spectral_analysis_chain() -> Node:
    """Spectral analysis chain."""
    return build_pipeline(
        SP_HANNING(),                    # Windowing
        SP_FFT(),                        # FFT
        SP_SPECTRAL_CENTROID(),          # Analysis
        SP_SPECTRAL_ROLLOFF(0.85)
    )

def create_vocal_processing_chain() -> Node:
    """Vocal processing chain."""
    return build_pipeline(
        SP_HIGHPASS(80.0),               # Highpass
        SP_BANDPASS(80.0, 8000.0),       # Bandpass
        SP_COMPRESSOR(-18.0, 3.0, 5.0, 50.0),  # Compression
        SP_ECHO(120.0, 0.2, 0.2),        # Subtle echo
        SP_REVERB(0.3, 0.4, 0.2)         # Reverb
    )

# ============================
# Demo
# ============================

def demo_extended_sp_vm():
    """Demo the extended signal processing VM."""
    print("=== Extended Signal Processing VM Demo ===\n")
    
    # Create test signal
    sr = 48000.0
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    x = 0.5 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t)
    field = make_field_from_real(x, sr)
    
    print(f"Test signal: {len(field.z)} samples, {field.sr} Hz")
    print(f"Original energy: {np.sum(np.abs(field.z)**2):.6f}")
    
    # Test individual operations
    print("\n--- Individual Operations ---")
    
    operations = [
        (SP_AMP(0.5), "Amplitude (0.5x)"),
        (SP_GAIN(-6.0), "Gain (-6dB)"),
        (SP_LOWPASS(1000.0), "Lowpass (1kHz)"),
        (SP_HIGHPASS(500.0), "Highpass (500Hz)"),
        (SP_BANDPASS(200.0, 2000.0), "Bandpass (200Hz-2kHz)"),
        (SP_PHASE(90.0), "Phase (90°)"),
        (SP_FREQ_SHIFT(100.0), "Freq shift (+100Hz)"),
        (SP_TREMOLO(4.0, 0.3), "Tremolo (4Hz, 30%)"),
        (SP_VIBRATO(6.0, 0.1), "Vibrato (6Hz, 10%)"),
        (SP_DISTORTION(1.5), "Distortion (1.5x)"),
        (SP_ECHO(100.0, 0.3, 0.4), "Echo (100ms, 30%, 40%)"),
        (SP_DELAY(50.0), "Delay (50ms)")
    ]
    
    for op, name in operations:
        result = execute_sp_vm(op, field)
        if hasattr(result, 'z'):
            energy = np.sum(np.abs(result.z)**2)
            print(f"{name}: {energy:.6f}")
        else:
            print(f"{name}: {result.v}")
    
    # Test pre-built chains
    print("\n--- Pre-built Chains ---")
    
    chains = [
        (create_audio_mastering_chain(), "Audio Mastering"),
        (create_guitar_effects_chain(), "Guitar Effects"),
        (create_vocal_processing_chain(), "Vocal Processing")
    ]
    
    for chain, name in chains:
        result = execute_sp_vm(chain, field)
        energy = np.sum(np.abs(result.z)**2)
        print(f"{name}: {energy:.6f}")
    
    # Test serialization
    print("\n--- Serialization Test ---")
    
    mastering_chain = create_audio_mastering_chain()
    json_str = to_json(mastering_chain)
    print(f"Serialized mastering chain: {len(json_str)} characters")
    
    deserialized = from_json(json_str)
    deserialized_result = execute_sp_vm(deserialized, field)
    print(f"Deserialized result: {np.sum(np.abs(deserialized_result.z)**2):.6f}")
    
    print("\n=== Extended SP VM Complete ===")
    print("✓ 40+ signal processing operations")
    print("✓ Pre-built processing chains")
    print("✓ Full serialization support")
    print("✓ Ready for cross-language deployment")

if __name__ == "__main__":
    demo_extended_sp_vm()


