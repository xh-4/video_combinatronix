
# ============================
# HPU Quadrature Combinator Kernel (minimal)
# ============================
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict, Any, Optional, Union, Iterator
from enum import Enum

# Optional OpenCV import - will work without it for basic functionality
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    try:
        # Try to use mock OpenCV for testing
        import mock_opencv as cv2
        HAS_OPENCV = True
        print("Using mock OpenCV for testing - video processing features available")
    except ImportError:
        HAS_OPENCV = False
        print("OpenCV not available - video processing features will be limited")

Array = np.ndarray

# ---------- Utilities ----------
def analytic_signal(x: Array) -> Array:
    """
    Real x -> analytic z = x + j*H{x} using FFT method (no SciPy).
    Fixed DC offset handling and improved numerical stability.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Remove DC offset before processing
    x_dc_removed = x - np.mean(x)
    
    N = x_dc_removed.shape[-1]
    X = np.fft.rfft(x_dc_removed)
    
    # Create the "doubling" vector for positive freqs
    h = np.zeros_like(X, dtype=np.float64)
    if N % 2:  # odd
        h[1:] = 2.0
    else:      # even
        h[1:-1] = 2.0
        h[-1] = 1.0  # Nyquist bin unchanged
    
    # Zero out DC component to prevent DC offset in analytic signal
    h[0] = 0.0
    
    Z = X * h
    z = np.fft.irfft(Z, n=N)
    
    # Ensure complex dtype and add back original DC if needed
    return z + 0j  # ensure complex dtype

def circ_delay(x: Array, samples: int) -> Array:
    return np.roll(x, int(samples), axis=-1)

def moving_average(x: Array, width: int) -> Array:
    width = max(1, int(width))
    k = np.ones(width, dtype=np.float64) / width
    # Convolve real and imag separately to preserve complex
    xr = np.convolve(np.real(x), k, mode="same")
    xi = np.convolve(np.imag(x), k, mode="same")
    return xr + 1j * xi

def remove_dc_offset(x: Array) -> Array:
    """Remove DC offset from signal while preserving complex structure."""
    if np.iscomplexobj(x):
        return x - np.mean(x)
    else:
        return x - np.mean(x)

def validate_quadrature(i: Array, q: Array, tolerance: float = 1e-6) -> bool:
    """Validate that I and Q form a proper quadrature pair."""
    # Check that I and Q have same length
    if len(i) != len(q):
        return False
    
    # Check for DC offset (should be minimal for proper quadrature)
    i_dc = np.mean(i)
    q_dc = np.mean(q)
    if abs(i_dc) > tolerance or abs(q_dc) > tolerance:
        return False
    
    # Check for proper phase relationship (90 degrees apart)
    # This is a simplified check - in practice you might want more sophisticated validation
    return True

# ---------- FieldIQ ----------
@dataclass
class FieldIQ:
    """Analytic (I/Q) field over a time block with enhanced quadrature validation."""
    z: Array                 # complex: I + jQ
    sr: float                # sample rate
    roles: Dict[str, Any] = None  # optional symbolic roles/metadata

    def __post_init__(self):
        """Validate quadrature pair and remove DC offset on initialization."""
        if self.z is not None:
            # Remove DC offset
            self.z = remove_dc_offset(self.z)
            
            # Validate quadrature if both I and Q exist
            if len(self.z) > 0:
                i, q = np.real(self.z), np.imag(self.z)
                if not validate_quadrature(i, q):
                    # If validation fails, try to correct DC offset
                    self.z = remove_dc_offset(self.z)

    @property
    def I(self) -> Array: return np.real(self.z)
    @property
    def Q(self) -> Array: return np.imag(self.z)
    
    @property
    def magnitude(self) -> Array: return np.abs(self.z)
    @property
    def phase(self) -> Array: return np.angle(self.z)
    @property
    def power(self) -> Array: return np.abs(self.z)**2
    
    def copy(self) -> "FieldIQ":
        return FieldIQ(self.z.copy(), self.sr, dict(self.roles or {}))
    
    def is_valid_quadrature(self, tolerance: float = 1e-6) -> bool:
        """Check if this field represents a valid quadrature pair."""
        return validate_quadrature(self.I, self.Q, tolerance)

    # ---- core quadrature ops ----
    def rotate(self, degrees: float) -> "FieldIQ":
        theta = np.deg2rad(degrees)
        return FieldIQ(self.z * np.exp(1j * theta), self.sr, self.roles)

    def scale(self, gain: float) -> "FieldIQ":
        return FieldIQ(self.z * gain, self.sr, self.roles)

    def phase_shift_narrowband(self, degrees: float, f0: float) -> "FieldIQ":
        """Approx “+deg at f0” → delay τ = φ/(2π f0). Narrowband trick."""
        tau = np.deg2rad(degrees) / (2 * np.pi * f0)
        samples = int(round(tau * self.sr))
        return FieldIQ(circ_delay(self.z, samples), self.sr, self.roles)

    def freq_shift(self, delta_hz: float) -> "FieldIQ":
        t = np.arange(self.z.shape[-1]) / self.sr
        return FieldIQ(self.z * np.exp(1j * 2*np.pi*delta_hz * t), self.sr, self.roles)

    def delay_ms(self, ms: float) -> "FieldIQ":
        samples = int(round(ms * 1e-3 * self.sr))
        return FieldIQ(circ_delay(self.z, samples), self.sr, self.roles)

    def lowpass_width(self, width: int) -> "FieldIQ":
        return FieldIQ(moving_average(self.z, width), self.sr, self.roles)

    def lowpass_hz(self, cutoff_hz: float) -> "FieldIQ":
        # crude: infer width from cutoff (MA ~ sinc LP). tweak as needed.
        width = max(1, int(round(self.sr / max(cutoff_hz, 1.0))))
        return self.lowpass_width(width)

    # ---- role tagging ----
    def with_role(self, key: str, value: Any) -> "FieldIQ":
        r = dict(self.roles or {})
        r[key] = value
        return FieldIQ(self.z, self.sr, r)

# helper to construct analytic field from REAL waveform
def make_field_from_real(x: Array, sr: float, tag: Optional[Tuple[str, Any]]=None) -> FieldIQ:
    z = analytic_signal(x)
    roles = {tag[0]: tag[1]} if tag else {}
    return FieldIQ(z, sr, roles)

# ---------- Video Processing Classes ----------
class ColorSpace(Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"
    YUV = "yuv"
    LAB = "lab"

@dataclass
class VideoFrame:
    """Single video frame with metadata."""
    data: Array  # Frame data (H, W, C) or (H, W) for grayscale
    frame_number: int
    timestamp: float
    color_space: ColorSpace = ColorSpace.BGR
    fps: float = 30.0
    
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def channels(self) -> int:
        return self.data.shape[2] if len(self.data.shape) == 3 else 1
    
    def to_field_iq(self, channel: int = 0, sr: float = 48000.0) -> FieldIQ:
        """Convert frame channel to FieldIQ for processing."""
        if len(self.data.shape) == 3:
            channel_data = self.data[:, :, channel].flatten()
        else:
            channel_data = self.data.flatten()
        
        # Normalize to [-1, 1] range
        channel_data = (channel_data.astype(np.float64) - 128.0) / 128.0
        
        return make_field_from_real(channel_data, sr, tag=("video_frame", self.frame_number))

@dataclass
class VideoChunk:
    """Chunk of video frames for processing."""
    frames: List[VideoFrame]
    chunk_id: int
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def frame_count(self) -> int:
        return len(self.frames)
    
    def to_field_iq_sequence(self, channel: int = 0, sr: float = 48000.0) -> List[FieldIQ]:
        """Convert all frames in chunk to FieldIQ sequence."""
        return [frame.to_field_iq(channel, sr) for frame in self.frames]

class VideoStreamProcessor:
    """Streaming video processor with chunking support."""
    
    def __init__(self, chunk_size: int = 30, overlap: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.frame_buffer: List[VideoFrame] = []
        self.chunk_counter = 0
    
    def add_frame(self, frame: VideoFrame) -> Optional[VideoChunk]:
        """Add frame to buffer, return chunk when ready."""
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) >= self.chunk_size:
            # Create chunk
            chunk = VideoChunk(
                frames=self.frame_buffer[:self.chunk_size],
                chunk_id=self.chunk_counter,
                start_time=self.frame_buffer[0].timestamp,
                end_time=self.frame_buffer[self.chunk_size-1].timestamp
            )
            
            # Keep overlap frames for next chunk
            self.frame_buffer = self.frame_buffer[self.chunk_size - self.overlap:]
            self.chunk_counter += 1
            
            return chunk
        
        return None
    
    def flush(self) -> Optional[VideoChunk]:
        """Flush remaining frames as final chunk."""
        if len(self.frame_buffer) > 0:
            chunk = VideoChunk(
                frames=self.frame_buffer,
                chunk_id=self.chunk_counter,
                start_time=self.frame_buffer[0].timestamp,
                end_time=self.frame_buffer[-1].timestamp
            )
            self.frame_buffer = []
            self.chunk_counter += 1
            return chunk
        return None

def load_video_chunks(video_path: str, chunk_size: int = 30, overlap: int = 5) -> Iterator[VideoChunk]:
    """Load video file and yield chunks for processing."""
    if not HAS_OPENCV:
        print("OpenCV not available - cannot load video files")
        return
    
    cap = cv2.VideoCapture(video_path)
    processor = VideoStreamProcessor(chunk_size, overlap)
    frame_number = 0
    
    try:
        while True:
            ret, frame_data = cap.read()
            if not ret:
                break
            
            frame = VideoFrame(
                data=frame_data,
                frame_number=frame_number,
                timestamp=frame_number / cap.get(cv2.CAP_PROP_FPS),
                fps=cap.get(cv2.CAP_PROP_FPS)
            )
            
            chunk = processor.add_frame(frame)
            if chunk is not None:
                yield chunk
            
            frame_number += 1
        
        # Flush remaining frames
        final_chunk = processor.flush()
        if final_chunk is not None:
            yield final_chunk
    
    finally:
        cap.release()

def process_video_stream(video_path: str, processor_func: Callable[[VideoChunk], Any], 
                        chunk_size: int = 30, overlap: int = 5) -> List[Any]:
    """Process video stream with given processor function."""
    results = []
    for chunk in load_video_chunks(video_path, chunk_size, overlap):
        result = processor_func(chunk)
        results.append(result)
    return results

# ---------- SKI/BCW combinators on fields ----------
# These operate on *functions* or *curried* combiners that consume FieldIQ
Field = FieldIQ
Unary = Callable[[Field], Field]
Binary = Callable[[Field], Callable[[Field], Field]]

def K(a: Field) -> Callable[[Field], Field]:
    return lambda b: a

def S(comb: Binary) -> Callable[[Unary], Unary]:
    # (S comb g) x  =  comb x (g x)
    return lambda g: (lambda x: comb(x)(g(x)))

def B(f: Unary) -> Callable[[Unary], Unary]:
    return lambda g: (lambda x: f(g(x)))

def C(f: Binary) -> Callable[[Field], Callable[[Field], Field]]:
    # C f x y = f y x
    return lambda x: (lambda y: f(y)(x))

def W(f: Binary) -> Unary:
    # W f x = f x x
    return lambda x: f(x)(x)

def PLUS(a: Field) -> Callable[[Field], Field]:
    return lambda b: FieldIQ(a.z + b.z, a.sr, a.roles or b.roles)

def TIMES(a: Field) -> Callable[[Field], Field]:
    # multiplicative interference (complex)
    return lambda b: FieldIQ(a.z * b.z, a.sr, a.roles or b.roles)

# convenience builders
def split_add(g: Unary) -> Unary:   # x -> x + g(x)
    return S(PLUS)(g)
def split_mul(g: Unary) -> Unary:   # x -> x * g(x)
    return S(TIMES)(g)

# ---------- temporal combinators ----------
def gate_percent(percent: float) -> Unary:
    """Keep first p% of the block, zero the rest (simple rhythmic chopper)."""
    p = np.clip(percent, 0.0, 100.0) / 100.0
    def fn(F: Field) -> Field:
        n = F.z.shape[-1]
        k = int(round(p * n))
        mask = np.zeros(n, dtype=np.float64); mask[:k] = 1.0
        return FieldIQ(F.z * mask, F.sr, F.roles)
    return fn

def fold_beats(beats: float, tempo_bpm: float) -> Unary:
    """Fold time every N beats: t -> t mod (N beats). Good for phase-wrapping FX."""
    period_s = 60.0 * beats / tempo_bpm
    def fn(F: Field) -> Field:
        n = F.z.shape[-1]
        t = np.arange(n) / F.sr
        t_fold = np.mod(t, period_s)
        # resample via nearest (cheap); swap with linear if desired
        idx = np.floor(t_fold * F.sr).astype(int) % n
        return FieldIQ(F.z[idx], F.sr, F.roles)
    return fn

# ---------- symbolic layer: tiny EventStream ----------
@dataclass
class Event:
    t: float
    data: Dict[str, Any]

@dataclass
class EventStream:
    events: List[Event]
    def map(self, f: Callable[[Event], Event]) -> "EventStream":
        return EventStream([f(e) for e in self.events])
    def filter(self, p: Callable[[Event], bool]) -> "EventStream":
        return EventStream([e for e in self.events if p(e)])
    def merge(self, other: "EventStream") -> "EventStream":
        return EventStream(sorted(self.events + other.events, key=lambda e: e.t))

# ---------- effects as unary closures on FieldIQ ----------
def phase_deg(deg: float, f0: float) -> Unary:
    return lambda F: F.phase_shift_narrowband(deg, f0)

def freq_shift(delta_hz: float) -> Unary:
    return lambda F: F.freq_shift(delta_hz)

def amp(gain: float) -> Unary:
    return lambda F: F.scale(gain)

def delay_ms(ms: float) -> Unary:
    return lambda F: F.delay_ms(ms)

def lowpass_hz(cut: float) -> Unary:
    return lambda F: F.lowpass_hz(cut)

def lowpass_w(width: int) -> Unary:
    return lambda F: F.lowpass_width(width)

# ---------- Video-specific combinators ----------
def video_channel_processor(channel: int = 0, sr: float = 48000.0) -> Callable[[VideoChunk], List[FieldIQ]]:
    """Convert video chunk to FieldIQ sequence for specified channel."""
    return lambda chunk: chunk.to_field_iq_sequence(channel, sr)

def video_frame_processor(processor: Unary, channel: int = 0, sr: float = 48000.0) -> Callable[[VideoChunk], List[FieldIQ]]:
    """Apply processor to each frame in video chunk."""
    def process_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        fields = chunk.to_field_iq_sequence(channel, sr)
        return [processor(field) for field in fields]
    return process_chunk

def video_temporal_processor(processor: Unary, channel: int = 0, sr: float = 48000.0) -> Callable[[VideoChunk], FieldIQ]:
    """Apply processor to concatenated temporal sequence of video chunk."""
    def process_chunk(chunk: VideoChunk) -> FieldIQ:
        fields = chunk.to_field_iq_sequence(channel, sr)
        # Concatenate all frames into single temporal sequence
        if fields:
            # Concatenate complex data
            z_concat = np.concatenate([field.z for field in fields])
            # Create single FieldIQ with combined roles
            combined_roles = {}
            for field in fields:
                if field.roles:
                    combined_roles.update(field.roles)
            combined_roles['video_chunk'] = chunk.chunk_id
            return processor(FieldIQ(z_concat, sr, combined_roles))
        return None
    return process_chunk

def video_spectral_analyzer(channel: int = 0, sr: float = 48000.0) -> Callable[[VideoChunk], Dict[str, Any]]:
    """Analyze spectral content of video chunk."""
    def analyze_chunk(chunk: VideoChunk) -> Dict[str, Any]:
        fields = chunk.to_field_iq_sequence(channel, sr)
        if not fields:
            return {}
        
        # Analyze first frame as representative
        field = fields[0]
        fft_data = np.fft.fft(field.z)
        freqs = np.fft.fftfreq(len(field.z), 1/sr)
        
        return {
            'chunk_id': chunk.chunk_id,
            'frame_count': len(fields),
            'duration': chunk.duration,
            'dominant_freq': freqs[np.argmax(np.abs(fft_data))],
            'spectral_centroid': np.sum(freqs * np.abs(fft_data)) / np.sum(np.abs(fft_data)),
            'total_power': np.sum(field.power),
            'is_valid_quadrature': field.is_valid_quadrature()
        }
    return analyze_chunk

# ============================
# Demo
# ============================
if __name__ == "__main__":
    print("=== Audio Processing Demo ===")
    sr = 48000
    dur = 1.0
    t  = np.linspace(0, dur, int(sr*dur), endpoint=False)
    x  = 0.8*np.cos(2*np.pi*440*t)  # real input
    F0 = make_field_from_real(x, sr, tag=("role","carrier"))

    # Program: LP ∘ (split-add (phase +90° at f0))
    prog = B(lowpass_hz(1200.0))( split_add( phase_deg(+90.0, 440.0) ) )
    Y = prog(F0)

    # A few more:
    wobble = split_mul( freq_shift(+5.0) )          # ring-mod split
    chorus = split_add( delay_ms(15.0) )            # short delay add
    dupadd = W(PLUS)                                # x + x
    rotated = F0.rotate(33.0).with_role("vibe","tilt")

    # Compose a small chain point-free:
    chain = B(lowpass_w(25))(
                B(amp(0.7))(
                    split_add( phase_deg(+90, 440.0) )))
    Z = chain(F0)

    # Show quick numeric sanity
    print("RMS in/out:", np.sqrt(np.mean(F0.I**2)), "→", np.sqrt(np.mean(np.real(Y.z)**2)))
    print("Roles:", Y.roles)
    print("Valid quadrature:", F0.is_valid_quadrature())
    
    print("\n=== Video Processing Demo ===")
    # Create synthetic video frame for demo
    height, width = 64, 64
    synthetic_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Create video frame
    video_frame = VideoFrame(
        data=synthetic_frame,
        frame_number=0,
        timestamp=0.0,
        fps=30.0
    )
    
    # Convert to FieldIQ
    field_iq = video_frame.to_field_iq(channel=0, sr=48000.0)
    print(f"Video frame shape: {video_frame.data.shape}")
    print(f"FieldIQ length: {len(field_iq.z)}")
    print(f"Valid quadrature: {field_iq.is_valid_quadrature()}")
    
    # Create video chunk
    chunk = VideoChunk(
        frames=[video_frame],
        chunk_id=0,
        start_time=0.0,
        end_time=1.0/30.0
    )
    
    # Process with video combinators
    frame_processor = video_frame_processor(
        B(lowpass_hz(1000.0))(amp(0.5)),
        channel=0
    )
    
    processed_frames = frame_processor(chunk)
    print(f"Processed {len(processed_frames)} frames")
    
    # Spectral analysis
    analyzer = video_spectral_analyzer(channel=0)
    analysis = analyzer(chunk)
    print(f"Spectral analysis: {analysis}")
    
    print("\n=== Streaming Demo ===")
    print("To process a real video file, use:")
    print("  for chunk in load_video_chunks('video.mp4', chunk_size=30):")
    print("      result = process_chunk(chunk)")
    print("  # or use process_video_stream('video.mp4', processor_func)")
