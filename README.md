# HPU Quadrature Combinator Kernel

A powerful functional programming kernel for quadrature (I/Q) signal processing with video processing capabilities, built on the SKI/BCW combinator calculus.

## Features

### 🎵 **Audio Processing**
- **DC Offset Removal**: Automatic DC bias correction in analytic signal generation
- **Full Quadrature Support**: Proper I/Q validation and phase coherence
- **Functional Combinators**: SKI/BCW calculus for signal processing
- **Real-time Processing**: Efficient streaming audio processing

### 🎬 **Video Processing**
- **Chunked Processing**: Memory-efficient video processing in configurable chunks
- **Streaming Support**: Real-time video stream processing
- **Multi-channel Analysis**: Process RGB channels independently
- **Spectral Analysis**: Frequency domain analysis of video content

### 🔧 **Core Capabilities**
- **Analytic Signal Generation**: Real-to-complex conversion with Hilbert transform
- **Phase Operations**: Rotation, shifting, and modulation
- **Frequency Operations**: Shifting, filtering, and spectral analysis
- **Temporal Operations**: Delay, gating, and beat folding
- **Combinator Calculus**: Functional composition of signal processors

## Installation

### Requirements
- Python 3.8+
- NumPy
- OpenCV (optional, for video processing)

### Install Dependencies
```bash
pip install numpy opencv-python
```

### Basic Installation
```bash
git clone https://github.com/yourusername/combinator-kernel.git
cd combinator-kernel
pip install -r requirements.txt
```

## Quick Start

### Audio Processing
```python
from Combinator_Kernel import make_field_from_real, lowpass_hz, phase_deg, split_add, B
import numpy as np

# Create a test signal
sr = 48000
t = np.linspace(0, 1, sr)
x = 0.8 * np.cos(2 * np.pi * 440 * t)  # 440 Hz tone

# Convert to FieldIQ
field = make_field_from_real(x, sr, tag=("role", "carrier"))

# Apply processing chain
processor = B(lowpass_hz(1200.0))(split_add(phase_deg(90.0, 440.0)))
result = processor(field)

print(f"Valid quadrature: {result.is_valid_quadrature()}")
print(f"RMS: {np.sqrt(np.mean(result.power))}")
```

### Video Processing
```python
from Combinator_Kernel import load_video_chunks, video_frame_processor, lowpass_hz, amp, B

# Process video file in chunks
for chunk in load_video_chunks('video.mp4', chunk_size=30, overlap=5):
    # Convert to FieldIQ and process
    processor = video_frame_processor(
        B(lowpass_hz(1000.0))(amp(0.8)),
        channel=0  # Blue channel
    )
    processed_frames = processor(chunk)
    print(f"Processed {len(processed_frames)} frames")
```

## Core Classes

### FieldIQ
The fundamental data structure representing a complex analytic signal:
```python
@dataclass
class FieldIQ:
    z: Array                 # complex: I + jQ
    sr: float                # sample rate
    roles: Dict[str, Any]    # metadata
```

**Properties:**
- `I`, `Q`: Real and imaginary components
- `magnitude`, `phase`, `power`: Signal properties
- `is_valid_quadrature()`: Validates I/Q pair quality

**Methods:**
- `rotate(degrees)`: Phase rotation
- `scale(gain)`: Amplitude scaling
- `freq_shift(delta_hz)`: Frequency shifting
- `delay_ms(ms)`: Time delay
- `lowpass_hz(cutoff)`: Low-pass filtering

### VideoFrame
Represents a single video frame:
```python
@dataclass
class VideoFrame:
    data: Array              # Frame data (H, W, C)
    frame_number: int
    timestamp: float
    color_space: ColorSpace
    fps: float
```

### VideoChunk
Manages chunks of video frames:
```python
@dataclass
class VideoChunk:
    frames: List[VideoFrame]
    chunk_id: int
    start_time: float
    end_time: float
```

## Combinator Calculus

The kernel implements the SKI/BCW combinator calculus for functional signal processing:

### Basic Combinators
- **K**: Constant function (`K a b = a`)
- **S**: Application combinator (`S f g x = f x (g x)`)
- **B**: Composition (`B f g x = f (g x)`)
- **C**: Argument swapping (`C f x y = f y x`)
- **W**: Duplication (`W f x = f x x`)

### Signal Combinators
- **PLUS**: Addition (`PLUS a b = a + b`)
- **TIMES**: Multiplication (`TIMES a b = a * b`)
- **split_add**: Split and add (`split_add f x = x + f(x)`)
- **split_mul**: Split and multiply (`split_mul f x = x * f(x)`)

### Example: Complex Processing Chain
```python
# Create a complex effect chain
wobble = split_mul(freq_shift(5.0))           # Ring modulation
chorus = split_add(delay_ms(15.0))            # Chorus effect
filter_chain = B(lowpass_hz(1000.0))(amp(0.7))

# Compose effects
complex_effect = B(filter_chain)(
    B(chorus)(wobble)
)

# Apply to signal
result = complex_effect(input_field)
```

## Video Processing Examples

### Frame-by-Frame Processing
```python
# Process each frame independently
frame_processor = video_frame_processor(
    B(lowpass_hz(800.0))(amp(0.7)),
    channel=0
)

for chunk in load_video_chunks('video.mp4'):
    processed_frames = frame_processor(chunk)
    # Process each frame...
```

### Temporal Processing
```python
# Process entire chunk as temporal sequence
temporal_processor = video_temporal_processor(
    B(lowpass_hz(1000.0))(
        split_add(phase_deg(45.0, 1000.0))
    ),
    channel=1
)

for chunk in load_video_chunks('video.mp4'):
    result = temporal_processor(chunk)
    # Process temporal sequence...
```

### Spectral Analysis
```python
# Analyze spectral content
analyzer = video_spectral_analyzer(channel=0)

for chunk in load_video_chunks('video.mp4'):
    analysis = analyzer(chunk)
    print(f"Dominant frequency: {analysis['dominant_freq']} Hz")
    print(f"Spectral centroid: {analysis['spectral_centroid']}")
    print(f"Total power: {analysis['total_power']}")
```

## Streaming Processing

### Real-time Video Processing
```python
from Combinator_Kernel import VideoStreamProcessor

# Create streaming processor
processor = VideoStreamProcessor(chunk_size=30, overlap=5)

# Process frames as they arrive
for frame in video_source:
    chunk = processor.add_frame(frame)
    if chunk is not None:
        # Process chunk
        result = process_chunk(chunk)
```

### Custom Stream Processing
```python
def custom_processor(chunk):
    fields = chunk.to_field_iq_sequence(channel=0)
    processed = [my_effect_chain(field) for field in fields]
    return {
        'chunk_id': chunk.chunk_id,
        'frame_count': len(processed),
        'avg_power': np.mean([np.sum(field.power) for field in processed])
    }

# Process video stream
results = process_video_stream('video.mp4', custom_processor)
```

## Advanced Features

### DC Offset Correction
The kernel automatically removes DC offset from signals:
```python
# DC offset is automatically removed
field = make_field_from_real(signal_with_dc_offset, sr)
print(f"DC removed: {np.mean(field.I):.6f}")  # Should be ~0
```

### Quadrature Validation
```python
# Check I/Q pair quality
if field.is_valid_quadrature(tolerance=1e-6):
    print("Valid quadrature pair")
else:
    print("Quadrature pair needs correction")
```

### Role-based Processing
```python
# Tag signals with metadata
field = field.with_role("instrument", "guitar")
field = field.with_role("effect", "reverb")

# Access roles
print(f"Roles: {field.roles}")
```

## Examples

See the `examples/` directory for:
- `audio_demo.py`: Audio processing examples
- `video_demo.py`: Video processing examples
- `streaming_demo.py`: Real-time processing examples

## Testing

Run the test suite:
```bash
python test_opencv.py      # Test OpenCV integration
python video_demo.py       # Test video processing
python Combinator_Kernel.py # Run built-in demos
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built on the SKI/BCW combinator calculus
- Inspired by functional signal processing techniques
- OpenCV integration for video processing
