#!/usr/bin/env python3
"""
Temporal Blend Effects Demo
Demonstrates various temporal blending effects on video chunks
"""

import sys
import os
import numpy as np
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Combinator_Kernel import (
    VideoFrame, VideoChunk, VideoStreamProcessor,
    load_video_chunks, process_video_stream
)
from effects import (
    temporal_blend_linear, temporal_blend_exponential, temporal_blend_sinusoidal,
    temporal_blend_adaptive, temporal_echo, temporal_glitch, temporal_morph,
    compose_effects, parallel_effects, blend_effects,
    create_temporal_blend_presets, EffectParams
)

def create_test_video(output_path="temporal_test.mp4", duration=3.0, fps=30):
    """Create a test video with interesting temporal patterns"""
    try:
        import cv2
    except ImportError:
        import mock_opencv as cv2
    
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_num in range(total_frames):
        # Create frame with temporal patterns
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving objects with different speeds
        t = frame_num / fps
        
        # Fast moving circle
        circle_x = int(width * (0.2 + 0.6 * (t * 2) % 1))
        circle_y = int(height * 0.3)
        cv2.circle(frame, (circle_x, circle_y), 25, (0, 255, 0), -1)
        
        # Slow moving rectangle
        rect_x = int(width * (0.1 + 0.8 * (t * 0.5) % 1))
        rect_y = int(height * 0.7)
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 40, rect_y + 30), (255, 0, 0), -1)
        
        # Pulsing triangle
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * t * 3)
        tri_size = int(20 * pulse)
        tri_x = int(width * 0.8)
        tri_y = int(height * 0.5)
        
        # Draw triangle
        pts = np.array([
            [tri_x, tri_y - tri_size],
            [tri_x - tri_size, tri_y + tri_size],
            [tri_x + tri_size, tri_y + tri_size]
        ], np.int32)
        cv2.fillPoly(frame, [pts], (0, 0, 255))
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Created test video: {output_path}")

def demo_linear_blend():
    """Demonstrate linear temporal blending"""
    print("\n=== Linear Temporal Blend Demo ===")
    
    # Create test video
    video_path = "linear_blend_test.mp4"
    create_test_video(video_path, duration=2.0)
    
    # Test different blend factors
    blend_factors = [0.2, 0.5, 0.8]
    
    for blend_factor in blend_factors:
        print(f"\nTesting linear blend with factor {blend_factor}")
        
        # Create effect
        effect = temporal_blend_linear(blend_factor)
        
        # Process video
        def process_chunk(chunk):
            return effect(chunk)
        
        results = process_video_stream(video_path, process_chunk, chunk_size=15, overlap=3)
        
        print(f"  Processed {len(results)} chunks")
        if results:
            sample_chunk = results[0]
            print(f"  Sample chunk: {len(sample_chunk)} frames")
            print(f"  First frame roles: {sample_chunk[0].roles}")

def demo_exponential_blend():
    """Demonstrate exponential temporal blending"""
    print("\n=== Exponential Temporal Blend Demo ===")
    
    video_path = "exp_blend_test.mp4"
    create_test_video(video_path, duration=2.0)
    
    decay_rates = [0.5, 0.7, 0.9]
    
    for decay_rate in decay_rates:
        print(f"\nTesting exponential blend with decay rate {decay_rate}")
        
        effect = temporal_blend_exponential(decay_rate)
        
        def process_chunk(chunk):
            return effect(chunk)
        
        results = process_video_stream(video_path, process_chunk, chunk_size=12, overlap=2)
        
        print(f"  Processed {len(results)} chunks")
        if results:
            sample_chunk = results[0]
            print(f"  Sample chunk: {len(sample_chunk)} frames")

def demo_sinusoidal_blend():
    """Demonstrate sinusoidal temporal blending"""
    print("\n=== Sinusoidal Temporal Blend Demo ===")
    
    video_path = "sin_blend_test.mp4"
    create_test_video(video_path, duration=2.0)
    
    frequencies = [1.0, 2.0, 4.0]
    
    for freq in frequencies:
        print(f"\nTesting sinusoidal blend with frequency {freq}")
        
        effect = temporal_blend_sinusoidal(freq, 0.0)
        
        def process_chunk(chunk):
            return effect(chunk)
        
        results = process_video_stream(video_path, process_chunk, chunk_size=10, overlap=1)
        
        print(f"  Processed {len(results)} chunks")

def demo_adaptive_blend():
    """Demonstrate adaptive temporal blending"""
    print("\n=== Adaptive Temporal Blend Demo ===")
    
    video_path = "adaptive_blend_test.mp4"
    create_test_video(video_path, duration=2.0)
    
    adaptation_rates = [0.1, 0.3, 0.5]
    
    for rate in adaptation_rates:
        print(f"\nTesting adaptive blend with rate {rate}")
        
        effect = temporal_blend_adaptive(rate)
        
        def process_chunk(chunk):
            return effect(chunk)
        
        results = process_video_stream(video_path, process_chunk, chunk_size=15, overlap=3)
        
        print(f"  Processed {len(results)} chunks")

def demo_echo_effect():
    """Demonstrate temporal echo effect"""
    print("\n=== Temporal Echo Demo ===")
    
    video_path = "echo_test.mp4"
    create_test_video(video_path, duration=2.0)
    
    echo_configs = [
        (2, 0.8, 0.2),  # delay, decay, feedback
        (3, 0.6, 0.3),
        (4, 0.7, 0.1)
    ]
    
    for delay, decay, feedback in echo_configs:
        print(f"\nTesting echo: delay={delay}, decay={decay}, feedback={feedback}")
        
        effect = temporal_echo(delay, decay, feedback)
        
        def process_chunk(chunk):
            return effect(chunk)
        
        results = process_video_stream(video_path, process_chunk, chunk_size=20, overlap=5)
        
        print(f"  Processed {len(results)} chunks")

def demo_glitch_effect():
    """Demonstrate temporal glitch effect"""
    print("\n=== Temporal Glitch Demo ===")
    
    video_path = "glitch_test.mp4"
    create_test_video(video_path, duration=2.0)
    
    glitch_configs = [
        (0.1, 0.3),  # probability, strength
        (0.2, 0.5),
        (0.15, 0.7)
    ]
    
    for prob, strength in glitch_configs:
        print(f"\nTesting glitch: prob={prob}, strength={strength}")
        
        effect = temporal_glitch(prob, strength)
        
        def process_chunk(chunk):
            return effect(chunk)
        
        results = process_video_stream(video_path, process_chunk, chunk_size=15, overlap=3)
        
        print(f"  Processed {len(results)} chunks")

def demo_effect_composition():
    """Demonstrate effect composition and blending"""
    print("\n=== Effect Composition Demo ===")
    
    video_path = "composition_test.mp4"
    create_test_video(video_path, duration=2.0)
    
    # Create individual effects
    smooth_blend = temporal_blend_linear(0.3)
    echo_effect = temporal_echo(2, 0.8, 0.2)
    glitch_effect = temporal_glitch(0.1, 0.3)
    
    # Compose effects in sequence
    print("\n1. Sequential composition (smooth -> echo -> glitch)")
    composed = compose_effects(smooth_blend, echo_effect, glitch_effect)
    
    def process_composed(chunk):
        return composed(chunk)
    
    results = process_video_stream(video_path, process_composed, chunk_size=12, overlap=2)
    print(f"  Processed {len(results)} chunks with composed effects")
    
    # Blend two effects
    print("\n2. Effect blending (smooth + echo)")
    blended = blend_effects(smooth_blend, echo_effect, 0.5)
    
    def process_blended(chunk):
        return blended(chunk)
    
    results = process_video_stream(video_path, process_blended, chunk_size=12, overlap=2)
    print(f"  Processed {len(results)} chunks with blended effects")
    
    # Parallel effects
    print("\n3. Parallel effects")
    parallel = parallel_effects(smooth_blend, echo_effect, glitch_effect)
    
    def process_parallel(chunk):
        results = parallel(chunk)
        print(f"    Parallel results: {len(results)} effect outputs")
        return results[0]  # Return first result
    
    results = process_video_stream(video_path, process_parallel, chunk_size=10, overlap=1)
    print(f"  Processed {len(results)} chunks with parallel effects")

def demo_effect_presets():
    """Demonstrate effect presets"""
    print("\n=== Effect Presets Demo ===")
    
    video_path = "presets_test.mp4"
    create_test_video(video_path, duration=1.5)
    
    presets = create_temporal_blend_presets()
    
    for name, effect in presets.items():
        print(f"\nTesting preset: {name}")
        
        def process_preset(chunk):
            return effect(chunk)
        
        results = process_video_stream(video_path, process_preset, chunk_size=8, overlap=1)
        print(f"  Processed {len(results)} chunks with '{name}' preset")

def demo_performance():
    """Demonstrate effect performance"""
    print("\n=== Performance Demo ===")
    
    video_path = "perf_test.mp4"
    create_test_video(video_path, duration=2.0)
    
    effects = [
        ("linear_blend", temporal_blend_linear(0.5)),
        ("exponential_blend", temporal_blend_exponential(0.7)),
        ("sinusoidal_blend", temporal_blend_sinusoidal(2.0)),
        ("adaptive_blend", temporal_blend_adaptive(0.2)),
        ("echo", temporal_echo(2, 0.8, 0.2)),
        ("glitch", temporal_glitch(0.1, 0.3))
    ]
    
    for name, effect in effects:
        start_time = time.time()
        
        def process_effect(chunk):
            return effect(chunk)
        
        results = process_video_stream(video_path, process_effect, chunk_size=15, overlap=3)
        
        end_time = time.time()
        total_frames = sum(len(chunk) for chunk in results)
        fps = total_frames / (end_time - start_time) if end_time > start_time else 0
        
        print(f"{name:20s}: {len(results):3d} chunks, {total_frames:4d} frames, {fps:6.1f} FPS")

def main():
    """Run all temporal blend demos"""
    print("=== Temporal Blend Effects Demo Suite ===")
    print("Demonstrating various temporal blending effects for video processing")
    
    try:
        demo_linear_blend()
        demo_exponential_blend()
        demo_sinusoidal_blend()
        demo_adaptive_blend()
        demo_echo_effect()
        demo_glitch_effect()
        demo_effect_composition()
        demo_effect_presets()
        demo_performance()
        
        print("\n🎉 All temporal blend demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()












