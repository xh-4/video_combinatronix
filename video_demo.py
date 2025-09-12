#!/usr/bin/env python3
"""
Comprehensive video processing demo using the Combinator Kernel
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Combinator_Kernel import (
    VideoFrame, VideoChunk, VideoStreamProcessor,
    load_video_chunks, process_video_stream,
    video_frame_processor, video_temporal_processor, video_spectral_analyzer,
    lowpass_hz, amp, phase_deg, freq_shift, B, split_add, split_mul
)

def demo_video_processing():
    """Demonstrate video processing capabilities"""
    print("=== Video Processing Demo with Combinator Kernel ===\n")
    
    # Create a test video file
    video_path = "demo_video.mp4"
    if not os.path.exists(video_path):
        print("Creating demo video...")
        create_demo_video(video_path)
    
    print("1. Basic Video Chunking")
    print("-" * 30)
    chunk_count = 0
    for chunk in load_video_chunks(video_path, chunk_size=20, overlap=5):
        chunk_count += 1
        print(f"Chunk {chunk.chunk_id}: {chunk.frame_count} frames, "
              f"duration {chunk.duration:.3f}s")
        if chunk_count >= 3:
            break
    
    print(f"\n✅ Processed {chunk_count} video chunks")
    
    print("\n2. Frame-by-Frame Processing")
    print("-" * 30)
    
    # Create a processor that applies lowpass filter and amplitude scaling
    frame_processor = video_frame_processor(
        B(lowpass_hz(800.0))(amp(0.7)),
        channel=0  # Blue channel
    )
    
    # Process first chunk
    chunk = next(load_video_chunks(video_path, chunk_size=10, overlap=2))
    processed_frames = frame_processor(chunk)
    
    print(f"✅ Processed {len(processed_frames)} frames")
    print(f"First frame FieldIQ length: {len(processed_frames[0].z)}")
    print(f"Valid quadrature: {processed_frames[0].is_valid_quadrature()}")
    
    print("\n3. Temporal Processing")
    print("-" * 30)
    
    # Create a temporal processor that works on the entire chunk as one signal
    temporal_processor = video_temporal_processor(
        B(lowpass_hz(1000.0))(
            split_add(phase_deg(45.0, 1000.0))  # Add 45-degree phase shift
        ),
        channel=1  # Green channel
    )
    
    chunk = next(load_video_chunks(video_path, chunk_size=15, overlap=3))
    temporal_result = temporal_processor(chunk)
    
    if temporal_result:
        print(f"✅ Temporal processing completed")
        print(f"Result length: {len(temporal_result.z)}")
        print(f"Valid quadrature: {temporal_result.is_valid_quadrature()}")
        print(f"Roles: {temporal_result.roles}")
    
    print("\n4. Spectral Analysis")
    print("-" * 30)
    
    # Analyze different channels
    for channel in [0, 1, 2]:  # RGB channels
        analyzer = video_spectral_analyzer(channel=channel)
        chunk = next(load_video_chunks(video_path, chunk_size=12, overlap=2))
        analysis = analyzer(chunk)
        
        print(f"Channel {channel} analysis:")
        print(f"  Dominant frequency: {analysis['dominant_freq']:.2f} Hz")
        print(f"  Spectral centroid: {analysis['spectral_centroid']:.4f}")
        print(f"  Total power: {analysis['total_power']:.2f}")
        print(f"  Valid quadrature: {analysis['is_valid_quadrature']}")
    
    print("\n5. Advanced Combinator Processing")
    print("-" * 30)
    
    # Create complex processing chains using combinators
    wobble_effect = split_mul(freq_shift(10.0))  # Ring modulation
    chorus_effect = split_add(phase_deg(90.0, 500.0))  # Chorus
    filter_chain = B(lowpass_hz(600.0))(amp(0.8))
    
    # Compose effects
    complex_processor = video_frame_processor(
        B(filter_chain)(
            B(chorus_effect)(wobble_effect)
        ),
        channel=0
    )
    
    chunk = next(load_video_chunks(video_path, chunk_size=8, overlap=1))
    complex_result = complex_processor(chunk)
    
    print(f"✅ Complex processing completed")
    print(f"Processed {len(complex_result)} frames with combined effects")
    
    print("\n6. Streaming Processing")
    print("-" * 30)
    
    # Define a streaming processor function
    def stream_processor(chunk: VideoChunk):
        # Convert to FieldIQ sequence
        fields = chunk.to_field_iq_sequence(channel=0)
        
        # Process each frame
        processed = []
        for field in fields:
            # Apply processing chain
            result = B(lowpass_hz(500.0))(
                split_add(phase_deg(30.0, 1000.0))
            )(field)
            processed.append(result)
        
        return {
            'chunk_id': chunk.chunk_id,
            'frame_count': len(processed),
            'avg_power': np.mean([np.sum(field.power) for field in processed]),
            'valid_quadrature': all(field.is_valid_quadrature() for field in processed)
        }
    
    # Process video stream
    results = process_video_stream(video_path, stream_processor, chunk_size=10, overlap=2)
    
    print(f"✅ Stream processing completed")
    print(f"Processed {len(results)} chunks")
    for i, result in enumerate(results[:3]):  # Show first 3 results
        print(f"  Chunk {result['chunk_id']}: {result['frame_count']} frames, "
              f"avg power {result['avg_power']:.2f}, "
              f"valid quadrature: {result['valid_quadrature']}")
    
    print("\n🎉 All video processing demos completed successfully!")

def create_demo_video(output_path="demo_video.mp4", duration=15.0, fps=30):
    """Create a demo video with interesting patterns"""
    try:
        import cv2
    except ImportError:
        import mock_opencv as cv2
    
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_num in range(total_frames):
        # Create frame with multiple moving patterns
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving circles
        t = frame_num / fps
        for i in range(3):
            center_x = int(width * (0.2 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * t * (i + 1)))))
            center_y = int(height * (0.2 + 0.6 * (0.5 + 0.5 * np.cos(2 * np.pi * t * (i + 1)))))
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]
            cv2.circle(frame, (center_x, center_y), 30, color, -1)
        
        # Moving rectangles
        rect_x = int(width * (0.1 + 0.8 * (frame_num / total_frames)))
        rect_y = int(height * 0.8)
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 40, rect_y + 30), (255, 255, 0), -1)
        
        # Add text
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Demo video created: {output_path}")

if __name__ == "__main__":
    demo_video_processing()
