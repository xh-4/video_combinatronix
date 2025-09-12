#!/usr/bin/env python3
"""
Streaming processing demo using the Combinator Kernel
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Combinator_Kernel import (
    VideoFrame, VideoStreamProcessor, VideoChunk,
    load_video_chunks, process_video_stream,
    video_frame_processor, video_temporal_processor, video_spectral_analyzer,
    lowpass_hz, amp, phase_deg, freq_shift, B, split_add, split_mul
)

def create_synthetic_frame(frame_number, width=320, height=240):
    """Create a synthetic video frame for testing"""
    # Create frame with moving patterns
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Moving circle
    t = frame_number / 30.0  # 30 FPS
    center_x = int(width * (0.5 + 0.3 * np.sin(2 * np.pi * t)))
    center_y = int(height * (0.5 + 0.3 * np.cos(2 * np.pi * t)))
    
    # Draw circle
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= 30**2
    frame[mask] = [0, 255, 0]  # Green circle
    
    # Moving rectangle
    rect_x = int(width * (0.1 + 0.8 * (frame_number % 60) / 60))
    rect_y = int(height * 0.8)
    frame[rect_y:rect_y+20, rect_x:rect_x+40] = [255, 0, 0]  # Red rectangle
    
    return frame

def demo_streaming_processing():
    """Demonstrate streaming processing capabilities"""
    print("=== Streaming Processing Demo ===\n")
    
    print("1. Real-time Frame Processing")
    print("-" * 30)
    
    # Create streaming processor
    processor = VideoStreamProcessor(chunk_size=15, overlap=3)
    
    # Simulate real-time frame processing
    print("Processing frames in real-time...")
    chunk_count = 0
    
    for frame_num in range(60):  # 2 seconds at 30 FPS
        # Create synthetic frame
        frame_data = create_synthetic_frame(frame_num)
        
        # Create VideoFrame
        video_frame = VideoFrame(
            data=frame_data,
            frame_number=frame_num,
            timestamp=frame_num / 30.0,
            fps=30.0
        )
        
        # Add frame to processor
        chunk = processor.add_frame(video_frame)
        
        if chunk is not None:
            chunk_count += 1
            print(f"  Chunk {chunk.chunk_id}: {chunk.frame_count} frames, "
                  f"duration {chunk.duration:.3f}s")
            
            # Process chunk
            process_chunk_realtime(chunk)
            
            if chunk_count >= 3:  # Limit output
                break
    
    print(f"✅ Processed {chunk_count} chunks in real-time")
    
    print("\n2. Chunked Video Processing")
    print("-" * 30)
    
    # Create a test video file
    video_path = "streaming_test.mp4"
    create_test_video(video_path)
    
    # Process video in chunks
    print(f"Processing video file: {video_path}")
    chunk_count = 0
    
    for chunk in load_video_chunks(video_path, chunk_size=20, overlap=5):
        chunk_count += 1
        print(f"  Processing chunk {chunk.chunk_id}: {chunk.frame_count} frames")
        
        # Process chunk
        result = process_chunk_batch(chunk)
        print(f"    Result: {result}")
        
        if chunk_count >= 3:  # Limit output
            break
    
    print(f"✅ Processed {chunk_count} video chunks")
    
    print("\n3. Streaming with Custom Processors")
    print("-" * 30)
    
    # Define custom processors
    def realtime_analyzer(chunk):
        """Real-time spectral analyzer"""
        fields = chunk.to_field_iq_sequence(channel=0)
        if not fields:
            return None
        
        # Analyze first frame
        field = fields[0]
        fft_data = np.fft.fft(field.z)
        freqs = np.fft.fftfreq(len(field.z), 1/48000)
        
        return {
            'chunk_id': chunk.chunk_id,
            'timestamp': chunk.start_time,
            'dominant_freq': freqs[np.argmax(np.abs(fft_data))],
            'power': np.sum(field.power),
            'valid_quadrature': field.is_valid_quadrature()
        }
    
    def effect_processor(chunk):
        """Apply effects to chunk"""
        processor = video_frame_processor(
            B(lowpass_hz(1000.0))(
                split_add(phase_deg(45.0, 1000.0))
            ),
            channel=1
        )
        
        processed_frames = processor(chunk)
        return {
            'chunk_id': chunk.chunk_id,
            'processed_frames': len(processed_frames),
            'avg_power': np.mean([np.sum(field.power) for field in processed_frames])
        }
    
    # Process with different processors
    print("Running real-time analyzer...")
    analyzer_results = process_video_stream(video_path, realtime_analyzer, chunk_size=10)
    
    print("Running effect processor...")
    effect_results = process_video_stream(video_path, effect_processor, chunk_size=10)
    
    print(f"✅ Analyzer processed {len(analyzer_results)} chunks")
    print(f"✅ Effect processor processed {len(effect_results)} chunks")
    
    # Show sample results
    if analyzer_results:
        sample = analyzer_results[0]
        print(f"Sample analysis: freq={sample['dominant_freq']:.2f}Hz, "
              f"power={sample['power']:.2f}")
    
    print("\n4. Performance Monitoring")
    print("-" * 30)
    
    # Monitor processing performance
    start_time = time.time()
    
    def performance_monitor(chunk):
        """Monitor processing performance"""
        process_start = time.time()
        
        # Simulate processing
        fields = chunk.to_field_iq_sequence(channel=0)
        processed = [B(lowpass_hz(500.0))(amp(0.8))(field) for field in fields]
        
        process_time = time.time() - process_start
        
        return {
            'chunk_id': chunk.chunk_id,
            'frame_count': len(processed),
            'process_time': process_time,
            'frames_per_second': len(processed) / process_time if process_time > 0 else 0
        }
    
    perf_results = process_video_stream(video_path, performance_monitor, chunk_size=15)
    
    total_time = time.time() - start_time
    total_frames = sum(result['frame_count'] for result in perf_results)
    
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Total frames processed: {total_frames}")
    print(f"Overall FPS: {total_frames / total_time:.2f}")
    
    if perf_results:
        avg_fps = np.mean([result['frames_per_second'] for result in perf_results])
        print(f"Average chunk FPS: {avg_fps:.2f}")
    
    print("\n🎉 Streaming processing demo completed!")

def process_chunk_realtime(chunk):
    """Process chunk in real-time"""
    # Convert to FieldIQ
    fields = chunk.to_field_iq_sequence(channel=0)
    
    # Apply simple processing
    processed = [B(lowpass_hz(1000.0))(amp(0.8))(field) for field in fields]
    
    # Calculate statistics
    avg_power = np.mean([np.sum(field.power) for field in processed])
    valid_quadrature = all(field.is_valid_quadrature() for field in processed)
    
    print(f"    Processed {len(processed)} frames, avg power: {avg_power:.2f}, "
          f"valid quadrature: {valid_quadrature}")

def process_chunk_batch(chunk):
    """Process chunk in batch mode"""
    # Apply spectral analysis
    analyzer = video_spectral_analyzer(channel=0)
    analysis = analyzer(chunk)
    
    return {
        'chunk_id': chunk.chunk_id,
        'frame_count': chunk.frame_count,
        'duration': chunk.duration,
        'dominant_freq': analysis['dominant_freq'],
        'total_power': analysis['total_power']
    }

def create_test_video(output_path, duration=2.0, fps=30):
    """Create a test video file"""
    try:
        import cv2
    except ImportError:
        import mock_opencv as cv2
    
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_num in range(total_frames):
        frame_data = create_synthetic_frame(frame_num, width, height)
        out.write(frame_data)
    
    out.release()
    print(f"Created test video: {output_path}")

if __name__ == "__main__":
    demo_streaming_processing()



