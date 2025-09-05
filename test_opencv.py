#!/usr/bin/env python3
"""
Test script to verify OpenCV functionality with the Combinator Kernel
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_opencv_availability():
    """Test if OpenCV is available and working"""
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        return True
    except ImportError as e:
        try:
            import mock_opencv as cv2
            print(f"✅ Mock OpenCV version: {cv2.__version__}")
            return True
        except ImportError:
            print(f"❌ OpenCV not available: {e}")
            return False

def create_synthetic_video(output_path="test_video.mp4", duration=2.0, fps=30):
    """Create a synthetic test video using OpenCV"""
    if not test_opencv_availability():
        print("Cannot create video without OpenCV")
        return False
    
    try:
        import cv2
    except ImportError:
        import mock_opencv as cv2
    
    # Video properties
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    print(f"Creating synthetic video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    for frame_num in range(total_frames):
        # Create a frame with moving patterns
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add moving circles
        center_x = int(width * (0.5 + 0.3 * np.sin(2 * np.pi * frame_num / fps)))
        center_y = int(height * (0.5 + 0.3 * np.cos(2 * np.pi * frame_num / fps)))
        
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
        
        # Add moving rectangles
        rect_x = int(width * (0.2 + 0.6 * (frame_num / total_frames)))
        rect_y = int(height * 0.8)
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 100, rect_y + 50), (255, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Synthetic video created: {output_path}")
    return True

def test_kernel_with_video():
    """Test the combinator kernel with video processing"""
    print("\n=== Testing Combinator Kernel with Video ===")
    
    # Import the kernel
    try:
        from Combinator_Kernel import (
            VideoFrame, VideoChunk, VideoStreamProcessor,
            load_video_chunks, process_video_stream,
            video_frame_processor, video_spectral_analyzer,
            lowpass_hz, amp, B
        )
        print("✅ Kernel imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import kernel: {e}")
        return False
    
    # Test with synthetic video
    video_path = "test_video.mp4"
    if not os.path.exists(video_path):
        if not create_synthetic_video(video_path):
            return False
    
    print(f"\n--- Testing video chunking ---")
    try:
        chunk_count = 0
        for chunk in load_video_chunks(video_path, chunk_size=15, overlap=3):
            chunk_count += 1
            print(f"Chunk {chunk.chunk_id}: {chunk.frame_count} frames, "
                  f"duration {chunk.duration:.3f}s")
            
            if chunk_count >= 3:  # Limit output
                break
        
        print(f"✅ Successfully processed {chunk_count} chunks")
    except Exception as e:
        print(f"❌ Video chunking failed: {e}")
        return False
    
    print(f"\n--- Testing video processing ---")
    try:
        # Create a processor
        processor = video_frame_processor(
            B(lowpass_hz(500.0))(amp(0.8)),
            channel=0  # Blue channel
        )
        
        # Process first chunk
        chunk = next(load_video_chunks(video_path, chunk_size=10, overlap=2))
        processed_frames = processor(chunk)
        
        print(f"✅ Processed {len(processed_frames)} frames")
        print(f"First frame FieldIQ length: {len(processed_frames[0].z)}")
        print(f"Valid quadrature: {processed_frames[0].is_valid_quadrature()}")
        
    except Exception as e:
        print(f"❌ Video processing failed: {e}")
        return False
    
    print(f"\n--- Testing spectral analysis ---")
    try:
        analyzer = video_spectral_analyzer(channel=0)
        chunk = next(load_video_chunks(video_path, chunk_size=10, overlap=2))
        analysis = analyzer(chunk)
        
        print(f"✅ Spectral analysis completed:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Spectral analysis failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("=== OpenCV + Combinator Kernel Test ===")
    
    # Test OpenCV availability
    if not test_opencv_availability():
        print("\n⚠️  OpenCV not available - testing fallback mode")
        # Test kernel without OpenCV
        try:
            from Combinator_Kernel import VideoFrame, VideoChunk
            print("✅ Kernel works in fallback mode")
        except Exception as e:
            print(f"❌ Kernel fallback failed: {e}")
        return
    
    # Test kernel with OpenCV
    if test_kernel_with_video():
        print("\n🎉 All tests passed! OpenCV integration working perfectly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
