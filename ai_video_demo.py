#!/usr/bin/env python3
"""
AI-Driven Video Effects Demo
Demonstrates the full integration of Combinatronix with video processing
"""

import sys
import os
import numpy as np
import time
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intelligent_video_effects import IntelligentVideoProcessor, create_test_video
from Combinator_Kernel import load_video_chunks, VideoFrame, VideoChunk

def create_diverse_test_video(output_path="ai_demo_video.mp4", duration=6.0, fps=30):
    """Create a diverse test video with different content types"""
    try:
        import cv2
    except ImportError:
        import mock_opencv as cv2
    
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        t = frame_num / fps
        
        # Create different content types over time
        if t < 1.0:
            # Static content - low motion, low complexity
            cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), -1)
            cv2.putText(frame, "STATIC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif t < 2.0:
            # Motion content - high motion, medium complexity
            x = int(50 + 100 * (t - 1.0))
            cv2.circle(frame, (x, 120), 30, (255, 0, 0), -1)
            cv2.putText(frame, "MOTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif t < 3.0:
            # Complex content - medium motion, high complexity
            for i in range(8):
                x = int(50 + i * 30 + 20 * np.sin(2 * np.pi * t * 2))
                y = int(100 + 20 * np.cos(2 * np.pi * t * 3))
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame, "COMPLEX", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif t < 4.0:
            # Dramatic content - high everything
            for i in range(5):
                x = int(50 + i * 50 + 30 * np.sin(2 * np.pi * t * 4))
                y = int(100 + 30 * np.cos(2 * np.pi * t * 5))
                cv2.rectangle(frame, (x-15, y-15), (x+15, y+15), (255, 255, 0), -1)
            cv2.putText(frame, "DRAMATIC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif t < 5.0:
            # Subtle content - low everything
            cv2.circle(frame, (160, 120), 20, (128, 128, 128), -1)
            cv2.putText(frame, "SUBTLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        else:
            # Rhythmic content - rhythmic patterns
            for i in range(6):
                x = int(50 + i * 40)
                y = int(120 + 20 * np.sin(2 * np.pi * t * 3 + i))
                cv2.circle(frame, (x, y), 15, (255, 0, 255), -1)
            cv2.putText(frame, "RHYTHMIC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame {frame_num}", (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Created diverse test video: {output_path}")

def demo_ai_video_processing():
    """Demonstrate AI-driven video processing"""
    print("=== AI-Driven Video Effects Demo ===")
    print("This demo shows how AI reasoning selects appropriate effects for different video content")
    
    # Create diverse test video
    video_path = "ai_demo_video.mp4"
    create_diverse_test_video(video_path, duration=6.0)
    
    # Create intelligent processor
    processor = IntelligentVideoProcessor()
    
    print(f"\nProcessing video: {video_path}")
    print("Analyzing content and selecting effects intelligently...\n")
    
    # Process video chunks
    chunk_count = 0
    processing_times = []
    
    for chunk in load_video_chunks(video_path, chunk_size=20, overlap=5):
        chunk_count += 1
        start_time = time.time()
        
        # Process chunk with AI
        processed_frames, info = processor.process_chunk(chunk)
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # Display results
        print(f"Chunk {chunk_count}:")
        print(f"  Time: {chunk.start_time:.2f}s - {chunk.end_time:.2f}s")
        print(f"  Content: {info['category']} (confidence: {info['confidence']:.3f})")
        print(f"  Motion: {info['context'].motion_level:.3f}, Complexity: {info['context'].complexity_level:.3f}")
        print(f"  AI Selected: {info['selected_effect']}")
        print(f"  Description: {info['effect_description']}")
        print(f"  Processing time: {processing_time*1000:.1f}ms")
        print(f"  Frames processed: {len(processed_frames)}")
        print()
        
        if chunk_count >= 8:  # Limit output
            break
    
    # Train ESN on accumulated data
    print("Training ESN on accumulated data...")
    processor.train_on_history()
    
    # Show statistics
    stats = processor.get_processing_stats()
    print(f"\n=== Processing Statistics ===")
    print(f"Total chunks processed: {stats['total_chunks']}")
    print(f"Average processing time: {np.mean(processing_times)*1000:.1f}ms")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    
    print(f"\nCategory Distribution:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count} chunks")
    
    print(f"\nEffect Distribution:")
    for effect, count in stats['effect_distribution'].items():
        print(f"  {effect}: {count} chunks")
    
    return processor

def demo_effect_learning():
    """Demonstrate how the system learns from experience"""
    print("\n=== Effect Learning Demo ===")
    print("Showing how the ESN learns to predict better effects over time")
    
    processor = IntelligentVideoProcessor()
    
    # Create test chunks with known characteristics
    test_chunks = []
    
    # Static chunk
    static_frame = create_test_frame("static", motion=0.1, complexity=0.2)
    test_chunks.append(("static", static_frame))
    
    # Motion chunk
    motion_frame = create_test_frame("motion", motion=0.8, complexity=0.4)
    test_chunks.append(("motion", motion_frame))
    
    # Complex chunk
    complex_frame = create_test_frame("complex", motion=0.5, complexity=0.9)
    test_chunks.append(("complex", complex_frame))
    
    print("Processing test chunks...")
    
    for i, (chunk_type, frame_data) in enumerate(test_chunks):
        # Create chunk
        frame = VideoFrame(
            data=frame_data,
            frame_number=i,
            timestamp=i * 0.1,
            fps=30.0
        )
        
        chunk = VideoChunk(
            frames=[frame],
            chunk_id=i,
            start_time=i * 0.1,
            end_time=(i + 1) * 0.1
        )
        
        # Process chunk
        processed_frames, info = processor.process_chunk(chunk)
        
        print(f"  {chunk_type.capitalize()} chunk:")
        print(f"    Category: {info['category']}")
        print(f"    Effect: {info['selected_effect']}")
        print(f"    Confidence: {info['confidence']:.3f}")
    
    # Train ESN
    processor.train_on_history()
    print("\n✅ ESN trained on test data")
    
    # Show learning results
    stats = processor.get_processing_stats()
    print(f"\nLearning Results:")
    print(f"  Categories learned: {list(stats['category_distribution'].keys())}")
    print(f"  Effects learned: {list(stats['effect_distribution'].keys())}")
    print(f"  Average confidence: {stats['avg_confidence']:.3f}")

def create_test_frame(frame_type: str, motion: float, complexity: float) -> np.ndarray:
    """Create a test frame with specific characteristics"""
    try:
        import cv2
    except ImportError:
        import mock_opencv as cv2
    
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    
    if frame_type == "static":
        # Static content
        cv2.rectangle(frame, (20, 20), (44, 44), (0, 255, 0), -1)
    elif frame_type == "motion":
        # Motion content
        x = int(32 + 20 * motion)
        cv2.circle(frame, (x, 32), 15, (255, 0, 0), -1)
    elif frame_type == "complex":
        # Complex content
        for i in range(int(5 * complexity)):
            x = int(32 + 20 * np.sin(i))
            y = int(32 + 20 * np.cos(i))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    return frame

def demo_real_time_ai_processing():
    """Demonstrate real-time AI processing"""
    print("\n=== Real-time AI Processing Demo ===")
    print("Simulating real-time video processing with AI effect selection")
    
    processor = IntelligentVideoProcessor()
    
    # Simulate real-time processing
    print("Simulating real-time video stream...")
    
    for frame_num in range(60):  # 2 seconds at 30 FPS
        # Create frame with varying content
        t = frame_num / 30.0
        
        if t < 0.5:
            motion = 0.2
            complexity = 0.3
        elif t < 1.0:
            motion = 0.8
            complexity = 0.4
        else:
            motion = 0.5
            complexity = 0.9
        
        frame_data = create_test_frame("dynamic", motion, complexity)
        
        # Create frame
        frame = VideoFrame(
            data=frame_data,
            frame_number=frame_num,
            timestamp=t,
            fps=30.0
        )
        
        # Process every 10 frames as a chunk
        if frame_num % 10 == 0 and frame_num > 0:
            # Create chunk from recent frames
            recent_frames = [frame]  # Simplified - in real implementation, collect frames
            
            chunk = VideoChunk(
                frames=recent_frames,
                chunk_id=frame_num // 10,
                start_time=t - 0.33,
                end_time=t
            )
            
            # Process with AI
            processed_frames, info = processor.process_chunk(chunk)
            
            print(f"  Frame {frame_num}: {info['category']} -> {info['selected_effect']}")
    
    print("✅ Real-time processing simulation completed")

def main():
    """Run all AI video processing demos"""
    print("=== AI-Driven Video Effects System Demo ===")
    print("Combining Combinatronix AI reasoning with video processing")
    print("=" * 60)
    
    try:
        # Import OpenCV for drawing
        try:
            import cv2
        except ImportError:
            import mock_opencv as cv2
        
        # Run demos
        processor = demo_ai_video_processing()
        demo_effect_learning()
        demo_real_time_ai_processing()
        
        print("\n🎉 All AI video processing demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✅ AI-driven content categorization")
        print("  ✅ Rule-based effect selection")
        print("  ✅ ESN temporal pattern learning")
        print("  ✅ LLM effect description generation")
        print("  ✅ Real-time processing capabilities")
        print("  ✅ Adaptive learning from experience")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
