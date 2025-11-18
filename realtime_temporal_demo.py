#!/usr/bin/env python3
"""
Real-time Temporal Blend Effects Demo
Shows temporal effects in real-time streaming mode
"""

import sys
import os
import numpy as np
import time
import threading
from queue import Queue

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Combinator_Kernel import VideoFrame, VideoStreamProcessor
from effects import (
    temporal_blend_linear, temporal_blend_exponential, temporal_blend_sinusoidal,
    temporal_blend_adaptive, temporal_echo, temporal_glitch,
    compose_effects, blend_effects, create_temporal_blend_presets
)

class RealtimeTemporalProcessor:
    """Real-time temporal effects processor"""
    
    def __init__(self, effect, chunk_size=15, overlap=5):
        self.effect = effect
        self.processor = VideoStreamProcessor(chunk_size, overlap)
        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.running = False
        self.stats = {
            'frames_processed': 0,
            'chunks_processed': 0,
            'start_time': None,
            'last_fps': 0
        }
    
    def start(self):
        """Start the real-time processor"""
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("✅ Real-time processor started")
    
    def stop(self):
        """Stop the real-time processor"""
        self.running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        print("⏹️ Real-time processor stopped")
    
    def add_frame(self, frame_data, frame_number, timestamp):
        """Add a frame for processing"""
        frame = VideoFrame(
            data=frame_data,
            frame_number=frame_number,
            timestamp=timestamp,
            fps=30.0
        )
        
        self.frame_queue.put(frame)
        return self.processor.add_frame(frame)
    
    def get_result(self, timeout=1.0):
        """Get the next processed result"""
        try:
            return self.result_queue.get(timeout=timeout)
        except:
            return None
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Add to processor
                chunk = self.processor.add_frame(frame)
                
                if chunk is not None:
                    # Process chunk with effect
                    processed_frames = self.effect(chunk)
                    
                    # Update stats
                    self.stats['chunks_processed'] += 1
                    self.stats['frames_processed'] += len(processed_frames)
                    
                    # Calculate FPS
                    if self.stats['start_time']:
                        elapsed = time.time() - self.stats['start_time']
                        if elapsed > 0:
                            self.stats['last_fps'] = self.stats['frames_processed'] / elapsed
                    
                    # Put result in queue
                    self.result_queue.put({
                        'chunk': chunk,
                        'processed_frames': processed_frames,
                        'stats': self.stats.copy()
                    })
                
            except:
                continue
    
    def get_stats(self):
        """Get current processing statistics"""
        return self.stats.copy()

def create_synthetic_frame(frame_number, width=320, height=240):
    """Create a synthetic video frame with temporal patterns"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    t = frame_number / 30.0  # 30 FPS
    
    # Multiple moving objects with different temporal characteristics
    objects = [
        # Fast moving circle
        {
            'type': 'circle',
            'center': (int(width * (0.2 + 0.6 * (t * 2) % 1)), int(height * 0.3)),
            'radius': 25,
            'color': (0, 255, 0)
        },
        # Slow moving rectangle
        {
            'type': 'rectangle',
            'pos': (int(width * (0.1 + 0.8 * (t * 0.5) % 1)), int(height * 0.7)),
            'size': (40, 30),
            'color': (255, 0, 0)
        },
        # Pulsing triangle
        {
            'type': 'triangle',
            'center': (int(width * 0.8), int(height * 0.5)),
            'size': int(20 * (0.5 + 0.5 * np.sin(2 * np.pi * t * 3))),
            'color': (0, 0, 255)
        },
        # Rotating square
        {
            'type': 'rotating_square',
            'center': (int(width * 0.5), int(height * 0.5)),
            'size': 30,
            'angle': t * 2 * np.pi,
            'color': (255, 255, 0)
        }
    ]
    
    # Draw objects
    for obj in objects:
        if obj['type'] == 'circle':
            cv2.circle(frame, obj['center'], obj['radius'], obj['color'], -1)
        elif obj['type'] == 'rectangle':
            x, y = obj['pos']
            w, h = obj['size']
            cv2.rectangle(frame, (x, y), (x + w, y + h), obj['color'], -1)
        elif obj['type'] == 'triangle':
            size = obj['size']
            center = obj['center']
            pts = np.array([
                [center[0], center[1] - size],
                [center[0] - size, center[1] + size],
                [center[0] + size, center[1] + size]
            ], np.int32)
            cv2.fillPoly(frame, [pts], obj['color'])
        elif obj['type'] == 'rotating_square':
            # Simple rotating square approximation
            size = obj['size']
            center = obj['center']
            angle = obj['angle']
            
            # Calculate rotated corners
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            half_size = size // 2
            
            corners = np.array([
                [-half_size, -half_size],
                [half_size, -half_size],
                [half_size, half_size],
                [-half_size, half_size]
            ])
            
            # Rotate corners
            rotated = np.array([
                [center[0] + x * cos_a - y * sin_a, center[1] + x * sin_a + y * cos_a]
                for x, y in corners
            ], np.int32)
            
            cv2.fillPoly(frame, [rotated], obj['color'])
    
    # Add frame info
    cv2.putText(frame, f"Frame {frame_number}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def demo_realtime_linear_blend():
    """Demo real-time linear blending"""
    print("\n=== Real-time Linear Blend Demo ===")
    
    # Create processor
    effect = temporal_blend_linear(0.4)
    processor = RealtimeTemporalProcessor(effect, chunk_size=12, overlap=3)
    
    try:
        processor.start()
        
        print("Generating frames and processing in real-time...")
        
        for frame_num in range(60):  # 2 seconds at 30 FPS
            # Create synthetic frame
            frame_data = create_synthetic_frame(frame_num)
            
            # Add frame to processor
            chunk = processor.add_frame(frame_data, frame_num, frame_num / 30.0)
            
            # Get results if available
            result = processor.get_result(timeout=0.1)
            if result:
                stats = result['stats']
                print(f"  Chunk {result['chunk'].chunk_id}: {len(result['processed_frames'])} frames, "
                      f"FPS: {stats['last_fps']:.1f}")
            
            # Simulate real-time frame rate
            time.sleep(1/30.0)
        
        # Get final stats
        final_stats = processor.get_stats()
        print(f"\nFinal stats:")
        print(f"  Total frames processed: {final_stats['frames_processed']}")
        print(f"  Total chunks processed: {final_stats['chunks_processed']}")
        print(f"  Average FPS: {final_stats['last_fps']:.1f}")
        
    finally:
        processor.stop()

def demo_realtime_adaptive_blend():
    """Demo real-time adaptive blending"""
    print("\n=== Real-time Adaptive Blend Demo ===")
    
    effect = temporal_blend_adaptive(0.3)
    processor = RealtimeTemporalProcessor(effect, chunk_size=15, overlap=5)
    
    try:
        processor.start()
        
        print("Generating frames with varying complexity...")
        
        for frame_num in range(90):  # 3 seconds
            # Create frame with varying complexity
            complexity = 0.5 + 0.5 * np.sin(2 * np.pi * frame_num / 30)
            frame_data = create_synthetic_frame(frame_num, complexity=complexity)
            
            processor.add_frame(frame_data, frame_num, frame_num / 30.0)
            
            # Show adaptive behavior
            result = processor.get_result(timeout=0.1)
            if result and result['chunk'].chunk_id % 3 == 0:  # Every 3rd chunk
                stats = result['stats']
                print(f"  Chunk {result['chunk'].chunk_id}: "
                      f"FPS: {stats['last_fps']:.1f}, "
                      f"Frames: {len(result['processed_frames'])}")
            
            time.sleep(1/30.0)
        
        final_stats = processor.get_stats()
        print(f"\nAdaptive processing completed:")
        print(f"  Frames: {final_stats['frames_processed']}, "
              f"Chunks: {final_stats['chunks_processed']}, "
              f"FPS: {final_stats['last_fps']:.1f}")
        
    finally:
        processor.stop()

def demo_realtime_effect_switching():
    """Demo switching between different effects in real-time"""
    print("\n=== Real-time Effect Switching Demo ===")
    
    # Create multiple effects
    effects = {
        'smooth': temporal_blend_linear(0.3),
        'sharp': temporal_blend_linear(0.8),
        'dreamy': temporal_blend_exponential(0.6),
        'rhythmic': temporal_blend_sinusoidal(2.0),
        'echo': temporal_echo(2, 0.8, 0.2),
        'glitch': temporal_glitch(0.1, 0.3)
    }
    
    current_effect = 'smooth'
    processor = RealtimeTemporalProcessor(effects[current_effect], chunk_size=10, overlap=2)
    
    try:
        processor.start()
        
        print("Switching effects every 30 frames...")
        
        for frame_num in range(180):  # 6 seconds
            # Switch effect every 30 frames
            if frame_num % 30 == 0 and frame_num > 0:
                effect_names = list(effects.keys())
                current_effect = effect_names[(frame_num // 30) % len(effect_names)]
                processor.effect = effects[current_effect]
                print(f"  Switched to effect: {current_effect}")
            
            # Create frame
            frame_data = create_synthetic_frame(frame_num)
            processor.add_frame(frame_data, frame_num, frame_num / 30.0)
            
            # Show results
            result = processor.get_result(timeout=0.1)
            if result and result['chunk'].chunk_id % 2 == 0:
                stats = result['stats']
                print(f"    Chunk {result['chunk'].chunk_id}: {current_effect}, "
                      f"FPS: {stats['last_fps']:.1f}")
            
            time.sleep(1/30.0)
        
        final_stats = processor.get_stats()
        print(f"\nEffect switching completed:")
        print(f"  Frames: {final_stats['frames_processed']}, "
              f"Chunks: {final_stats['chunks_processed']}, "
              f"FPS: {final_stats['last_fps']:.1f}")
        
    finally:
        processor.stop()

def demo_realtime_composition():
    """Demo real-time effect composition"""
    print("\n=== Real-time Effect Composition Demo ===")
    
    # Create composed effect
    base_effect = temporal_blend_linear(0.5)
    echo_effect = temporal_echo(2, 0.8, 0.2)
    composed = compose_effects(base_effect, echo_effect)
    
    processor = RealtimeTemporalProcessor(composed, chunk_size=12, overlap=3)
    
    try:
        processor.start()
        
        print("Processing with composed effects (blend + echo)...")
        
        for frame_num in range(75):  # 2.5 seconds
            frame_data = create_synthetic_frame(frame_num)
            processor.add_frame(frame_data, frame_num, frame_num / 30.0)
            
            result = processor.get_result(timeout=0.1)
            if result and result['chunk'].chunk_id % 2 == 0:
                stats = result['stats']
                print(f"  Chunk {result['chunk'].chunk_id}: "
                      f"FPS: {stats['last_fps']:.1f}, "
                      f"Frames: {len(result['processed_frames'])}")
            
            time.sleep(1/30.0)
        
        final_stats = processor.get_stats()
        print(f"\nComposed effect processing completed:")
        print(f"  Frames: {final_stats['frames_processed']}, "
              f"Chunks: {final_stats['chunks_processed']}, "
              f"FPS: {final_stats['last_fps']:.1f}")
        
    finally:
        processor.stop()

def main():
    """Run all real-time temporal demos"""
    print("=== Real-time Temporal Blend Effects Demo ===")
    print("Demonstrating real-time temporal effects processing")
    
    try:
        # Import OpenCV for drawing
        try:
            import cv2
        except ImportError:
            import mock_opencv as cv2
        
        demo_realtime_linear_blend()
        demo_realtime_adaptive_blend()
        demo_realtime_effect_switching()
        demo_realtime_composition()
        
        print("\n🎉 All real-time temporal demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()












