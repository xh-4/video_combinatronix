#!/usr/bin/env python3
"""
Test the integration between Combinatronix and video processing
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Combinatronix components
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SI'))
from SI_Combinatronix import Categorizer, Reasoner, Rule, ESN, ESNState, LLM

# Import video components
from Combinator_Kernel import VideoFrame, VideoChunk, FieldIQ, make_field_from_real

def test_combinatronix_imports():
    """Test that Combinatronix components can be imported"""
    print("=== Testing Combinatronix Integration ===")
    
    try:
        # Test Categorizer
        def phi(x):
            return np.array([len(str(x)), hash(str(x)) % 100], dtype=float)
        
        prototypes = [("test", np.array([5.0, 50.0]))]
        cat = Categorizer.init(phi, prototypes)
        
        label, score = cat.predict("test_input")
        print(f"✅ Categorizer working: {label} (score: {score:.3f})")
        
        # Test Reasoner
        rules = [Rule(premises=(("test", ()),), conclude=("result", ()))]
        reasoner = Reasoner.init(rules)
        
        kb = {("test", ())}
        derived_kb, proofs = reasoner.run(kb)
        print(f"✅ Reasoner working: {len(derived_kb)} facts derived")
        
        # Test ESN
        esn, state = ESN.init(n_in=2, n_res=8)
        x = np.array([1.0, 2.0])
        new_state = esn.update(state, x)
        print(f"✅ ESN working: state shape {new_state.r.shape}")
        
        # Test LLM
        def logits_fn(ctx, seq):
            return np.random.randn(64)
        
        llm = LLM(logits_fn, vocab_size=64)
        ctx = np.array([1.0, 2.0, 3.0])
        prompt = [1, 2, 3]
        result = llm.decode(ctx, prompt, max_len=8)
        print(f"✅ LLM working: generated {len(result)} tokens")
        
        print("\n🎉 All Combinatronix components working!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_processing():
    """Test video processing components"""
    print("\n=== Testing Video Processing ===")
    
    try:
        # Create synthetic video frame
        frame_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        frame = VideoFrame(
            data=frame_data,
            frame_number=0,
            timestamp=0.0,
            fps=30.0
        )
        
        print(f"✅ VideoFrame created: {frame.data.shape}")
        
        # Convert to FieldIQ
        field = frame.to_field_iq(channel=0)
        print(f"✅ FieldIQ created: {len(field.z)} samples")
        print(f"   Valid quadrature: {field.is_valid_quadrature()}")
        
        # Create video chunk
        chunk = VideoChunk(
            frames=[frame],
            chunk_id=0,
            start_time=0.0,
            end_time=1.0/30.0
        )
        
        print(f"✅ VideoChunk created: {chunk.frame_count} frames")
        
        # Test effects
        from effects import temporal_blend_linear
        effect = temporal_blend_linear(0.5)
        processed = effect(chunk)
        
        print(f"✅ Effect applied: {len(processed)} processed frames")
        
        print("\n🎉 Video processing working!")
        return True
        
    except Exception as e:
        print(f"❌ Video processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intelligent_processor():
    """Test the intelligent video processor"""
    print("\n=== Testing Intelligent Processor ===")
    
    try:
        from intelligent_video_effects import IntelligentVideoProcessor
        
        # Create processor
        processor = IntelligentVideoProcessor()
        print("✅ IntelligentVideoProcessor created")
        
        # Create test chunk
        frame_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        frame = VideoFrame(
            data=frame_data,
            frame_number=0,
            timestamp=0.0,
            fps=30.0
        )
        
        chunk = VideoChunk(
            frames=[frame],
            chunk_id=0,
            start_time=0.0,
            end_time=1.0/30.0
        )
        
        # Process chunk
        processed_frames, info = processor.process_chunk(chunk)
        
        print(f"✅ Chunk processed: {len(processed_frames)} frames")
        print(f"   Category: {info['category']}")
        print(f"   Selected effect: {info['selected_effect']}")
        print(f"   Confidence: {info['confidence']:.3f}")
        
        # Get stats
        stats = processor.get_processing_stats()
        print(f"✅ Stats: {stats}")
        
        print("\n🎉 Intelligent processor working!")
        return True
        
    except Exception as e:
        print(f"❌ Intelligent processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("=== Combinatronix + Video Processing Integration Test ===")
    
    success = True
    
    success &= test_combinatronix_imports()
    success &= test_video_processing()
    success &= test_intelligent_processor()
    
    if success:
        print("\n🎉 All integration tests passed!")
        print("Your Combinatronix system is successfully integrated with video processing!")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()












