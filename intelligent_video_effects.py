#!/usr/bin/env python3
"""
Intelligent Video Effects System
Integrates SI_Combinatronix with Combinator Kernel for AI-driven video processing
"""

import sys
import os
import numpy as np
from typing import Any, Callable, List, Dict, Tuple, Optional, Sequence
from dataclasses import dataclass
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Combinatronix components
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SI'))
from SI_Combinatronix import (
    Categorizer, Prototype, Reasoner, Rule, ESN, ESNState, ESNParams,
    LLM, WeaveOutputs, Atom, KB, Vector, Matrix, Label
)

# Import video processing components
from Combinator_Kernel import (
    VideoFrame, VideoChunk, VideoStreamProcessor, FieldIQ,
    load_video_chunks, process_video_stream, video_frame_processor,
    lowpass_hz, amp, phase_deg, freq_shift, B, split_add, split_mul
)

# Import effects
from effects import (
    temporal_blend_linear, temporal_blend_exponential, temporal_blend_sinusoidal,
    temporal_blend_adaptive, temporal_echo, temporal_glitch, temporal_morph,
    compose_effects, blend_effects, create_temporal_blend_presets, EffectParams
)

# ---------- Video-Specific Combinatronix Extensions ----------

@dataclass
class VideoEffect:
    """Represents a video effect with metadata"""
    name: str
    effect_func: Callable[[VideoChunk], List[FieldIQ]]
    parameters: Dict[str, Any]
    category: str
    intensity: float = 1.0
    temporal_stability: float = 0.5  # How stable the effect is over time

@dataclass
class VideoContext:
    """Context for video processing decisions"""
    frame_features: Vector
    temporal_features: Vector
    scene_category: Label
    motion_level: float
    complexity_level: float
    color_dominance: Vector
    audio_features: Optional[Vector] = None

class VideoCategorizer:
    """Specialized categorizer for video content analysis"""
    
    def __init__(self):
        # Create video-specific prototypes
        self.prototypes = [
            ("static", np.array([0.1, 0.2, 0.1, 0.05])),      # Low motion, low complexity
            ("motion", np.array([0.8, 0.6, 0.3, 0.2])),       # High motion, medium complexity
            ("complex", np.array([0.4, 0.3, 0.9, 0.7])),      # Medium motion, high complexity
            ("dramatic", np.array([0.9, 0.8, 0.8, 0.9])),     # High everything
            ("subtle", np.array([0.2, 0.1, 0.2, 0.1])),       # Low everything
            ("rhythmic", np.array([0.6, 0.7, 0.4, 0.6])),     # Medium-high motion, rhythmic
        ]
        
        self.categorizer = Categorizer.init(self._extract_features, self.prototypes)
    
    def _extract_features(self, video_chunk: VideoChunk) -> Vector:
        """Extract features from video chunk for categorization"""
        if not video_chunk.frames:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Convert to FieldIQ for analysis
        fields = video_chunk.to_field_iq_sequence(channel=0)
        
        # Calculate motion features (2 features)
        motion_features = self._calculate_motion_features(fields)
        
        # Calculate complexity features (2 features)
        complexity_features = self._calculate_complexity_features(fields)
        
        # Calculate color features (2 features)
        color_features = self._calculate_color_features(video_chunk.frames)
        
        # Calculate temporal features (2 features)
        temporal_features = self._calculate_temporal_features(fields)
        
        # Return only 4 features to match prototypes
        return np.concatenate([motion_features, complexity_features])
    
    def _calculate_motion_features(self, fields: List[FieldIQ]) -> Vector:
        """Calculate motion-related features"""
        if len(fields) < 2:
            return np.array([0.0, 0.0])
        
        # Calculate frame-to-frame differences
        diffs = []
        for i in range(1, len(fields)):
            diff = np.mean(np.abs(fields[i].z - fields[i-1].z))
            diffs.append(diff)
        
        motion_level = np.mean(diffs) if diffs else 0.0
        motion_variance = np.var(diffs) if len(diffs) > 1 else 0.0
        
        return np.array([motion_level, motion_variance])
    
    def _calculate_complexity_features(self, fields: List[FieldIQ]) -> Vector:
        """Calculate complexity-related features"""
        if not fields:
            return np.array([0.0, 0.0])
        
        # Use first field as representative
        field = fields[0]
        
        # Spectral complexity
        fft_data = np.fft.fft(field.z)
        freqs = np.fft.fftfreq(len(field.z), 1/48000)
        
        # Calculate spectral entropy
        power_spectrum = np.abs(fft_data) ** 2
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        spectral_entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))
        
        # Calculate frequency spread
        freq_spread = np.std(freqs[np.abs(fft_data) > np.max(np.abs(fft_data)) * 0.1])
        
        return np.array([spectral_entropy, freq_spread])
    
    def _calculate_color_features(self, frames: List[VideoFrame]) -> Vector:
        """Calculate color-related features"""
        if not frames:
            return np.array([0.0, 0.0])
        
        # Use first frame as representative
        frame = frames[0]
        
        # Calculate color variance
        if len(frame.data.shape) == 3:
            color_variance = np.var(frame.data.reshape(-1, 3), axis=0)
            color_dominance = np.mean(color_variance)
        else:
            color_dominance = 0.0
        
        # Calculate brightness
        brightness = np.mean(frame.data) / 255.0
        
        return np.array([color_dominance, brightness])
    
    def _calculate_temporal_features(self, fields: List[FieldIQ]) -> Vector:
        """Calculate temporal features"""
        if len(fields) < 2:
            return np.array([0.0, 0.0])
        
        # Calculate temporal consistency
        powers = [np.sum(field.power) for field in fields]
        temporal_consistency = 1.0 / (1.0 + np.std(powers))
        
        # Calculate rhythm
        power_diffs = [abs(powers[i] - powers[i-1]) for i in range(1, len(powers))]
        rhythm_strength = np.mean(power_diffs) if power_diffs else 0.0
        
        return np.array([temporal_consistency, rhythm_strength])
    
    def categorize(self, video_chunk: VideoChunk) -> Tuple[Label, float, VideoContext]:
        """Categorize video chunk and return context"""
        category, confidence = self.categorizer.predict(video_chunk)
        
        # Extract detailed context
        features = self._extract_features(video_chunk)
        
        # Get additional features for context
        fields = video_chunk.to_field_iq_sequence(channel=0)
        color_features = self._calculate_color_features(video_chunk.frames)
        temporal_features = self._calculate_temporal_features(fields)
        
        context = VideoContext(
            frame_features=features,
            temporal_features=temporal_features,
            scene_category=category,
            motion_level=features[0],
            complexity_level=features[2],
            color_dominance=color_features
        )
        
        return category, confidence, context

class VideoReasoner:
    """Specialized reasoner for video effect selection"""
    
    def __init__(self):
        # Define rules for effect selection
        self.rules = [
            # Motion-based rules
            Rule(
                premises=(("high_motion", ()), ("scene_type", ("action",))),
                conclude=("recommend_effect", ("sharp_blend",))
            ),
            Rule(
                premises=(("low_motion", ()), ("scene_type", ("static",))),
                conclude=("recommend_effect", ("smooth_blend",))
            ),
            
            # Complexity-based rules
            Rule(
                premises=(("high_complexity", ()), ("scene_type", ("complex",))),
                conclude=("recommend_effect", ("adaptive_blend",))
            ),
            Rule(
                premises=(("low_complexity", ()), ("scene_type", ("subtle",))),
                conclude=("recommend_effect", ("linear_blend",))
            ),
            
            # Dramatic scenes
            Rule(
                premises=(("scene_type", ("dramatic",)), ("high_motion", ())),
                conclude=("recommend_effect", ("echo",))
            ),
            Rule(
                premises=(("scene_type", ("dramatic",)), ("high_complexity", ())),
                conclude=("recommend_effect", ("glitch",))
            ),
            
            # Rhythmic scenes
            Rule(
                premises=(("scene_type", ("rhythmic",)), ("high_motion", ())),
                conclude=("recommend_effect", ("sinusoidal_blend",))
            ),
            
            # Fallback rules
            Rule(
                premises=(("scene_type", ("unknown",)),),
                conclude=("recommend_effect", ("linear_blend",))
            ),
        ]
        
        self.reasoner = Reasoner.init(self.rules, budget=16)
    
    def reason_about_effects(self, context: VideoContext) -> List[Atom]:
        """Reason about which effects to apply based on context"""
        # Build knowledge base from context
        kb = set()
        
        # Add motion facts
        if context.motion_level > 0.6:
            kb.add(("high_motion", ()))
        else:
            kb.add(("low_motion", ()))
        
        # Add complexity facts
        if context.complexity_level > 0.6:
            kb.add(("high_complexity", ()))
        else:
            kb.add(("low_complexity", ()))
        
        # Add scene type
        kb.add(("scene_type", (context.scene_category,)))
        
        # Run reasoning
        derived_kb, proofs = self.reasoner.run(kb)
        
        # Extract effect recommendations
        effect_recommendations = [atom for atom in derived_kb if atom[0] == "recommend_effect"]
        
        return effect_recommendations

class VideoESN:
    """Specialized ESN for learning temporal video patterns"""
    
    def __init__(self, n_features: int = 6, n_reservoir: int = 32):
        self.esn, self.state = ESN.init(
            n_in=n_features,
            n_res=n_reservoir,
            spectral_radius=0.9,
            seed=42
        )
        self.training_buffer = ESN.begin_train()
        self.is_trained = False
    
    def update(self, context: VideoContext) -> ESNState:
        """Update ESN with new context"""
        # Create input vector from context
        x = np.concatenate([
            context.frame_features[:2],  # Motion features
            context.temporal_features,   # Temporal features
            context.color_dominance      # Color features
        ])
        
        # Update state
        self.state = self.esn.update(self.state, x)
        return self.state
    
    def accumulate_training(self, context: VideoContext, target: Vector):
        """Accumulate training data"""
        x = np.concatenate([
            context.frame_features[:2],
            context.temporal_features,
            context.color_dominance
        ])
        ESN.accum(self.training_buffer, self.state, x, target)
    
    def train(self, lambda_reg: float = 1e-3):
        """Train the ESN readout layer"""
        if len(self.training_buffer["H"]) > 0:
            self.esn.fit_wout(self.training_buffer, lam=lambda_reg)
            self.is_trained = True
    
    def predict(self, context: VideoContext) -> Vector:
        """Predict effect parameters"""
        if not self.is_trained:
            # Return default prediction
            return np.array([0.5, 0.5, 0.5, 0.5])  # Default effect parameters
        
        x = np.concatenate([
            context.frame_features[:2],
            context.temporal_features,
            context.color_dominance
        ])
        
        return self.esn.read(self.state, x)

class VideoLLM:
    """Specialized LLM for generating effect descriptions and parameters"""
    
    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.llm = LLM(self._video_logits, vocab_size)
    
    def _video_logits(self, ctx: Vector, seq: List[int]) -> Vector:
        """Generate logits for video effect parameter prediction"""
        z = np.zeros((self.vocab_size,), dtype=float)
        
        # Context-based bias
        if len(ctx) > 0:
            # Use context to bias towards certain effect types
            context_bias = int(np.sum(ctx) * 10) % self.vocab_size
            z[context_bias] += 2.0
        
        # Sequence continuation bias
        if len(seq) > 0:
            last_token = seq[-1]
            z[last_token] += 1.5
        
        # Add some randomness
        z += np.random.normal(0, 0.1, self.vocab_size)
        
        return z
    
    def generate_effect_description(self, context: VideoContext, esn_state: ESNState) -> str:
        """Generate a description of the effect to apply"""
        ctx = LLM.build_ctx(
            history=[f"scene_{context.scene_category}", f"motion_{context.motion_level:.2f}"],
            tools=["blend", "echo", "glitch", "morph"],
            esn_state=esn_state
        )
        
        prompt = [1, 2, 3]  # Start tokens
        seq = self.llm.decode(ctx, prompt, max_len=16, temperature=0.8)
        
        # Convert sequence to description
        descriptions = {
            1: "smooth", 2: "sharp", 3: "dramatic", 4: "subtle",
            5: "rhythmic", 6: "complex", 7: "simple", 8: "intense"
        }
        
        desc_parts = [descriptions.get(tok, "unknown") for tok in seq[:4]]
        return "_".join(desc_parts)

class IntelligentVideoProcessor:
    """Main intelligent video processing system"""
    
    def __init__(self):
        self.categorizer = VideoCategorizer()
        self.reasoner = VideoReasoner()
        self.esn = VideoESN()
        self.llm = VideoLLM()
        
        # Effect registry
        self.effects = self._create_effect_registry()
        
        # Processing history
        self.processing_history = []
    
    def _create_effect_registry(self) -> Dict[str, VideoEffect]:
        """Create registry of available effects"""
        return {
            "linear_blend": VideoEffect(
                "Linear Blend",
                temporal_blend_linear(0.5),
                {"blend_factor": 0.5},
                "temporal",
                intensity=0.5,
                temporal_stability=0.8
            ),
            "smooth_blend": VideoEffect(
                "Smooth Blend",
                temporal_blend_linear(0.3),
                {"blend_factor": 0.3},
                "temporal",
                intensity=0.3,
                temporal_stability=0.9
            ),
            "sharp_blend": VideoEffect(
                "Sharp Blend",
                temporal_blend_linear(0.8),
                {"blend_factor": 0.8},
                "temporal",
                intensity=0.8,
                temporal_stability=0.6
            ),
            "adaptive_blend": VideoEffect(
                "Adaptive Blend",
                temporal_blend_adaptive(0.3),
                {"adaptation_rate": 0.3},
                "temporal",
                intensity=0.6,
                temporal_stability=0.7
            ),
            "sinusoidal_blend": VideoEffect(
                "Sinusoidal Blend",
                temporal_blend_sinusoidal(2.0, 0.0),
                {"frequency": 2.0, "phase": 0.0},
                "temporal",
                intensity=0.7,
                temporal_stability=0.5
            ),
            "echo": VideoEffect(
                "Echo",
                temporal_echo(2, 0.8, 0.2),
                {"delay": 2, "decay": 0.8, "feedback": 0.2},
                "temporal",
                intensity=0.8,
                temporal_stability=0.4
            ),
            "glitch": VideoEffect(
                "Glitch",
                temporal_glitch(0.1, 0.3),
                {"probability": 0.1, "strength": 0.3},
                "temporal",
                intensity=0.9,
                temporal_stability=0.2
            ),
        }
    
    def process_chunk(self, chunk: VideoChunk) -> Tuple[List[FieldIQ], Dict[str, Any]]:
        """Process a video chunk with intelligent effect selection"""
        # 1. Categorize the chunk
        category, confidence, context = self.categorizer.categorize(chunk)
        
        # 2. Reason about effects
        effect_recommendations = self.reasoner.reason_about_effects(context)
        
        # 3. Update ESN
        esn_state = self.esn.update(context)
        
        # 4. Generate effect description
        effect_description = self.llm.generate_effect_description(context, esn_state)
        
        # 5. Select effect based on recommendations and ESN prediction
        selected_effect = self._select_effect(effect_recommendations, context, esn_state)
        
        # 6. Apply effect
        processed_frames = selected_effect.effect_func(chunk)
        
        # 7. Accumulate training data
        target = self._create_target_vector(selected_effect, context)
        self.esn.accumulate_training(context, target)
        
        # 8. Store processing info
        processing_info = {
            "category": category,
            "confidence": confidence,
            "selected_effect": selected_effect.name,
            "effect_description": effect_description,
            "context": context,
            "recommendations": effect_recommendations
        }
        
        self.processing_history.append(processing_info)
        
        return processed_frames, processing_info
    
    def _select_effect(self, recommendations: List[Atom], context: VideoContext, esn_state: ESNState) -> VideoEffect:
        """Select the best effect based on recommendations and ESN prediction"""
        # Get ESN prediction
        esn_prediction = self.esn.predict(context)
        
        # If we have recommendations, use them
        if recommendations:
            effect_name = recommendations[0][1][0]  # Get first recommendation
            if effect_name in self.effects:
                return self.effects[effect_name]
        
        # Fallback to ESN prediction or default
        if esn_prediction is not None and len(esn_prediction) >= 4:
            # Use ESN prediction to select effect
            effect_scores = []
            for name, effect in self.effects.items():
                # Simple scoring based on ESN prediction
                score = np.dot(esn_prediction[:4], [
                    effect.intensity,
                    effect.temporal_stability,
                    context.motion_level,
                    context.complexity_level
                ])
                effect_scores.append((score, name, effect))
            
            # Select highest scoring effect
            effect_scores.sort(reverse=True)
            return effect_scores[0][2]
        
        # Default fallback
        return self.effects["linear_blend"]
    
    def _create_target_vector(self, effect: VideoEffect, context: VideoContext) -> Vector:
        """Create target vector for ESN training"""
        return np.array([
            effect.intensity,
            effect.temporal_stability,
            context.motion_level,
            context.complexity_level
        ])
    
    def train_on_history(self):
        """Train the ESN on accumulated processing history"""
        self.esn.train()
        print(f"✅ ESN trained on {len(self.processing_history)} samples")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processing"""
        if not self.processing_history:
            return {"total_chunks": 0}
        
        categories = [info["category"] for info in self.processing_history]
        effects = [info["selected_effect"] for info in self.processing_history]
        
        return {
            "total_chunks": len(self.processing_history),
            "category_distribution": {cat: categories.count(cat) for cat in set(categories)},
            "effect_distribution": {eff: effects.count(eff) for eff in set(effects)},
            "avg_confidence": np.mean([info["confidence"] for info in self.processing_history])
        }

def demo_intelligent_video_processing():
    """Demonstrate intelligent video processing"""
    print("=== Intelligent Video Processing Demo ===")
    
    # Create test video
    video_path = "intelligent_test.mp4"
    create_test_video(video_path, duration=3.0)
    
    # Create intelligent processor
    processor = IntelligentVideoProcessor()
    
    print("Processing video with AI-driven effect selection...")
    
    # Process video chunks
    chunk_count = 0
    for chunk in load_video_chunks(video_path, chunk_size=15, overlap=3):
        chunk_count += 1
        
        # Process chunk
        processed_frames, info = processor.process_chunk(chunk)
        
        print(f"\nChunk {chunk_count}:")
        print(f"  Category: {info['category']} (confidence: {info['confidence']:.3f})")
        print(f"  Selected effect: {info['selected_effect']}")
        print(f"  Description: {info['effect_description']}")
        print(f"  Motion level: {info['context'].motion_level:.3f}")
        print(f"  Complexity: {info['context'].complexity_level:.3f}")
        
        if chunk_count >= 5:  # Limit output
            break
    
    # Train ESN on accumulated data
    processor.train_on_history()
    
    # Show statistics
    stats = processor.get_processing_stats()
    print(f"\nProcessing Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Category distribution: {stats['category_distribution']}")
    print(f"  Effect distribution: {stats['effect_distribution']}")
    print(f"  Average confidence: {stats['avg_confidence']:.3f}")

def create_test_video(output_path="intelligent_test.mp4", duration=3.0, fps=30):
    """Create a test video with varying content for intelligent processing"""
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
        
        # Create different types of content over time
        if t < 1.0:
            # Static content
            cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), -1)
        elif t < 2.0:
            # Motion content
            x = int(50 + 100 * (t - 1.0))
            cv2.circle(frame, (x, 120), 30, (255, 0, 0), -1)
        else:
            # Complex content
            for i in range(5):
                x = int(50 + i * 50 + 20 * np.sin(2 * np.pi * t * 2))
                y = int(100 + 20 * np.cos(2 * np.pi * t * 3))
                cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)
        
        # Add frame info
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Created test video: {output_path}")

if __name__ == "__main__":
    demo_intelligent_video_processing()
