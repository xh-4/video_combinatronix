#!/usr/bin/env python3
"""
Video Effects System for Combinator Kernel
Advanced temporal and spatial effects using functional combinators
"""

import numpy as np
from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import from the main kernel
from Combinator_Kernel import (
    FieldIQ, VideoFrame, VideoChunk, 
    Unary, Binary, B, split_add, split_mul,
    lowpass_hz, amp, phase_deg, freq_shift, delay_ms
)

# ---------- Effect System Architecture ----------

class EffectType(Enum):
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SPECTRAL = "spectral"
    COMBINED = "combined"

@dataclass
class EffectParams:
    """Parameters for effect configuration"""
    strength: float = 1.0
    duration: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0
    blend_mode: str = "linear"
    custom: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom is None:
            self.custom = {}

# ---------- Temporal Blend Effects ----------

def temporal_blend_linear(blend_factor: float = 0.5) -> Callable[[VideoChunk], List[FieldIQ]]:
    """
    Linear temporal blend between consecutive frames.
    blend_factor: 0.0 = previous frame, 1.0 = current frame, 0.5 = 50/50 blend
    """
    def blend_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        if len(chunk.frames) < 2:
            return chunk.to_field_iq_sequence(channel=0)
        
        fields = chunk.to_field_iq_sequence(channel=0)
        blended = []
        
        for i in range(len(fields)):
            if i == 0:
                # First frame - no previous to blend with
                blended.append(fields[i])
            else:
                # Blend current with previous
                prev_field = fields[i-1]
                curr_field = fields[i]
                
                # Linear interpolation in complex domain
                blended_z = (1 - blend_factor) * prev_field.z + blend_factor * curr_field.z
                blended_field = FieldIQ(
                    blended_z, 
                    curr_field.sr, 
                    {**curr_field.roles, 'temporal_blend': blend_factor}
                )
                blended.append(blended_field)
        
        return blended
    
    return blend_chunk

def temporal_blend_exponential(decay_rate: float = 0.8) -> Callable[[VideoChunk], List[FieldIQ]]:
    """
    Exponential temporal blend - more recent frames have more influence.
    decay_rate: 0.0 = only current frame, 1.0 = equal weight, >1.0 = more weight to older frames
    """
    def blend_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        if len(chunk.frames) < 2:
            return chunk.to_field_iq_sequence(channel=0)
        
        fields = chunk.to_field_iq_sequence(channel=0)
        blended = []
        
        for i in range(len(fields)):
            if i == 0:
                blended.append(fields[i])
            else:
                # Calculate weights for exponential blend
                weights = np.array([decay_rate ** (i - j) for j in range(i + 1)])
                weights = weights / np.sum(weights)  # Normalize
                
                # Blend all previous frames with current
                blended_z = np.zeros_like(fields[i].z)
                for j, weight in enumerate(weights):
                    frame_idx = i - j
                    blended_z += weight * fields[frame_idx].z
                
                blended_field = FieldIQ(
                    blended_z,
                    fields[i].sr,
                    {**fields[i].roles, 'temporal_blend_exp': decay_rate}
                )
                blended.append(blended_field)
        
        return blended
    
    return blend_chunk

def temporal_blend_sinusoidal(frequency: float = 1.0, phase: float = 0.0) -> Callable[[VideoChunk], List[FieldIQ]]:
    """
    Sinusoidal temporal blend - creates rhythmic blending patterns.
    frequency: blend oscillation frequency (cycles per frame)
    phase: phase offset in radians
    """
    def blend_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        if len(chunk.frames) < 2:
            return chunk.to_field_iq_sequence(channel=0)
        
        fields = chunk.to_field_iq_sequence(channel=0)
        blended = []
        
        for i in range(len(fields)):
            if i == 0:
                blended.append(fields[i])
            else:
                # Calculate sinusoidal blend factor
                t = i / len(fields)  # Normalized time
                blend_factor = 0.5 * (1 + np.sin(2 * np.pi * frequency * t + phase))
                
                # Blend with previous frame
                prev_field = fields[i-1]
                curr_field = fields[i]
                blended_z = (1 - blend_factor) * prev_field.z + blend_factor * curr_field.z
                
                blended_field = FieldIQ(
                    blended_z,
                    curr_field.sr,
                    {**curr_field.roles, 'temporal_blend_sin': (frequency, phase)}
                )
                blended.append(blended_field)
        
        return blended
    
    return blend_chunk

def temporal_blend_adaptive(adaptation_rate: float = 0.1) -> Callable[[VideoChunk], List[FieldIQ]]:
    """
    Adaptive temporal blend - blend factor adapts based on frame differences.
    adaptation_rate: how quickly to adapt (0.0 = no adaptation, 1.0 = instant adaptation)
    """
    def blend_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        if len(chunk.frames) < 2:
            return chunk.to_field_iq_sequence(channel=0)
        
        fields = chunk.to_field_iq_sequence(channel=0)
        blended = []
        adaptive_blend = 0.5  # Start with 50/50 blend
        
        for i in range(len(fields)):
            if i == 0:
                blended.append(fields[i])
            else:
                # Calculate frame difference
                prev_field = fields[i-1]
                curr_field = fields[i]
                frame_diff = np.mean(np.abs(curr_field.z - prev_field.z))
                
                # Adapt blend factor based on difference
                # High difference -> more current frame (higher blend factor)
                # Low difference -> more previous frame (lower blend factor)
                target_blend = np.tanh(frame_diff * 10)  # Scale and normalize
                adaptive_blend = (1 - adaptation_rate) * adaptive_blend + adaptation_rate * target_blend
                
                # Apply adaptive blend
                blended_z = (1 - adaptive_blend) * prev_field.z + adaptive_blend * curr_field.z
                
                blended_field = FieldIQ(
                    blended_z,
                    curr_field.sr,
                    {**curr_field.roles, 'temporal_blend_adaptive': adaptive_blend}
                )
                blended.append(blended_field)
        
        return blended
    
    return blend_chunk

# ---------- Advanced Temporal Effects ----------

def temporal_echo(delay_frames: int = 3, decay: float = 0.7, feedback: float = 0.3) -> Callable[[VideoChunk], List[FieldIQ]]:
    """
    Temporal echo effect - creates delayed copies of frames with decay.
    delay_frames: number of frames to delay
    decay: amplitude decay per echo
    feedback: feedback amount for multiple echoes
    """
    def echo_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        fields = chunk.to_field_iq_sequence(channel=0)
        echoed = []
        echo_buffer = []
        
        for i, field in enumerate(fields):
            # Add current field to echo buffer
            echo_buffer.append(field.z.copy())
            
            # Calculate echo
            echoed_z = field.z.copy()
            for j in range(len(echo_buffer)):
                if j > 0:  # Skip current frame
                    delay = j * delay_frames
                    if i >= delay:
                        echo_strength = (decay ** j) * (feedback ** j)
                        echoed_z += echo_strength * echo_buffer[-(j+1)]
            
            echoed_field = FieldIQ(
                echoed_z,
                field.sr,
                {**field.roles, 'temporal_echo': (delay_frames, decay, feedback)}
            )
            echoed.append(echoed_field)
            
            # Limit buffer size
            if len(echo_buffer) > 10:
                echo_buffer.pop(0)
        
        return echoed
    
    return echo_chunk

def temporal_glitch(glitch_probability: float = 0.1, glitch_strength: float = 0.5) -> Callable[[VideoChunk], List[FieldIQ]]:
    """
    Temporal glitch effect - randomly distorts frames.
    glitch_probability: probability of glitch per frame (0.0 to 1.0)
    glitch_strength: intensity of glitch effect (0.0 to 1.0)
    """
    def glitch_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        fields = chunk.to_field_iq_sequence(channel=0)
        glitched = []
        
        for i, field in enumerate(fields):
            if np.random.random() < glitch_probability:
                # Apply glitch
                glitch_z = field.z.copy()
                
                # Random phase distortion
                phase_shift = np.random.uniform(-glitch_strength * np.pi, glitch_strength * np.pi)
                glitch_z *= np.exp(1j * phase_shift)
                
                # Random amplitude modulation
                amp_mod = 1.0 + np.random.uniform(-glitch_strength, glitch_strength)
                glitch_z *= amp_mod
                
                # Random frequency shift
                freq_shift = np.random.uniform(-glitch_strength * 1000, glitch_strength * 1000)
                t = np.arange(len(glitch_z)) / field.sr
                glitch_z *= np.exp(1j * 2 * np.pi * freq_shift * t)
                
                glitched_field = FieldIQ(
                    glitch_z,
                    field.sr,
                    {**field.roles, 'temporal_glitch': glitch_strength}
                )
                glitched.append(glitched_field)
            else:
                glitched.append(field)
        
        return glitched
    
    return glitch_chunk

def temporal_morph(target_chunk: VideoChunk, morph_strength: float = 0.5) -> Callable[[VideoChunk], List[FieldIQ]]:
    """
    Temporal morph effect - morphs current chunk towards target chunk.
    target_chunk: target chunk to morph towards
    morph_strength: morph intensity (0.0 = no morph, 1.0 = full morph)
    """
    def morph_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        source_fields = chunk.to_field_iq_sequence(channel=0)
        target_fields = target_chunk.to_field_iq_sequence(channel=0)
        
        if len(source_fields) != len(target_fields):
            # Pad shorter sequence with zeros
            max_len = max(len(source_fields), len(target_fields))
            if len(source_fields) < max_len:
                source_fields.extend([FieldIQ(np.zeros_like(source_fields[0].z), source_fields[0].sr, {})] * (max_len - len(source_fields)))
            if len(target_fields) < max_len:
                target_fields.extend([FieldIQ(np.zeros_like(target_fields[0].z), target_fields[0].sr, {})] * (max_len - len(target_fields)))
        
        morphed = []
        for i, (src_field, tgt_field) in enumerate(zip(source_fields, target_fields)):
            # Linear interpolation between source and target
            morphed_z = (1 - morph_strength) * src_field.z + morph_strength * tgt_field.z
            
            morphed_field = FieldIQ(
                morphed_z,
                src_field.sr,
                {**src_field.roles, 'temporal_morph': morph_strength}
            )
            morphed.append(morphed_field)
        
        return morphed
    
    return morph_chunk

# ---------- Effect Combinators ----------

def compose_effects(*effects) -> Callable[[VideoChunk], List[FieldIQ]]:
    """
    Compose multiple effects in sequence.
    """
    def composed_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        result = chunk
        for effect in effects:
            if hasattr(effect, '__call__'):
                result = effect(result)
            else:
                # If it's not callable, assume it's a chunk and apply next effect
                result = effect
        return result
    
    return composed_chunk

def parallel_effects(*effects) -> Callable[[VideoChunk], List[List[FieldIQ]]]:
    """
    Apply multiple effects in parallel and return all results.
    """
    def parallel_chunk(chunk: VideoChunk) -> List[List[FieldIQ]]:
        results = []
        for effect in effects:
            if hasattr(effect, '__call__'):
                results.append(effect(chunk))
            else:
                results.append(chunk.to_field_iq_sequence(channel=0))
        return results
    
    return parallel_chunk

def blend_effects(effect1: Callable, effect2: Callable, blend_factor: float = 0.5) -> Callable[[VideoChunk], List[FieldIQ]]:
    """
    Blend the results of two effects.
    """
    def blended_chunk(chunk: VideoChunk) -> List[FieldIQ]:
        result1 = effect1(chunk)
        result2 = effect2(chunk)
        
        if len(result1) != len(result2):
            min_len = min(len(result1), len(result2))
            result1 = result1[:min_len]
            result2 = result2[:min_len]
        
        blended = []
        for f1, f2 in zip(result1, result2):
            blended_z = (1 - blend_factor) * f1.z + blend_factor * f2.z
            blended_field = FieldIQ(
                blended_z,
                f1.sr,
                {**f1.roles, 'effect_blend': blend_factor}
            )
            blended.append(blended_field)
        
        return blended
    
    return blended_chunk

# ---------- Effect Presets ----------

def create_temporal_blend_presets() -> Dict[str, Callable[[VideoChunk], List[FieldIQ]]]:
    """Create a collection of temporal blend presets"""
    return {
        'smooth': temporal_blend_linear(0.3),
        'sharp': temporal_blend_linear(0.8),
        'dreamy': temporal_blend_exponential(0.6),
        'rhythmic': temporal_blend_sinusoidal(2.0, 0.0),
        'adaptive': temporal_blend_adaptive(0.2),
        'echo': temporal_echo(2, 0.8, 0.2),
        'glitch': temporal_glitch(0.15, 0.3),
    }

def create_effect_chain(name: str, params: EffectParams) -> Callable[[VideoChunk], List[FieldIQ]]:
    """Create effect chain from name and parameters"""
    presets = create_temporal_blend_presets()
    
    if name in presets:
        return presets[name]
    elif name == 'custom_blend':
        return temporal_blend_linear(params.strength)
    elif name == 'custom_echo':
        return temporal_echo(
            int(params.duration * 30),  # Convert duration to frames
            params.strength,
            params.frequency
        )
    else:
        raise ValueError(f"Unknown effect: {name}")

# ---------- Demo Functions ----------

def demo_temporal_effects():
    """Demonstrate temporal effects"""
    print("=== Temporal Effects Demo ===")
    
    # This would be used with actual video chunks
    print("Available temporal effects:")
    presets = create_temporal_blend_presets()
    for name, effect in presets.items():
        print(f"  - {name}: {effect.__name__}")
    
    print("\nEffect combinators:")
    print("  - compose_effects: Chain effects in sequence")
    print("  - parallel_effects: Apply multiple effects simultaneously")
    print("  - blend_effects: Blend results of two effects")

if __name__ == "__main__":
    demo_temporal_effects()



