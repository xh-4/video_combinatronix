# ============================
# Enhanced FieldIQ Pipeline with PhaseWheel Realms
# ============================
"""
Enhanced AI pipeline system using PhaseWheel realms for more flexible
FieldIQ signal processing with continuous phase states and adaptive behavior.
"""

import numpy as np
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from queue import Queue, Empty

# Import existing systems
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf
from Combinator_Kernel import FieldIQ, make_field_from_real
from phase_wheel_realms import (
    create_phase_wheel_realm, create_phase_wheel_activation_realm, 
    create_adaptive_phase_wheel_realm, PhaseWheelRealm, compose_phase_wheel_realms,
    calculate_phase_wheel_resonance
)

# Import resonance scheduler
import sys
sys.path.append(r'c:\Users\The School\Desktop\Code\HPU')
from resonance_singularity import ResonanceScheduler, Task, Field as ResonanceField

# ============================
# Enhanced Pipeline Stage Definitions
# ============================

class EnhancedPipelineStage(Enum):
    INGEST = "ingest"
    DENOISE = "denoise" 
    PROJECT = "project"
    PHASE_ENHANCE = "phase_enhance"  # New phase enhancement stage
    RESET = "reset"

@dataclass
class EnhancedPipelineState:
    """Enhanced state of the AI pipeline with PhaseWheel capabilities."""
    current_stage: EnhancedPipelineStage
    data_buffer: List[FieldIQ]
    processed_data: List[FieldIQ]
    stage_history: List[EnhancedPipelineStage]
    resonance_energy: float
    carrier_load: float
    phase_coherence: float  # New: phase coherence metric
    spectral_richness: float  # New: spectral richness metric
    timestamp: float
    
    def __post_init__(self):
        if self.data_buffer is None:
            self.data_buffer = []
        if self.processed_data is None:
            self.processed_data = []
        if self.stage_history is None:
            self.stage_history = []
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class EnhancedHPUCarrier:
    """Enhanced HPU carrier with PhaseWheel capabilities."""
    carrier_id: str
    capacity: float
    current_load: float
    active_stage: Optional[EnhancedPipelineStage]
    phase_wheel_registry: Dict[str, PhaseWheelRealm]
    resonance_scheduler: ResonanceScheduler
    phase_adaptation_rate: float  # New: phase adaptation capability
    
    def __post_init__(self):
        if self.current_load is None:
            self.current_load = 0.0
        if self.phase_wheel_registry is None:
            self.phase_wheel_registry = {}
        if self.resonance_scheduler is None:
            self.resonance_scheduler = ResonanceScheduler()
        if self.phase_adaptation_rate is None:
            self.phase_adaptation_rate = 0.1

# ============================
# Enhanced AI Pipeline Stage Implementations
# ============================

class EnhancedAIPipelineStages:
    """Enhanced implementation using PhaseWheel realms for more flexible processing."""
    
    def __init__(self):
        self.stage_realms = self._create_enhanced_stage_realms()
    
    def _create_enhanced_stage_realms(self) -> Dict[EnhancedPipelineStage, List[PhaseWheelRealm]]:
        """Create PhaseWheel realms for each enhanced pipeline stage."""
        return {
            EnhancedPipelineStage.INGEST: [
                create_phase_wheel_activation_realm(n_phases=8, phase_increment=np.pi/50),
                create_phase_wheel_realm(base_frequency=1.0, phase_increment=np.pi/100, n_slots=16)
            ],
            EnhancedPipelineStage.DENOISE: [
                create_adaptive_phase_wheel_realm(n_slots=12, base_frequency=0.5, adaptation_rate=0.3),
                create_phase_wheel_activation_realm(n_phases=16, phase_increment=np.pi/75)
            ],
            EnhancedPipelineStage.PROJECT: [
                create_phase_wheel_realm(base_frequency=2.0, phase_increment=np.pi/25, n_slots=24),
                create_adaptive_phase_wheel_realm(n_slots=20, base_frequency=1.5, adaptation_rate=0.2)
            ],
            EnhancedPipelineStage.PHASE_ENHANCE: [
                create_phase_wheel_realm(base_frequency=3.0, phase_increment=np.pi/17, n_slots=32),
                create_phase_wheel_activation_realm(n_phases=24, phase_increment=np.pi/30),
                create_adaptive_phase_wheel_realm(n_slots=28, base_frequency=2.5, adaptation_rate=0.4)
            ],
            EnhancedPipelineStage.RESET: [
                create_phase_wheel_activation_realm(n_phases=4, phase_increment=np.pi/200),
                create_phase_wheel_realm(base_frequency=0.1, phase_increment=np.pi/500, n_slots=8)
            ]
        }
    
    def create_stage_pipeline(self, stage: EnhancedPipelineStage) -> PhaseWheelRealm:
        """Create a pipeline of PhaseWheel realms for a specific stage."""
        stage_realms = self.stage_realms[stage]
        if len(stage_realms) == 1:
            return stage_realms[0]
        
        # Compose multiple realms into a pipeline
        pipeline = stage_realms[0]
        for realm in stage_realms[1:]:
            pipeline = compose_phase_wheel_realms(pipeline, realm)
        
        return pipeline
    
    def calculate_stage_resonance(self, field: FieldIQ, stage: EnhancedPipelineStage) -> float:
        """Calculate resonance between field and enhanced pipeline stage."""
        stage_realms = self.stage_realms[stage]
        
        # Calculate average resonance across stage realms
        resonances = []
        for realm in stage_realms:
            resonance = calculate_phase_wheel_resonance(field, realm)
            resonances.append(resonance)
        
        base_resonance = np.mean(resonances) if resonances else 0.0
        
        # Add stage-specific resonance adjustments
        if stage == EnhancedPipelineStage.INGEST:
            # Ingest resonates with raw signal energy and phase complexity
            field_energy = np.sum(np.abs(field.z) ** 2)
            phase_complexity = np.std(np.angle(field.z))
            energy_resonance = min(1.0, field_energy / 1000.0)
            phase_resonance = min(1.0, phase_complexity * 2.0)
            return (base_resonance + energy_resonance + phase_resonance) / 3.0
            
        elif stage == EnhancedPipelineStage.DENOISE:
            # Denoise resonates with noise characteristics and phase irregularities
            noise_estimate = np.std(field.z)
            phase_smoothness = 1.0 / (1.0 + np.std(np.diff(np.angle(field.z))))
            noise_resonance = min(1.0, noise_estimate * 10.0)
            return (base_resonance + noise_resonance + phase_smoothness) / 3.0
            
        elif stage == EnhancedPipelineStage.PROJECT:
            # Project resonates with feature richness and spectral complexity
            field_energy = np.sum(np.abs(field.z) ** 2)
            fft = np.fft.fft(field.z)
            spectral_energy = np.sum(np.abs(fft) ** 2)
            spectral_resonance = min(1.0, spectral_energy / (field_energy + 1e-8))
            return (base_resonance + spectral_resonance) / 2.0
            
        elif stage == EnhancedPipelineStage.PHASE_ENHANCE:
            # Phase enhance resonates with phase complexity and coherence
            phase_complexity = np.std(np.angle(field.z))
            phase_coherence = np.mean(np.abs(np.diff(np.angle(field.z))))
            phase_resonance = min(1.0, phase_complexity * 2.0)
            coherence_resonance = min(1.0, phase_coherence * 5.0)
            return (base_resonance + phase_resonance + coherence_resonance) / 3.0
            
        elif stage == EnhancedPipelineStage.RESET:
            # Reset resonates with processed state
            return 0.8  # Always ready to reset
        
        return base_resonance
    
    def calculate_phase_coherence(self, field: FieldIQ) -> float:
        """Calculate phase coherence metric for the field."""
        phase_diff = np.diff(np.angle(field.z))
        phase_consistency = 1.0 / (1.0 + np.std(phase_diff))
        phase_smoothness = np.mean(np.abs(phase_diff))
        return (phase_consistency + phase_smoothness) / 2.0
    
    def calculate_spectral_richness(self, field: FieldIQ) -> float:
        """Calculate spectral richness metric for the field."""
        fft = np.fft.fft(field.z)
        spectral_energy = np.sum(np.abs(fft) ** 2)
        field_energy = np.sum(np.abs(field.z) ** 2)
        return min(1.0, spectral_energy / (field_energy + 1e-8))

# ============================
# Enhanced Continuous Pipeline Orchestrator
# ============================

class EnhancedContinuousAIPipeline:
    """Enhanced orchestrator with PhaseWheel capabilities for more flexible processing."""
    
    def __init__(self, num_carriers: int = 4):
        self.num_carriers = num_carriers
        self.carriers: List[EnhancedHPUCarrier] = []
        self.pipeline_stages = EnhancedAIPipelineStages()
        self.data_queue = Queue()
        self.running = False
        self.pipeline_state = EnhancedPipelineState(
            current_stage=EnhancedPipelineStage.INGEST,
            data_buffer=[],
            processed_data=[],
            stage_history=[],
            resonance_energy=0.0,
            carrier_load=0.0,
            phase_coherence=0.0,
            spectral_richness=0.0,
            timestamp=time.time()
        )
        
        # Initialize carriers
        self._initialize_carriers()
    
    def _initialize_carriers(self):
        """Initialize enhanced HPU carriers with PhaseWheel registries."""
        for i in range(self.num_carriers):
            carrier_id = f"enhanced_carrier_{i}"
            scheduler = ResonanceScheduler()
            
            # Register all stage realms with each carrier
            for stage, realms in self.pipeline_stages.stage_realms.items():
                for j, realm in enumerate(realms):
                    key = f"{stage.value}_{realm.name}_{j}"
                    scheduler.register_realm(realm)
            
            carrier = EnhancedHPUCarrier(
                carrier_id=carrier_id,
                capacity=1.0,
                current_load=0.0,
                active_stage=None,
                phase_wheel_registry={},
                resonance_scheduler=scheduler,
                phase_adaptation_rate=0.1 + i * 0.05  # Vary adaptation rates
            )
            
            # Populate phase wheel registry
            for stage, realms in self.pipeline_stages.stage_realms.items():
                for j, realm in enumerate(realms):
                    key = f"{stage.value}_{realm.name}_{j}"
                    carrier.phase_wheel_registry[key] = realm
            
            self.carriers.append(carrier)
    
    def ingest_data(self, data: Any) -> FieldIQ:
        """Enhanced data ingestion with phase analysis."""
        if isinstance(data, np.ndarray):
            sr = 48000
            if len(data.shape) == 1:
                field = make_field_from_real(data, sr, tag=("enhanced_ingest", "raw"))
            else:
                field = make_field_from_real(data.flatten(), sr, tag=("enhanced_ingest", "raw"))
        else:
            # Create synthetic data with phase complexity for demo
            sr = 48000
            dur = 1.0
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            # Create more complex synthetic signal with phase variations
            base_freq = 440
            phase_mod = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Phase modulation
            synthetic = (0.5 * np.cos(2 * np.pi * base_freq * t + phase_mod) + 
                       0.3 * np.cos(2 * np.pi * base_freq * 2 * t + phase_mod * 2) +
                       0.1 * np.random.randn(len(t)))
            field = make_field_from_real(synthetic, sr, tag=("enhanced_ingest", "synthetic"))
        
        return field
    
    def should_transition_stage(self, current_stage: EnhancedPipelineStage, field: FieldIQ) -> EnhancedPipelineStage:
        """Enhanced stage transition logic with PhaseWheel considerations."""
        # Calculate resonance for each stage
        stage_resonances = {}
        for stage in EnhancedPipelineStage:
            resonance = self.pipeline_stages.calculate_stage_resonance(field, stage)
            stage_resonances[stage] = resonance
        
        # Find stage with highest resonance
        next_stage = max(stage_resonances.keys(), key=lambda s: stage_resonances[s])
        
        # Enhanced state machine logic with phase considerations
        if current_stage == EnhancedPipelineStage.INGEST:
            if stage_resonances[EnhancedPipelineStage.DENOISE] > 0.6:
                return EnhancedPipelineStage.DENOISE
        elif current_stage == EnhancedPipelineStage.DENOISE:
            if stage_resonances[EnhancedPipelineStage.PROJECT] > 0.7:
                return EnhancedPipelineStage.PROJECT
        elif current_stage == EnhancedPipelineStage.PROJECT:
            if stage_resonances[EnhancedPipelineStage.PHASE_ENHANCE] > 0.6:
                return EnhancedPipelineStage.PHASE_ENHANCE
        elif current_stage == EnhancedPipelineStage.PHASE_ENHANCE:
            if stage_resonances[EnhancedPipelineStage.RESET] > 0.5:
                return EnhancedPipelineStage.RESET
        elif current_stage == EnhancedPipelineStage.RESET:
            if stage_resonances[EnhancedPipelineStage.INGEST] > 0.4:
                return EnhancedPipelineStage.INGEST
        
        return current_stage
    
    def process_stage(self, field: FieldIQ, stage: EnhancedPipelineStage, carrier: EnhancedHPUCarrier) -> FieldIQ:
        """Enhanced stage processing with PhaseWheel realms."""
        print(f"🔄 Processing {stage.value} on {carrier.carrier_id}")
        
        # Get stage pipeline
        stage_pipeline = self.pipeline_stages.create_stage_pipeline(stage)
        
        # Process field through stage
        processed_field = stage_pipeline.field_processor(field)
        
        # Update carrier load
        carrier.current_load = min(1.0, carrier.current_load + 0.15)
        carrier.active_stage = stage
        
        # Add enhanced processing metadata
        processed_field = processed_field.with_role("pipeline_stage", stage.value)
        processed_field = processed_field.with_role("carrier_id", carrier.carrier_id)
        processed_field = processed_field.with_role("phase_adaptation_rate", carrier.phase_adaptation_rate)
        processed_field = processed_field.with_role("timestamp", time.time())
        
        return processed_field
    
    def reset_carrier(self, carrier: EnhancedHPUCarrier):
        """Reset carrier state after processing."""
        carrier.current_load = 0.0
        carrier.active_stage = None
        print(f"🔄 Reset {carrier.carrier_id}")
    
    async def enhanced_continuous_pipeline_cycle(self):
        """Enhanced continuous pipeline cycle with PhaseWheel capabilities."""
        print("🚀 Starting Enhanced AI Pipeline Cycle...")
        print("   INGEST → DENOISE → PROJECT → PHASE_ENHANCE → RESET → INGEST...")
        self.running = True
        
        cycle_count = 0
        while self.running:
            cycle_count += 1
            print(f"\n--- Enhanced Cycle {cycle_count} ---")
            
            # Update pipeline state
            self.pipeline_state.timestamp = time.time()
            
            # Process available data
            try:
                data = self.data_queue.get_nowait()
                field = self.ingest_data(data)
                self.pipeline_state.data_buffer.append(field)
                print(f"📥 Ingested data: {len(field.z)} samples")
            except Empty:
                # Create enhanced synthetic data with phase complexity
                sr = 48000
                dur = 0.5
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                
                # More complex synthetic signal
                freq = 440 + 50 * np.sin(cycle_count * 0.1)
                phase_mod = 0.3 * np.sin(cycle_count * 0.2)
                noise_level = 0.1 + 0.05 * np.sin(cycle_count * 0.3)
                
                synthetic = (0.5 * np.cos(2 * np.pi * freq * t + phase_mod) + 
                           0.2 * np.cos(2 * np.pi * freq * 1.5 * t + phase_mod * 1.5) +
                           noise_level * np.random.randn(len(t)))
                
                field = make_field_from_real(synthetic, sr, tag=("enhanced_synthetic", f"cycle_{cycle_count}"))
                self.pipeline_state.data_buffer.append(field)
                print(f"📥 Generated enhanced synthetic data: {freq:.1f}Hz, phase_mod={phase_mod:.3f}")
            
            # Process through pipeline stages
            current_field = self.pipeline_state.data_buffer[-1]
            
            # Find available carrier
            available_carrier = min(self.carriers, key=lambda c: c.current_load)
            
            if available_carrier.current_load < 0.8:
                # Determine next stage
                next_stage = self.should_transition_stage(self.pipeline_state.current_stage, current_field)
                
                if next_stage != self.pipeline_state.current_stage:
                    print(f"🔄 Transitioning: {self.pipeline_state.current_stage.value} → {next_stage.value}")
                    self.pipeline_state.current_stage = next_stage
                    self.pipeline_state.stage_history.append(next_stage)
                
                # Process field through current stage
                processed_field = self.process_stage(current_field, next_stage, available_carrier)
                self.pipeline_state.processed_data.append(processed_field)
                
                # If we just completed a full cycle, reset
                if next_stage == EnhancedPipelineStage.RESET:
                    self.reset_carrier(available_carrier)
                    self.pipeline_state.current_stage = EnhancedPipelineStage.INGEST
                    print("🔄 Full enhanced pipeline cycle completed, resetting to ingest")
            
            # Update enhanced pipeline state
            if self.pipeline_state.data_buffer:
                recent_fields = self.pipeline_state.data_buffer[-3:]
                self.pipeline_state.resonance_energy = np.mean([
                    self.pipeline_stages.calculate_stage_resonance(f, self.pipeline_state.current_stage)
                    for f in recent_fields
                ])
                self.pipeline_state.phase_coherence = np.mean([
                    self.pipeline_stages.calculate_phase_coherence(f)
                    for f in recent_fields
                ])
                self.pipeline_state.spectral_richness = np.mean([
                    self.pipeline_stages.calculate_spectral_richness(f)
                    for f in recent_fields
                ])
            else:
                self.pipeline_state.resonance_energy = 0.0
                self.pipeline_state.phase_coherence = 0.0
                self.pipeline_state.spectral_richness = 0.0
            
            self.pipeline_state.carrier_load = np.mean([c.current_load for c in self.carriers])
            
            # Print enhanced status
            print(f"📊 State: {self.pipeline_state.current_stage.value} | "
                  f"Resonance: {self.pipeline_state.resonance_energy:.3f} | "
                  f"Phase Coherence: {self.pipeline_state.phase_coherence:.3f} | "
                  f"Spectral Richness: {self.pipeline_state.spectral_richness:.3f}")
            
            # Show carrier status
            for carrier in self.carriers:
                stage_str = carrier.active_stage.value if carrier.active_stage else "idle"
                print(f"   {carrier.carrier_id}: {stage_str} (load: {carrier.current_load:.2f}, adapt: {carrier.phase_adaptation_rate:.2f})")
            
            # Small delay
            await asyncio.sleep(0.5)
    
    def add_data(self, data: Any):
        """Add data to the enhanced pipeline queue."""
        self.data_queue.put(data)
    
    def stop(self):
        """Stop the enhanced continuous pipeline."""
        self.running = False
        print("🛑 Stopping enhanced AI pipeline...")

# ============================
# Demo Functions
# ============================

async def demo_enhanced_pipeline():
    """Demonstrate the enhanced AI pipeline with PhaseWheel capabilities."""
    print("🔷 Enhanced AI Pipeline with PhaseWheel Realms Demo")
    print("=" * 60)
    
    # Create enhanced pipeline
    pipeline = EnhancedContinuousAIPipeline(num_carriers=3)
    
    print(f"Created enhanced pipeline with {len(pipeline.carriers)} carriers")
    print(f"Enhanced pipeline stages: {[stage.value for stage in EnhancedPipelineStage]}")
    
    # Add test data
    test_data = np.random.randn(1000) * 0.5
    pipeline.add_data(test_data)
    
    # Run for a few cycles
    try:
        await asyncio.wait_for(pipeline.enhanced_continuous_pipeline_cycle(), timeout=20.0)
    except asyncio.TimeoutError:
        print("\n⏰ Enhanced demo completed after 20 seconds")
    
    pipeline.stop()
    
    # Print final statistics
    print(f"\n📊 Enhanced Pipeline Statistics:")
    print(f"  Data processed: {len(pipeline.pipeline_state.processed_data)}")
    print(f"  Stage history: {[s.value for s in pipeline.pipeline_state.stage_history]}")
    print(f"  Final resonance: {pipeline.pipeline_state.resonance_energy:.3f}")
    print(f"  Final phase coherence: {pipeline.pipeline_state.phase_coherence:.3f}")
    print(f"  Final spectral richness: {pipeline.pipeline_state.spectral_richness:.3f}")

if __name__ == "__main__":
    print("🚀 Enhanced FieldIQ Pipeline with PhaseWheel Realms")
    print("=" * 60)
    
    # Run enhanced demo
    asyncio.run(demo_enhanced_pipeline())
    
    print(f"\n🔷 Enhanced AI Pipeline System Complete")
    print("✓ PhaseWheel realms integrated")
    print("✓ Enhanced phase processing")
    print("✓ Adaptive phase behavior")
    print("✓ Continuous processing pipeline")

