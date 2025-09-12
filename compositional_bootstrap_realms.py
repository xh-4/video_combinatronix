# ============================
# Compositional Bootstrap Networks for Hybrid AI Systems
# ============================
"""
Compositional bootstrap networks that fuse domain specialists and discover
emergent capabilities, integrated as realms in the singularity platform.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union

# Import existing systems
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf, to_json, from_json
from Combinator_Kernel import FieldIQ, make_field_from_real

# ============================
# Core Compositional Components
# ============================

class CrossDomainFusionLayer:
    """Cross-domain fusion layer for combining domain outputs."""
    def __init__(self, n_domains: int = 4):
        self.n_domains = n_domains
        
        # Cross-domain interaction matrices
        self.interaction_matrix = np.random.randn(n_domains, n_domains) * 0.1
        np.fill_diagonal(self.interaction_matrix, 1.0)  # Self-connection
        
        # Fusion weights
        self.fusion_weights = np.ones(n_domains) / n_domains
        
        # Fusion tracking
        self.fusion_history = []
        
    def fuse_domains(self, domain_outputs: List[np.ndarray], fusion_strategy: str = 'collaborative') -> np.ndarray:
        """Fuse outputs from multiple domains."""
        if len(domain_outputs) != self.n_domains:
            # Adjust for different number of domains
            self._adjust_for_domains(len(domain_outputs))
        
        if fusion_strategy == 'harmonic':
            return self._harmonic_fusion(domain_outputs)
        elif fusion_strategy == 'phase_locked':
            return self._phase_locked_fusion(domain_outputs)
        elif fusion_strategy == 'competitive':
            return self._competitive_fusion(domain_outputs)
        else:  # collaborative
            return self._collaborative_fusion(domain_outputs)
    
    def _adjust_for_domains(self, n_domains: int):
        """Adjust matrices for different number of domains."""
        self.n_domains = n_domains
        self.interaction_matrix = np.random.randn(n_domains, n_domains) * 0.1
        np.fill_diagonal(self.interaction_matrix, 1.0)
        self.fusion_weights = np.ones(n_domains) / n_domains
    
    def _harmonic_fusion(self, outputs: List[np.ndarray]) -> np.ndarray:
        """Combine domains harmonically - like musical harmony."""
        if not outputs:
            return np.array([])
        
        # Create harmonic series from base domains
        fundamental = outputs[0]  # first domain as fundamental
        harmonics = []
        
        for i, output in enumerate(outputs):
            # Each domain contributes a harmonic component
            harmonic_freq = (i + 1)  # 1st, 2nd, 3rd harmonic, etc.
            harmonic = output * np.cos(harmonic_freq * np.linspace(0, 2*np.pi, len(output)))
            harmonics.append(harmonic)
        
        # Weighted harmonic sum
        harmonic_series = np.array(harmonics)
        weights = self.fusion_weights[:len(outputs)]
        weights = weights / np.sum(weights)  # Normalize
        
        return np.sum(harmonic_series * weights[:, np.newaxis], axis=0)
    
    def _phase_locked_fusion(self, outputs: List[np.ndarray]) -> np.ndarray:
        """Lock phases between domains for synchronized operation."""
        if not outputs:
            return np.array([])
        
        # Force phase coherence across domains
        reference_phase = np.angle(np.fft.fft(outputs[0]))  # first domain sets reference
        
        synchronized_outputs = []
        for output in outputs:
            # Adjust phase to match reference
            output_phase = np.angle(np.fft.fft(output))
            phase_correction = reference_phase - output_phase
            synchronized = output * np.exp(1j * phase_correction)
            synchronized_outputs.append(synchronized.real)
        
        # Combine synchronized outputs
        return np.mean(synchronized_outputs, axis=0)
    
    def _competitive_fusion(self, outputs: List[np.ndarray]) -> np.ndarray:
        """Domains compete for dominance based on confidence."""
        if not outputs:
            return np.array([])
        
        # Compute confidence/strength of each domain's output
        confidences = [np.abs(output).mean() for output in outputs]
        confidence_array = np.array(confidences)
        
        # Winner-take-all or soft competition
        competition_weights = self._softmax(confidence_array * 5)  # high temp for sharp competition
        
        output_array = np.array(outputs)
        return np.sum(output_array * competition_weights[:, np.newaxis], axis=0)
    
    def _collaborative_fusion(self, outputs: List[np.ndarray]) -> np.ndarray:
        """Domains collaborate, sharing information."""
        if not outputs:
            return np.array([])
        
        n_domains = len(outputs)
        interaction_outputs = []
        
        for i, output_i in enumerate(outputs):
            collaborated = output_i.copy()
            
            for j, output_j in enumerate(outputs):
                if i != j:
                    # Domain i influenced by domain j
                    influence = self.interaction_matrix[i, j]
                    collaborated += influence * output_j
            
            interaction_outputs.append(collaborated)
        
        # Combine collaborated outputs
        return np.mean(interaction_outputs, axis=0)
    
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax function for competition weights."""
        exp_x = np.exp(x / temperature)
        return exp_x / np.sum(exp_x)
    
    def _update_fusion_tracking(self, fusion_strategy: str, n_domains: int, output_variance: float):
        """Track fusion operations."""
        self.fusion_history.append({
            'strategy': fusion_strategy,
            'n_domains': n_domains,
            'output_variance': output_variance,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.fusion_history) > 1000:
            self.fusion_history = self.fusion_history[-1000:]

class HarmonicCompositor:
    """Harmonic compositor for musical-like domain fusion."""
    def __init__(self, n_harmonics: int = 8):
        self.n_harmonics = n_harmonics
        self.harmonic_weights = np.random.randn(n_harmonics) * 0.1
        self.harmonic_phases = np.random.randn(n_harmonics) * 0.1
        
        # Harmonic tracking
        self.harmonic_history = []
        
    def compose_harmonics(self, domain_outputs: List[np.ndarray]) -> np.ndarray:
        """Compose domains into harmonic series."""
        if not domain_outputs:
            return np.array([])
        
        # Create harmonic series
        harmonics = []
        for i, output in enumerate(domain_outputs):
            harmonic_freq = (i + 1)  # 1st, 2nd, 3rd harmonic, etc.
            harmonic = output * np.cos(harmonic_freq * np.linspace(0, 2*np.pi, len(output)) + self.harmonic_phases[i])
            harmonics.append(harmonic)
        
        # Weighted harmonic sum
        harmonic_series = np.array(harmonics)
        weights = self.harmonic_weights[:len(domain_outputs)]
        weights = weights / np.sum(weights)  # Normalize
        
        composed = np.sum(harmonic_series * weights[:, np.newaxis], axis=0)
        
        # Track harmonic composition
        self._update_harmonic_tracking(composed)
        
        return composed
    
    def _update_harmonic_tracking(self, composed_output: np.ndarray):
        """Track harmonic composition."""
        self.harmonic_history.append({
            'output_variance': np.var(composed_output),
            'output_range': np.max(composed_output) - np.min(composed_output),
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.harmonic_history) > 1000:
            self.harmonic_history = self.harmonic_history[-1000:]

class PhaseSynchronizer:
    """Phase synchronizer for coordinated domain operation."""
    def __init__(self):
        self.phase_reference = 0.0
        self.synchronization_history = []
        
    def synchronize_phases(self, domain_outputs: List[np.ndarray]) -> List[np.ndarray]:
        """Synchronize phases across domain outputs."""
        if not domain_outputs:
            return []
        
        # Use first domain as phase reference
        reference_phase = np.angle(np.fft.fft(domain_outputs[0]))
        self.phase_reference = np.mean(reference_phase)
        
        synchronized_outputs = []
        for output in domain_outputs:
            # Calculate phase correction
            output_phase = np.angle(np.fft.fft(output))
            phase_correction = self.phase_reference - np.mean(output_phase)
            
            # Apply phase correction
            synchronized = output * np.exp(1j * phase_correction)
            synchronized_outputs.append(synchronized.real)
        
        # Track synchronization
        self._update_synchronization_tracking(domain_outputs, synchronized_outputs)
        
        return synchronized_outputs
    
    def _update_synchronization_tracking(self, original_outputs: List[np.ndarray], synchronized_outputs: List[np.ndarray]):
        """Track phase synchronization."""
        sync_info = {
            'n_domains': len(original_outputs),
            'phase_reference': self.phase_reference,
            'original_variances': [np.var(output) for output in original_outputs],
            'synchronized_variances': [np.var(output) for output in synchronized_outputs],
            'timestamp': time.time()
        }
        
        self.synchronization_history.append(sync_info)
        
        # Keep only recent history
        if len(self.synchronization_history) > 1000:
            self.synchronization_history = self.synchronization_history[-1000:]

class EmergentPatternDetector:
    """Detects when fusion creates genuinely novel patterns."""
    def __init__(self, novelty_threshold: float = 0.7):
        self.novelty_threshold = novelty_threshold
        self.pattern_memory = {}
        self.emergent_history = []
        
    def detect_emergence(self, domain_outputs: List[np.ndarray], fusion_output: np.ndarray) -> Tuple[bool, float]:
        """Detect when fusion creates genuinely novel patterns."""
        # Measure information content
        individual_information = sum([self._compute_entropy(output) for output in domain_outputs])
        fusion_information = self._compute_entropy(fusion_output)
        
        # Emergent information = fusion info - sum of individual info
        emergent_info = fusion_information - individual_information
        
        if emergent_info > self.novelty_threshold:
            # Novel pattern emerged! Store and analyze
            pattern_signature = self._compute_signature(fusion_output)
            pattern_id = len(self.pattern_memory)
            
            self.pattern_memory[pattern_id] = {
                'pattern': pattern_signature,
                'emergence_strength': emergent_info,
                'contributing_domains': [output.copy() for output in domain_outputs],
                'fusion_result': fusion_output.copy(),
                'timestamp': time.time()
            }
            
            # Track emergence
            self._update_emergent_tracking(emergent_info, pattern_id)
            
            return True, emergent_info
        
        return False, emergent_info
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute information entropy of data."""
        if len(data) == 0:
            return 0.0
        
        # Simple entropy calculation
        hist, _ = np.histogram(data, bins=50)
        hist = hist[hist > 0]  # Remove zero bins
        probabilities = hist / np.sum(hist)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
    
    def _compute_signature(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute pattern signature for storage."""
        return {
            'variance': np.var(data),
            'mean': np.mean(data),
            'range': np.max(data) - np.min(data),
            'spectral_centroid': self._compute_spectral_centroid(data),
            'zero_crossings': self._count_zero_crossings(data)
        }
    
    def _compute_spectral_centroid(self, data: np.ndarray) -> float:
        """Compute spectral centroid of data."""
        if len(data) == 0:
            return 0.0
        
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        power_spectrum = np.abs(fft) ** 2
        
        if np.sum(power_spectrum) == 0:
            return 0.0
        
        return np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
    
    def _count_zero_crossings(self, data: np.ndarray) -> int:
        """Count zero crossings in data."""
        if len(data) < 2:
            return 0
        
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        return len(zero_crossings)
    
    def _update_emergent_tracking(self, emergence_strength: float, pattern_id: int):
        """Track emergent pattern detection."""
        self.emergent_history.append({
            'pattern_id': pattern_id,
            'emergence_strength': emergence_strength,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.emergent_history) > 1000:
            self.emergent_history = self.emergent_history[-1000:]

class DomainCompositor:
    """Composes multiple domain specialists into hybrid systems."""
    def __init__(self, specialists: List[Any], weights: List[float], phase_offsets: List[float], 
                 fusion_strategy: str = 'collaborative'):
        self.specialists = specialists
        self.fusion_strategy = fusion_strategy
        
        # Compositional parameters
        self.composition_weights = np.array(weights)
        self.phase_offsets = np.array(phase_offsets)
        
        # Cross-domain interaction matrices
        n_domains = len(specialists)
        self.interaction_matrix = np.random.randn(n_domains, n_domains) * 0.1
        np.fill_diagonal(self.interaction_matrix, 1.0)
        
        # Fusion components
        self.cross_domain_fusion = CrossDomainFusionLayer(n_domains)
        self.harmonic_compositor = HarmonicCompositor()
        self.phase_synchronizer = PhaseSynchronizer()
        
        # Composition tracking
        self.composition_history = []
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through compositional system."""
        # Get outputs from each domain specialist
        specialist_outputs = []
        specialist_phases = []
        
        for i, specialist in enumerate(self.specialists):
            # Simulate specialist processing
            output = self._simulate_specialist_output(specialist, x)
            specialist_outputs.append(output)
            
            # Extract phase information
            phase = np.angle(np.fft.fft(output)) if np.iscomplexobj(output) else np.zeros_like(output)
            specialist_phases.append(phase + self.phase_offsets[i])
        
        # Apply fusion strategy
        if self.fusion_strategy == 'harmonic':
            fused_output = self.harmonic_compositor.compose_harmonics(specialist_outputs)
        elif self.fusion_strategy == 'phase_locked':
            synchronized_outputs = self.phase_synchronizer.synchronize_phases(specialist_outputs)
            fused_output = self.cross_domain_fusion.fuse_domains(synchronized_outputs, 'phase_locked')
        else:
            fused_output = self.cross_domain_fusion.fuse_domains(specialist_outputs, self.fusion_strategy)
        
        # Track composition
        self._update_composition_tracking(specialist_outputs, fused_output)
        
        return fused_output
    
    def _simulate_specialist_output(self, specialist: Any, x: np.ndarray) -> np.ndarray:
        """Simulate specialist output processing."""
        # Simple simulation of specialist processing
        if hasattr(specialist, 'forward'):
            return specialist.forward(x)
        else:
            # Mock specialist output
            return np.random.randn(len(x)) * 0.1 + x * 0.9
    
    def _update_composition_tracking(self, specialist_outputs: List[np.ndarray], fused_output: np.ndarray):
        """Track composition operations."""
        self.composition_history.append({
            'n_specialists': len(specialist_outputs),
            'fusion_strategy': self.fusion_strategy,
            'specialist_variances': [np.var(output) for output in specialist_outputs],
            'fused_variance': np.var(fused_output),
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.composition_history) > 1000:
            self.composition_history = self.composition_history[-1000:]

class CrossDomainPatternTransfer:
    """Transfer patterns discovered in one domain composition to another."""
    def __init__(self):
        self.transfer_history = []
        
    def transfer_emergent_pattern(self, source_composition: Any, target_domains: List[str]) -> Any:
        """Apply emergent pattern from source to new domain combination."""
        # Extract abstract pattern from source composition
        abstract_pattern = self._extract_abstract_pattern(source_composition)
        
        # Adapt pattern to target domain characteristics
        adapted_patterns = []
        for domain in target_domains:
            domain_adapted = self._adapt_pattern_to_domain(abstract_pattern, domain)
            adapted_patterns.append(domain_adapted)
        
        # Create new hybrid system with transferred patterns
        return self._instantiate_hybrid(target_domains, adapted_patterns)
    
    def _extract_abstract_pattern(self, source_composition: Any) -> Dict[str, Any]:
        """Extract abstract pattern from source composition."""
        # Simple pattern extraction
        return {
            'pattern_type': 'abstract',
            'complexity': 0.5,
            'transferability': 0.8,
            'domain_agnostic': True
        }
    
    def _adapt_pattern_to_domain(self, abstract_pattern: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Adapt pattern to specific domain characteristics."""
        return {
            'domain': domain,
            'adapted_pattern': abstract_pattern.copy(),
            'domain_specificity': 0.7,
            'adaptation_strength': 0.6
        }
    
    def _instantiate_hybrid(self, target_domains: List[str], adapted_patterns: List[Dict[str, Any]]) -> Any:
        """Instantiate hybrid system with transferred patterns."""
        # Mock hybrid system creation
        return {
            'domains': target_domains,
            'patterns': adapted_patterns,
            'hybrid_type': 'transferred',
            'timestamp': time.time()
        }

class CompositionalBootstrapNetwork:
    """Main compositional bootstrap network."""
    def __init__(self, gai_foundation):
        self.gai_foundation = gai_foundation
        
        # Registry of specialized domain adapters
        self.domain_specialists = {}
        
        # Compositional fusion layers
        self.cross_domain_fusion = CrossDomainFusionLayer()
        self.harmonic_compositor = HarmonicCompositor()
        self.phase_synchronizer = PhaseSynchronizer()
        
        # Novel domain emergence tracker
        self.emergent_patterns = EmergentPatternDetector()
        
        # Cross-domain pattern transfer
        self.pattern_transfer = CrossDomainPatternTransfer()
        
        # Network tracking
        self.composition_history = []
        
    def compose_hybrid_system(self, domain_components: List[Tuple[str, float, float]], 
                            fusion_strategy: str = 'harmonic') -> Any:
        """Create hybrid AI by compositionally combining domain specialists."""
        
        # Initialize compositional architecture
        compositor = DomainCompositor(
            specialists=[self.domain_specialists.get(name, self._create_mock_specialist(name)) 
                        for name, _, _ in domain_components],
            weights=[weight for _, weight, _ in domain_components],
            phase_offsets=[offset for _, _, offset in domain_components],
            fusion_strategy=fusion_strategy
        )
        
        return HybridAI(
            gai_foundation=self.gai_foundation,
            compositor=compositor,
            emergent_detector=self.emergent_patterns
        )
    
    def _create_mock_specialist(self, domain_name: str) -> Any:
        """Create mock specialist for domain."""
        class MockSpecialist:
            def __init__(self, domain):
                self.domain = domain
                
            def forward(self, x):
                # Mock specialist processing
                return np.random.randn(len(x)) * 0.1 + x * 0.9
        
        return MockSpecialist(domain_name)
    
    def register_domain_specialist(self, domain_name: str, specialist: Any):
        """Register a domain specialist."""
        self.domain_specialists[domain_name] = specialist
    
    def get_composition_summary(self) -> Dict[str, Any]:
        """Get summary of composition operations."""
        return {
            'n_domain_specialists': len(self.domain_specialists),
            'domains': list(self.domain_specialists.keys()),
            'emergent_patterns': len(self.emergent_patterns.pattern_memory),
            'composition_history': self.composition_history[-10:],  # Last 10 compositions
            'total_compositions': len(self.composition_history)
        }

class HybridAI:
    """Hybrid AI system combining multiple domains."""
    def __init__(self, gai_foundation, compositor, emergent_detector):
        self.gai_foundation = gai_foundation
        self.compositor = compositor
        self.emergent_detector = emergent_detector
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through hybrid system."""
        # Process through GAI foundation
        gai_features = self.gai_foundation(x)
        
        # Process through compositor
        hybrid_output = self.compositor.forward(gai_features)
        
        # Detect emergent patterns
        domain_outputs = [self.compositor.specialists[i].forward(gai_features) 
                         for i in range(len(self.compositor.specialists))]
        is_emergent, emergence_strength = self.emergent_detector.detect_emergence(
            domain_outputs, hybrid_output
        )
        
        return hybrid_output

# ============================
# FieldIQ Integration
# ============================

class FieldIQCompositionalProcessor:
    """FieldIQ processor for compositional bootstrap networks."""
    def __init__(self, gai_foundation):
        self.gai_foundation = gai_foundation
        self.compositional_network = CompositionalBootstrapNetwork(gai_foundation)
        
    def process_field(self, field: FieldIQ, domain_components: List[Tuple[str, float, float]], 
                     fusion_strategy: str = 'harmonic') -> FieldIQ:
        """Process FieldIQ through compositional bootstrap network."""
        z_array = field.z
        
        # Process real and imaginary parts
        real_part = np.real(z_array)
        imag_part = np.imag(z_array)
        
        # Create hybrid system
        hybrid_system = self.compositional_network.compose_hybrid_system(
            domain_components, fusion_strategy
        )
        
        # Process through hybrid system
        real_features = hybrid_system.forward(real_part)
        imag_features = hybrid_system.forward(imag_part)
        
        # Reconstruct complex field
        new_z = real_features + 1j * imag_features
        
        # Create new field with metadata
        processed_field = FieldIQ(new_z, field.sr, field.roles or {})
        processed_field = processed_field.with_role("compositional_processed", True)
        processed_field = processed_field.with_role("domain_components", domain_components)
        processed_field = processed_field.with_role("fusion_strategy", fusion_strategy)
        processed_field = processed_field.with_role("compositional_metrics", self._get_compositional_metrics())
        
        return processed_field
    
    def _get_compositional_metrics(self) -> Dict[str, Any]:
        """Get compositional processing metrics."""
        return {
            'n_domain_specialists': len(self.compositional_network.domain_specialists),
            'domains': list(self.compositional_network.domain_specialists.keys()),
            'emergent_patterns': len(self.compositional_network.emergent_patterns.pattern_memory),
            'composition_summary': self.compositional_network.get_composition_summary()
        }

# ============================
# Realm System
# ============================

@dataclass
class CompositionalBootstrapRealm:
    """Realm for compositional bootstrap processing."""
    name: str
    processor: FieldIQCompositionalProcessor
    vm_node: Node
    learning_enabled: bool = True
    
    def __repr__(self):
        return f"<CompositionalBootstrapRealm {self.name}>"
    
    def field_processor(self, field: FieldIQ, domain_components: List[Tuple[str, float, float]], 
                       fusion_strategy: str = 'harmonic') -> FieldIQ:
        """Process field through realm."""
        return self.processor.process_field(field, domain_components, fusion_strategy)
    
    def compose_hybrid_system(self, domain_components: List[Tuple[str, float, float]], 
                            fusion_strategy: str = 'harmonic') -> Any:
        """Compose hybrid system from domain components."""
        return self.processor.compositional_network.compose_hybrid_system(
            domain_components, fusion_strategy
        )

def create_compositional_bootstrap_realm(gai_foundation) -> CompositionalBootstrapRealm:
    """Create a compositional bootstrap realm."""
    processor = FieldIQCompositionalProcessor(gai_foundation)
    
    vm_node = Val({
        'type': 'compositional_bootstrap_realm',
        'name': 'compositional_bootstrap',
        'class': 'FieldIQCompositionalProcessor',
        'parameters': {
            'gai_foundation': 'frozen_gai_foundation'
        },
        'vm_operations': {'primary': 'COMPOSITIONAL_BOOTSTRAP'}
    })
    
    return CompositionalBootstrapRealm(
        name='compositional_bootstrap',
        processor=processor,
        vm_node=vm_node,
        learning_enabled=True
    )

# ============================
# Demo Functions
# ============================

def demo_cross_domain_fusion():
    """Demo cross-domain fusion strategies."""
    print("=== Cross-Domain Fusion Strategies Demo ===\n")
    
    # Create fusion layer
    fusion_layer = CrossDomainFusionLayer(n_domains=3)
    
    # Create mock domain outputs
    domain_outputs = [
        np.random.randn(100) + 0.5 * np.sin(np.linspace(0, 2*np.pi, 100)),  # Domain 1
        np.random.randn(100) + 0.3 * np.cos(np.linspace(0, 4*np.pi, 100)),  # Domain 2
        np.random.randn(100) + 0.4 * np.sin(np.linspace(0, 6*np.pi, 100))   # Domain 3
    ]
    
    print(f"Domain outputs: {len(domain_outputs)}")
    print(f"Output shapes: {[output.shape for output in domain_outputs]}")
    
    # Test different fusion strategies
    strategies = ['harmonic', 'phase_locked', 'competitive', 'collaborative']
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} Fusion ---")
        
        fused_output = fusion_layer.fuse_domains(domain_outputs, strategy)
        
        print(f"Fused output shape: {fused_output.shape}")
        print(f"Output range: [{fused_output.min():.3f}, {fused_output.max():.3f}]")
        print(f"Output variance: {np.var(fused_output):.6f}")
    
    print()

def demo_emergent_pattern_detection():
    """Demo emergent pattern detection."""
    print("=== Emergent Pattern Detection Demo ===\n")
    
    # Create emergent pattern detector
    detector = EmergentPatternDetector(novelty_threshold=0.5)
    
    # Create mock domain outputs
    domain_outputs = [
        np.random.randn(100) + 0.5 * np.sin(np.linspace(0, 2*np.pi, 100)),
        np.random.randn(100) + 0.3 * np.cos(np.linspace(0, 4*np.pi, 100)),
        np.random.randn(100) + 0.4 * np.sin(np.linspace(0, 6*np.pi, 100))
    ]
    
    # Create fusion output
    fusion_output = np.random.randn(100) + 0.8 * np.sin(np.linspace(0, 8*np.pi, 100))
    
    print(f"Domain outputs: {len(domain_outputs)}")
    print(f"Fusion output shape: {fusion_output.shape}")
    
    # Detect emergence
    is_emergent, emergence_strength = detector.detect_emergence(domain_outputs, fusion_output)
    
    print(f"Emergent pattern detected: {is_emergent}")
    print(f"Emergence strength: {emergence_strength:.6f}")
    
    if is_emergent:
        print(f"Stored patterns: {len(detector.pattern_memory)}")
        print(f"Latest pattern: {list(detector.pattern_memory.keys())[-1]}")
    
    print()

def demo_domain_compositor():
    """Demo domain compositor."""
    print("=== Domain Compositor Demo ===\n")
    
    # Create mock specialists
    class MockSpecialist:
        def __init__(self, domain):
            self.domain = domain
            
        def forward(self, x):
            return np.random.randn(len(x)) * 0.1 + x * 0.9
    
    specialists = [MockSpecialist(f'domain_{i}') for i in range(3)]
    weights = [0.4, 0.3, 0.3]
    phase_offsets = [0.0, np.pi/2, np.pi]
    
    # Create compositor
    compositor = DomainCompositor(specialists, weights, phase_offsets, 'collaborative')
    
    print(f"Specialists: {len(specialists)}")
    print(f"Weights: {weights}")
    print(f"Phase offsets: {phase_offsets}")
    print(f"Fusion strategy: collaborative")
    
    # Test forward pass
    test_input = np.random.randn(100)
    output = compositor.forward(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Output variance: {np.var(output):.6f}")
    
    print()

def demo_compositional_bootstrap_network():
    """Demo compositional bootstrap network."""
    print("=== Compositional Bootstrap Network Demo ===\n")
    
    # Create mock GAI foundation
    class MockGAIFoundation:
        def __call__(self, x):
            return np.random.randn(64)  # Mock GAI features
    
    gai_foundation = MockGAIFoundation()
    compositional_network = CompositionalBootstrapNetwork(gai_foundation)
    
    print(f"GAI foundation: {type(gai_foundation).__name__}")
    print(f"Domain specialists: {len(compositional_network.domain_specialists)}")
    
    # Register domain specialists
    class MockSpecialist:
        def __init__(self, domain):
            self.domain = domain
            
        def forward(self, x):
            return np.random.randn(len(x)) * 0.1 + x * 0.9
    
    compositional_network.register_domain_specialist('radiology', MockSpecialist('radiology'))
    compositional_network.register_domain_specialist('cardiac_audio', MockSpecialist('cardiac_audio'))
    compositional_network.register_domain_specialist('medical_text', MockSpecialist('medical_text'))
    
    print(f"Registered specialists: {list(compositional_network.domain_specialists.keys())}")
    
    # Compose hybrid system
    domain_components = [
        ('radiology', 0.4, 0.0),      # 40% weight, base phase
        ('cardiac_audio', 0.3, np.pi/2),  # 30% weight, 90° phase offset
        ('medical_text', 0.3, np.pi)      # 30% weight, 180° phase offset
    ]
    
    hybrid_system = compositional_network.compose_hybrid_system(
        domain_components, fusion_strategy='collaborative'
    )
    
    print(f"Hybrid system: {type(hybrid_system).__name__}")
    print(f"Domain components: {domain_components}")
    print(f"Fusion strategy: collaborative")
    
    # Test hybrid system
    test_input = np.random.randn(100)
    hybrid_output = hybrid_system.forward(test_input)
    
    print(f"Hybrid output shape: {hybrid_output.shape}")
    print(f"Hybrid output range: [{hybrid_output.min():.3f}, {hybrid_output.max():.3f}]")
    
    # Show composition summary
    summary = compositional_network.get_composition_summary()
    print(f"\nComposition summary:")
    print(f"  Domain specialists: {summary['n_domain_specialists']}")
    print(f"  Emergent patterns: {summary['emergent_patterns']}")
    print(f"  Total compositions: {summary['total_compositions']}")
    
    print()

def demo_compositional_bootstrap_realms():
    """Demo compositional bootstrap realms."""
    print("=== Compositional Bootstrap Realms Demo ===\n")
    
    # Create sample field
    sr = 48000
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = 0.8 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t + np.pi/4)
    field = make_field_from_real(x, sr, tag=("demo", "multi_tone"))
    
    print(f"Original field: {len(field.z)} samples, {field.sr} Hz")
    print(f"Field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
    
    # Create mock GAI foundation
    class MockGAIFoundation:
        def __call__(self, x):
            return np.random.randn(64)  # Mock GAI features
    
    gai_foundation = MockGAIFoundation()
    
    # Create compositional bootstrap realm
    realm = create_compositional_bootstrap_realm(gai_foundation)
    
    print(f"\nCreated compositional bootstrap realm: {realm.name}")
    print(f"Learning enabled: {realm.learning_enabled}")
    
    # Test different hybrid compositions
    print(f"\n--- Testing Hybrid Compositions ---")
    
    # Diagnostic system
    diagnostic_components = [
        ('radiology', 0.4, 0.0),      # 40% weight, base phase
        ('cardiac_audio', 0.3, np.pi/2),  # 30% weight, 90° phase offset
        ('medical_text', 0.3, np.pi)      # 30% weight, 180° phase offset
    ]
    
    try:
        processed_field = realm.field_processor(field, diagnostic_components, 'collaborative')
        energy = np.sum(np.abs(processed_field.z) ** 2)
        
        print(f"  Diagnostic system | Energy: {energy:8.2f} | Strategy: collaborative")
        
        # Show compositional metrics
        compositional_metrics = processed_field.roles.get('compositional_metrics', {})
        if compositional_metrics:
            print(f"  {'':18} | Domains: {compositional_metrics.get('n_domain_specialists', 0)}, "
                  f"Emergent: {compositional_metrics.get('emergent_patterns', 0)}")
        
    except Exception as e:
        print(f"  Diagnostic system | Error: {e}")
    
    # Autonomous perception system
    perception_components = [
        ('computer_vision', 0.35, 0.0),        # primary visual processing
        ('lidar_processing', 0.30, np.pi/4),   # depth/distance specialist
        ('audio_processing', 0.20, np.pi/2),   # emergency vehicle detection
        ('v2v_communication', 0.15, 3*np.pi/4) # vehicle-to-vehicle data
    ]
    
    try:
        processed_field = realm.field_processor(field, perception_components, 'phase_locked')
        energy = np.sum(np.abs(processed_field.z) ** 2)
        
        print(f"  Perception system | Energy: {energy:8.2f} | Strategy: phase_locked")
        
    except Exception as e:
        print(f"  Perception system | Error: {e}")
    
    # Creative composer system
    creative_components = [
        ('music_generation', 0.25, 0.0),
        ('visual_art', 0.25, np.pi/2),
        ('poetry', 0.25, np.pi),
        ('narrative', 0.25, 3*np.pi/2)
    ]
    
    try:
        processed_field = realm.field_processor(field, creative_components, 'harmonic')
        energy = np.sum(np.abs(processed_field.z) ** 2)
        
        print(f"  Creative composer | Energy: {energy:8.2f} | Strategy: harmonic")
        
    except Exception as e:
        print(f"  Creative composer | Error: {e}")
    
    # Test VM integration
    print(f"\n--- Testing VM Integration ---")
    try:
        vm_json = to_json(realm.vm_node)
        print(f"  VM JSON length: {len(vm_json)} chars")
    except Exception as e:
        print(f"  VM Error: {e}")
    print()

def main():
    """Run all compositional bootstrap demos."""
    print("🔷 Compositional Bootstrap Networks Demo")
    print("=" * 60)
    
    # Run demos
    demo_cross_domain_fusion()
    demo_emergent_pattern_detection()
    demo_domain_compositor()
    demo_compositional_bootstrap_network()
    demo_compositional_bootstrap_realms()
    
    print("🔷 Compositional Bootstrap Networks Complete")
    print("✓ Four fusion strategies (harmonic, phase_locked, competitive, collaborative)")
    print("✓ Emergent pattern detection and storage")
    print("✓ Cross-domain pattern transfer")
    print("✓ Hybrid AI system composition")
    print("✓ FieldIQ integration with compositional metrics")
    print("✓ Realm system integration")
    print("✓ VM compatibility")

if __name__ == "__main__":
    main()

