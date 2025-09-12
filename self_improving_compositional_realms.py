# ============================
# Self-Improving Compositional AI for Autonomous Evolution
# ============================
"""
Self-improving compositional AI that continuously evolves itself and discovers
new forms of intelligence, integrated as realms in the singularity platform.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union

# Import existing systems
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf, to_json, from_json
from Combinator_Kernel import FieldIQ, make_field_from_real

# ============================
# Core Self-Improving Components
# ============================

class EmergentPatternHarvester:
    """Harvests emergent patterns from experience for self-improvement."""
    def __init__(self, utility_threshold: float = 0.7):
        self.utility_threshold = utility_threshold
        self.pattern_memory = {}
        self.harvest_history = []
        
    def harvest_from_experience(self, experience_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Continuously mine experience for valuable emergent patterns."""
        harvested_patterns = []
        
        for experience in experience_batch:
            # Analyze multi-domain interactions in this experience
            domain_activations = self._extract_domain_activations(experience)
            
            # Look for novel activation patterns
            novel_patterns = self._detect_novelty(domain_activations)
            
            # Evaluate utility of novel patterns
            for pattern in novel_patterns:
                utility_score = self._evaluate_utility(pattern, experience)
                
                if utility_score > self.utility_threshold:
                    # Distill pattern into reusable domain component
                    distilled_domain = self._distill_pattern(pattern)
                    
                    harvested_patterns.append({
                        'pattern': pattern,
                        'distilled_domain': distilled_domain,
                        'utility_score': utility_score,
                        'source_experience': experience
                    })
        
        # Track harvesting
        self._update_harvest_tracking(harvested_patterns)
        
        return harvested_patterns
    
    def _extract_domain_activations(self, experience: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract domain activations from experience."""
        # Mock domain activation extraction
        domains = experience.get('domains', ['domain1', 'domain2', 'domain3'])
        activations = {}
        
        for domain in domains:
            # Simulate domain activation
            activation = np.random.randn(100) * 0.1 + np.sin(np.linspace(0, 2*np.pi, 100))
            activations[domain] = activation
        
        return activations
    
    def _detect_novelty(self, domain_activations: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect novel patterns in domain activations."""
        novel_patterns = []
        
        # Simple novelty detection based on variance and correlation
        for domain, activation in domain_activations.items():
            variance = np.var(activation)
            spectral_centroid = self._compute_spectral_centroid(activation)
            
            # Check if this is novel compared to memory
            is_novel = self._check_novelty(variance, spectral_centroid)
            
            if is_novel:
                novel_patterns.append({
                    'domain': domain,
                    'activation': activation,
                    'variance': variance,
                    'spectral_centroid': spectral_centroid,
                    'novelty_score': variance * spectral_centroid
                })
        
        return novel_patterns
    
    def _evaluate_utility(self, pattern: Dict[str, Any], experience: Dict[str, Any]) -> float:
        """Evaluate utility of a pattern."""
        # Simple utility evaluation based on pattern characteristics
        novelty_score = pattern.get('novelty_score', 0)
        experience_complexity = experience.get('complexity', 1.0)
        
        # Higher novelty and complexity = higher utility
        utility = novelty_score * experience_complexity * 0.1
        
        return min(1.0, utility)  # Cap at 1.0
    
    def _distill_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Distill pattern into reusable domain component."""
        return {
            'domain_name': f"emergent_{pattern['domain']}_{len(self.pattern_memory)}",
            'pattern_type': 'emergent',
            'activation_template': pattern['activation'].copy(),
            'variance_threshold': pattern['variance'],
            'spectral_centroid': pattern['spectral_centroid'],
            'creation_timestamp': time.time()
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
    
    def _check_novelty(self, variance: float, spectral_centroid: float) -> bool:
        """Check if pattern is novel compared to memory."""
        # Simple novelty check - in practice would compare to stored patterns
        return variance > 0.5 and abs(spectral_centroid) > 0.1
    
    def _update_harvest_tracking(self, harvested_patterns: List[Dict[str, Any]]):
        """Track pattern harvesting."""
        self.harvest_history.append({
            'n_patterns': len(harvested_patterns),
            'timestamp': time.time(),
            'utility_scores': [p['utility_score'] for p in harvested_patterns]
        })
        
        # Keep only recent history
        if len(self.harvest_history) > 1000:
            self.harvest_history = self.harvest_history[-1000:]

class ArchitectureEvolver:
    """Evolves the overall system architecture to incorporate improvements."""
    def __init__(self):
        self.evolution_history = []
        self.architecture_memory = {}
        
    def evolve_architecture(self, current_domains: List[str], new_domains: List[Dict[str, Any]], 
                          performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve architecture to optimally integrate new capabilities."""
        
        # Generate candidate architectures
        candidate_architectures = self._generate_candidates(current_domains, new_domains)
        
        # Optimize connections between domains
        for candidate in candidate_architectures:
            candidate['connections'] = self._optimize_connections(candidate['domains'])
            candidate['phase_relationships'] = self._evolve_phase_relationships(candidate['domains'])
        
        # Select best candidate
        best_architecture = self._select_best_architecture(candidate_architectures)
        
        # Track evolution
        self._update_evolution_tracking(best_architecture)
        
        return best_architecture
    
    def _generate_candidates(self, current_domains: List[str], new_domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate candidate architectures."""
        candidates = []
        
        # Candidate 1: Add all new domains
        candidate1 = {
            'domains': current_domains + [d['domain_name'] for d in new_domains],
            'strategy': 'add_all',
            'complexity': len(current_domains) + len(new_domains)
        }
        candidates.append(candidate1)
        
        # Candidate 2: Add only high-utility domains
        high_utility_domains = [d for d in new_domains if d.get('utility_score', 0) > 0.8]
        candidate2 = {
            'domains': current_domains + [d['domain_name'] for d in high_utility_domains],
            'strategy': 'selective',
            'complexity': len(current_domains) + len(high_utility_domains)
        }
        candidates.append(candidate2)
        
        # Candidate 3: Replace low-performing domains
        candidate3 = {
            'domains': current_domains[1:] + [d['domain_name'] for d in new_domains[:2]],
            'strategy': 'replacement',
            'complexity': len(current_domains) - 1 + min(2, len(new_domains))
        }
        candidates.append(candidate3)
        
        return candidates
    
    def _optimize_connections(self, domains: List[str]) -> Dict[str, List[str]]:
        """Optimize connections between domains."""
        connections = {}
        
        for i, domain in enumerate(domains):
            # Connect to nearby domains
            connected_domains = []
            for j, other_domain in enumerate(domains):
                if i != j and abs(i - j) <= 2:  # Connect to nearby domains
                    connected_domains.append(other_domain)
            
            connections[domain] = connected_domains
        
        return connections
    
    def _evolve_phase_relationships(self, domains: List[str]) -> Dict[str, float]:
        """Evolve phase relationships between domains."""
        phase_relationships = {}
        
        for i, domain in enumerate(domains):
            # Distribute phases evenly
            phase = (i * 2 * np.pi) / len(domains)
            phase_relationships[domain] = phase
        
        return phase_relationships
    
    def _select_best_architecture(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best architecture through virtual testing."""
        # Simple selection based on complexity and strategy
        best_candidate = min(candidates, key=lambda c: c['complexity'])
        
        # Add performance prediction
        best_candidate['predicted_performance'] = 0.8 - (best_candidate['complexity'] * 0.01)
        
        return best_candidate
    
    def _update_evolution_tracking(self, architecture: Dict[str, Any]):
        """Track architecture evolution."""
        self.evolution_history.append({
            'architecture': architecture,
            'timestamp': time.time(),
            'generation': len(self.evolution_history)
        })
        
        # Keep only recent history
        if len(self.evolution_history) > 100:
            self.evolution_history = self.evolution_history[-100:]

class RecursiveEvolutionEngine:
    """Engine that improves its own improvement processes."""
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.improvement_levels = {}
        self.recursion_history = []
        
    def recursive_self_improvement(self, depth_limit: int = 5) -> Dict[str, Any]:
        """Recursive improvement up to specified depth."""
        improvements = {}
        
        for depth in range(min(depth_limit, self.max_depth)):
            if depth == 0:
                # Base level: improve performance
                improvement = self._improve_performance()
            else:
                # Meta levels: improve the improver at depth-1
                improvement = self._improve_improver(depth)
            
            improvements[f'level_{depth}'] = improvement
            
            # Check if this level discovered ways to improve higher levels
            if improvement.get('suggests_meta_improvement', False):
                # Recursively create new improvement levels
                self._spawn_higher_level_improver(depth + 1, improvement.get('meta_suggestions', []))
        
        # Track recursion
        self._update_recursion_tracking(improvements)
        
        return improvements
    
    def _improve_performance(self) -> Dict[str, Any]:
        """Base level: improve performance."""
        return {
            'improvement_type': 'performance',
            'improvement_score': np.random.rand() * 0.3 + 0.7,  # 0.7-1.0
            'suggests_meta_improvement': np.random.rand() > 0.8,
            'meta_suggestions': ['optimize_learning_rate', 'adjust_architecture'] if np.random.rand() > 0.8 else []
        }
    
    def _improve_improver(self, depth: int) -> Dict[str, Any]:
        """Meta level: improve the improver at depth-1."""
        return {
            'improvement_type': f'meta_improvement_level_{depth}',
            'improvement_score': np.random.rand() * 0.2 + 0.8,  # 0.8-1.0
            'suggests_meta_improvement': np.random.rand() > 0.9,
            'meta_suggestions': ['optimize_meta_learning', 'adjust_recursion_depth'] if np.random.rand() > 0.9 else []
        }
    
    def _spawn_higher_level_improver(self, depth: int, suggestions: List[str]):
        """Spawn new improvement level based on suggestions."""
        if depth < self.max_depth:
            self.improvement_levels[f'level_{depth}'] = {
                'suggestions': suggestions,
                'created_at': time.time()
            }
    
    def _update_recursion_tracking(self, improvements: Dict[str, Any]):
        """Track recursive improvement."""
        self.recursion_history.append({
            'improvements': improvements,
            'timestamp': time.time(),
            'max_depth': len(improvements)
        })
        
        # Keep only recent history
        if len(self.recursion_history) > 100:
            self.recursion_history = self.recursion_history[-100:]

class StabilityController:
    """Prevents runaway self-improvement from destabilizing the system."""
    def __init__(self, stability_threshold: float = 0.7):
        self.stability_threshold = stability_threshold
        self.stability_history = []
        
    def controlled_improvement(self, proposed_improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvements with safety controls."""
        
        # Test improvement in sandbox
        sandbox_result = self._test_in_sandbox(proposed_improvement)
        
        # Check for instability
        stability_score = self._assess_stability(sandbox_result)
        
        if stability_score < self.stability_threshold:
            # Reject potentially destabilizing improvement
            return self._generate_conservative_alternative(proposed_improvement)
        
        # Apply with gradual rollout
        return self._gradual_rollout(proposed_improvement)
    
    def _test_in_sandbox(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Test improvement in sandbox environment."""
        # Mock sandbox testing
        return {
            'performance_impact': np.random.rand() * 0.5 + 0.5,  # 0.5-1.0
            'stability_impact': np.random.rand() * 0.3 + 0.7,    # 0.7-1.0
            'resource_usage': np.random.rand() * 0.2 + 0.8,      # 0.8-1.0
            'compatibility_score': np.random.rand() * 0.4 + 0.6  # 0.6-1.0
        }
    
    def _assess_stability(self, sandbox_result: Dict[str, Any]) -> float:
        """Assess stability of sandbox result."""
        # Weighted stability assessment
        weights = {
            'performance_impact': 0.3,
            'stability_impact': 0.4,
            'resource_usage': 0.2,
            'compatibility_score': 0.1
        }
        
        stability_score = sum(sandbox_result[key] * weights[key] for key in weights)
        return stability_score
    
    def _generate_conservative_alternative(self, proposed_improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conservative alternative to proposed improvement."""
        return {
            'improvement_type': 'conservative',
            'original_proposal': proposed_improvement,
            'conservative_factor': 0.5,
            'safety_measures': ['gradual_rollout', 'monitoring', 'rollback_ready']
        }
    
    def _gradual_rollout(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvement with gradual rollout."""
        return {
            'improvement_type': 'gradual_rollout',
            'rollout_phases': ['test', 'limited', 'full'],
            'monitoring_required': True,
            'rollback_available': True
        }

class SelfImprovingCompositionalAI:
    """Main self-improving compositional AI system."""
    def __init__(self, initial_gai_foundation, improvement_threshold: float = 0.8):
        self.gai_foundation = initial_gai_foundation
        self.improvement_threshold = improvement_threshold
        
        # Core evolving components
        self.domain_specialists = {}
        self.emergent_domains = {}
        self.meta_domains = {}
        
        # Self-improvement machinery
        self.pattern_harvester = EmergentPatternHarvester()
        self.architecture_evolver = ArchitectureEvolver()
        self.performance_monitor = PerformanceMonitor()
        self.recursive_engine = RecursiveEvolutionEngine()
        self.stability_controller = StabilityController()
        
        # Evolution tracking
        self.generation = 0
        self.evolution_history = []
        self.improvement_metrics = {}
        
    def self_improve_cycle(self, experience_batch: List[Dict[str, Any]]) -> bool:
        """Complete self-improvement cycle."""
        
        # Phase 1: Harvest emergent patterns from recent experience
        new_patterns = self.pattern_harvester.harvest_from_experience(experience_batch)
        
        # Phase 2: Evaluate patterns for integration potential
        valuable_patterns = self._evaluate_patterns(new_patterns)
        
        # Phase 3: Integrate valuable patterns as new domains
        new_domains = self._integrate_patterns_as_domains(valuable_patterns)
        
        # Phase 4: Evolve architecture to accommodate new domains
        improved_architecture = self.architecture_evolver.evolve_architecture(
            current_domains=list(self.domain_specialists.keys()) + list(self.emergent_domains.keys()),
            new_domains=new_domains,
            performance_history=self.performance_monitor.get_history()
        )
        
        # Phase 5: Test and validate improvements
        improvement_validated = self._validate_improvements(improved_architecture)
        
        if improvement_validated:
            self._commit_improvements(improved_architecture, new_domains)
            self.generation += 1
            
        # Track evolution
        self._update_evolution_tracking(improvement_validated, new_domains)
        
        return improvement_validated
    
    def _evaluate_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate patterns for integration potential."""
        return [p for p in patterns if p['utility_score'] > self.improvement_threshold]
    
    def _integrate_patterns_as_domains(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Integrate valuable patterns as new domains."""
        new_domains = []
        
        for pattern in patterns:
            domain_name = pattern['distilled_domain']['domain_name']
            self.emergent_domains[domain_name] = pattern['distilled_domain']
            new_domains.append(pattern['distilled_domain'])
        
        return new_domains
    
    def _validate_improvements(self, architecture: Dict[str, Any]) -> bool:
        """Test and validate improvements."""
        # Mock validation - in practice would test in sandbox
        predicted_performance = architecture.get('predicted_performance', 0.5)
        return predicted_performance > 0.7
    
    def _commit_improvements(self, architecture: Dict[str, Any], new_domains: List[Dict[str, Any]]):
        """Commit validated improvements."""
        # Update domain specialists
        for domain in architecture['domains']:
            if domain not in self.domain_specialists and domain not in self.emergent_domains:
                self.domain_specialists[domain] = self._create_domain_specialist(domain)
        
        # Update architecture
        self.current_architecture = architecture
    
    def _create_domain_specialist(self, domain_name: str) -> Any:
        """Create domain specialist."""
        class DomainSpecialist:
            def __init__(self, name):
                self.name = name
                
            def forward(self, x):
                return np.random.randn(len(x)) * 0.1 + x * 0.9
        
        return DomainSpecialist(domain_name)
    
    def _update_evolution_tracking(self, improvement_validated: bool, new_domains: List[Dict[str, Any]]):
        """Track evolution process."""
        self.evolution_history.append({
            'generation': self.generation,
            'improvement_validated': improvement_validated,
            'new_domains': len(new_domains),
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-1000:]

class PerformanceMonitor:
    """Monitors performance for self-improvement decisions."""
    def __init__(self):
        self.performance_history = []
        
    def get_history(self) -> List[Dict[str, Any]]:
        """Get performance history."""
        return self.performance_history
    
    def record_performance(self, performance_metrics: Dict[str, Any]):
        """Record performance metrics."""
        self.performance_history.append({
            'metrics': performance_metrics,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

# ============================
# FieldIQ Integration
# ============================

class FieldIQSelfImprovingProcessor:
    """FieldIQ processor for self-improving compositional AI."""
    def __init__(self, gai_foundation):
        self.gai_foundation = gai_foundation
        self.self_improving_ai = SelfImprovingCompositionalAI(gai_foundation)
        
    def process_field(self, field: FieldIQ, domain_components: List[Tuple[str, float, float]], 
                     fusion_strategy: str = 'harmonic') -> FieldIQ:
        """Process FieldIQ through self-improving AI."""
        z_array = field.z
        
        # Process real and imaginary parts
        real_part = np.real(z_array)
        imag_part = np.imag(z_array)
        
        # Process through self-improving AI
        real_features = self._process_with_self_improving_ai(real_part, domain_components, fusion_strategy)
        imag_features = self._process_with_self_improving_ai(imag_part, domain_components, fusion_strategy)
        
        # Reconstruct complex field
        new_z = real_features + 1j * imag_features
        
        # Create new field with metadata
        processed_field = FieldIQ(new_z, field.sr, field.roles or {})
        processed_field = processed_field.with_role("self_improving_processed", True)
        processed_field = processed_field.with_role("domain_components", domain_components)
        processed_field = processed_field.with_role("fusion_strategy", fusion_strategy)
        processed_field = processed_field.with_role("generation", self.self_improving_ai.generation)
        processed_field = processed_field.with_role("self_improving_metrics", self._get_self_improving_metrics())
        
        return processed_field
    
    def _process_with_self_improving_ai(self, data: np.ndarray, domain_components: List[Tuple[str, float, float]], 
                                      fusion_strategy: str) -> np.ndarray:
        """Process data through self-improving AI."""
        # Simulate self-improving AI processing
        if len(data) > 64:
            # Downsample to fixed size
            indices = np.linspace(0, len(data)-1, 64, dtype=int)
            return data[indices]
        else:
            # Pad to fixed size
            padded = np.pad(data, (0, 64 - len(data)), mode='constant')
            return padded
    
    def _get_self_improving_metrics(self) -> Dict[str, Any]:
        """Get self-improving AI metrics."""
        return {
            'generation': self.self_improving_ai.generation,
            'n_domain_specialists': len(self.self_improving_ai.domain_specialists),
            'n_emergent_domains': len(self.self_improving_ai.emergent_domains),
            'n_meta_domains': len(self.self_improving_ai.meta_domains),
            'evolution_history_length': len(self.self_improving_ai.evolution_history)
        }

# ============================
# Realm System
# ============================

@dataclass
class SelfImprovingCompositionalRealm:
    """Realm for self-improving compositional AI."""
    name: str
    processor: FieldIQSelfImprovingProcessor
    vm_node: Node
    learning_enabled: bool = True
    
    def __repr__(self):
        return f"<SelfImprovingCompositionalRealm {self.name}>"
    
    def field_processor(self, field: FieldIQ, domain_components: List[Tuple[str, float, float]], 
                       fusion_strategy: str = 'harmonic') -> FieldIQ:
        """Process field through realm."""
        return self.processor.process_field(field, domain_components, fusion_strategy)
    
    def self_improve_cycle(self, experience_batch: List[Dict[str, Any]]) -> bool:
        """Run self-improvement cycle."""
        return self.processor.self_improving_ai.self_improve_cycle(experience_batch)

def create_self_improving_compositional_realm(gai_foundation) -> SelfImprovingCompositionalRealm:
    """Create a self-improving compositional realm."""
    processor = FieldIQSelfImprovingProcessor(gai_foundation)
    
    vm_node = Val({
        'type': 'self_improving_compositional_realm',
        'name': 'self_improving_compositional',
        'class': 'FieldIQSelfImprovingProcessor',
        'parameters': {
            'gai_foundation': 'frozen_gai_foundation'
        },
        'vm_operations': {'primary': 'SELF_IMPROVING_COMPOSITIONAL'}
    })
    
    return SelfImprovingCompositionalRealm(
        name='self_improving_compositional',
        processor=processor,
        vm_node=vm_node,
        learning_enabled=True
    )

# ============================
# Demo Functions
# ============================

def demo_emergent_pattern_harvester():
    """Demo emergent pattern harvester."""
    print("=== Emergent Pattern Harvester Demo ===\n")
    
    harvester = EmergentPatternHarvester(utility_threshold=0.7)
    
    # Create mock experience batch
    experience_batch = [
        {'domains': ['radiology', 'pathology'], 'complexity': 0.8},
        {'domains': ['cardiac_audio', 'medical_text'], 'complexity': 0.9},
        {'domains': ['vision', 'audio'], 'complexity': 0.7}
    ]
    
    print(f"Experience batch: {len(experience_batch)} experiences")
    
    # Harvest patterns
    harvested_patterns = harvester.harvest_from_experience(experience_batch)
    
    print(f"Harvested patterns: {len(harvested_patterns)}")
    
    for i, pattern in enumerate(harvested_patterns):
        print(f"  Pattern {i+1}:")
        print(f"    Domain: {pattern['distilled_domain']['domain_name']}")
        print(f"    Utility score: {pattern['utility_score']:.3f}")
        print(f"    Pattern type: {pattern['distilled_domain']['pattern_type']}")
    
    print()

def demo_architecture_evolver():
    """Demo architecture evolver."""
    print("=== Architecture Evolver Demo ===\n")
    
    evolver = ArchitectureEvolver()
    
    # Mock current domains and new domains
    current_domains = ['radiology', 'pathology', 'clinical_notes']
    new_domains = [
        {'domain_name': 'emergent_radiology_pathology_0', 'utility_score': 0.8},
        {'domain_name': 'emergent_cardiac_audio_1', 'utility_score': 0.9},
        {'domain_name': 'emergent_vision_audio_2', 'utility_score': 0.7}
    ]
    performance_history = [{'performance': 0.8}, {'performance': 0.85}]
    
    print(f"Current domains: {current_domains}")
    print(f"New domains: {[d['domain_name'] for d in new_domains]}")
    
    # Evolve architecture
    evolved_architecture = evolver.evolve_architecture(current_domains, new_domains, performance_history)
    
    print(f"Evolved architecture:")
    print(f"  Domains: {evolved_architecture['domains']}")
    print(f"  Strategy: {evolved_architecture['strategy']}")
    print(f"  Complexity: {evolved_architecture['complexity']}")
    print(f"  Predicted performance: {evolved_architecture.get('predicted_performance', 0):.3f}")
    
    print()

def demo_recursive_evolution_engine():
    """Demo recursive evolution engine."""
    print("=== Recursive Evolution Engine Demo ===\n")
    
    engine = RecursiveEvolutionEngine(max_depth=3)
    
    # Run recursive improvement
    improvements = engine.recursive_self_improvement(depth_limit=3)
    
    print(f"Recursive improvements: {len(improvements)} levels")
    
    for level, improvement in improvements.items():
        print(f"  {level}:")
        print(f"    Type: {improvement['improvement_type']}")
        print(f"    Score: {improvement['improvement_score']:.3f}")
        print(f"    Suggests meta: {improvement['suggests_meta_improvement']}")
        if improvement['meta_suggestions']:
            print(f"    Suggestions: {improvement['meta_suggestions']}")
    
    print()

def demo_self_improving_compositional_ai():
    """Demo self-improving compositional AI."""
    print("=== Self-Improving Compositional AI Demo ===\n")
    
    # Create mock GAI foundation
    class MockGAIFoundation:
        def __call__(self, x):
            return np.random.randn(64)  # Mock GAI features
    
    gai_foundation = MockGAIFoundation()
    self_improving_ai = SelfImprovingCompositionalAI(gai_foundation)
    
    print(f"Initial generation: {self_improving_ai.generation}")
    print(f"Initial domains: {len(self_improving_ai.domain_specialists)}")
    
    # Create mock experience batch
    experience_batch = [
        {'domains': ['radiology', 'pathology'], 'complexity': 0.8},
        {'domains': ['cardiac_audio', 'medical_text'], 'complexity': 0.9},
        {'domains': ['vision', 'audio'], 'complexity': 0.7}
    ]
    
    print(f"Experience batch: {len(experience_batch)} experiences")
    
    # Run self-improvement cycle
    improvement_validated = self_improving_ai.self_improve_cycle(experience_batch)
    
    print(f"Improvement validated: {improvement_validated}")
    print(f"New generation: {self_improving_ai.generation}")
    print(f"Domain specialists: {len(self_improving_ai.domain_specialists)}")
    print(f"Emergent domains: {len(self_improving_ai.emergent_domains)}")
    print(f"Meta domains: {len(self_improving_ai.meta_domains)}")
    
    print()

def demo_self_improving_compositional_realms():
    """Demo self-improving compositional realms."""
    print("=== Self-Improving Compositional Realms Demo ===\n")
    
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
    
    # Create self-improving compositional realm
    realm = create_self_improving_compositional_realm(gai_foundation)
    
    print(f"\nCreated self-improving compositional realm: {realm.name}")
    print(f"Learning enabled: {realm.learning_enabled}")
    print(f"Initial generation: {realm.processor.self_improving_ai.generation}")
    
    # Test processing
    domain_components = [
        ('radiology', 0.4, 0.0),
        ('cardiac_audio', 0.3, np.pi/2),
        ('medical_text', 0.3, np.pi)
    ]
    
    try:
        processed_field = realm.field_processor(field, domain_components, 'collaborative')
        energy = np.sum(np.abs(processed_field.z) ** 2)
        
        print(f"\nProcessed field:")
        print(f"  Energy: {energy:8.2f}")
        print(f"  Generation: {processed_field.roles.get('generation', 0)}")
        
        # Show self-improving metrics
        metrics = processed_field.roles.get('self_improving_metrics', {})
        if metrics:
            print(f"  Domain specialists: {metrics.get('n_domain_specialists', 0)}")
            print(f"  Emergent domains: {metrics.get('n_emergent_domains', 0)}")
            print(f"  Meta domains: {metrics.get('n_meta_domains', 0)}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test self-improvement cycle
    print(f"\n--- Testing Self-Improvement Cycle ---")
    
    experience_batch = [
        {'domains': ['radiology', 'pathology'], 'complexity': 0.8},
        {'domains': ['cardiac_audio', 'medical_text'], 'complexity': 0.9}
    ]
    
    try:
        improvement_validated = realm.self_improve_cycle(experience_batch)
        print(f"Self-improvement cycle completed: {improvement_validated}")
        print(f"New generation: {realm.processor.self_improving_ai.generation}")
        
    except Exception as e:
        print(f"Self-improvement error: {e}")
    
    # Test VM integration
    print(f"\n--- Testing VM Integration ---")
    try:
        vm_json = to_json(realm.vm_node)
        print(f"  VM JSON length: {len(vm_json)} chars")
    except Exception as e:
        print(f"  VM Error: {e}")
    print()

def main():
    """Run all self-improving compositional demos."""
    print("🔷 Self-Improving Compositional AI Demo")
    print("=" * 60)
    
    # Run demos
    demo_emergent_pattern_harvester()
    demo_architecture_evolver()
    demo_recursive_evolution_engine()
    demo_self_improving_compositional_ai()
    demo_self_improving_compositional_realms()
    
    print("🔷 Self-Improving Compositional AI Complete")
    print("✓ Emergent pattern harvesting from experience")
    print("✓ Architecture evolution for new capabilities")
    print("✓ Recursive evolution engine for meta-improvement")
    print("✓ Stability control for safe evolution")
    print("✓ FieldIQ integration with self-improving metrics")
    print("✓ Realm system integration")
    print("✓ VM compatibility")

if __name__ == "__main__":
    main()

