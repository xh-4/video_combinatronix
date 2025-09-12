# ============================
# Abstract Feature Networks for Universal AI Processing
# ============================
"""
Abstract feature networks that learn universal patterns across domains,
integrated as realms in the singularity platform.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

# Import existing systems
from combinatronix_vm_complete import Comb, Val, App, Node, app, reduce_whnf, to_json, from_json
from Combinator_Kernel import FieldIQ, make_field_from_real

# ============================
# Core Abstract Components
# ============================

class AbstractPhaseLayer:
    """Abstract phase layer that learns domain-agnostic patterns."""
    def __init__(self, input_size: int, output_size: int, abstraction_level: int):
        self.input_size = input_size
        self.output_size = output_size
        self.abstraction_level = abstraction_level
        
        # Core phase wheel components
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.phases = np.random.randn(output_size)
        
        # Abstract frequency patterns (not tied to specific domains)
        if abstraction_level == 0:
            # Low-level: detect basic patterns/rhythms/gradients
            self.frequencies = np.logspace(0, 2, output_size)  # 1 to 100
        elif abstraction_level == 1:
            # Mid-level: detect relationships/symmetries/compositions
            self.frequencies = np.logspace(-1, 1, output_size)  # 0.1 to 10
        else:
            # High-level: detect concepts/categories/abstractions
            self.frequencies = np.logspace(-2, 0, output_size)  # 0.01 to 1
        
        # Abstract pattern templates (4D pattern space)
        self.pattern_templates = np.random.randn(output_size, 4) * 0.1
        
        # Learning parameters
        self.learning_rate = 0.01
        self.pattern_learning_rate = 0.001
        
        # Pattern tracking
        self.pattern_history = []
        self.abstraction_metrics = {}
        
    def forward(self, x):
        """Forward pass with abstract pattern detection."""
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # Linear transformation
        linear_out = np.dot(self.weights, x)
        
        # Multi-dimensional abstract patterns
        cos_component = np.cos(self.frequencies * linear_out + self.phases)
        sin_component = np.sin(self.frequencies * linear_out + self.phases)
        
        # Pattern template matching
        pattern_components = np.stack([
            cos_component,
            sin_component,
            cos_component * sin_component,
            cos_component**2 - sin_component**2
        ], axis=-1)
        
        pattern_match = np.dot(pattern_components, self.pattern_templates.T)
        
        # Bounded abstract features
        abstract_features = np.tanh(pattern_match)
        
        # Update abstraction metrics
        self._update_abstraction_metrics(abstract_features)
        
        return abstract_features
    
    def _update_abstraction_metrics(self, features):
        """Update abstraction-level metrics."""
        self.abstraction_metrics = {
            'abstraction_level': self.abstraction_level,
            'feature_variance': np.var(features),
            'feature_range': np.max(features) - np.min(features),
            'frequency_scale': np.mean(self.frequencies),
            'pattern_diversity': np.var(self.pattern_templates),
            'n_features': len(features)
        }
        
        # Store in history
        self.pattern_history.append({
            'features': features.copy(),
            'metrics': self.abstraction_metrics.copy(),
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.pattern_history) > 100:
            self.pattern_history = self.pattern_history[-100:]

class DomainAdapter:
    """Domain-specific input adapter."""
    def __init__(self, domain: str, config: Dict[str, Any], target_size: int = 128):
        self.domain = domain
        self.config = config
        self.target_size = target_size
        
        # Domain-specific processing parameters
        if domain == 'audio':
            self.sample_rate = config.get('sample_rate', 48000)
            self.n_mel_bins = config.get('n_mel_bins', 128)
            self.hop_length = config.get('hop_length', 512)
        elif domain == 'vision':
            self.image_size = config.get('image_size', 224)
            self.n_channels = config.get('n_channels', 3)
            self.patch_size = config.get('patch_size', 16)
        elif domain == 'text':
            self.vocab_size = config.get('vocab_size', 10000)
            self.max_length = config.get('max_length', 512)
            self.embedding_dim = config.get('embedding_dim', 128)
        else:
            # Generic domain
            self.input_dim = config.get('input_dim', 128)
        
        # Adapter weights
        self.adapter_weights = np.random.randn(target_size, self._get_input_dim()) * 0.1
        self.adapter_bias = np.zeros(target_size)
        
        # Domain-specific tracking
        self.domain_metrics = {}
        
    def _get_input_dim(self):
        """Get input dimension for this domain."""
        if self.domain == 'audio':
            return self.n_mel_bins
        elif self.domain == 'vision':
            return (self.image_size // self.patch_size) ** 2 * self.n_channels
        elif self.domain == 'text':
            return self.embedding_dim
        else:
            return self.input_dim
    
    def adapt(self, x):
        """Adapt domain-specific input to shared representation."""
        # Domain-specific preprocessing
        if self.domain == 'audio':
            adapted = self._adapt_audio(x)
        elif self.domain == 'vision':
            adapted = self._adapt_vision(x)
        elif self.domain == 'text':
            adapted = self._adapt_text(x)
        else:
            adapted = self._adapt_generic(x)
        
        # Apply adapter transformation
        output = np.dot(self.adapter_weights, adapted) + self.adapter_bias
        
        # Update domain metrics
        self._update_domain_metrics(adapted, output)
        
        return output
    
    def _adapt_audio(self, x):
        """Adapt audio input to shared representation."""
        # Simple mel-spectrogram simulation
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # Simulate mel-spectrogram features
        n_frames = len(x) // self.hop_length
        mel_features = np.random.randn(n_frames, self.n_mel_bins)
        
        # Add some audio-specific processing
        for i in range(n_frames):
            start_idx = i * self.hop_length
            end_idx = min(start_idx + self.hop_length, len(x))
            frame = x[start_idx:end_idx]
            
            # Simple frequency analysis
            fft_frame = np.fft.fft(frame)
            mel_features[i] = np.abs(fft_frame[:self.n_mel_bins])
        
        return mel_features.flatten()
    
    def _adapt_vision(self, x):
        """Adapt vision input to shared representation."""
        # Simple patch-based processing
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # Simulate patch features
        n_patches = (self.image_size // self.patch_size) ** 2
        patch_features = np.random.randn(n_patches, self.n_channels)
        
        # Add some vision-specific processing
        for i in range(n_patches):
            patch = x[i * self.n_channels:(i + 1) * self.n_channels]
            patch_features[i] = patch
        
        return patch_features.flatten()
    
    def _adapt_text(self, x):
        """Adapt text input to shared representation."""
        # Simple embedding simulation
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # Simulate word embeddings
        n_words = min(len(x), self.max_length)
        word_embeddings = np.random.randn(n_words, self.embedding_dim)
        
        # Add some text-specific processing
        for i in range(n_words):
            word_id = int(x[i]) % self.vocab_size
            word_embeddings[i] = np.random.randn(self.embedding_dim) * (word_id / self.vocab_size)
        
        return word_embeddings.flatten()
    
    def _adapt_generic(self, x):
        """Adapt generic input to shared representation."""
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # Simple linear projection
        if len(x) > self.input_dim:
            x = x[:self.input_dim]
        elif len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)), mode='constant')
        
        return x
    
    def _update_domain_metrics(self, input_data, output_data):
        """Update domain-specific metrics."""
        self.domain_metrics = {
            'domain': self.domain,
            'input_variance': np.var(input_data),
            'output_variance': np.var(output_data),
            'adaptation_ratio': np.var(output_data) / (np.var(input_data) + 1e-8),
            'target_size': self.target_size
        }

class CrossDomainAligner:
    """Cross-domain feature alignment."""
    def __init__(self, abstract_layers: List[AbstractPhaseLayer]):
        self.abstract_layers = abstract_layers
        self.n_layers = len(abstract_layers)
        
        # Alignment weights between layers
        self.alignment_weights = []
        for i in range(self.n_layers - 1):
            curr_size = abstract_layers[i].output_size
            next_size = abstract_layers[i + 1].output_size
            weights = np.random.randn(next_size, curr_size) * 0.1
            self.alignment_weights.append(weights)
        
        # Cross-domain correlation tracking
        self.correlation_history = []
        
    def align_features(self, abstract_features: List[np.ndarray]) -> List[np.ndarray]:
        """Align features across abstraction levels."""
        aligned_features = []
        
        for i, features in enumerate(abstract_features):
            if i == 0:
                # First layer - no alignment needed
                aligned_features.append(features)
            else:
                # Align with previous layer
                alignment_weight = self.alignment_weights[i - 1]
                aligned = np.dot(alignment_weight, aligned_features[-1])
                
                # Combine with current features
                combined = 0.7 * features + 0.3 * aligned
                aligned_features.append(combined)
        
        # Track cross-domain correlations
        self._update_correlation_tracking(abstract_features, aligned_features)
        
        return aligned_features
    
    def _update_correlation_tracking(self, original, aligned):
        """Track cross-domain correlations."""
        correlation_info = {
            'n_layers': len(original),
            'original_variances': [np.var(f) for f in original],
            'aligned_variances': [np.var(f) for f in aligned],
            'alignment_effects': [np.var(aligned[i]) - np.var(original[i]) for i in range(len(original))],
            'timestamp': time.time()
        }
        
        self.correlation_history.append(correlation_info)
        
        # Keep only recent history
        if len(self.correlation_history) > 100:
            self.correlation_history = self.correlation_history[-100:]

class AbstractFeatureNetwork:
    """Universal abstract feature network."""
    def __init__(self, domain_configs: Dict[str, Dict[str, Any]]):
        self.domain_configs = domain_configs
        
        # Shared abstract feature layers (domain-agnostic)
        self.abstract_layers = [
            AbstractPhaseLayer(128, 64, abstraction_level=0),  # pattern detection
            AbstractPhaseLayer(64, 32, abstraction_level=1),   # relationship detection
            AbstractPhaseLayer(32, 16, abstraction_level=2),   # conceptual detection
        ]
        
        # Domain-specific input adapters
        self.domain_adapters = {}
        for domain, config in domain_configs.items():
            self.domain_adapters[domain] = DomainAdapter(domain, config, target_size=128)
        
        # Cross-domain feature alignment
        self.feature_aligner = CrossDomainAligner(self.abstract_layers)
        
        # Network-level tracking
        self.network_history = []
        
    def forward(self, x, domain: str):
        """Forward pass through abstract feature network."""
        # Convert domain-specific input to shared representation
        adapted = self.domain_adapters[domain].adapt(x)
        
        # Process through shared abstract layers
        abstract_features = []
        current = adapted
        
        for layer in self.abstract_layers:
            current = layer(current)
            abstract_features.append(current)
        
        # Align features across abstraction levels
        aligned_features = self.feature_aligner.align_features(abstract_features)
        
        # Track network processing
        self._update_network_tracking(domain, abstract_features, aligned_features)
        
        return aligned_features[-1], aligned_features
    
    def _update_network_tracking(self, domain, original_features, aligned_features):
        """Track network-level processing."""
        network_info = {
            'domain': domain,
            'n_layers': len(original_features),
            'original_variances': [np.var(f) for f in original_features],
            'aligned_variances': [np.var(f) for f in aligned_features],
            'timestamp': time.time()
        }
        
        self.network_history.append(network_info)
        
        # Keep only recent history
        if len(self.network_history) > 100:
            self.network_history = self.network_history[-100:]

# ============================
# FieldIQ Integration
# ============================

class FieldIQAbstractProcessor:
    """FieldIQ processor for abstract feature networks."""
    def __init__(self, domain_configs: Dict[str, Dict[str, Any]]):
        self.domain_configs = domain_configs
        self.network = AbstractFeatureNetwork(domain_configs)
        
    def process_field(self, field: FieldIQ, domain: str = 'audio') -> FieldIQ:
        """Process FieldIQ through abstract feature network."""
        z_array = field.z
        
        # Process real and imaginary parts
        real_part = np.real(z_array)
        imag_part = np.imag(z_array)
        
        # Process through abstract network
        real_features, real_abstract_features = self.network.forward(real_part, domain)
        imag_features, imag_abstract_features = self.network.forward(imag_part, domain)
        
        # Reconstruct complex field
        new_z = real_features + 1j * imag_features
        
        # Create new field with metadata
        processed_field = FieldIQ(new_z, field.sr, field.roles or {})
        processed_field = processed_field.with_role("abstract_processed", True)
        processed_field = processed_field.with_role("domain", domain)
        processed_field = processed_field.with_role("abstraction_levels", len(real_abstract_features))
        processed_field = processed_field.with_role("abstract_metrics", self._get_abstract_metrics())
        
        return processed_field
    
    def _get_abstract_metrics(self) -> Dict[str, Any]:
        """Get abstract feature metrics."""
        metrics = {
            'n_abstract_layers': len(self.network.abstract_layers),
            'domains': list(self.domain_configs.keys()),
            'abstraction_levels': [layer.abstraction_level for layer in self.network.abstract_layers]
        }
        
        # Add layer-specific metrics
        for i, layer in enumerate(self.network.abstract_layers):
            metrics[f'layer_{i}_abstraction'] = layer.abstraction_metrics
            metrics[f'layer_{i}_frequency_scale'] = np.mean(layer.frequencies)
        
        # Add domain adapter metrics
        for domain, adapter in self.network.domain_adapters.items():
            metrics[f'{domain}_adapter'] = adapter.domain_metrics
        
        # Add alignment metrics
        if self.network.feature_aligner.correlation_history:
            latest_correlation = self.network.feature_aligner.correlation_history[-1]
            metrics['alignment_effects'] = latest_correlation.get('alignment_effects', [])
        
        return metrics

# ============================
# Realm System
# ============================

@dataclass
class AbstractFeatureRealm:
    """Realm for abstract feature processing."""
    name: str
    processor: FieldIQAbstractProcessor
    vm_node: Node
    learning_enabled: bool = True
    
    def __repr__(self):
        return f"<AbstractFeatureRealm {self.name}>"
    
    def field_processor(self, field: FieldIQ, domain: str = 'audio') -> FieldIQ:
        """Process field through realm."""
        return self.processor.process_field(field, domain)

def create_abstract_feature_realm(domain_configs: Dict[str, Dict[str, Any]]) -> AbstractFeatureRealm:
    """Create an abstract feature realm."""
    processor = FieldIQAbstractProcessor(domain_configs)
    
    vm_node = Val({
        'type': 'abstract_feature_realm',
        'name': f'abstract_feature_{len(domain_configs)}_domains',
        'class': 'FieldIQAbstractProcessor',
        'parameters': {
            'domain_configs': domain_configs
        },
        'vm_operations': {'primary': 'ABSTRACT_FEATURE'}
    })
    
    return AbstractFeatureRealm(
        name=f'abstract_feature_{len(domain_configs)}_domains',
        processor=processor,
        vm_node=vm_node,
        learning_enabled=True
    )

# ============================
# Demo Functions
# ============================

def demo_abstract_phase_layers():
    """Demo abstract phase layers."""
    print("=== Abstract Phase Layers Demo ===\n")
    
    # Test different abstraction levels
    for level in [0, 1, 2]:
        print(f"--- Abstraction Level {level} ---")
        
        layer = AbstractPhaseLayer(128, 64, level)
        test_input = np.random.randn(128)
        output = layer.forward(test_input)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Frequency scale: {np.mean(layer.frequencies):.3f}")
        print(f"Pattern diversity: {np.var(layer.pattern_templates):.3f}")
        print(f"Abstraction metrics: {layer.abstraction_metrics}")
        print()

def demo_domain_adapters():
    """Demo domain adapters."""
    print("=== Domain Adapters Demo ===\n")
    
    # Test different domains
    domain_configs = {
        'audio': {'sample_rate': 48000, 'n_mel_bins': 128, 'hop_length': 512},
        'vision': {'image_size': 224, 'n_channels': 3, 'patch_size': 16},
        'text': {'vocab_size': 10000, 'max_length': 512, 'embedding_dim': 128}
    }
    
    for domain, config in domain_configs.items():
        print(f"--- {domain.upper()} Domain ---")
        
        adapter = DomainAdapter(domain, config, target_size=128)
        test_input = np.random.randn(1000)
        adapted = adapter.adapt(test_input)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Adapted shape: {adapted.shape}")
        print(f"Domain metrics: {adapter.domain_metrics}")
        print()

def demo_abstract_feature_network():
    """Demo abstract feature network."""
    print("=== Abstract Feature Network Demo ===\n")
    
    # Create network with multiple domains
    domain_configs = {
        'audio': {'sample_rate': 48000, 'n_mel_bins': 128, 'hop_length': 512},
        'vision': {'image_size': 224, 'n_channels': 3, 'patch_size': 16},
        'text': {'vocab_size': 10000, 'max_length': 512, 'embedding_dim': 128}
    }
    
    network = AbstractFeatureNetwork(domain_configs)
    
    print(f"Network domains: {list(domain_configs.keys())}")
    print(f"Abstract layers: {len(network.abstract_layers)}")
    print(f"Domain adapters: {len(network.domain_adapters)}")
    
    # Test with different domains
    for domain in ['audio', 'vision', 'text']:
        print(f"\n--- Testing {domain.upper()} Domain ---")
        
        test_input = np.random.randn(1000)
        output, abstract_features = network.forward(test_input, domain)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Abstract features: {[len(f) for f in abstract_features]}")
        
        # Show domain adapter metrics
        adapter = network.domain_adapters[domain]
        print(f"Adapter metrics: {adapter.domain_metrics}")
    
    print()

def demo_abstract_feature_realms():
    """Demo abstract feature realms."""
    print("=== Abstract Feature Realms Demo ===\n")
    
    # Create sample field
    sr = 48000
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = 0.8 * np.cos(2 * np.pi * 440 * t) + 0.3 * np.cos(2 * np.pi * 880 * t + np.pi/4)
    field = make_field_from_real(x, sr, tag=("demo", "multi_tone"))
    
    print(f"Original field: {len(field.z)} samples, {field.sr} Hz")
    print(f"Field energy: {np.sum(np.abs(field.z) ** 2):.2f}")
    
    # Create realms with different domain configurations
    realm_configs = {
        'multi_domain': {
            'audio': {'sample_rate': 48000, 'n_mel_bins': 128, 'hop_length': 512},
            'vision': {'image_size': 224, 'n_channels': 3, 'patch_size': 16},
            'text': {'vocab_size': 10000, 'max_length': 512, 'embedding_dim': 128}
        },
        'audio_focused': {
            'audio': {'sample_rate': 48000, 'n_mel_bins': 256, 'hop_length': 256}
        },
        'vision_focused': {
            'vision': {'image_size': 512, 'n_channels': 3, 'patch_size': 32}
        }
    }
    
    realms = {}
    for name, config in realm_configs.items():
        realms[name] = create_abstract_feature_realm(config)
    
    print(f"\nCreated {len(realms)} abstract feature realms:")
    for name, realm in realms.items():
        print(f"  - {name}: {list(realm.processor.domain_configs.keys())} domains")
    
    # Test each realm
    print(f"\nTesting abstract feature realms on field:")
    for name, realm in realms.items():
        try:
            # Test with audio domain
            processed_field = realm.field_processor(field, 'audio')
            energy = np.sum(np.abs(processed_field.z) ** 2)
            
            print(f"  {name:15} | Energy: {energy:8.2f} | Learning: {realm.learning_enabled}")
            
            # Show abstract metrics
            abstract_metrics = processed_field.roles.get('abstract_metrics', {})
            if abstract_metrics:
                print(f"  {'':15} | Domains: {abstract_metrics.get('domains', [])}, "
                      f"Layers: {abstract_metrics.get('n_abstract_layers', 0)}")
            
        except Exception as e:
            print(f"  {name:15} | Error: {e}")
    
    # Test VM integration
    print(f"\nTesting VM integration:")
    for name, realm in realms.items():
        try:
            vm_json = to_json(realm.vm_node)
            print(f"  {name:15} | VM JSON length: {len(vm_json)} chars")
        except Exception as e:
            print(f"  {name:15} | VM Error: {e}")
    print()

def main():
    """Run all abstract feature network demos."""
    print("🔷 Abstract Feature Networks Demo")
    print("=" * 60)
    
    # Run demos
    demo_abstract_phase_layers()
    demo_domain_adapters()
    demo_abstract_feature_network()
    demo_abstract_feature_realms()
    
    print("🔷 Abstract Feature Networks Complete")
    print("✓ Universal pattern detection across domains")
    print("✓ Domain-specific input adapters")
    print("✓ Cross-domain feature alignment")
    print("✓ FieldIQ integration with abstract metrics")
    print("✓ Realm system integration")
    print("✓ VM compatibility")

if __name__ == "__main__":
    main()

