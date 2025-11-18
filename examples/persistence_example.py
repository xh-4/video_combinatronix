"""
Persistent Homology Example

This example demonstrates topological data analysis using persistent homology:
1. Basic persistence diagram computation
2. Betti number analysis
3. Persistence entropy and statistics
4. Visualization and comparison
5. Real-world applications with channel states
"""

import numpy as np
import matplotlib.pyplot as plt
from channelpy.topology.persistence import (
    PersistenceDiagram, compute_betti_numbers, compute_persistence_entropy,
    compute_total_persistence, compare_persistence, plot_comparison
)
from channelpy.examples.datasets import make_classification_data, make_trading_data
from channelpy import StateArray, EMPTY, DELTA, PHI, PSI


def generate_test_data():
    """Generate test data for persistent homology"""
    print("=== Generating Test Data ===")
    
    # Generate different types of data
    datasets = {}
    
    # 1. Circle data
    t = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(t)
    circle_y = np.sin(t)
    datasets['circle'] = np.column_stack([circle_x, circle_y])
    
    # 2. Torus data
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, 2*np.pi, 50)
    U, V = np.meshgrid(u, v)
    torus_x = (2 + np.cos(V)) * np.cos(U)
    torus_y = (2 + np.cos(V)) * np.sin(U)
    torus_z = np.sin(V)
    datasets['torus'] = np.column_stack([torus_x.flatten(), torus_y.flatten(), torus_z.flatten()])
    
    # 3. Sphere data
    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    PHI, THETA = np.meshgrid(phi, theta)
    sphere_x = np.sin(PHI) * np.cos(THETA)
    sphere_y = np.sin(PHI) * np.sin(THETA)
    sphere_z = np.cos(PHI)
    datasets['sphere'] = np.column_stack([sphere_x.flatten(), sphere_y.flatten(), sphere_z.flatten()])
    
    # 4. Random data
    datasets['random'] = np.random.randn(100, 3)
    
    # 5. Classification data
    X, y = make_classification_data(n_samples=200, n_features=3, n_classes=2)
    datasets['classification'] = X
    
    # 6. Trading data
    df = make_trading_data(n_samples=100)
    datasets['trading'] = df[['close', 'volume']].values
    
    print("Generated datasets:")
    for name, data in datasets.items():
        print(f"  {name}: {data.shape}")
    
    return datasets


def demonstrate_basic_persistence(datasets):
    """Demonstrate basic persistence diagram computation"""
    print("\n=== Basic Persistence Analysis ===")
    
    # Test with circle data
    circle_data = datasets['circle']
    print(f"Analyzing circle data: {circle_data.shape}")
    
    # Create persistence diagram
    diagram = PersistenceDiagram()
    diagram.compute(circle_data, max_dim=1)
    
    if diagram.is_computed:
        print("Persistence diagram computed successfully!")
        
        # Get Betti numbers
        betti = diagram.get_betti_numbers()
        print(f"Betti numbers: {betti}")
        
        # Get persistence entropy
        entropy = diagram.get_persistence_entropy()
        print(f"Persistence entropy: {entropy}")
        
        # Get total persistence
        total_pers = diagram.get_total_persistence()
        print(f"Total persistence: {total_pers}")
        
        # Get lifespan statistics
        stats = diagram.get_lifespan_statistics()
        print(f"Lifespan statistics: {stats}")
        
        # Get summary
        summary = diagram.get_summary()
        print(f"Summary: {summary}")
    else:
        print("Failed to compute persistence diagram")


def demonstrate_visualization(datasets):
    """Demonstrate persistence diagram visualization"""
    print("\n=== Persistence Visualization ===")
    
    # Test with different datasets
    test_datasets = ['circle', 'torus', 'sphere', 'random']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, dataset_name in enumerate(test_datasets):
        data = datasets[dataset_name]
        print(f"Computing persistence for {dataset_name}...")
        
        diagram = PersistenceDiagram()
        diagram.compute(data, max_dim=1)
        
        if diagram.is_computed:
            # Plot persistence diagram
            diagram.plot(title=f'{dataset_name.title()} Persistence', ax=axes[i])
        else:
            axes[i].text(0.5, 0.5, f'{dataset_name}\n(No data)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{dataset_name.title()} Persistence')
    
    plt.tight_layout()
    plt.savefig('persistence_diagrams.png', dpi=150, bbox_inches='tight')
    print("Saved: persistence_diagrams.png")


def demonstrate_betti_curves(datasets):
    """Demonstrate Betti number curves"""
    print("\n=== Betti Number Curves ===")
    
    # Test with circle data
    circle_data = datasets['circle']
    diagram = PersistenceDiagram()
    diagram.compute(circle_data, max_dim=1)
    
    if diagram.is_computed:
        # Plot Betti curves
        fig, ax = diagram.plot_betti_curves()
        plt.savefig('betti_curves.png', dpi=150, bbox_inches='tight')
        print("Saved: betti_curves.png")
        
        # Test with different epsilon values
        epsilons = np.linspace(0, 2, 50)
        betti_curves = {}
        
        for eps in epsilons:
            betti = diagram.get_betti_numbers(eps)
            for dim, value in betti.items():
                if dim not in betti_curves:
                    betti_curves[dim] = []
                betti_curves[dim].append(value)
        
        print("Betti number evolution:")
        for dim, curve in betti_curves.items():
            print(f"  Dimension {dim}: min={min(curve)}, max={max(curve)}")


def demonstrate_comparison(datasets):
    """Demonstrate persistence diagram comparison"""
    print("\n=== Persistence Comparison ===")
    
    # Compare different datasets
    test_datasets = ['circle', 'torus', 'sphere']
    diagrams = []
    labels = []
    
    for name in test_datasets:
        data = datasets[name]
        diagram = PersistenceDiagram()
        diagram.compute(data, max_dim=1)
        
        if diagram.is_computed:
            diagrams.append(diagram)
            labels.append(name)
    
    if diagrams:
        # Compare persistence diagrams
        comparison = compare_persistence(diagrams, labels)
        print("Comparison results:")
        for label in labels:
            print(f"  {label}:")
            print(f"    Betti numbers: {comparison['betti_numbers'][label]}")
            print(f"    Persistence entropy: {comparison['persistence_entropy'][label]}")
            print(f"    Total persistence: {comparison['total_persistence'][label]}")
        
        # Plot comparison
        fig, axes = plot_comparison(diagrams, labels)
        plt.savefig('persistence_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: persistence_comparison.png")


def demonstrate_channel_state_analysis(datasets):
    """Demonstrate persistent homology with channel states"""
    print("\n=== Channel State Analysis ===")
    
    # Generate channel states from data
    data = datasets['classification']
    states = []
    
    # Simple encoding: use first two features
    for i in range(min(100, len(data))):
        feature1 = data[i, 0]
        feature2 = data[i, 1]
        
        # Encode to states
        if feature1 > 0:
            i_bit = 1
        else:
            i_bit = 0
        
        if feature2 > 0:
            q_bit = 1
        else:
            q_bit = 0
        
        state = StateArray.from_bits(i=[i_bit], q=[q_bit])[0]
        states.append(state)
    
    print(f"Generated {len(states)} channel states")
    
    # Convert states to coordinates for persistence analysis
    state_coords = []
    for state in states:
        # Map states to 2D coordinates
        if state == EMPTY:
            coords = [0, 0]
        elif state == DELTA:
            coords = [1, 0]
        elif state == PHI:
            coords = [0, 1]
        elif state == PSI:
            coords = [1, 1]
        else:
            coords = [0.5, 0.5]  # Unknown state
        
        state_coords.append(coords)
    
    state_coords = np.array(state_coords)
    print(f"State coordinates shape: {state_coords.shape}")
    
    # Compute persistence for state coordinates
    diagram = PersistenceDiagram()
    diagram.compute(state_coords, max_dim=1)
    
    if diagram.is_computed:
        print("Channel state persistence computed!")
        
        # Get Betti numbers
        betti = diagram.get_betti_numbers()
        print(f"Betti numbers: {betti}")
        
        # Get persistence entropy
        entropy = diagram.get_persistence_entropy()
        print(f"Persistence entropy: {entropy}")
        
        # Plot persistence diagram
        fig, ax = diagram.plot(title='Channel State Persistence')
        plt.savefig('channel_state_persistence.png', dpi=150, bbox_inches='tight')
        print("Saved: channel_state_persistence.png")
        
        # Plot Betti curves
        fig, ax = diagram.plot_betti_curves()
        plt.savefig('channel_state_betti_curves.png', dpi=150, bbox_inches='tight')
        print("Saved: channel_state_betti_curves.png")


def demonstrate_real_world_application(datasets):
    """Demonstrate real-world application with trading data"""
    print("\n=== Real-World Application: Trading Data ===")
    
    # Use trading data
    trading_data = datasets['trading']
    print(f"Trading data shape: {trading_data.shape}")
    
    # Compute persistence
    diagram = PersistenceDiagram()
    diagram.compute(trading_data, max_dim=1)
    
    if diagram.is_computed:
        print("Trading data persistence computed!")
        
        # Get comprehensive analysis
        summary = diagram.get_summary()
        print("Trading data persistence summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Plot persistence diagram
        fig, ax = diagram.plot(title='Trading Data Persistence')
        plt.savefig('trading_persistence.png', dpi=150, bbox_inches='tight')
        print("Saved: trading_persistence.png")
        
        # Plot Betti curves
        fig, ax = diagram.plot_betti_curves()
        plt.savefig('trading_betti_curves.png', dpi=150, bbox_inches='tight')
        print("Saved: trading_betti_curves.png")
        
        # Analyze at different scales
        scales = [0.1, 0.5, 1.0, 2.0]
        print("\nBetti numbers at different scales:")
        for scale in scales:
            betti = diagram.get_betti_numbers(scale)
            print(f"  Scale {scale}: {betti}")


def demonstrate_advanced_analysis(datasets):
    """Demonstrate advanced persistent homology analysis"""
    print("\n=== Advanced Analysis ===")
    
    # Test with torus data (higher dimensional)
    torus_data = datasets['torus']
    print(f"Analyzing torus data: {torus_data.shape}")
    
    diagram = PersistenceDiagram()
    diagram.compute(torus_data, max_dim=2)  # Higher dimension
    
    if diagram.is_computed:
        print("Torus persistence computed!")
        
        # Get Betti numbers at different scales
        scales = np.linspace(0, 3, 20)
        betti_evolution = {0: [], 1: [], 2: []}
        
        for scale in scales:
            betti = diagram.get_betti_numbers(scale)
            for dim in [0, 1, 2]:
                betti_evolution[dim].append(betti.get(dim, 0))
        
        # Plot Betti evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        for dim, curve in betti_evolution.items():
            ax.plot(scales, curve, label=f'β{dim}', linewidth=2)
        
        ax.set_xlabel('Scale (Epsilon)', fontsize=12)
        ax.set_ylabel('Betti Number', fontsize=12)
        ax.set_title('Betti Number Evolution (Torus)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig('torus_betti_evolution.png', dpi=150, bbox_inches='tight')
        print("Saved: torus_betti_evolution.png")
        
        # Analyze persistence features
        print("\nPersistence feature analysis:")
        for dim, dgm in diagram.diagrams.items():
            if len(dgm) > 0:
                print(f"  Dimension {dim}: {len(dgm)} features")
                
                # Filter finite features
                finite_features = [(b, d) for b, d in dgm if d < np.inf]
                if finite_features:
                    lifespans = [d - b for b, d in finite_features]
                    print(f"    Lifespan stats: min={min(lifespans):.3f}, "
                          f"max={max(lifespans):.3f}, mean={np.mean(lifespans):.3f}")


def demonstrate_quick_functions(datasets):
    """Demonstrate quick persistence functions"""
    print("\n=== Quick Functions ===")
    
    # Test quick functions
    circle_data = datasets['circle']
    
    print("1. Quick Betti numbers:")
    betti = compute_betti_numbers(circle_data, max_dim=1, epsilon=0.5)
    print(f"   Betti numbers: {betti}")
    
    print("\n2. Quick persistence entropy:")
    entropy = compute_persistence_entropy(circle_data, max_dim=1)
    print(f"   Persistence entropy: {entropy}")
    
    print("\n3. Quick total persistence:")
    total_pers = compute_total_persistence(circle_data, max_dim=1)
    print(f"   Total persistence: {total_pers}")
    
    # Test with different epsilon values
    print("\n4. Betti numbers at different scales:")
    epsilons = [0.1, 0.5, 1.0, 2.0]
    for eps in epsilons:
        betti = compute_betti_numbers(circle_data, max_dim=1, epsilon=eps)
        print(f"   Epsilon {eps}: {betti}")


def main():
    """Main persistence example function"""
    print("ChannelPy Persistent Homology Example")
    print("=" * 50)
    
    # 1. Generate test data
    datasets = generate_test_data()
    
    # 2. Basic persistence analysis
    demonstrate_basic_persistence(datasets)
    
    # 3. Visualization
    demonstrate_visualization(datasets)
    
    # 4. Betti curves
    demonstrate_betti_curves(datasets)
    
    # 5. Comparison
    demonstrate_comparison(datasets)
    
    # 6. Channel state analysis
    demonstrate_channel_state_analysis(datasets)
    
    # 7. Real-world application
    demonstrate_real_world_application(datasets)
    
    # 8. Advanced analysis
    demonstrate_advanced_analysis(datasets)
    
    # 9. Quick functions
    demonstrate_quick_functions(datasets)
    
    print("\n" + "=" * 50)
    print("Persistent homology example completed successfully!")
    print("\nGenerated files:")
    print("- persistence_diagrams.png: Persistence diagrams for different datasets")
    print("- betti_curves.png: Betti number curves")
    print("- persistence_comparison.png: Comparison of persistence diagrams")
    print("- channel_state_persistence.png: Channel state persistence analysis")
    print("- channel_state_betti_curves.png: Channel state Betti curves")
    print("- trading_persistence.png: Trading data persistence analysis")
    print("- trading_betti_curves.png: Trading data Betti curves")
    print("- torus_betti_evolution.png: Betti number evolution for torus data")
    print("\nNext steps:")
    print("- Experiment with different datasets and parameters")
    print("- Use persistent homology for feature extraction")
    print("- Apply topological analysis to your specific domain")
    print("- Explore advanced topological data analysis techniques")


if __name__ == "__main__":
    main()







