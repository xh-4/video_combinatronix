"""
Lattice Structure Example

This example demonstrates the lattice structure of channel states:
1. Partial order relationships
2. Lattice operations (meet, join)
3. Chain and antichain analysis
4. Lattice properties and visualization
5. Real-world applications
"""

import numpy as np
import matplotlib.pyplot as plt
from channelpy import (
    State, StateArray, EMPTY, DELTA, PHI, PSI,
    partial_order, are_comparable, meet, join, lattice_distance,
    complement, is_atom, is_coatom, ChannelLattice, get_lattice,
    lattice_operations
)
from channelpy.examples.datasets import make_classification_data


def demonstrate_basic_lattice_operations():
    """Demonstrate basic lattice operations"""
    print("=== Basic Lattice Operations ===")
    
    # Test partial order
    print("1. Partial Order Relationships:")
    print(f"   EMPTY ≤ DELTA: {partial_order(EMPTY, DELTA)}")
    print(f"   DELTA ≤ EMPTY: {partial_order(DELTA, EMPTY)}")
    print(f"   EMPTY ≤ PSI: {partial_order(EMPTY, PSI)}")
    print(f"   DELTA ≤ PHI: {partial_order(DELTA, PHI)}")
    
    # Test comparability
    print("\n2. Comparability:")
    print(f"   EMPTY and PSI comparable: {are_comparable(EMPTY, PSI)}")
    print(f"   DELTA and PHI comparable: {are_comparable(DELTA, PHI)}")
    
    # Test meet and join
    print("\n3. Meet and Join Operations:")
    print(f"   meet(DELTA, PHI) = {meet(DELTA, PHI)}")
    print(f"   join(DELTA, PHI) = {join(DELTA, PHI)}")
    print(f"   meet(PSI, DELTA) = {meet(PSI, DELTA)}")
    print(f"   join(EMPTY, PSI) = {join(EMPTY, PSI)}")
    
    # Test lattice distance
    print("\n4. Lattice Distance:")
    print(f"   distance(EMPTY, PSI) = {lattice_distance(EMPTY, PSI)}")
    print(f"   distance(DELTA, PHI) = {lattice_distance(DELTA, PHI)}")
    print(f"   distance(EMPTY, DELTA) = {lattice_distance(EMPTY, DELTA)}")


def demonstrate_lattice_structure():
    """Demonstrate the complete lattice structure"""
    print("\n=== Lattice Structure ===")
    
    lattice = get_lattice()
    
    # Basic properties
    print("1. Basic Properties:")
    print(f"   Bottom element: {lattice.bottom}")
    print(f"   Top element: {lattice.top}")
    print(f"   All elements: {lattice.elements}")
    
    # Successors and predecessors
    print("\n2. Successors and Predecessors:")
    for state in lattice.elements:
        successors = lattice.get_successors(state)
        predecessors = lattice.get_predecessors(state)
        print(f"   {state}: successors={successors}, predecessors={predecessors}")
    
    # Atoms and coatoms
    print("\n3. Atoms and Coatoms:")
    print(f"   Atoms: {lattice.get_atoms()}")
    print(f"   Coatoms: {lattice.get_coatoms()}")
    
    # Heights and depths
    print("\n4. Heights and Depths:")
    for state in lattice.elements:
        height = lattice.get_height(state)
        depth = lattice.get_depth(state)
        level = lattice.get_level(state)
        print(f"   {state}: height={height}, depth={depth}, level={level}")


def demonstrate_chain_analysis():
    """Demonstrate chain analysis in the lattice"""
    print("\n=== Chain Analysis ===")
    
    lattice = get_lattice()
    
    # Find all chains from EMPTY to PSI
    print("1. All chains from EMPTY to PSI:")
    chains = lattice.compute_chain(EMPTY, PSI)
    for i, chain in enumerate(chains):
        print(f"   Chain {i+1}: {' → '.join(str(s) for s in chain)}")
    
    # Shortest and longest chains
    print("\n2. Shortest and Longest Chains:")
    shortest = lattice.get_shortest_chain(EMPTY, PSI)
    longest = lattice.get_longest_chain(EMPTY, PSI)
    print(f"   Shortest: {' → '.join(str(s) for s in shortest)}")
    print(f"   Longest: {' → '.join(str(s) for s in longest)}")
    
    # Test chain property
    print("\n3. Chain Property Tests:")
    test_chains = [
        [EMPTY, DELTA, PSI],
        [EMPTY, PHI, PSI],
        [DELTA, PHI],  # Not a chain
        [EMPTY, PSI]   # Not a chain (skips intermediate)
    ]
    
    for chain in test_chains:
        is_chain = lattice.is_chain(chain)
        print(f"   {chain} is chain: {is_chain}")


def demonstrate_antichain_analysis():
    """Demonstrate antichain analysis"""
    print("\n=== Antichain Analysis ===")
    
    lattice = get_lattice()
    
    # Test different sets for antichains
    test_sets = [
        [EMPTY, PSI],      # Antichain
        [DELTA, PHI],      # Antichain
        [EMPTY, DELTA],    # Not antichain
        [DELTA, PSI],      # Not antichain
        [EMPTY, DELTA, PHI, PSI]  # Not antichain
    ]
    
    print("1. Antichain Tests:")
    for states in test_sets:
        antichain = lattice.get_antichain(states)
        print(f"   {states} → antichain: {antichain}")
    
    # Find maximal antichain
    print("\n2. Maximal Antichain:")
    all_states = lattice.elements
    maximal_antichain = lattice.get_antichain(all_states)
    print(f"   Maximal antichain: {maximal_antichain}")


def demonstrate_lattice_properties():
    """Demonstrate lattice properties"""
    print("\n=== Lattice Properties ===")
    
    lattice = get_lattice()
    properties = lattice.get_lattice_properties()
    
    print("1. Lattice Properties:")
    for prop, value in properties.items():
        print(f"   {prop}: {value}")
    
    # Test distributive property
    print("\n2. Distributive Property Test:")
    x, y, z = DELTA, PHI, PSI
    left = meet(x, join(y, z))
    right = join(meet(x, y), meet(x, z))
    print(f"   x ∧ (y ∨ z) = {left}")
    print(f"   (x ∧ y) ∨ (x ∧ z) = {right}")
    print(f"   Distributive: {left == right}")
    
    # Test complement property
    print("\n3. Complement Property Test:")
    for state in lattice.elements:
        comp_state = complement(state)
        print(f"   complement({state}) = {comp_state}")
        print(f"   meet({state}, {comp_state}) = {meet(state, comp_state)}")
        print(f"   join({state}, {comp_state}) = {join(state, comp_state)}")


def demonstrate_lattice_visualization():
    """Demonstrate lattice visualization"""
    print("\n=== Lattice Visualization ===")
    
    lattice = get_lattice()
    
    # Text visualization
    print("1. Text Visualization:")
    print(lattice.visualize())
    
    # Cover relations
    print("\n2. Cover Relations:")
    covers = lattice.get_cover_relations()
    for start, end in covers:
        print(f"   {start} → {end}")


def demonstrate_lattice_operations_on_sets():
    """Demonstrate lattice operations on sets of states"""
    print("\n=== Lattice Operations on Sets ===")
    
    # Test different sets of states
    test_sets = [
        [EMPTY, DELTA],
        [DELTA, PHI],
        [EMPTY, DELTA, PHI, PSI],
        [DELTA, PSI],
        [PHI, PSI]
    ]
    
    for states in test_sets:
        print(f"\nStates: {states}")
        operations = lattice_operations(states)
        
        print(f"   Meet: {operations['meet']}")
        print(f"   Join: {operations['join']}")
        print(f"   Antichain: {operations['antichain']}")
        print(f"   Is chain: {operations['is_chain']}")
        print(f"   Atoms: {operations['atoms']}")
        print(f"   Coatoms: {operations['coatoms']}")


def demonstrate_real_world_application():
    """Demonstrate real-world application of lattice operations"""
    print("\n=== Real-World Application ===")
    
    # Generate some data
    X, y = make_classification_data(n_samples=100, n_features=2, n_classes=2)
    
    # Create states based on data
    print("1. Creating states from data:")
    states = []
    for i in range(min(20, len(X))):
        # Simple encoding: use first feature
        feature = X[i, 0]
        if feature > 0.5:
            i_bit = 1
        else:
            i_bit = 0
        
        if feature > 0.8:
            q_bit = 1
        else:
            q_bit = 0
        
        state = State(i_bit, q_bit)
        states.append(state)
        print(f"   Sample {i}: feature={feature:.2f} → {state}")
    
    # Analyze lattice properties
    print("\n2. Lattice Analysis:")
    operations = lattice_operations(states)
    print(f"   Meet of all states: {operations['meet']}")
    print(f"   Join of all states: {operations['join']}")
    print(f"   Antichain: {operations['antichain']}")
    print(f"   Is chain: {operations['is_chain']}")
    
    # Find intervals
    print("\n3. Interval Analysis:")
    lattice = get_lattice()
    
    # Find interval between meet and join
    meet_state = operations['meet']
    join_state = operations['join']
    interval = lattice.get_interval(meet_state, join_state)
    print(f"   Interval [{meet_state}, {join_state}]: {interval}")
    
    # Count states at each level
    print("\n4. Level Distribution:")
    for level in range(3):  # 0, 1, 2
        level_states = lattice.get_states_at_level(level)
        count = sum(1 for s in states if lattice.get_level(s) == level)
        print(f"   Level {level}: {count} states {level_states}")


def create_lattice_visualization():
    """Create a visual representation of the lattice"""
    print("\n=== Creating Lattice Visualization ===")
    
    # Create a simple plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Define positions for each state
    positions = {
        EMPTY: (0, 0),
        DELTA: (-1, 1),
        PHI: (1, 1),
        PSI: (0, 2)
    }
    
    # Plot states
    for state, (x, y) in positions.items():
        ax.scatter(x, y, s=200, c='lightblue', edgecolors='black', linewidth=2)
        ax.text(x, y, str(state), ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Plot edges (cover relations)
    lattice = get_lattice()
    covers = lattice.get_cover_relations()
    
    for start, end in covers:
        start_pos = positions[start]
        end_pos = positions[end]
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                'k-', alpha=0.5, linewidth=1)
    
    # Add arrows to show direction
    for start, end in covers:
        start_pos = positions[start]
        end_pos = positions[end]
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        ax.arrow(start_pos[0] + dx*0.3, start_pos[1] + dy*0.3, 
                dx*0.4, dy*0.4, head_width=0.1, head_length=0.1, 
                fc='black', ec='black', alpha=0.7)
    
    ax.set_title('Channel Lattice Structure', fontsize=16, fontweight='bold')
    ax.set_xlabel('Lattice Position', fontsize=12)
    ax.set_ylabel('Level', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 2.5)
    
    # Add level labels
    for level in range(3):
        ax.text(-1.3, level, f'Level {level}', ha='right', va='center', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('lattice_structure.png', dpi=150, bbox_inches='tight')
    print("Saved: lattice_structure.png")


def demonstrate_advanced_lattice_operations():
    """Demonstrate advanced lattice operations"""
    print("\n=== Advanced Lattice Operations ===")
    
    lattice = get_lattice()
    
    # Test interval operations
    print("1. Interval Operations:")
    intervals = [
        (EMPTY, PSI),
        (DELTA, PSI),
        (EMPTY, DELTA),
        (PHI, PSI)
    ]
    
    for start, end in intervals:
        interval = lattice.get_interval(start, end)
        print(f"   [{start}, {end}]: {interval}")
    
    # Test dual lattice
    print("\n2. Dual Lattice:")
    dual = lattice.get_dual()
    print(f"   Dual lattice elements: {dual.elements}")
    print(f"   Dual bottom: {dual.bottom}")
    print(f"   Dual top: {dual.top}")
    
    # Test complement operations
    print("\n3. Complement Operations:")
    for state in lattice.elements:
        comp_state = complement(state)
        print(f"   complement({state}) = {comp_state}")
        print(f"   meet({state}, {comp_state}) = {meet(state, comp_state)}")
        print(f"   join({state}, {comp_state}) = {join(state, comp_state)}")


def main():
    """Main lattice example function"""
    print("ChannelPy Lattice Structure Example")
    print("=" * 50)
    
    # 1. Basic lattice operations
    demonstrate_basic_lattice_operations()
    
    # 2. Lattice structure
    demonstrate_lattice_structure()
    
    # 3. Chain analysis
    demonstrate_chain_analysis()
    
    # 4. Antichain analysis
    demonstrate_antichain_analysis()
    
    # 5. Lattice properties
    demonstrate_lattice_properties()
    
    # 6. Lattice visualization
    demonstrate_lattice_visualization()
    
    # 7. Lattice operations on sets
    demonstrate_lattice_operations_on_sets()
    
    # 8. Real-world application
    demonstrate_real_world_application()
    
    # 9. Create visualization
    create_lattice_visualization()
    
    # 10. Advanced operations
    demonstrate_advanced_lattice_operations()
    
    print("\n" + "=" * 50)
    print("Lattice example completed successfully!")
    print("\nGenerated files:")
    print("- lattice_structure.png: Visual representation of the lattice")
    print("\nNext steps:")
    print("- Experiment with different lattice operations")
    print("- Use lattice analysis for state sequence analysis")
    print("- Apply lattice theory to your specific domain")
    print("- Explore advanced lattice properties and applications")


if __name__ == "__main__":
    main()







