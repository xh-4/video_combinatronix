"""
Persistent homology for channel state data

Provides topological data analysis capabilities for channel states
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt


class PersistenceDiagram:
    """
    Persistence diagram computation and analysis
    
    Computes persistent homology features from channel state data
    and provides visualization and analysis capabilities.
    
    Examples
    --------
    >>> diagram = PersistenceDiagram()
    >>> diagram.compute(data)
    >>> diagram.plot()
    >>> betti = diagram.get_betti_numbers()
    """
    
    def __init__(self):
        self.diagrams = {}  # dimension -> [(birth, death)]
        self.betti_numbers = {}
        self.is_computed = False
    
    def compute(self, data: np.ndarray, max_dim: int = 1, 
                metric: str = 'euclidean', **kwargs):
        """
        Compute persistence diagram
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_samples, n_features)
        max_dim : int
            Maximum homology dimension to compute
        metric : str
            Distance metric for ripser
        **kwargs
            Additional arguments for ripser
            
        Returns
        -------
        self : PersistenceDiagram
        """
        try:
            from ripser import ripser
            
            # Reshape if needed
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Compute persistence
            result = ripser(data, maxdim=max_dim, metric=metric, **kwargs)
            
            # Store diagrams
            for dim in range(max_dim + 1):
                self.diagrams[dim] = result['dgms'][dim]
            
            # Compute Betti numbers
            self._compute_betti_numbers()
            self.is_computed = True
            
        except ImportError:
            print("ripser not installed. Install with: pip install ripser")
            self.is_computed = False
        except Exception as e:
            print(f"Error computing persistence: {e}")
            self.is_computed = False
        
        return self
    
    def _compute_betti_numbers(self, epsilon: float = 0.1):
        """Compute Betti numbers at epsilon"""
        self.betti_numbers = {}
        for dim, dgm in self.diagrams.items():
            # Count features alive at epsilon
            alive = sum(1 for birth, death in dgm 
                       if birth <= epsilon < death)
            self.betti_numbers[dim] = alive
    
    def get_betti_numbers(self, epsilon: float = 0.1) -> Dict[int, int]:
        """
        Get Betti numbers at specific epsilon
        
        Parameters
        ----------
        epsilon : float
            Scale parameter
            
        Returns
        -------
        betti : dict
            Betti numbers by dimension
        """
        if not self.is_computed:
            return {}
        
        betti = {}
        for dim, dgm in self.diagrams.items():
            alive = sum(1 for birth, death in dgm 
                       if birth <= epsilon < death)
            betti[dim] = alive
        
        return betti
    
    def get_persistence_entropy(self) -> Dict[int, float]:
        """
        Compute persistence entropy for each dimension
        
        Returns
        -------
        entropy : dict
            Persistence entropy by dimension
        """
        if not self.is_computed:
            return {}
        
        entropy = {}
        for dim, dgm in self.diagrams.items():
            # Filter out infinite death times
            dgm_finite = [(b, d) for b, d in dgm if d < np.inf]
            
            if not dgm_finite:
                entropy[dim] = 0.0
                continue
            
            # Compute persistence lengths
            lengths = [d - b for b, d in dgm_finite]
            total_length = sum(lengths)
            
            if total_length == 0:
                entropy[dim] = 0.0
                continue
            
            # Compute entropy
            probs = [l / total_length for l in lengths]
            entropy[dim] = -sum(p * np.log(p) for p in probs if p > 0)
        
        return entropy
    
    def get_total_persistence(self) -> Dict[int, float]:
        """
        Compute total persistence for each dimension
        
        Returns
        -------
        total_persistence : dict
            Total persistence by dimension
        """
        if not self.is_computed:
            return {}
        
        total_persistence = {}
        for dim, dgm in self.diagrams.items():
            # Filter out infinite death times
            dgm_finite = [(b, d) for b, d in dgm if d < np.inf]
            total_persistence[dim] = sum(d - b for b, d in dgm_finite)
        
        return total_persistence
    
    def get_lifespan_statistics(self) -> Dict[int, Dict[str, float]]:
        """
        Compute lifespan statistics for each dimension
        
        Returns
        -------
        stats : dict
            Statistics by dimension
        """
        if not self.is_computed:
            return {}
        
        stats = {}
        for dim, dgm in self.diagrams.items():
            # Filter out infinite death times
            dgm_finite = [(b, d) for b, d in dgm if d < np.inf]
            
            if not dgm_finite:
                stats[dim] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
                continue
            
            lifespans = [d - b for b, d in dgm_finite]
            stats[dim] = {
                'mean': np.mean(lifespans),
                'std': np.std(lifespans),
                'min': np.min(lifespans),
                'max': np.max(lifespans)
            }
        
        return stats
    
    def plot(self, figsize: Tuple[int, int] = (10, 6), 
             title: str = 'Persistence Diagram', **kwargs):
        """
        Plot persistence diagram
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        title : str
            Plot title
        **kwargs
            Additional arguments for plotting
            
        Returns
        -------
        fig, ax : matplotlib objects
        """
        if not self.is_computed:
            print("No persistence diagram computed. Call compute() first.")
            return None, None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for dim, dgm in self.diagrams.items():
            # Filter out infinite death times
            dgm_finite = [(b, d) for b, d in dgm if d < np.inf]
            if dgm_finite:
                births, deaths = zip(*dgm_finite)
                ax.scatter(births, deaths, 
                          label=f'H{dim}', 
                          color=colors[dim % len(colors)],
                          alpha=0.6, s=30)
        
        # Diagonal line
        if self.diagrams:
            max_val = max(max(deaths) for dgm in self.diagrams.values() 
                         for b, d in dgm if d < np.inf)
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Birth', fontsize=12)
        ax.set_ylabel('Death', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_betti_curves(self, epsilons: np.ndarray = None, 
                          figsize: Tuple[int, int] = (10, 6)):
        """
        Plot Betti number curves
        
        Parameters
        ----------
        epsilons : np.ndarray, optional
            Epsilon values to plot
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib objects
        """
        if not self.is_computed:
            print("No persistence diagram computed. Call compute() first.")
            return None, None
        
        if epsilons is None:
            # Generate epsilon values
            max_death = max(max(deaths) for dgm in self.diagrams.values() 
                           for b, d in dgm if d < np.inf)
            epsilons = np.linspace(0, max_death, 100)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for dim, dgm in self.diagrams.items():
            betti_curve = []
            for eps in epsilons:
                betti = sum(1 for birth, death in dgm 
                           if birth <= eps < death)
                betti_curve.append(betti)
            
            ax.plot(epsilons, betti_curve, 
                label=f'β{dim}', color=colors[dim % len(colors)], linewidth=2)
        
        ax.set_xlabel('Epsilon', fontsize=12)
        ax.set_ylabel('Betti Number', fontsize=12)
        ax.set_title('Betti Number Curves', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive summary of persistence analysis
        
        Returns
        -------
        summary : dict
            Summary statistics
        """
        if not self.is_computed:
            return {}
        
        summary = {
            'betti_numbers': self.get_betti_numbers(),
            'persistence_entropy': self.get_persistence_entropy(),
            'total_persistence': self.get_total_persistence(),
            'lifespan_stats': self.get_lifespan_statistics(),
            'num_features': {dim: len(dgm) for dim, dgm in self.diagrams.items()}
        }
        
        return summary


def compute_betti_numbers(data: np.ndarray, max_dim: int = 1, 
                         epsilon: float = 0.1) -> Dict[int, int]:
    """
    Quick function to compute Betti numbers
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    max_dim : int
        Maximum homology dimension
    epsilon : float
        Scale parameter
        
    Returns
    -------
    betti : dict
        Betti numbers by dimension
    """
    diagram = PersistenceDiagram()
    diagram.compute(data, max_dim)
    return diagram.get_betti_numbers(epsilon)


def compute_persistence_entropy(data: np.ndarray, max_dim: int = 1) -> Dict[int, float]:
    """
    Quick function to compute persistence entropy
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    max_dim : int
        Maximum homology dimension
        
    Returns
    -------
    entropy : dict
        Persistence entropy by dimension
    """
    diagram = PersistenceDiagram()
    diagram.compute(data, max_dim)
    return diagram.get_persistence_entropy()


def compute_total_persistence(data: np.ndarray, max_dim: int = 1) -> Dict[int, float]:
    """
    Quick function to compute total persistence
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    max_dim : int
        Maximum homology dimension
        
    Returns
    -------
    total_persistence : dict
        Total persistence by dimension
    """
    diagram = PersistenceDiagram()
    diagram.compute(data, max_dim)
    return diagram.get_total_persistence()


def compare_persistence(diagrams: List[PersistenceDiagram], 
                       labels: List[str] = None) -> Dict:
    """
    Compare multiple persistence diagrams
    
    Parameters
    ----------
    diagrams : List[PersistenceDiagram]
        List of persistence diagrams to compare
    labels : List[str], optional
        Labels for each diagram
        
    Returns
    -------
    comparison : dict
        Comparison statistics
    """
    if labels is None:
        labels = [f'Diagram {i+1}' for i in range(len(diagrams))]
    
    comparison = {
        'labels': labels,
        'betti_numbers': {},
        'persistence_entropy': {},
        'total_persistence': {},
        'num_features': {}
    }
    
    for i, (diagram, label) in enumerate(zip(diagrams, labels)):
        if diagram.is_computed:
            comparison['betti_numbers'][label] = diagram.get_betti_numbers()
            comparison['persistence_entropy'][label] = diagram.get_persistence_entropy()
            comparison['total_persistence'][label] = diagram.get_total_persistence()
            comparison['num_features'][label] = {
                dim: len(dgm) for dim, dgm in diagram.diagrams.items()
            }
    
    return comparison


def plot_comparison(diagrams: List[PersistenceDiagram], 
                   labels: List[str] = None,
                   figsize: Tuple[int, int] = (15, 5)):
    """
    Plot comparison of multiple persistence diagrams
    
    Parameters
    ----------
    diagrams : List[PersistenceDiagram]
        List of persistence diagrams
    labels : List[str], optional
        Labels for each diagram
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, axes : matplotlib objects
    """
    if labels is None:
        labels = [f'Diagram {i+1}' for i in range(len(diagrams))]
    
    n_diagrams = len(diagrams)
    fig, axes = plt.subplots(1, n_diagrams, figsize=figsize)
    
    if n_diagrams == 1:
        axes = [axes]
    
    for i, (diagram, label) in enumerate(zip(diagrams, labels)):
        if diagram.is_computed:
            diagram.plot(title=label, ax=axes[i])
        else:
            axes[i].text(0.5, 0.5, f'{label}\n(No data)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(label)
    
    plt.tight_layout()
    return fig, axes







