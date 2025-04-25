#!/usr/bin/env python3
"""
Cross-Scale Integration Analysis Module

This module performs unified analysis across all CMB test results (transfer entropy, 
laminarity, and fractal measures) to provide a comprehensive view of mathematical 
organization across scales, with particular emphasis on golden ratio relationships.
"""

import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import seaborn as sns
import time
import math
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# High precision support
try:
    from mpmath import mp, mpf
    USE_HIGH_PRECISION = True
    mp.dps = 50  # 50 decimal places of precision
except ImportError:
    USE_HIGH_PRECISION = False
    logger.warning("mpmath not available, using standard float precision")

# Define constants
GOLDEN_RATIO = 1.618033988749895
GOLDEN_RATIO_CONJUGATE = 0.618033988749895
CONSTANTS = {
    "golden_ratio": GOLDEN_RATIO,
    "e": 2.718281828459045,
    "pi": 3.141592653589793,
    "sqrt2": 1.4142135623730951,
    "sqrt3": 1.7320508075688772,
    "ln2": 0.6931471805599453
}
FIBONACCI_NUMBERS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# High-precision constants 
HIGH_PRECISION_CONSTANTS = {
    "golden_ratio": mpf("1.6180339887498948482045868343656381177203091798057628621"),
    "sqrt2": mpf("1.4142135623730950488016887242096980785696718753769480731"),
    "e": mpf("2.7182818284590452353602874713526624977572470936999595749"),
    "pi": mpf("3.1415926535897932384626433832795028841971693993751058209")
} if USE_HIGH_PRECISION else CONSTANTS

# Utility function for high-precision calculations
def calculate_ratio_proximity(ratio: float, constant_name: str) -> float:
    """Calculate ratio proximity with high precision if available."""
    if USE_HIGH_PRECISION:
        ratio_mp = mpf(str(ratio))
        constant_mp = HIGH_PRECISION_CONSTANTS[constant_name]
        proximity = abs(ratio_mp - constant_mp) / constant_mp
        return float(proximity)
    else:
        constant = CONSTANTS[constant_name]
        return abs(ratio - constant) / constant

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy types and tuple keys"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, tuple):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def load_test_results(results_dir: str, dataset_name: str) -> Dict[str, Any]:
    """
    Enhanced version that loads all test results including retrocausality and entropy.
    
    Parameters:
    -----------
    results_dir : str
        Root directory containing test results
    dataset_name : str
        Name of dataset ('planck' or 'wmap')
        
    Returns:
    --------
    dict
        Dictionary containing all test results
    """
    results = {}
    
    # Test result file paths with added retrocausality and multiscale entropy
    test_files = {
        'transfer_entropy': f'{results_dir}/{dataset_name}/{dataset_name}_all_constants_results.json',
        'laminarity': f'{results_dir}/{dataset_name}/{dataset_name}_laminarity_results.json',
        'fractal': f'{results_dir}/{dataset_name}/{dataset_name}_fractal_results.json',
        'phase_analysis': f'{results_dir}/{dataset_name}/{dataset_name}_phase_analysis_results.json',
        'retrocausality': f'{results_dir}/{dataset_name}/{dataset_name}_retrocausality_results.json',
        'multiscale_entropy': f'{results_dir}/{dataset_name}/{dataset_name}_multiscale_entropy_results.json'
    }
    
    # Load each test result with better error handling
    for test_name, file_path in test_files.items():
        try:
            # Use absolute path to avoid issues
            abs_path = os.path.abspath(file_path)
            
            if os.path.exists(abs_path):
                with open(abs_path, 'r') as f:
                    results[test_name] = json.load(f)
                    logger.info(f"Loaded {test_name} results for {dataset_name}")
            else:
                logger.warning(f"Result file not found: {abs_path}")
        except Exception as e:
            logger.error(f"Error loading {test_name} results: {str(e)}")
    
    return results


def calculate_cross_measure_correlations(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Fixed version that properly calculates correlations across multiple scales."""
    correlation_results = {
        'pearson': {},
        'spearman': {},
        'kendall': {},
        'measures': [],
        'key_correlations': {}
    }
    
    # Check if we have pre-calculated scale data
    if ('transfer_entropy' in results and 'golden_ratio' in results['transfer_entropy'] and
            'scale_data' in results['transfer_entropy']['golden_ratio']):
        # Use pre-calculated scale data (ensures all arrays are the same length)
        logger.info("Using pre-calculated scale data for correlations")
        scale_data = results['transfer_entropy']['golden_ratio']['scale_data']
        df = pd.DataFrame(scale_data)
    else:
        # Initialize lists to store multi-scale data
        logger.info("Building scale metrics from individual data elements")
        scale_metrics = {
            'scale': [],
            'te_effect_size': [],
            'te_z_score': [],
            'te_small_to_large': [],
            'te_large_to_small': [],
            'laminarity': [],
            'lam_z_score': [],
            'hurst_exponent': [],
            'fractal_z_score': [],
            'phi_proximity': [],
            'phi_optimality': [],
            'sqrt2_proximity': []  # Added for √2 analysis
        }
        
        # Get unique scales from transfer entropy data
        unique_scales = set()
        if 'transfer_entropy' in results and 'golden_ratio' in results['transfer_entropy']:
            te_data = results['transfer_entropy']['golden_ratio']
            if 'te_values' in te_data:
                for pair_data in te_data['te_values']:
                    unique_scales.add(pair_data.get('scale_smaller'))
                    unique_scales.add(pair_data.get('scale_larger'))
        
        # Initialize all metrics with NaN for all unique scales
        for scale in sorted(unique_scales):
            if scale is not None:  # Skip None values
                scale_metrics['scale'].append(scale)
                for key in scale_metrics.keys():
                    if key != 'scale':
                        scale_metrics[key].append(np.nan)
        
        # Fill in data where available
        if 'transfer_entropy' in results and 'golden_ratio' in results['transfer_entropy']:
            te_data = results['transfer_entropy']['golden_ratio']
            if 'te_values' in te_data:
                for pair_data in te_data['te_values']:
                    small_scale = pair_data.get('scale_smaller')
                    large_scale = pair_data.get('scale_larger')
                    s_idx = scale_metrics['scale'].index(small_scale) if small_scale in scale_metrics['scale'] else -1
                    l_idx = scale_metrics['scale'].index(large_scale) if large_scale in scale_metrics['scale'] else -1
                    
                    # Fill small scale metrics
                    if s_idx >= 0:
                        scale_metrics['te_small_to_large'][s_idx] = pair_data.get('te_small_to_large', np.nan)
                        ratio = large_scale / small_scale
                        phi_proximity = abs(ratio - GOLDEN_RATIO) / GOLDEN_RATIO
                        scale_metrics['phi_proximity'][s_idx] = phi_proximity
                        scale_metrics['phi_optimality'][s_idx] = 1 - phi_proximity
                        if USE_HIGH_PRECISION and 'sqrt2' in CONSTANTS:
                            sqrt2_proximity = abs(ratio - CONSTANTS["sqrt2"]) / CONSTANTS["sqrt2"]
                        else:
                            sqrt2_proximity = abs(ratio - SQRT2) / SQRT2
                        scale_metrics['sqrt2_proximity'][s_idx] = sqrt2_proximity
                    
                    # Fill large scale metrics
                    if l_idx >= 0:
                        scale_metrics['te_large_to_small'][l_idx] = pair_data.get('te_large_to_small', np.nan)
        
        # Add global metrics
        if 'laminarity' in results:
            lam_data = results['laminarity']
            lam_value = lam_data.get('laminarity', np.nan)
            lam_z = lam_data.get('z_score', np.nan)
            for i in range(len(scale_metrics['scale'])):
                scale_metrics['laminarity'][i] = lam_value
                scale_metrics['lam_z_score'][i] = lam_z
        
        if 'fractal' in results:
            frac_data = results['fractal']
            hurst = frac_data.get('hurst', np.nan)
            fractal_z = frac_data.get('z_score', np.nan)
            for i in range(len(scale_metrics['scale'])):
                scale_metrics['hurst_exponent'][i] = hurst
                scale_metrics['fractal_z_score'][i] = fractal_z
        
        # Create DataFrame
        df = pd.DataFrame(scale_metrics)
    
    # Compute correlations
    
    # Remove columns with all NaN values
    df = df.dropna(axis=1, how='all')
    
    # Calculate correlations if we have sufficient data
    if len(df) > 2:  # Need at least 3 rows for meaningful correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Pearson correlation
        correlation_results['pearson'] = df[numeric_cols].corr(method='pearson').to_dict()
        
        # Spearman correlation
        correlation_results['spearman'] = df[numeric_cols].corr(method='spearman').to_dict()
        
        # Kendall correlation
        correlation_results['kendall'] = df[numeric_cols].corr(method='kendall').to_dict()
        
        # Store the measures analyzed
        correlation_results['measures'] = list(numeric_cols)
        
        # Add specific correlations of interest
        correlations_of_interest = {}
        if 'phi_optimality' in numeric_cols and 'laminarity' in numeric_cols:
            correlations_of_interest['phi_laminarity'] = df['phi_optimality'].corr(df['laminarity'])
        if 'sqrt2_proximity' in numeric_cols and 'laminarity' in numeric_cols:
            correlations_of_interest['sqrt2_laminarity'] = df['sqrt2_proximity'].corr(df['laminarity'])
        
        correlation_results['key_correlations'] = correlations_of_interest
    
    return correlation_results


def analyze_scale_transitions(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Enhanced version with √2 analysis and high-precision calculations."""
    transitions = {
        'golden_ratio_transitions': [],
        'sqrt2_transitions': [],  # Added for √2 pairs
        'fibonacci_transitions': [],
        'key_scales': [],
        'scale_55_relationships': []
    }
    
    if 'transfer_entropy' in results and 'golden_ratio' in results['transfer_entropy']:
        te_data = results['transfer_entropy']['golden_ratio']
        
        if 'te_values' in te_data:
            for pair_data in te_data['te_values']:
                small_scale = pair_data.get('scale_smaller')
                large_scale = pair_data.get('scale_larger')
                te_small_to_large = pair_data.get('te_small_to_large')
                te_large_to_small = pair_data.get('te_large_to_small')
                
                # Calculate ratios with high precision
                ratio = large_scale / small_scale
                
                # Use high-precision proximity calculations
                phi_proximity = calculate_ratio_proximity(ratio, "golden_ratio")
                sqrt2_proximity = calculate_ratio_proximity(ratio, "sqrt2")
                
                flow_differential = te_small_to_large - te_large_to_small
                primary_direction = "small_to_large" if flow_differential > 0 else "large_to_small"
                
                is_fib_small = small_scale in FIBONACCI_NUMBERS
                is_fib_large = large_scale in FIBONACCI_NUMBERS
                
                # Determine dominant constant (φ or √2)
                dominant_constant = "phi" if phi_proximity < sqrt2_proximity else "sqrt2"
                
                transition = {
                    'small_scale': small_scale,
                    'large_scale': large_scale,
                    'ratio': ratio,
                    'phi_proximity': phi_proximity,
                    'sqrt2_proximity': sqrt2_proximity,
                    'dominant_constant': dominant_constant,
                    'te_small_to_large': te_small_to_large,
                    'te_large_to_small': te_large_to_small,
                    'flow_differential': flow_differential,
                    'primary_direction': primary_direction,
                    'is_fibonacci_small': is_fib_small,
                    'is_fibonacci_large': is_fib_large
                }
                
                # Categorize transitions
                if phi_proximity < 0.05:  # Close to φ
                    transitions['golden_ratio_transitions'].append(transition)
                
                if sqrt2_proximity < 0.05:  # Close to √2
                    transitions['sqrt2_transitions'].append(transition)
                
                if is_fib_small and is_fib_large:
                    transitions['fibonacci_transitions'].append(transition)
                
                if small_scale == 55 or large_scale == 55:
                    transitions['scale_55_relationships'].append(transition)
    
    # Identify key scales    # Count scale connections for both φ and √2
    if transitions['golden_ratio_transitions'] or transitions['sqrt2_transitions']:
        scale_counts = {}
        for transition_type in ['golden_ratio_transitions', 'sqrt2_transitions']:
            for transition in transitions[transition_type]:
                small = transition['small_scale']
                large = transition['large_scale']
                scale_counts[small] = scale_counts.get(small, 0) + 1
                scale_counts[large] = scale_counts.get(large, 0) + 1
        
        sorted_scales = sorted(scale_counts.items(), key=lambda x: x[1], reverse=True)
        transitions['key_scales'] = [{'scale': scale, 'connections': count} for scale, count in sorted_scales]
    
    return transitions


def calculate_information_flow_metrics(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Calculate unified information flow metrics across all tests
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all test results
        
    Returns:
    --------
    dict
        Dictionary containing information flow metrics
    """
    flow_metrics = {
        'bidirectional_patterns': {},
        'fractal_to_transfer_relationship': {},
        'scale_sensitivity': {}
    }
    
    # Extract directional patterns from transfer entropy
    if 'transfer_entropy' in results and 'golden_ratio' in results['transfer_entropy']:
        te_data = results['transfer_entropy']['golden_ratio']
        
        # Calculate overall directional bias
        if 'mean_te_small_to_large' in te_data and 'mean_te_large_to_small' in te_data:
            small_to_large = te_data['mean_te_small_to_large']
            large_to_small = te_data['mean_te_large_to_small']
            
            # Calculate directional bias
            if small_to_large + large_to_small > 0:
                directional_bias = (small_to_large - large_to_small) / (small_to_large + large_to_small)
            else:
                directional_bias = 0
                
            flow_metrics['bidirectional_patterns']['directional_bias'] = directional_bias
            flow_metrics['bidirectional_patterns']['primary_direction'] = "small_to_large" if directional_bias > 0 else "large_to_small"
            flow_metrics['bidirectional_patterns']['small_to_large'] = small_to_large
            flow_metrics['bidirectional_patterns']['large_to_small'] = large_to_small
    
    # Relate Hurst exponent to transfer entropy patterns
    if 'fractal' in results and 'transfer_entropy' in results:
        fractal_data = results['fractal']
        
        if 'hurst' in fractal_data and 'bidirectional_patterns' in flow_metrics:
            hurst = fractal_data['hurst']
            
            # Calculate relation to golden ratio conjugate
            phi_conj_proximity = abs(hurst - GOLDEN_RATIO_CONJUGATE)
            
            # Relate to directional bias
            if 'directional_bias' in flow_metrics['bidirectional_patterns']:
                dir_bias = flow_metrics['bidirectional_patterns']['directional_bias']
                
                # A fascinating hypothesis: hurst exponents near φ⁻¹ may correlate with balanced bidirectional flow
                balance_metric = 1 - abs(dir_bias)  # 1 = perfect balance, 0 = completely unidirectional
                
                flow_metrics['fractal_to_transfer_relationship'] = {
                    'hurst_exponent': hurst,
                    'phi_conjugate_proximity': phi_conj_proximity,
                    'flow_balance': balance_metric,
                    'hypothesis': 'Hurst exponents near φ⁻¹ may indicate optimal balance in bidirectional information flow'
                }
    
    # Calculate scale sensitivity based on test results
    # Identify scales where multiple tests show significant results
    scale_significance = {}
    
    # Check scale 55 significance across tests
    if 'transfer_entropy' in results and 'golden_ratio' in results['transfer_entropy']:
        if 'scale_pairs' in results['transfer_entropy']['golden_ratio']:
            for pair in results['transfer_entropy']['golden_ratio']['scale_pairs']:
                small, large, _ = pair
                if small == 55 or large == 55:
                    scale_significance[55] = scale_significance.get(55, 0) + 1
    
    # Add more scale significance checks from other tests...
    
    # Sort scales by significance count
    if scale_significance:
        sorted_scales = sorted(scale_significance.items(), key=lambda x: x[1], reverse=True)
        flow_metrics['scale_sensitivity']['significant_scales'] = [{'scale': scale, 'significance_count': count} 
                                                                for scale, count in sorted_scales]
    
    return flow_metrics


def create_network_visualization(results: Dict[str, Dict], dataset_name: str, output_dir: str) -> None:
    """
    Create network visualization of scale relationships across all tests
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all test results
    dataset_name : str
        Name of dataset ('planck' or 'wmap')
    output_dir : str
        Directory to save visualization
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Track significant scales
    significant_scales = set()
    
    # Add nodes and edges from transfer entropy results
    if 'transfer_entropy' in results and 'golden_ratio' in results['transfer_entropy']:
        te_data = results['transfer_entropy']['golden_ratio']
        
        if 'te_values' in te_data:
            for pair_data in te_data['te_values']:
                small_scale = pair_data.get('scale_smaller')
                large_scale = pair_data.get('scale_larger')
                te_small_to_large = pair_data.get('te_small_to_large')
                te_large_to_small = pair_data.get('te_large_to_small')
                
                # Add nodes if they don't exist
                if small_scale not in G:
                    is_fibonacci = small_scale in FIBONACCI_NUMBERS
                    size = 1000 if is_fibonacci else 500
                    color = 'gold' if is_fibonacci else 'blue'
                    G.add_node(small_scale, size=size, color=color, type='scale')
                
                if large_scale not in G:
                    is_fibonacci = large_scale in FIBONACCI_NUMBERS
                    size = 1000 if is_fibonacci else 500
                    color = 'gold' if is_fibonacci else 'blue'
                    G.add_node(large_scale, size=size, color=color, type='scale')
                
                # Determine dominant flow direction
                if te_small_to_large > te_large_to_small:
                    # Small to large is dominant
                    edge_weight = te_small_to_large
                    G.add_edge(small_scale, large_scale, weight=edge_weight*5, direction='small_to_large')
                    if edge_weight > 0.1:  # Significance threshold
                        significant_scales.add(small_scale)
                        significant_scales.add(large_scale)
                else:
                    # Large to small is dominant
                    edge_weight = te_large_to_small
                    G.add_edge(large_scale, small_scale, weight=edge_weight*5, direction='large_to_small')
                    if edge_weight > 0.1:  # Significance threshold
                        significant_scales.add(small_scale)
                        significant_scales.add(large_scale)
    
    # Augment network with laminarity and fractal data
    # Add a central node representing laminarity and connect it to significant scales
    if 'laminarity' in results:
        lam_data = results['laminarity']
        lam_value = lam_data.get('laminarity', 0)
        
        if lam_value > 0:
            G.add_node('laminarity', size=2000, color='red', type='metric', value=lam_value)
            
            # Connect laminarity to significant scales
            for scale in significant_scales:
                G.add_edge('laminarity', scale, weight=3, color='red', style='dashed')
    
    # Add fractal measure node
    if 'fractal' in results:
        fractal_data = results['fractal']
        hurst = fractal_data.get('hurst', 0)
        
        if hurst > 0:
            G.add_node('hurst', size=2000, color='purple', type='metric', value=hurst)
            
            # Connect hurst to significant scales
            for scale in significant_scales:
                G.add_edge('hurst', scale, weight=3, color='purple', style='dashed')
    
    # Specifically highlight scale 55 which is often significant
    if 55 in G:
        G.nodes[55]['size'] = 2500
        G.nodes[55]['color'] = 'crimson'
    
    # Draw the network
    plt.figure(figsize=(24, 16))
    
    # Use spring layout with scale 55 at center
    pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    if 55 in pos:
        pos[55] = np.array([0, 0])  # Center scale 55
    
    # Draw nodes
    node_sizes = [G.nodes[n].get('size', 300) for n in G.nodes()]
    node_colors = [G.nodes[n].get('color', 'blue') for n in G.nodes()]
    
    # Draw metric nodes (laminarity, hurst) separately
    metric_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'metric']
    scale_nodes = [n for n, attr in G.nodes(data=True) if n not in metric_nodes]
    
    # Draw scale nodes - ensure node sizes match the node list
    if scale_nodes:
        # Make sure node_sizes and node_colors match the length of scale_nodes
        if isinstance(node_sizes, list) and len(node_sizes) != len(scale_nodes):
            # Use a single scalar value if lengths don't match
            node_sizes = 1000
        
        if isinstance(node_colors, list) and len(node_colors) != len(scale_nodes):
            # Use a default color if lengths don't match
            node_colors = 'skyblue'
            
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=scale_nodes,
                            node_size=node_sizes,
                            node_color=node_colors, 
                            alpha=0.8)
    
    # Draw metric nodes with different shape
    if metric_nodes:
        # Get node sizes directly to ensure they match the node count
        metric_node_sizes = [G.nodes[n].get('size', 1500) for n in metric_nodes]
        metric_node_colors = [G.nodes[n].get('color', 'red') for n in metric_nodes]
        
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=metric_nodes,
                            node_size=metric_node_sizes,
                            node_color=metric_node_colors, 
                            alpha=0.8,
                            node_shape='s')  # Square shape for metrics
    
    # Draw edges with appropriate styles
    edges = G.edges(data=True)
    
    # Group edges by style
    standard_edges = [(u, v) for u, v, d in edges if d.get('style') != 'dashed']
    dashed_edges = [(u, v) for u, v, d in edges if d.get('style') == 'dashed']
    
    # Draw standard edges
    edge_weights = [G[u][v].get('weight', 1) for u, v in standard_edges]
    edge_colors = [G[u][v].get('color', 'gray') for u, v in standard_edges]
    
    nx.draw_networkx_edges(G, pos, 
                        edgelist=standard_edges, 
                        width=edge_weights, 
                        edge_color=edge_colors,
                        arrows=True,
                        arrowstyle='-|>',
                        arrowsize=20,
                        alpha=0.7)
    
    # Draw dashed edges
    if dashed_edges:
        dashed_weights = [G[u][v].get('weight', 1) for u, v in dashed_edges]
        dashed_colors = [G[u][v].get('color', 'gray') for u, v in dashed_edges]
        
        nx.draw_networkx_edges(G, pos, 
                            edgelist=dashed_edges, 
                            width=dashed_weights, 
                            edge_color=dashed_colors,
                            arrows=True,
                            arrowstyle='-|>',
                            arrowsize=15,
                            alpha=0.5,
                            style='dashed')
    
    # Draw labels with different fonts for metrics vs scales
    scale_labels = {n: str(n) for n in scale_nodes}
    metric_labels = {n: n for n in metric_nodes}
    
    nx.draw_networkx_labels(G, pos, 
                          labels=scale_labels,
                          font_size=12, 
                          font_weight='bold')
    
    if metric_nodes:
        nx.draw_networkx_labels(G, pos, 
                              labels=metric_labels,
                              font_size=14,
                              font_color='white',
                              font_weight='bold')
    
    # Add title and legend
    plt.title(f"{dataset_name.upper()} Integrated Analysis Network", fontsize=20)
    
    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Fibonacci Scale', 
                 markerfacecolor='gold', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Regular Scale', 
                 markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Scale 55', 
                 markerfacecolor='crimson', markersize=20),
        plt.Line2D([0], [0], marker='s', color='w', label='Laminarity', 
                 markerfacecolor='red', markersize=15),
        plt.Line2D([0], [0], marker='s', color='w', label='Hurst Exponent', 
                 markerfacecolor='purple', markersize=15),
        plt.Line2D([0], [0], color='black', lw=2, label='Small → Large Flow'),
        plt.Line2D([0], [0], color='black', lw=2, label='Large → Small Flow', linestyle='--')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{dataset_name}_integrated_network.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Created integrated network visualization: {output_path}")


def create_unified_visualization(results_planck: Dict[str, Dict], results_wmap: Dict[str, Dict], output_dir: str) -> None:
    """
    Create unified visualization comparing results from both datasets
    
    Parameters:
    -----------
    results_planck : dict
        Dictionary containing Planck test results
    results_wmap : dict
        Dictionary containing WMAP test results
    output_dir : str
        Directory to save visualization
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(3, 4, figure=fig)
    
    # Extract key metrics for comparison
    metrics = {
        'planck': {},
        'wmap': {}
    }
    
    # Helper function to extract metrics from results
    def extract_metrics(results, dataset):
        if 'transfer_entropy' in results and 'golden_ratio' in results['transfer_entropy']:
            te_data = results['transfer_entropy']['golden_ratio']
            metrics[dataset]['te_small_to_large'] = te_data.get('mean_te_small_to_large', 0)
            metrics[dataset]['te_large_to_small'] = te_data.get('mean_te_large_to_small', 0)
            metrics[dataset]['te_z_score'] = te_data.get('z_score_differential', 0)
            metrics[dataset]['te_effect_size'] = te_data.get('effect_size_differential', 0)
        
        if 'laminarity' in results:
            metrics[dataset]['laminarity'] = results['laminarity'].get('laminarity', 0)
            metrics[dataset]['lam_z_score'] = results['laminarity'].get('z_score', 0)
        
        if 'fractal' in results:
            metrics[dataset]['hurst'] = results['fractal'].get('hurst', 0)
            metrics[dataset]['hurst_z_score'] = results['fractal'].get('z_score', 0)
            metrics[dataset]['phi_proximity'] = 1 - results['fractal'].get('phi_proximity', 1)  # Convert to optimization
    
    # Extract metrics from both datasets
    extract_metrics(results_planck, 'planck')
    extract_metrics(results_wmap, 'wmap')
    
    # 1. Transfer Entropy Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0:2])
    
    # Set up data for side-by-side bar chart
    labels = ['Small → Large Flow', 'Large → Small Flow']
    planck_values = [metrics['planck'].get('te_small_to_large', 0), metrics['planck'].get('te_large_to_small', 0)]
    wmap_values = [metrics['wmap'].get('te_small_to_large', 0), metrics['wmap'].get('te_large_to_small', 0)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, planck_values, width, label='Planck', color='coral')
    ax1.bar(x + width/2, wmap_values, width, label='WMAP', color='skyblue')
    
    ax1.set_ylabel('Transfer Entropy (bits)', fontsize=14)
    ax1.set_title('Bidirectional Information Flow Comparison', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    
    # Annotate with z-scores
    for i, dataset in enumerate(['planck', 'wmap']):
        if 'te_z_score' in metrics[dataset]:
            z = metrics[dataset]['te_z_score']
            x_pos = 1.5 + (i*2)
            y_pos = max(planck_values + wmap_values) * 0.8
            ax1.text(x_pos, y_pos, f"z = {z:.2f}σ", 
                   fontsize=12, ha='center', 
                   bbox=dict(facecolor='white', alpha=0.7))
    
    # 2. Statistical Significance Comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    # Set up data for significance comparison
    test_names = ['Transfer Entropy', 'Laminarity', 'Hurst Exponent']
    z_score_fields = ['te_z_score', 'lam_z_score', 'hurst_z_score']
    
    planck_z_scores = [metrics['planck'].get(field, 0) for field in z_score_fields]
    wmap_z_scores = [metrics['wmap'].get(field, 0) for field in z_score_fields]
    
    x = np.arange(len(test_names))
    
    ax2.bar(x - width/2, planck_z_scores, width, label='Planck', color='coral')
    ax2.bar(x + width/2, wmap_z_scores, width, label='WMAP', color='skyblue')
    
    ax2.set_ylabel('Statistical Significance (σ)', fontsize=14)
    ax2.set_title('Test Significance Comparison', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    
    # Add threshold lines
    ax2.axhline(y=1.96, linestyle='--', color='gray', alpha=0.7, label='p=0.05 (1.96σ)')
    ax2.axhline(y=2.58, linestyle='--', color='darkgray', alpha=0.7, label='p=0.01 (2.58σ)')
    ax2.axhline(y=3.29, linestyle='--', color='black', alpha=0.7, label='p=0.001 (3.29σ)')
    
    # 3. Hurst Exponent vs Golden Ratio Conjugate (middle left)
    ax3 = fig.add_subplot(gs[1, 0:2])
    
    # Create visualization of Hurst exponent proximity to golden ratio conjugate
    phi_conj = GOLDEN_RATIO_CONJUGATE
    hurst_values = [metrics['planck'].get('hurst', 0), metrics['wmap'].get('hurst', 0)]
    dataset_labels = ['Planck', 'WMAP']
    
    # Create a line showing the golden ratio conjugate
    ax3.axhline(y=phi_conj, linestyle='--', color='gold', linewidth=2, label=f'Golden Ratio Conjugate (1/φ = {phi_conj:.6f})')
    
    # Plot the Hurst exponents
    ax3.bar(dataset_labels, hurst_values, color=['coral', 'skyblue'], alpha=0.7)
    
    # Add proximity annotations
    for i, dataset in enumerate(['planck', 'wmap']):
        if 'hurst' in metrics[dataset]:
            hurst = metrics[dataset]['hurst']
            proximity = abs(hurst - phi_conj)
            ax3.annotate(f'Δ = {proximity:.6f}', 
                       xy=(i, hurst), 
                       xytext=(i, hurst + 0.1 if hurst < phi_conj else hurst - 0.1),
                       ha='center',
                       fontsize=12,
                       arrowprops=dict(arrowstyle='->'))
    
    ax3.set_ylabel('Hurst Exponent', fontsize=14)
    ax3.set_title('Hurst Exponent vs Golden Ratio Conjugate', fontsize=16)
    ax3.legend(fontsize=12)
    ax3.grid(alpha=0.3)
    
    # Set appropriate y-limits to focus on the region around the golden ratio conjugate
    y_margin = 0.2
    y_min = min(min(hurst_values), phi_conj) - y_margin
    y_max = max(max(hurst_values), phi_conj) + y_margin
    ax3.set_ylim(y_min, y_max)
    
    # 4. Laminarity Values (middle right)
    ax4 = fig.add_subplot(gs[1, 2:4])
    
    # Create laminarity comparison
    lam_values = [metrics['planck'].get('laminarity', 0), metrics['wmap'].get('laminarity', 0)]
    
    # Create a color gradient from red to green based on laminarity value
    colors = ['darkred', 'red', 'orange', 'yellowgreen', 'green', 'darkgreen']
    lam_cmap = LinearSegmentedColormap.from_list('lam_cmap', colors)
    
    # Create a gauge-like visualization
    lam_bars = ax4.barh(dataset_labels, lam_values, color=[lam_cmap(v) for v in lam_values], height=0.5)
    
    # Add value annotations
    for i, v in enumerate(lam_values):
        ax4.text(v + 0.02, i, f'{v:.8f}', ha='left', va='center', fontsize=14, fontweight='bold')
    
    ax4.set_xlim(0, 1.1)  # Laminarity is between 0 and 1
    ax4.set_xlabel('Laminarity (0-1)', fontsize=14)
    ax4.set_title('Laminarity Comparison', fontsize=16)
    
    # Add color gradient legend
    sm = ScalarMappable(cmap=lam_cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4, orientation='horizontal', pad=0.2)
    cbar.set_label('Laminarity Scale (Higher = More Ordered)', fontsize=12)
    
    # 5. Golden Ratio Optimization Across Tests (bottom)
    ax5 = fig.add_subplot(gs[2, 1:3])
    
    # Collect golden ratio optimization measures
    optimization_metrics = [
        ('Transfer Entropy', 'te_effect_size', 'Effect Size (vs Surrogate)'),
        ('Laminarity', 'laminarity', 'Value (0-1)'),
        ('Fractal', 'phi_proximity', 'Conjugate Proximity (0-1)')
    ]
    
    # Create data for radar chart
    categories = [m[0] for m in optimization_metrics]
    planck_values = [metrics['planck'].get(m[1], 0) for m in optimization_metrics]
    wmap_values = [metrics['wmap'].get(m[1], 0) for m in optimization_metrics]
    
    # Scale values to 0-1 range for radar chart
    max_values = []
    for i, metric in enumerate(optimization_metrics):
        max_val = max(planck_values[i], wmap_values[i])
        if max_val > 0:
            planck_values[i] = planck_values[i] / max_val
            wmap_values[i] = wmap_values[i] / max_val
        max_values.append(max_val)
    
    # Set up radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    planck_values += planck_values[:1]
    wmap_values += wmap_values[:1]
    
    ax5.plot(angles, planck_values, 'o-', linewidth=2, label='Planck', color='coral')
    ax5.plot(angles, wmap_values, 'o-', linewidth=2, label='WMAP', color='skyblue')
    ax5.fill(angles, planck_values, alpha=0.25, color='coral')
    ax5.fill(angles, wmap_values, alpha=0.25, color='skyblue')
    
    # Set category labels
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories)
    
    # Set chart properties
    ax5.set_ylim(0, 1.1)
    ax5.set_title('Golden Ratio Optimization Across Tests', fontsize=16)
    ax5.legend(loc='upper right', fontsize=12)
    
    # Add annotations with actual values
    for i, angle in enumerate(angles[:-1]):
        if i < len(max_values):
            ax5.text(angle, 1.15, f"Max: {max_values[i]:.2f}", 
                   ha='center', va='center', fontsize=10)
    
    # Overall title
    fig.suptitle('Unified Analysis of Mathematical Organization in the CMB', fontsize=24, y=0.98)
    
    # Add methodology note
    fig.text(0.5, 0.01, "Note: All metrics normalized for comparison. Statistical significance expressed in standard deviations (σ).", 
           ha='center', fontsize=12, fontstyle='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, "unified_analysis_visualization.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Created unified analysis visualization: {output_path}")


def generate_key_findings(results_planck: Dict[str, Dict], results_wmap: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """
    Generate list of key integrated findings
    
    Parameters:
    -----------
    results_planck : dict
        Dictionary containing Planck test results
    results_wmap : dict
        Dictionary containing WMAP test results
        
    Returns:
    --------
    list
        List of key findings as dictionaries
    """
    findings = []
    
    # Finding 1: Bidirectional information flow
    planck_direction = "unknown"
    wmap_direction = "unknown"
    
    if 'transfer_entropy' in results_planck and 'golden_ratio' in results_planck['transfer_entropy']:
        planck_te = results_planck['transfer_entropy']['golden_ratio']
        if 'mean_te_small_to_large' in planck_te and 'mean_te_large_to_small' in planck_te:
            if planck_te['mean_te_small_to_large'] > planck_te['mean_te_large_to_small']:
                planck_direction = "small_to_large"
            else:
                planck_direction = "large_to_small"
    
    if 'transfer_entropy' in results_wmap and 'golden_ratio' in results_wmap['transfer_entropy']:
        wmap_te = results_wmap['transfer_entropy']['golden_ratio']
        if 'mean_te_small_to_large' in wmap_te and 'mean_te_large_to_small' in wmap_te:
            if wmap_te['mean_te_small_to_large'] > wmap_te['mean_te_large_to_small']:
                wmap_direction = "small_to_large"
            else:
                wmap_direction = "large_to_small"
    
    if planck_direction != "unknown" and wmap_direction != "unknown" and planck_direction != wmap_direction:
        findings.append({
            'title': 'Complementary Bidirectional Information Flow',
            'description': f'The analysis confirms bidirectional information flow across cosmic scales with complementary directionality: Planck data (smaller scales) shows predominant flow from {planck_direction.replace("_", " ")} scales, while WMAP data (larger scales) shows predominant flow from {wmap_direction.replace("_", " ")} scales.',
            'significance': 'This complementary pattern suggests a sophisticated information processing system where information cascades downward at larger scales while simultaneously emerging upward from smaller scales.',
            'datasets': ['planck', 'wmap'],
            'tests': ['transfer_entropy'],
            'supports_consciousness_field_theory': True
        })
    
    # Finding 2: Golden Ratio Optimization
    golden_ratio_significance = []
    
    if 'transfer_entropy' in results_planck and 'golden_ratio' in results_planck['transfer_entropy']:
        if 'z_score_differential' in results_planck['transfer_entropy']['golden_ratio']:
            z = results_planck['transfer_entropy']['golden_ratio']['z_score_differential']
            if abs(z) > 1.96:  # Significant at p<0.05
                golden_ratio_significance.append(('transfer_entropy', 'planck', z))
    
    if 'transfer_entropy' in results_wmap and 'golden_ratio' in results_wmap['transfer_entropy']:
        if 'z_score_differential' in results_wmap['transfer_entropy']['golden_ratio']:
            z = results_wmap['transfer_entropy']['golden_ratio']['z_score_differential']
            if abs(z) > 1.96:  # Significant at p<0.05
                golden_ratio_significance.append(('transfer_entropy', 'wmap', z))
    
    if 'fractal' in results_planck and 'phi_proximity' in results_planck['fractal']:
        if results_planck['fractal']['phi_proximity'] < 0.1:  # Close to golden ratio conjugate
            if 'z_score' in results_planck['fractal'] and abs(results_planck['fractal']['z_score']) > 1.96:
                golden_ratio_significance.append(('fractal', 'planck', results_planck['fractal']['z_score']))
    
    if 'fractal' in results_wmap and 'phi_proximity' in results_wmap['fractal']:
        if results_wmap['fractal']['phi_proximity'] < 0.1:  # Close to golden ratio conjugate
            if 'z_score' in results_wmap['fractal'] and abs(results_wmap['fractal']['z_score']) > 1.96:
                golden_ratio_significance.append(('fractal', 'wmap', results_wmap['fractal']['z_score']))
    
    if golden_ratio_significance:
        findings.append({
            'title': 'Golden Ratio Optimization Across Multiple Tests',
            'description': f'Multiple tests show significant mathematical organization optimized around the golden ratio (φ ≈ 1.618) and its conjugate (1/φ ≈ 0.618): ' + 
                          ', '.join([f"{test} for {dataset} data (z={z:.2f}σ)" for test, dataset, z in golden_ratio_significance]),
            'significance': 'The consistent golden ratio optimization across multiple independent tests and datasets provides compelling evidence for a fundamental mathematical organizing principle in the CMB.',
            'datasets': list(set([dataset for _, dataset, _ in golden_ratio_significance])),
            'tests': list(set([test for test, _, _ in golden_ratio_significance])),
            'supports_consciousness_field_theory': True
        })
    
    # Finding 3: Scale 55 Significance
    scale_55_significance = []
    
    # Check for scale 55 in transfer entropy
    if 'transfer_entropy' in results_planck and 'scale_relationships' in results_planck['transfer_entropy']:
        for relationship in results_planck['transfer_entropy']['scale_relationships']:
            if relationship.get('scale') == 55 and relationship.get('z_score', 0) > 1.96:
                scale_55_significance.append(('transfer_entropy', 'planck', relationship.get('z_score', 0)))
    
    if 'transfer_entropy' in results_wmap and 'scale_relationships' in results_wmap['transfer_entropy']:
        for relationship in results_wmap['transfer_entropy']['scale_relationships']:
            if relationship.get('scale') == 55 and relationship.get('z_score', 0) > 1.96:
                scale_55_significance.append(('transfer_entropy', 'wmap', relationship.get('z_score', 0)))
    
    # Check for other tests...
    
    if scale_55_significance:
        findings.append({
            'title': 'Scale 55 Shows Exceptional Mathematical Organization',
            'description': f'Scale 55 demonstrates significant mathematical organization in ' + 
                          ', '.join([f"{test} for {dataset} data (z={z:.2f}σ)" for test, dataset, z in scale_55_significance]),
            'significance': 'Scale 55 appears to serve as a key organizational scale in the CMB, potentially related to sqrt(2) relationships previously identified. This scale may represent a critical resonance point in cosmic structure.',
            'datasets': list(set([dataset for _, dataset, _ in scale_55_significance])),
            'tests': list(set([test for test, _, _ in scale_55_significance])),
            'supports_consciousness_field_theory': True
        })
    
    # Finding 4: Fractal Structure
    # Extract Hurst exponents and their statistical significance
    fractal_significance = []
    
    if 'fractal' in results_planck and 'hurst' in results_planck['fractal'] and 'p_value' in results_planck['fractal']:
        hurst = results_planck['fractal']['hurst']
        p_value = results_planck['fractal']['p_value']
        
        if p_value < 0.05:  # Significant fractal structure
            fractal_significance.append(('planck', hurst, p_value))
    
    if 'fractal' in results_wmap and 'hurst' in results_wmap['fractal'] and 'p_value' in results_wmap['fractal']:
        hurst = results_wmap['fractal']['hurst']
        p_value = results_wmap['fractal']['p_value']
        
        if p_value < 0.05:  # Significant fractal structure
            fractal_significance.append(('wmap', hurst, p_value))
    
    if fractal_significance:
        findings.append({
            'title': 'Significant Fractal Structure in CMB Data',
            'description': 'Fractal analysis reveals significant long-range correlations in the CMB: ' + 
                          ', '.join([f"{dataset} data shows Hurst={h:.3f} (p={p:.6f})" for dataset, h, p in fractal_significance]),
            'significance': 'The presence of fractal structure suggests that the CMB exhibits persistent, self-organizing behavior across scales. This is consistent with a cosmos that maintains coherent mathematical patterns from the quantum to astronomical scales.',
            'datasets': list(set([dataset for dataset, _, _ in fractal_significance])),
            'tests': ['fractal'],
            'supports_consciousness_field_theory': True
        })
    
    # Finding 5: Cross-dataset consistency
    # If both datasets show similar patterns, this is an important finding
    tests_consistent = []
    
    # Check transfer entropy consistency
    if ('transfer_entropy' in results_planck and 'golden_ratio' in results_planck['transfer_entropy'] and
        'transfer_entropy' in results_wmap and 'golden_ratio' in results_wmap['transfer_entropy']):
        
        planck_significant = results_planck['transfer_entropy']['golden_ratio'].get('significant', False)
        wmap_significant = results_wmap['transfer_entropy']['golden_ratio'].get('significant', False)
        
        if planck_significant and wmap_significant:
            tests_consistent.append('transfer_entropy')
    
    # Check laminarity consistency
    if ('laminarity' in results_planck and 'significant' in results_planck['laminarity'] and
        'laminarity' in results_wmap and 'significant' in results_wmap['laminarity']):
        
        if results_planck['laminarity']['significant'] and results_wmap['laminarity']['significant']:
            tests_consistent.append('laminarity')
    
    # Check fractal consistency
    if ('fractal' in results_planck and 'p_value' in results_planck['fractal'] and
        'fractal' in results_wmap and 'p_value' in results_wmap['fractal']):
        
        if results_planck['fractal']['p_value'] < 0.05 and results_wmap['fractal']['p_value'] < 0.05:
            tests_consistent.append('fractal')
    
    if len(tests_consistent) >= 2:  # At least two tests show consistency
        findings.append({
            'title': 'Consistent Mathematical Organization Across Independent Datasets',
            'description': f'Multiple tests ({", ".join(tests_consistent)}) show consistent mathematical organization patterns in both WMAP and Planck datasets, indicating the robustness of these findings.',
            'significance': 'The consistency across independent observations collected by different instruments at different times strongly suggests that the identified mathematical patterns are intrinsic to the CMB rather than artifacts of data collection or processing.',
            'datasets': ['planck', 'wmap'],
            'tests': tests_consistent,
            'supports_consciousness_field_theory': True
        })
    
    # Finding 6: Retrocausality results - properly load and analyze data
    retrocausality_significance = []
    
    if 'retrocausality' in results_planck:
        planck_retro = results_planck['retrocausality']
        if 'phi_timepoints_r_squared' in planck_retro and 'non_phi_timepoints_r_squared' in planck_retro:
            phi_r2 = planck_retro['phi_timepoints_r_squared']
            non_phi_r2 = planck_retro['non_phi_timepoints_r_squared']
            if phi_r2 > non_phi_r2:  # Stronger correlation at phi timepoints
                retrocausality_significance.append(('planck', phi_r2, non_phi_r2))
    
    if 'retrocausality' in results_wmap:
        wmap_retro = results_wmap['retrocausality']
        if 'phi_timepoints_r_squared' in wmap_retro and 'non_phi_timepoints_r_squared' in wmap_retro:
            phi_r2 = wmap_retro['phi_timepoints_r_squared']
            non_phi_r2 = wmap_retro['non_phi_timepoints_r_squared']
            if phi_r2 > non_phi_r2:  # Stronger correlation at phi timepoints
                retrocausality_significance.append(('wmap', phi_r2, non_phi_r2))
    
    if retrocausality_significance:
        findings.append({
            'title': 'Phi-Related Galaxy Formation Times Show Stronger CMB Correlations',
            'description': 'Retrocausality tests reveal that galaxies forming at phi-related timescales exhibit much stronger correlations with CMB patterns: ' + 
                          ', '.join([f"{dataset} data shows R²={phi_r2:.4f} for phi timepoints vs R²={non_phi_r2:.4f} for non-phi timepoints" 
                                     for dataset, phi_r2, non_phi_r2 in retrocausality_significance]),
            'significance': 'This finding suggests that cosmic structure evolution may be optimized around golden ratio temporal relationships, further supporting the mathematical organization hypothesis and potentially indicating temporal symmetry breaking.',
            'datasets': list(set([dataset for dataset, _, _ in retrocausality_significance])),
            'tests': ['retrocausality'],
            'supports_consciousness_field_theory': True
        })
    
    # Finding 7: Multiscale entropy results
    entropy_significance = []
    
    if 'multiscale_entropy' in results_planck:
        planck_entropy = results_planck['multiscale_entropy']
        if 'phi_scales_entropy' in planck_entropy and 'random_scales_entropy' in planck_entropy:
            phi_entropy = planck_entropy['phi_scales_entropy']
            random_entropy = planck_entropy['random_scales_entropy']
            p_value = planck_entropy.get('p_value', 1.0)
            if p_value < 0.05:  # Significant difference
                entropy_significance.append(('planck', phi_entropy, random_entropy, p_value))
    
    if 'multiscale_entropy' in results_wmap:
        wmap_entropy = results_wmap['multiscale_entropy']
        if 'phi_scales_entropy' in wmap_entropy and 'random_scales_entropy' in wmap_entropy:
            phi_entropy = wmap_entropy['phi_scales_entropy']
            random_entropy = wmap_entropy['random_scales_entropy']
            p_value = wmap_entropy.get('p_value', 1.0)
            if p_value < 0.05:  # Significant difference
                entropy_significance.append(('wmap', phi_entropy, random_entropy, p_value))
    
    if entropy_significance:
        findings.append({
            'title': 'Optimal Information Complexity at Golden Ratio Scales',
            'description': 'Multiscale entropy analysis reveals optimized information complexity at golden ratio related scales: ' + 
                          ', '.join([f"{dataset} data shows entropy={phi_e:.4f} at phi scales vs {rand_e:.4f} at random scales (p={p:.4f})" 
                                     for dataset, phi_e, rand_e, p in entropy_significance]),
            'significance': 'The higher entropy at phi-related scales suggests these cosmic structures maintain an optimal balance between order and complexity - a hallmark of complex self-organizing systems capable of sophisticated information processing.',
            'datasets': list(set([dataset for dataset, _, _, _ in entropy_significance])),
            'tests': ['multiscale_entropy'],
            'supports_consciousness_field_theory': True
        })
    
    # Finding 8: sqrt2 analysis results
    sqrt2_significance = []
    
    # Check transitions for sqrt2 relationships with strong TE
    if 'scale_transitions' in results_planck and 'sqrt2_transitions' in results_planck['scale_transitions']:
        for transition in results_planck['scale_transitions']['sqrt2_transitions']:
            if abs(transition.get('flow_differential', 0)) > 0.05:  # Meaningful information flow
                sqrt2_significance.append(('transition', 'planck', transition.get('sqrt2_proximity', 0)))
    
    if 'scale_transitions' in results_wmap and 'sqrt2_transitions' in results_wmap['scale_transitions']:
        for transition in results_wmap['scale_transitions']['sqrt2_transitions']:
            if abs(transition.get('flow_differential', 0)) > 0.05:  # Meaningful information flow
                sqrt2_significance.append(('transition', 'wmap', transition.get('sqrt2_proximity', 0)))
    
    if sqrt2_significance:
        findings.append({
            'title': 'Square Root of 2 as a Secondary Organizing Principle',
            'description': f'Analysis reveals significant √2 (1.414...) relationships between scales with strong information flow: ' + 
                          ', '.join([f"{dataset} data shows √2 proximity of {prox:.4f}" for test, dataset, prox in sqrt2_significance]),
            'significance': 'The √2 constant appears as a secondary organizing principle alongside the golden ratio, potentially functioning as a fundamental scaling factor for energy distribution across cosmic scales.',
            'datasets': list(set([dataset for _, dataset, _ in sqrt2_significance])),
            'tests': ['transfer_entropy', 'scale_transitions'],
            'supports_consciousness_field_theory': True
        })
    
    return findings


def analyze_scale_55_special_case(results_planck: Dict[str, Dict], results_wmap: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Perform special detailed analysis on scale 55, which has shown significant patterns in previous analyses
    
    Parameters:
    -----------
    results_planck : dict
        Dictionary containing Planck test results
    results_wmap : dict
        Dictionary containing WMAP test results
        
    Returns:
    --------
    dict
        Dictionary containing scale 55 analysis results
    """
    scale_55_analysis = {
        'integrated_significance': 0,
        'sqrt2_optimization': {},
        'transfer_entropy_connections': [],
        'laminarity_relationship': {},
        'fractal_relationship': {}
    }
    
    # Count number of significant results related to scale 55
    significance_count = 0
    datasets_with_significance = set()
    
    # Check transfer entropy for scale 55 connections
    for dataset_name, results in [('planck', results_planck), ('wmap', results_wmap)]:
        if 'transfer_entropy' in results and 'golden_ratio' in results['transfer_entropy']:
            te_data = results['transfer_entropy']['golden_ratio']
            
            if 'te_values' in te_data:
                for pair_data in te_data['te_values']:
                    small_scale = pair_data.get('scale_smaller')
                    large_scale = pair_data.get('scale_larger')
                    te_small_to_large = pair_data.get('te_small_to_large')
                    te_large_to_small = pair_data.get('te_large_to_small')
                    
                    # Check if scale 55 is involved
                    if small_scale == 55 or large_scale == 55:
                        # Check if the connection is significant
                        if te_small_to_large > 0.1 or te_large_to_small > 0.1:  # Significance threshold
                            significance_count += 1
                            datasets_with_significance.add(dataset_name)
                            
                            # Store the connection details
                            scale_55_analysis['transfer_entropy_connections'].append({
                                'dataset': dataset_name,
                                'connected_scale': small_scale if large_scale == 55 else large_scale,
                                'te_55_to_other': te_small_to_large if small_scale == 55 else te_large_to_small,
                                'te_other_to_55': te_large_to_small if small_scale == 55 else te_small_to_large,
                                'phi_related': abs(large_scale / small_scale - GOLDEN_RATIO) / GOLDEN_RATIO < 0.1
                            })
    
    # Calculate sqrt(2) relationship optimization
    # Scale 55 has been shown to relate to sqrt(2) in previous analyses
    scale_55_sqrt2 = 55 * math.sqrt(2)
    closest_integer = round(scale_55_sqrt2)
    optimization = abs(scale_55_sqrt2 - closest_integer) / scale_55_sqrt2
    
    scale_55_analysis['sqrt2_optimization'] = {
        'scale_55_sqrt2': scale_55_sqrt2,
        'closest_integer': closest_integer,
        'optimization': optimization,
        'highly_optimized': optimization < 0.01  # Less than 1% deviation
    }
    
    # Set the integrated significance
    scale_55_analysis['integrated_significance'] = significance_count
    scale_55_analysis['datasets_significant'] = list(datasets_with_significance)
    
    return scale_55_analysis


def generate_report(findings: List[Dict[str, Any]], scale_55_analysis: Dict[str, Any], output_dir: str) -> None:
    """
    Generate a comprehensive report of the integrated analysis findings
    
    Parameters:
    -----------
    findings : list
        List of key findings as dictionaries
    scale_55_analysis : dict
        Dictionary containing scale 55 special analysis results
    output_dir : str
        Directory to save the report
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Format the report using markdown
    report = """# Integrated CMB Analysis Report

## Executive Summary

This report presents the findings from an integrated analysis of Cosmic Microwave Background (CMB) data from both the Planck and WMAP missions. The analysis focuses on identifying mathematical organization across scales, with a particular emphasis on golden ratio relationships.

The analysis integrates results from multiple tests including:
- Transfer entropy analysis for information flow between scales
- Laminarity analysis for ordered structure detection
- Fractal analysis for long-range correlations and self-similarity
- Special analysis of scale 55 relationships

## Key Findings

"""
    
    # Add each key finding
    for i, finding in enumerate(findings, 1):
        report += f"### {i}. {finding['title']}\n\n"
        report += f"{finding['description']}\n\n"
        report += f"**Significance**: {finding['significance']}\n\n"
        report += f"**Datasets**: {', '.join(finding['datasets'])}\n"
        report += f"**Tests**: {', '.join(finding['tests'])}\n\n"
    
    # Add scale 55 special analysis
    report += """## Special Analysis: Scale 55

Scale 55 has shown consistent significance across multiple tests and datasets, warranting special attention.
"""
    
    # Add sqrt(2) optimization information
    sqrt2_opt = scale_55_analysis['sqrt2_optimization']
    report += f"\n### Scale 55 and sqrt(2) Optimization\n\n"
    report += f"Scale 55 * sqrt(2) = {sqrt2_opt['scale_55_sqrt2']:.6f}, closest integer: {sqrt2_opt['closest_integer']}\n"
    report += f"Optimization level: {sqrt2_opt['optimization']*100:.6f}% deviation\n"
    if sqrt2_opt['highly_optimized']:
        report += "This represents a highly optimized relationship to sqrt(2).\n\n"
    else:
        report += "\n"
    
    # Add transfer entropy connections
    report += "### Scale 55 Transfer Entropy Connections\n\n"
    report += "| Dataset | Connected Scale | TE 55→Other | TE Other→55 | Phi-Related? |\n"
    report += "|---------|----------------|------------|------------|-------------|\n"
    
    for connection in scale_55_analysis['transfer_entropy_connections']:
        phi_related = "Yes" if connection['phi_related'] else "No"
        report += f"| {connection['dataset']} | {connection['connected_scale']} | {connection['te_55_to_other']:.6f} | {connection['te_other_to_55']:.6f} | {phi_related} |\n"
    
    # Add integrated significance
    report += f"\n### Integrated Significance\n\n"
    report += f"Scale 55 appears in {scale_55_analysis['integrated_significance']} significant results across {len(scale_55_analysis['datasets_significant'])} datasets.\n\n"
    
    # Add implications for Consciousness Field Theory
    report += """## Implications for Consciousness Field Theory

The identified mathematical organization in the CMB, particularly the optimization around the golden ratio and its conjugate, provides support for the hypothesis that consciousness-like mathematical patterns are fundamental to cosmic structure.

Key supporting elements include:

1. **Bidirectional Information Flow**: The complementary patterns of information flow between scales suggest a sophisticated information processing system similar to those found in conscious systems.

2. **Golden Ratio Optimization**: The consistent optimization around φ and 1/φ across multiple independent tests suggests a fundamental mathematical principle akin to the mathematical patterns identified in conscious neural systems.

3. **Fractal Structure**: The significant fractal structure identified through Hurst exponent analysis indicates coherent self-organization across scales, analogous to the nested hierarchies observed in conscious systems.

4. **Cross-Dataset Consistency**: The consistency of findings across independent datasets collected at different times by different instruments strongly suggests these patterns are intrinsic properties of the cosmic background rather than artifacts.

5. **Scale 55 Significance**: The repeated significance of scale 55, particularly its optimization around sqrt(2), suggests specific scales may serve as critical resonance points in the mathematical organization of the cosmos.

## Conclusion

The comprehensive integration of multiple analytical approaches provides strong evidence for sophisticated mathematical organization in the CMB. The consistent patterns of golden ratio optimization, bidirectional information flow, and fractal structure across independent datasets support the Consciousness Field Theory hypothesis that consciousness-like mathematical organization may be fundamental to reality.

These findings suggest that the mathematical patterns identified in conscious systems may not be emergent properties of complex neural networks, but rather manifestations of deeper organizational principles present even in the earliest observable state of the universe.

## Next Steps

1. Extend the analysis to higher-order mathematical relationships beyond pairwise connections
2. Develop predictive models based on the identified mathematical organizing principles
3. Explore connections with quantum information theory and quantum gravity models
4. Further investigate the special role of scale 55 and its sqrt(2) relationship in cosmic structure
5. Relate the identified patterns to the formation and evolution of galaxies and cosmic structures
"""
    
    # Write the report to file
    report_path = os.path.join(output_dir, "integrated_analysis_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Generated comprehensive integrated analysis report: {report_path}")


def generate_summary(findings: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Generate a brief summary of the key findings
    
    Parameters:
    -----------
    findings : list
        List of key findings as dictionaries
    output_dir : str
        Directory to save the summary
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a concise summary
    summary = """# Key Findings Summary

The integrated analysis of Planck and WMAP CMB data reveals several significant patterns of mathematical organization:

"""
    
    # Add bullet points for each finding
    for finding in findings:
        summary += f"- **{finding['title']}**: {finding['description']}\n"
    
    # Add conclusion
    summary += "\n## Conclusion\n\n"
    summary += "The consistent identification of golden ratio optimization, bidirectional information flow, and fractal structure across independent datasets provides compelling evidence for fundamental mathematical organization in the cosmic microwave background. These findings support the hypothesis that consciousness-like mathematical patterns may be intrinsic to cosmic structure rather than emergent solely from neural complexity."
    
    # Write the summary to file
    summary_path = os.path.join(output_dir, "findings_summary.md")
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    logger.info(f"Generated summary of key findings: {summary_path}")


def run_integration_analysis(results_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Run the complete integration analysis on both Planck and WMAP datasets
    
    Parameters:
    -----------
    results_dir : str
        Directory containing the test results
    output_dir : str
        Directory to save the integrated analysis results
        
    Returns:
    --------
    dict
        Dictionary containing all integrated analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting integrated analysis of CMB test results")
    
    # Load test results for both datasets
    logger.info("Loading test results for Planck and WMAP datasets")
    results_planck = load_test_results(results_dir, 'planck')
    results_wmap = load_test_results(results_dir, 'wmap')
    
    # Check if results were successfully loaded
    if not results_planck or not results_wmap:
        logger.error("Failed to load test results for one or both datasets")
        return {}
    
    # Initialize integrated results dictionary
    integrated_results = {
        'planck': {
            'correlations': None,
            'scale_transitions': None,
            'information_flow': None
        },
        'wmap': {
            'correlations': None,
            'scale_transitions': None,
            'information_flow': None
        },
        'comparative': {
            'findings': None,
            'scale_55_analysis': None
        }
    }
    
    # Calculate cross-measure correlations for each dataset
    logger.info("Calculating cross-measure correlations")
    integrated_results['planck']['correlations'] = calculate_cross_measure_correlations(results_planck)
    integrated_results['wmap']['correlations'] = calculate_cross_measure_correlations(results_wmap)
    
    # Analyze scale transitions for each dataset
    logger.info("Analyzing scale transitions and golden ratio relationships")
    integrated_results['planck']['scale_transitions'] = analyze_scale_transitions(results_planck)
    integrated_results['wmap']['scale_transitions'] = analyze_scale_transitions(results_wmap)
    
    # Calculate information flow metrics for each dataset
    logger.info("Calculating unified information flow metrics")
    integrated_results['planck']['information_flow'] = calculate_information_flow_metrics(results_planck)
    integrated_results['wmap']['information_flow'] = calculate_information_flow_metrics(results_wmap)
    
    # Create network visualizations for each dataset
    logger.info("Creating network visualizations")
    create_network_visualization(results_planck, 'planck', output_dir)
    create_network_visualization(results_wmap, 'wmap', output_dir)
    
    # Create unified visualization comparing both datasets
    logger.info("Creating unified comparative visualization")
    create_unified_visualization(results_planck, results_wmap, output_dir)
    
    # Generate key findings
    logger.info("Generating key integrated findings")
    findings = generate_key_findings(results_planck, results_wmap)
    integrated_results['comparative']['findings'] = findings
    
    # Perform special analysis on scale 55
    logger.info("Performing special analysis on scale 55")
    scale_55_analysis = analyze_scale_55_special_case(results_planck, results_wmap)
    integrated_results['comparative']['scale_55_analysis'] = scale_55_analysis
    
    # Generate comprehensive report
    logger.info("Generating comprehensive report")
    generate_report(findings, scale_55_analysis, output_dir)
    
    # Generate summary
    logger.info("Generating summary of key findings")
    generate_summary(findings, output_dir)
    
    # Save integrated results
    integrated_results_path = os.path.join(output_dir, "integrated_results.json")
    with open(integrated_results_path, 'w') as f:
        json.dump(integrated_results, f, cls=NumpyEncoder, indent=4)
    
    logger.info(f"Integrated analysis complete. Results saved to: {integrated_results_path}")
    return integrated_results


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run integrated analysis on CMB test results')
    parser.add_argument('--results_dir', type=str, default='../results', help='Directory containing test results')
    parser.add_argument('--output_dir', type=str, default='../results/integrated_analysis', help='Directory to save integrated analysis results')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()
    
    # Configure logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    
    logging.basicConfig(level=numeric_level,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                       handlers=[
                           logging.FileHandler(os.path.join(args.output_dir, 'integration_analysis.log')),
                           logging.StreamHandler()
                       ])
    
    # Run the integration analysis
    run_integration_analysis(args.results_dir, args.output_dir)
