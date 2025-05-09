#!/usr/bin/env python3
"""
Dual Dataset KSG Transfer Entropy Analysis for CMB Data

This script runs the enhanced KSG-based transfer entropy analysis on both
WMAP and Planck datasets simultaneously with consistent protocols.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import KSG transfer entropy module
from cmb.transfer_entropy_ksg import enhanced_transfer_entropy_analysis

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder for numpy values
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_cmb_data(file_path):
    """Load CMB power spectrum data from a file"""
    try:
        data = np.loadtxt(file_path, delimiter=None)
        
        # Ensure we have at least l and Cl
        if data.shape[1] < 2:
            raise ValueError("Data must have at least l and Cl columns")
            
        # If no error column, add zeros
        if data.shape[1] == 2:
            logger.warning(f"No error column found in {file_path}. Adding zeros.")
            errors = np.zeros((data.shape[0], 1))
            data = np.hstack((data, errors))
            
        return data
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def generate_scale_pairs():
    """Generate scale pairs based on mathematical relationships"""
    # Generate Fibonacci sequence
    fibonacci = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    
    # Generate powers of 2
    powers_of_two = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    # Key scales identified in previous research
    key_scales = [55, 89, 144, 233, 377]
    
    pairs = []
    pair_types = {}
    
    # Add consecutive Fibonacci pairs
    for i in range(len(fibonacci) - 1):
        pairs.append((fibonacci[i], fibonacci[i+1]))
        pair_types[(fibonacci[i], fibonacci[i+1])] = ["fibonacci_consecutive"]
    
    # Add consecutive powers of 2 pairs
    for i in range(len(powers_of_two) - 1):
        pairs.append((powers_of_two[i], powers_of_two[i+1]))
        pair_types[(powers_of_two[i], powers_of_two[i+1])] = ["power_of_two_consecutive"]
    
    # Add key scale combinations
    for scale_a in key_scales:
        for scale_b in fibonacci + powers_of_two:
            if scale_a != scale_b and (scale_a, scale_b) not in pairs and (scale_b, scale_a) not in pairs:
                pairs.append((scale_a, scale_b))
                pair_types[(scale_a, scale_b)] = ["key_scale_pair"]
    
    # Add pairs with golden ratio relationship
    golden_ratio = (1 + np.sqrt(5)) / 2
    sqrt2 = np.sqrt(2)
    
    # Find pairs that approximate phi
    for scale_a in range(2, 400):
        scale_b = int(round(scale_a * golden_ratio))
        if scale_b <= 610:
            ratio = scale_b / scale_a
            if abs(ratio - golden_ratio) / golden_ratio < 0.01:  # Within 1% of golden ratio
                pairs.append((scale_a, scale_b))
                pair_types[(scale_a, scale_b)] = ["golden_ratio"]
    
    # Find pairs that approximate sqrt(2)
    for scale_a in range(2, 400):
        scale_b = int(round(scale_a * sqrt2))
        if scale_b <= 610:
            ratio = scale_b / scale_a
            if abs(ratio - sqrt2) / sqrt2 < 0.01:  # Within 1% of sqrt(2)
                pairs.append((scale_a, scale_b))
                pair_types[(scale_a, scale_b)] = ["sqrt2"]
    
    # Remove duplicates and sort
    unique_pairs = list(set(pairs))
    unique_pairs.sort()
    
    return unique_pairs, pair_types

def run_dual_ksg_analysis(n_surrogates=10000):
    """Run KSG transfer entropy analysis on both WMAP and Planck datasets"""
    start_time = time.time()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"ksg_transfer_entropy_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    planck_file = os.path.join(project_root, 'data/processed/planck_power_spectrum.txt')
    wmap_file = os.path.join(project_root, 'data/processed/wmap_power_spectrum.txt')
    
    logger.info("Running KSG Transfer Entropy Analysis on WMAP and Planck datasets")
    logger.info(f"Number of surrogates: {n_surrogates}")
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Load datasets
    logger.info(f"Loading Planck data from {planck_file}")
    planck_data = load_cmb_data(planck_file)
    logger.info(f"Loaded Planck data with {len(planck_data)} multipoles")
    
    logger.info(f"Loading WMAP data from {wmap_file}")
    wmap_data = load_cmb_data(wmap_file)
    logger.info(f"Loaded WMAP data with {len(wmap_data)} multipoles")
    
    # Generate scale pairs
    scale_pairs, pair_types = generate_scale_pairs()
    logger.info(f"Generated {len(scale_pairs)} scale pairs to test")
    
    # Filter pairs based on available scales
    planck_scales = set(planck_data[:, 0].astype(int))
    wmap_scales = set(wmap_data[:, 0].astype(int))
    available_pairs = []
    
    for scale_a, scale_b in scale_pairs:
        if scale_a in planck_scales and scale_b in planck_scales and \
           scale_a in wmap_scales and scale_b in wmap_scales:
            available_pairs.append((scale_a, scale_b))
    
    logger.info(f"{len(available_pairs)} pairs available in both datasets")
    
    # Create subdirectories for each dataset
    planck_dir = os.path.join(output_dir, "planck")
    wmap_dir = os.path.join(output_dir, "wmap")
    os.makedirs(planck_dir, exist_ok=True)
    os.makedirs(wmap_dir, exist_ok=True)
    
    # Run analysis on Planck data
    logger.info("\n" + "="*80)
    logger.info("ANALYZING PLANCK DATASET")
    logger.info("="*80)
    
    planck_results = enhanced_transfer_entropy_analysis(
        planck_data, 
        available_pairs, 
        n_surrogates=n_surrogates
    )
    
    # Add pair types to results
    if not planck_results.empty:
        planck_results['pair_types'] = planck_results.apply(
            lambda row: ', '.join(pair_types.get((row['scale_a'], row['scale_b']), [])),
            axis=1
        )
    
    # Save Planck results
    planck_results.to_csv(os.path.join(planck_dir, "planck_ksg_te_results.csv"), index=False)
    
    # Run analysis on WMAP data
    logger.info("\n" + "="*80)
    logger.info("ANALYZING WMAP DATASET")
    logger.info("="*80)
    
    wmap_results = enhanced_transfer_entropy_analysis(
        wmap_data, 
        available_pairs, 
        n_surrogates=n_surrogates
    )
    
    # Add pair types to results
    if not wmap_results.empty:
        wmap_results['pair_types'] = wmap_results.apply(
            lambda row: ', '.join(pair_types.get((row['scale_a'], row['scale_b']), [])),
            axis=1
        )
    
    # Save WMAP results
    wmap_results.to_csv(os.path.join(wmap_dir, "wmap_ksg_te_results.csv"), index=False)
    
    # Compare results between datasets
    logger.info("\n" + "="*80)
    logger.info("COMPARING RESULTS ACROSS DATASETS")
    logger.info("="*80)
    
    # Get significant results
    if not planck_results.empty:
        planck_sig = planck_results[planck_results["any_significant"]]
    else:
        planck_sig = pd.DataFrame()
    
    if not wmap_results.empty:
        wmap_sig = wmap_results[wmap_results["any_significant"]]
    else:
        wmap_sig = pd.DataFrame()
    
    logger.info(f"Planck significant pairs: {len(planck_sig)}/{len(planck_results)} ({len(planck_sig)/max(1, len(planck_results))*100:.1f}%)")
    logger.info(f"WMAP significant pairs: {len(wmap_sig)}/{len(wmap_results)} ({len(wmap_sig)/max(1, len(wmap_results))*100:.1f}%)")
    
    # Find common significant pairs
    if not planck_sig.empty and not wmap_sig.empty:
        planck_sig_pairs = set(tuple(row) for _, row in planck_sig[["scale_a", "scale_b"]].iterrows())
        wmap_sig_pairs = set(tuple(row) for _, row in wmap_sig[["scale_a", "scale_b"]].iterrows())
        common_pairs = planck_sig_pairs.intersection(wmap_sig_pairs)
        
        logger.info(f"Common significant pairs: {len(common_pairs)}")
        
        if common_pairs:
            logger.info("\nCommon significant scale pairs:")
            for scale_a, scale_b in common_pairs:
                p_row = planck_results[(planck_results["scale_a"] == scale_a) & (planck_results["scale_b"] == scale_b)].iloc[0]
                w_row = wmap_results[(wmap_results["scale_a"] == scale_a) & (wmap_results["scale_b"] == scale_b)].iloc[0]
                
                logger.info(f"Scale {scale_a}-{scale_b} (ratio: {scale_b/scale_a:.4f}):")
                logger.info(f"  Planck: Forward Z={p_row['forward_z']:.4f}, Reverse Z={p_row['reverse_z']:.4f}")
                logger.info(f"  WMAP: Forward Z={w_row['forward_z']:.4f}, Reverse Z={w_row['reverse_z']:.4f}")
                logger.info(f"  Types: {p_row['pair_types']}")
        
        # Save combined results
        combined_results = {
            "planck_significant_count": len(planck_sig),
            "wmap_significant_count": len(wmap_sig),
            "common_significant_count": len(common_pairs),
            "common_pairs": [{"scale_a": int(a), "scale_b": int(b), "ratio": float(b/a)} for a, b in common_pairs],
            "n_surrogates": n_surrogates,
            "runtime_minutes": (time.time() - start_time) / 60
        }
        
        combined_file = os.path.join(output_dir, "combined_results.json")
        with open(combined_file, 'w') as f:
            json.dump(combined_results, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Combined results saved to {combined_file}")
    else:
        logger.info("No common significant pairs found (one or both datasets had no significant results)")
    
    # Create visualization of results
    try:
        create_comparison_visualization(planck_results, wmap_results, output_dir)
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
    
    runtime_minutes = (time.time() - start_time) / 60
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Total runtime: {runtime_minutes:.2f} minutes")
    logger.info("="*80)
    
    return output_dir

def create_comparison_visualization(planck_results, wmap_results, output_dir):
    """Create visualizations comparing results between datasets"""
    if planck_results.empty or wmap_results.empty:
        return  # Skip if any dataset has no results
    
    # Create scatter plot of z-scores by ratio
    plt.figure(figsize=(15, 10))
    
    # Plot Planck forward z-scores
    plt.scatter(planck_results['ratio'], planck_results['forward_z'], 
               alpha=0.7, s=50, marker='o', label="Planck Forward TE", c='blue')
    
    # Plot Planck reverse z-scores
    plt.scatter(1/planck_results['ratio'], planck_results['reverse_z'], 
               alpha=0.7, s=50, marker='s', label="Planck Reverse TE", c='lightblue')
    
    # Plot WMAP forward z-scores
    plt.scatter(wmap_results['ratio'], wmap_results['forward_z'], 
               alpha=0.7, s=50, marker='^', label="WMAP Forward TE", c='red')
    
    # Plot WMAP reverse z-scores
    plt.scatter(1/wmap_results['ratio'], wmap_results['reverse_z'], 
               alpha=0.7, s=50, marker='v', label="WMAP Reverse TE", c='lightcoral')
    
    # Add significant thresholds
    plt.axhline(y=1.96, color='green', linestyle='--', alpha=0.8, label="p=0.05 (Z=1.96)")
    plt.axhline(y=-1.96, color='green', linestyle='--', alpha=0.8)
    
    # Add mathematical constant lines
    plt.axvline(x=(1 + np.sqrt(5)) / 2, color='gold', linestyle='--', alpha=0.8, label="Golden Ratio (φ)")
    plt.axvline(x=np.sqrt(2), color='purple', linestyle='--', alpha=0.8, label="√2")
    plt.axvline(x=np.pi/2, color='brown', linestyle='--', alpha=0.8, label="π/2")
    plt.axvline(x=np.e/2, color='orange', linestyle='--', alpha=0.8, label="e/2")
    
    plt.xlabel("Scale Ratio")
    plt.ylabel("Z-Score")
    plt.title("Transfer Entropy Z-Scores by Scale Ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_z_scores.png"), dpi=300)
    plt.close()
    
    # Create bar chart of significant pairs by type
    planck_sig = planck_results[planck_results['any_significant']]
    wmap_sig = wmap_results[wmap_results['any_significant']]
    
    if len(planck_sig) == 0 and len(wmap_sig) == 0:
        return  # Skip if no significant results
    
    # Collect types from significant pairs
    all_types = set()
    type_counts = {'Planck': {}, 'WMAP': {}}
    
    for _, row in planck_sig.iterrows():
        types = row['pair_types'].split(', ')
        for t in types:
            if t:
                all_types.add(t)
                type_counts['Planck'][t] = type_counts['Planck'].get(t, 0) + 1
    
    for _, row in wmap_sig.iterrows():
        types = row['pair_types'].split(', ')
        for t in types:
            if t:
                all_types.add(t)
                type_counts['WMAP'][t] = type_counts['WMAP'].get(t, 0) + 1
    
    # Create the bar chart
    if all_types:
        all_types = sorted(list(all_types))
        
        plt.figure(figsize=(14, 8))
        x = np.arange(len(all_types))
        width = 0.35
        
        planck_values = [type_counts['Planck'].get(t, 0) for t in all_types]
        wmap_values = [type_counts['WMAP'].get(t, 0) for t in all_types]
        
        plt.bar(x - width/2, planck_values, width, label='Planck', color='blue', alpha=0.7)
        plt.bar(x + width/2, wmap_values, width, label='WMAP', color='red', alpha=0.7)
        
        plt.xlabel('Pair Type')
        plt.ylabel('Number of Significant Pairs')
        plt.title('Significant Pairs by Type and Dataset')
        plt.xticks(x, all_types, rotation=45, ha='right')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "significant_pairs_by_type.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run KSG Transfer Entropy Analysis on both WMAP and Planck datasets")
    parser.add_argument("--surrogates", type=int, default=10000, help="Number of surrogate datasets (default: 10000)")
    
    args = parser.parse_args()
    
    run_dual_ksg_analysis(n_surrogates=args.surrogates)
