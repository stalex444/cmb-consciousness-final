#!/usr/bin/env python3
"""
Analyze KSG Transfer Entropy Results with Benjamini-Hochberg Correction

This script applies the Benjamini-Hochberg correction to the transfer entropy
results and analyzes the significant pairs after correction, focusing especially
on mathematical constants like the Golden Ratio and Square Root of 2.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import BH correction module
from cmb.bh_correction import apply_bh_to_te_results, analyze_bh_corrected_results

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

def find_latest_te_results():
    """Find the most recent transfer entropy results directory"""
    results_dir = os.path.join(project_root, 'results')
    te_dirs = [d for d in os.listdir(results_dir) if d.startswith('ksg_transfer_entropy_')]
    
    if not te_dirs:
        raise FileNotFoundError("No transfer entropy results found")
        
    # Sort by timestamp (latest first)
    te_dirs.sort(reverse=True)
    
    return os.path.join(results_dir, te_dirs[0])

def create_bh_visualizations(planck_df, wmap_df, output_dir):
    """Create visualizations of BH-corrected results"""
    # Extract significant pairs after BH correction
    planck_sig = planck_df[planck_df['any_significant_bh']]
    wmap_sig = wmap_df[wmap_df['any_significant_bh']]
    
    # Z-score vs Ratio scatter plot
    plt.figure(figsize=(15, 10))
    
    # Plot Planck forward z-scores
    if not planck_sig.empty:
        plt.scatter(planck_sig['ratio'], planck_sig['forward_z'], 
                   alpha=0.7, s=50, marker='o', label="Planck Forward TE", c='blue')
        
        # Plot Planck reverse z-scores
        plt.scatter(1/planck_sig['ratio'], planck_sig['reverse_z'], 
                   alpha=0.7, s=50, marker='s', label="Planck Reverse TE", c='lightblue')
    
    # Plot WMAP forward z-scores
    if not wmap_sig.empty:
        plt.scatter(wmap_sig['ratio'], wmap_sig['forward_z'], 
                   alpha=0.7, s=50, marker='^', label="WMAP Forward TE", c='red')
        
        # Plot WMAP reverse z-scores
        plt.scatter(1/wmap_sig['ratio'], wmap_sig['reverse_z'], 
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
    plt.title("BH-Corrected Transfer Entropy Z-Scores by Scale Ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bh_corrected_z_scores.png"), dpi=300)
    plt.close()
    
    # Create bar chart of significant pairs by mathematical relationship
    # Define proximity ranges
    golden_ratio = (1 + np.sqrt(5)) / 2
    sqrt2 = np.sqrt(2)
    
    # Function to classify ratio
    def classify_ratio(ratio):
        if abs(ratio - golden_ratio) < 0.01:
            return "Golden Ratio (φ)"
        elif abs(ratio - sqrt2) < 0.01:
            return "Square Root of 2 (√2)"
        elif abs(ratio - 2.0) < 0.01:
            return "Power of 2"
        else:
            return "Other"
    
    # Add classification column to dataframes
    if not planck_sig.empty:
        planck_sig['ratio_class'] = planck_sig['ratio'].apply(classify_ratio)
        planck_counts = planck_sig['ratio_class'].value_counts()
    else:
        planck_counts = pd.Series(dtype=int)
    
    if not wmap_sig.empty:
        wmap_sig['ratio_class'] = wmap_sig['ratio'].apply(classify_ratio)
        wmap_counts = wmap_sig['ratio_class'].value_counts()
    else:
        wmap_counts = pd.Series(dtype=int)
    
    # Get all categories
    all_categories = set(list(planck_counts.index) + list(wmap_counts.index))
    
    if all_categories:
        # Sort categories logically
        category_order = ["Golden Ratio (φ)", "Square Root of 2 (√2)", "Power of 2", "Other"]
        all_categories = [c for c in category_order if c in all_categories]
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        # Set up bar positions
        x = np.arange(len(all_categories))
        width = 0.35
        
        # Get counts
        planck_values = [planck_counts.get(cat, 0) for cat in all_categories]
        wmap_values = [wmap_counts.get(cat, 0) for cat in all_categories]
        
        # Create bars
        plt.bar(x - width/2, planck_values, width, label='Planck', color='blue', alpha=0.7)
        plt.bar(x + width/2, wmap_values, width, label='WMAP', color='red', alpha=0.7)
        
        plt.xlabel('Mathematical Relationship')
        plt.ylabel('Number of Significant Pairs (BH-corrected)')
        plt.title('Significant Pairs by Mathematical Relationship after BH Correction')
        plt.xticks(x, all_categories)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bh_corrected_pairs_by_type.png"), dpi=300)
        plt.close()

def run_bh_analysis(results_dir=None, alpha=0.05):
    """Run Benjamini-Hochberg analysis on transfer entropy results"""
    # Find latest results if not specified
    if results_dir is None:
        results_dir = find_latest_te_results()
    
    print(f"Analyzing results from: {results_dir}")
    
    # Create output directory for BH-corrected results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"bh_corrected_te_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Planck and WMAP results
    planck_file = os.path.join(results_dir, "planck", "planck_ksg_te_results.csv")
    wmap_file = os.path.join(results_dir, "wmap", "wmap_ksg_te_results.csv")
    
    if not os.path.exists(planck_file) or not os.path.exists(wmap_file):
        raise FileNotFoundError(f"Results files not found in {results_dir}")
    
    planck_results = pd.read_csv(planck_file)
    wmap_results = pd.read_csv(wmap_file)
    
    print(f"Loaded {len(planck_results)} Planck pairs and {len(wmap_results)} WMAP pairs")
    
    # Apply BH correction and analyze
    print(f"Applying Benjamini-Hochberg correction with alpha={alpha}...")
    bh_results = analyze_bh_corrected_results(planck_results, wmap_results, alpha)
    
    # Print summary
    print(f"\nSUMMARY OF BH-CORRECTED RESULTS:")
    print(f"Planck significant pairs: {bh_results['planck_total_significant']}/{len(planck_results)} ({bh_results['planck_percent_significant']:.1f}%)")
    print(f"WMAP significant pairs: {bh_results['wmap_total_significant']}/{len(wmap_results)} ({bh_results['wmap_percent_significant']:.1f}%)")
    print(f"Common significant pairs: {bh_results['common_significant_count']}")
    
    # Print details of common significant pairs
    if bh_results['common_significant_details']:
        print("\nCommon Significant Pairs Details:")
        for pair in bh_results['common_significant_details']:
            print(f"Scale {pair['scale_a']}-{pair['scale_b']} (ratio: {pair['ratio']:.4f})")
            print(f"  Planck: Forward Z={pair['planck_forward_z']:.4f}, Reverse Z={pair['planck_reverse_z']:.4f}")
            print(f"  WMAP: Forward Z={pair['wmap_forward_z']:.4f}, Reverse Z={pair['wmap_reverse_z']:.4f}")
    
    # Print golden ratio relationships
    print(f"\nPlanck Golden Ratio Pairs: {len(bh_results['planck_golden_ratio_pairs'])}")
    for pair in bh_results['planck_golden_ratio_pairs']:
        print(f"  Scale {pair[0]}-{pair[1]} (ratio: {pair[2]:.4f})")
    
    print(f"\nWMAP Golden Ratio Pairs: {len(bh_results['wmap_golden_ratio_pairs'])}")
    for pair in bh_results['wmap_golden_ratio_pairs']:
        print(f"  Scale {pair[0]}-{pair[1]} (ratio: {pair[2]:.4f})")
    
    # Print square root of 2 relationships
    print(f"\nPlanck Square Root of 2 Pairs: {len(bh_results['planck_sqrt2_pairs'])}")
    for pair in bh_results['planck_sqrt2_pairs']:
        print(f"  Scale {pair[0]}-{pair[1]} (ratio: {pair[2]:.4f})")
    
    print(f"\nWMAP Square Root of 2 Pairs: {len(bh_results['wmap_sqrt2_pairs'])}")
    for pair in bh_results['wmap_sqrt2_pairs']:
        print(f"  Scale {pair[0]}-{pair[1]} (ratio: {pair[2]:.4f})")
    
    # Save BH-corrected results
    planck_bh = bh_results['planck_df']
    wmap_bh = bh_results['wmap_df']
    
    planck_bh.to_csv(os.path.join(output_dir, "planck_bh_corrected.csv"), index=False)
    wmap_bh.to_csv(os.path.join(output_dir, "wmap_bh_corrected.csv"), index=False)
    
    # Save summary results to JSON
    summary = {
        "planck_significant_count": int(bh_results['planck_total_significant']),
        "wmap_significant_count": int(bh_results['wmap_total_significant']),
        "common_significant_count": bh_results['common_significant_count'],
        "common_pairs": [
            {
                "scale_a": int(pair['scale_a']), 
                "scale_b": int(pair['scale_b']), 
                "ratio": float(pair['ratio'])
            } 
            for pair in bh_results['common_significant_details']
        ],
        "planck_golden_ratio_pairs": [
            {"scale_a": int(a), "scale_b": int(b), "ratio": float(r)} 
            for a, b, r in bh_results['planck_golden_ratio_pairs']
        ],
        "wmap_golden_ratio_pairs": [
            {"scale_a": int(a), "scale_b": int(b), "ratio": float(r)} 
            for a, b, r in bh_results['wmap_golden_ratio_pairs']
        ],
        "planck_sqrt2_pairs": [
            {"scale_a": int(a), "scale_b": int(b), "ratio": float(r)} 
            for a, b, r in bh_results['planck_sqrt2_pairs']
        ],
        "wmap_sqrt2_pairs": [
            {"scale_a": int(a), "scale_b": int(b), "ratio": float(r)} 
            for a, b, r in bh_results['wmap_sqrt2_pairs']
        ],
        "alpha": alpha
    }
    
    with open(os.path.join(output_dir, "bh_corrected_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    # Create visualizations
    create_bh_visualizations(planck_bh, wmap_bh, output_dir)
    
    print(f"\nBH-corrected results saved to: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply Benjamini-Hochberg correction to transfer entropy results")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing transfer entropy results (default: most recent)")
    parser.add_argument("--alpha", type=float, default=0.05, help="False discovery rate (default: 0.05)")
    
    args = parser.parse_args()
    
    run_bh_analysis(args.results_dir, args.alpha)
