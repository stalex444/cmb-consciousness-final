#!/usr/bin/env python3
"""
Standalone Transfer Entropy Test for CMB Data

This script runs a focused transfer entropy analysis on both WMAP and Planck datasets
simultaneously with 10,000 simulations. This analysis examines information flow 
between scales in the CMB data, with special focus on scales related by 
mathematical constants like the Golden Ratio and Square Root of 2.

Uses improved transfer entropy module with:
- Adaptive binning based on data size
- Enhanced synthetic time series generation
- Gaussian p-values for extreme significance
- More robust statistical validation
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import json
from datetime import datetime
import concurrent.futures
from scipy.stats import norm

# Get project root for robust path handling
project_root = os.path.abspath(os.path.dirname(__file__))

# Add project root to Python path
sys.path.insert(0, project_root)

# Configure timestamped logging to prevent overwrites
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"transfer_entropy_test_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import improved transfer entropy functions
try:
    from cmb.improved_transfer_entropy import (
        calculate_transfer_entropy,
        calculate_te_significance,
        identify_scale_pairs_by_constant
    )
    logger.info("Using improved_transfer_entropy with adaptive binning, enhanced synthetic series, Gaussian p-values")
except ImportError as e:
    logger.error(f"Failed to import improved transfer entropy module: {e}")
    sys.exit(1)

def validate_data(data, name, focus_scales=None):
    """
    Validate the CMB power spectrum data
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data array to validate
    name : str
        Name of the dataset for error messages
    focus_scales : list, optional
        List of scales to check for presence
    
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    # Check data shape and values
    if data.shape[1] < 2:
        logger.error(f"Invalid {name} data: requires at least 2 columns [l, Cl]")
        return False
    
    # Check for NaN or infinite values
    if np.any(np.isnan(data[:, 1])) or np.any(np.isinf(data[:, 1])):
        logger.error(f"{name} data contains NaN or infinite Cl values")
        return False
    
    # Check for too many negative power spectrum values
    neg_count = np.sum(data[:, 1] < 0)
    if neg_count > 0:
        neg_percent = (neg_count / len(data)) * 100
        logger.warning(f"{name} data contains {neg_count} negative Cl values ({neg_percent:.1f}%)")
        if neg_percent > 30:
            logger.error(f"{name} data has too many negative values ({neg_percent:.1f}%). Check data processing.")
            return False
    
    # Check for sufficient multipoles
    if len(data) < 500:
        logger.error(f"{name} data too small: {len(data)} multipoles. Required: >= 500")
        return False
    
    # Check for presence of focus scales
    if focus_scales:
        available = [int(l) for l in data[:, 0]]
        missing = [s for s in focus_scales if s not in available]
        if missing:
            logger.warning(f"{name} data missing focus scales: {missing}")
    
    return True

def generate_surrogate(data, scale_val, noise_scale=0.005):
    """
    Generate a surrogate value for a specific scale using phase randomization
    
    Parameters:
    -----------
    data : numpy.ndarray
        Full CMB power spectrum data
    scale_val : int
        Multipole (l) to generate surrogate for
    noise_scale : float
        Noise scale for additional randomization
    
    Returns:
    --------
    float
        Surrogate value for the specified scale
    """
    cl = data[:, 1].copy()
    n = len(cl)
    fft_vals = np.fft.rfft(cl, n=2**int(np.ceil(np.log2(n))))
    magnitudes = np.abs(fft_vals)
    phases = np.random.uniform(0, 2*np.pi, len(magnitudes))
    fft_vals_new = magnitudes * np.exp(1j * phases)
    surrogate_cl = np.fft.irfft(fft_vals_new, n=2**int(np.ceil(np.log2(n))))[:n]
    surrogate_cl = (surrogate_cl - np.mean(surrogate_cl)) / np.std(surrogate_cl) * np.std(cl) + np.mean(cl)
    idx = np.where(data[:, 0] == scale_val)[0][0]
    return surrogate_cl[idx] + np.random.normal(0, noise_scale * abs(surrogate_cl[idx]) or 0.005)

def run_transfer_entropy_analysis(data, output_dir, dataset_type, n_surrogates=10000, focus_scales=None):
    """
    Run transfer entropy analysis on CMB data
    
    Parameters:
    -----------
    data : numpy.ndarray
        Power spectrum data
    output_dir : str
        Directory to save results
    dataset_type : str
        Type of dataset (planck or wmap)
    n_surrogates : int
        Number of surrogate datasets to generate
    focus_scales : list
        Optional list of scales to focus on
        
    Returns:
    --------
    dict
        Analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default key scales to focus on if none provided
    if focus_scales is None:
        focus_scales = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    
    # Identify φ and √2 related scale pairs
    scale_pairs = identify_scale_pairs_by_constant(data, tolerance=0.05)
    phi_pairs = [(p[0], p[1]) for p in scale_pairs.get("golden_ratio", []) if p[0] in focus_scales and p[1] in focus_scales]
    sqrt2_pairs = [(p[0], p[1]) for p in scale_pairs.get("sqrt2", []) if p[0] in focus_scales and p[1] in focus_scales]
    priority_pairs = [(s1, s2) for s1, s2 in phi_pairs + sqrt2_pairs if s1 == 55 or s2 == 55]
    
    # Step 1: Identify key scales in the data
    available_scales = sorted([int(l) for l in data[:, 0]])
    scales_to_analyze = sorted(set([s for pair in priority_pairs for s in pair] + [s for s in focus_scales if s in available_scales]))
    
    if not scales_to_analyze:
        logger.error(f"None of the focus scales {focus_scales} found in {dataset_type} data")
        return {}
    
    logger.info(f"Analyzing {len(scales_to_analyze)} scales in {dataset_type} data: {scales_to_analyze}")
    
    # Step 2: Calculate transfer entropy between all pairs of key scales
    results = {"priority_pairs": {"phi": phi_pairs, "sqrt2": sqrt2_pairs}}
    total_pairs = len(scales_to_analyze) * (len(scales_to_analyze) - 1) // 2
    pair_counter = 0
    
    logger.info(f"Analyzing {total_pairs} scale pairs with {n_surrogates} surrogates each")
    start_time = time.time()
    
    # Create results structure
    for i, scale1 in enumerate(scales_to_analyze):
        for j, scale2 in enumerate(scales_to_analyze[i+1:], i+1):
            pair_counter += 1
            
            # Extract power spectrum values for the scales
            scale1_idx = np.where(data[:, 0] == scale1)[0][0]
            scale2_idx = np.where(data[:, 0] == scale2)[0][0]
            
            scale1_val = data[scale1_idx, 1]
            scale2_val = data[scale2_idx, 1]
            
            # Calculate TE in both directions
            te_forward = calculate_transfer_entropy(scale1_val, scale2_val)
            te_reverse = calculate_transfer_entropy(scale2_val, scale1_val)
            
            # Determine dominant direction
            if abs(te_forward) > abs(te_reverse):
                te_value = te_forward
                direction = f"{scale1} → {scale2}"
                source_val = scale1_val
                target_val = scale2_val
            else:
                te_value = te_reverse
                direction = f"{scale2} → {scale1}"
                source_val = scale2_val
                target_val = scale1_val
            
            # Generate surrogate data for significance testing
            surrogate_tes = []
            noise_scale_source = abs(source_val) * 0.005 if source_val != 0 else 0.005
            noise_scale_target = abs(target_val) * 0.005 if target_val != 0 else 0.005
            
            for k in range(n_surrogates):
                # Generate surrogate with phase randomization
                surrogate_source = generate_surrogate(data, scale1, noise_scale_source)
                surrogate_target = generate_surrogate(data, scale2, noise_scale_target)
                
                # Calculate TE for surrogate
                surrogate_te = calculate_transfer_entropy(surrogate_source, surrogate_target)
                surrogate_tes.append(surrogate_te)
                
                # Show progress every 1000 surrogates
                if (k + 1) % 1000 == 0 and k > 0:
                    elapsed = time.time() - start_time
                    pairs_per_sec = pair_counter / elapsed
                    surrogates_per_sec = (pair_counter * (k + 1)) / elapsed
                    remaining_pairs = total_pairs - pair_counter
                    remaining_surrogates = (total_pairs * n_surrogates) - (pair_counter * (k + 1)) - (remaining_pairs * (k + 1))
                    remaining_time = remaining_surrogates / surrogates_per_sec / 60  # minutes
                    
                    logger.info(f"Progress: Pair {pair_counter}/{total_pairs}, Surrogate {k+1}/{n_surrogates} - "
                               f"Est. {remaining_time:.1f} minutes remaining")
            
            # Calculate significance
            p_value, z_score, effect_size = calculate_te_significance(te_value, surrogate_tes)
            
            # Store results
            pair_key = f"{scale1}-{scale2}"
            results[pair_key] = {
                "scale1": scale1,
                "scale2": scale2,
                "dominant_direction": direction,
                "te_value": te_value,
                "p_value": p_value,
                "z_score": z_score,
                "effect_size": effect_size,
                "n_surrogates": n_surrogates,
                "significant": p_value < 0.05
            }
            
            # Log significant results immediately
            if p_value < 0.05:
                logger.info(f"Significant result: {direction}, TE={te_value:.4f}, p={p_value:.6f}, z={z_score:.4f}")
            
            # Show progress
            if pair_counter % 20 == 0 or pair_counter == total_pairs:
                elapsed = time.time() - start_time
                pairs_per_sec = pair_counter / elapsed
                remaining_pairs = total_pairs - pair_counter
                remaining_time = remaining_pairs / pairs_per_sec / 60  # minutes
                
                logger.info(f"Progress: {pair_counter}/{total_pairs} pairs processed "
                          f"({pair_counter/total_pairs*100:.1f}%) - "
                          f"Est. {remaining_time:.1f} minutes remaining")
    
    # Step 3: Identify significant results
    significant_pairs = {k: v for k, v in results.items() if isinstance(v, dict) and v.get('significant')}
    
    # Step 4: Save results
    results_summary = {
        "dataset_type": dataset_type,
        "n_surrogates": n_surrogates,
        "scales_analyzed": scales_to_analyze,
        "significant_count": len(significant_pairs),
        "total_pairs": total_pairs,
        "most_significant": sorted(
            list(significant_pairs.values()), 
            key=lambda x: x['p_value']
        )[:10] if significant_pairs else [],
        "runtime_minutes": (time.time() - start_time) / 60
    }
    
    # Save detailed results
    detail_file = os.path.join(output_dir, f"{dataset_type}_transfer_entropy_detailed.json")
    with open(detail_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Save summary results
    summary_file = os.path.join(output_dir, f"{dataset_type}_transfer_entropy_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Analysis completed for {dataset_type} data. "
               f"Found {len(significant_pairs)}/{total_pairs} significant pairs.")
    
    return results_summary

# Custom JSON encoder for numpy values
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def run_dual_transfer_entropy_test(planck_file, wmap_file, n_surrogates=10000, 
                                  output_dir="results/transfer_entropy"):
    """
    Run transfer entropy test simultaneously on both WMAP and Planck datasets
    
    Parameters:
    -----------
    planck_file : str
        Path to Planck dataset
    wmap_file : str
        Path to WMAP dataset
    n_surrogates : int
        Number of surrogate simulations to run
    output_dir : str
        Directory for saving results
        
    Returns:
    --------
    str
        Output directory path with results
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create separate directories for each dataset
    planck_dir = os.path.join(output_dir, "planck")
    wmap_dir = os.path.join(output_dir, "wmap")
    os.makedirs(planck_dir, exist_ok=True)
    os.makedirs(wmap_dir, exist_ok=True)
    
    start_time = time.time()
    
    logger.info("="*80)
    logger.info(f"TRANSFER ENTROPY TEST - {timestamp}")
    logger.info("="*80)
    logger.info(f"Planck data: {planck_file}")
    logger.info(f"WMAP data: {wmap_file}")
    logger.info(f"Number of surrogates: {n_surrogates}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)
    
    # Load and validate data
    try:
        planck_data = np.loadtxt(planck_file)
        wmap_data = np.loadtxt(wmap_file)
        
        logger.info(f"Loaded Planck data: {planck_data.shape}")
        logger.info(f"Loaded WMAP data: {wmap_data.shape}")
        
        # Validate data formats
        if not validate_data(planck_data, "Planck") or not validate_data(wmap_data, "WMAP"):
            logger.error("Data validation failed. Please check your input files.")
            return None
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    # Run analyses in parallel for both datasets (simultaneous testing protocol)
    logger.info("\n" + "="*80)
    logger.info("ANALYZING BOTH DATASETS SIMULTANEOUSLY")
    logger.info("="*80)
    
    # Define focus scales (Fibonacci sequence plus key scales)
    focus_scales = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    
    planck_results = {}
    wmap_results = {}
    
    # Use ProcessPoolExecutor to run both analyses in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Submit both analyses
        planck_future = executor.submit(
            run_transfer_entropy_analysis, 
            planck_data, 
            planck_dir, 
            "planck", 
            n_surrogates,
            focus_scales
        )
        
        wmap_future = executor.submit(
            run_transfer_entropy_analysis, 
            wmap_data, 
            wmap_dir, 
            "wmap", 
            n_surrogates,
            focus_scales
        )
        
        # Get results (will wait for completion)
        try:
            planck_results = planck_future.result()
            wmap_results = wmap_future.result()
        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")
    
    # Compare results across datasets
    try:
        logger.info("\n" + "="*80)
        logger.info("COMPARING RESULTS ACROSS DATASETS")
        logger.info("="*80)
        
        # Create comparison report
        comparison = {
            "planck": planck_results,
            "wmap": wmap_results,
            "metadata": {
                "planck_data_points": planck_data.shape[0],
                "wmap_data_points": wmap_data.shape[0],
                "n_surrogates": n_surrogates,
                "timestamp": timestamp,
                "runtime_seconds": time.time() - start_time
            }
        }
        
        # Save comparison results
        with open(os.path.join(output_dir, "combined_results.json"), 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Calculate total runtime
        total_time = time.time() - start_time
        logger.info(f"Total runtime: {total_time/60:.2f} minutes")
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)
    
    return output_dir

def main():
    """Main function to parse arguments and run the test"""
    parser = argparse.ArgumentParser(description="Run Transfer Entropy Test on CMB data")
    
    # Use absolute paths for defaults
    default_planck = os.path.join(project_root, 'data/processed/planck_power_spectrum.txt')
    default_wmap = os.path.join(project_root, 'data/processed/wmap_power_spectrum.txt')
    
    parser.add_argument('--planck', type=str, default=default_planck,
                       help='Path to processed Planck data file')
    parser.add_argument('--wmap', type=str, default=default_wmap,
                       help='Path to processed WMAP data file')
    parser.add_argument('--output', type=str, default='results/transfer_entropy',
                       help='Output directory')
    parser.add_argument('--surrogates', type=int, default=10000,
                       help='Number of surrogate datasets')
    
    args = parser.parse_args()
    
    # Validate arguments
    for file_path, name in [(args.planck, "Planck"), (args.wmap, "WMAP")]:
        if not os.path.exists(file_path):
            logger.error(f"{name} file not found: {file_path}")
            sys.exit(1)
    
    if args.surrogates < 1000:
        logger.warning(f"Low surrogate count ({args.surrogates}) - recommend >= 10000 for statistical robustness")
    
    # Run the test
    run_dual_transfer_entropy_test(
        args.planck, 
        args.wmap, 
        n_surrogates=args.surrogates, 
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
