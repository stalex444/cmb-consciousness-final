#!/usr/bin/env python3
"""
Hierarchical Information Architecture Test for CMB Data

This script runs a comprehensive analysis of how different mathematical constants
(phi, sqrt2, sqrt3, ln2, e, pi) organize aspects of the hierarchical information
structure in the Cosmic Microwave Background. It analyzes both WMAP and Planck
datasets simultaneously with 10,000 simulations without early stopping.

Key features:
- Parallel processing of both datasets
- Adaptive binning based on data size
- Enhanced synthetic time series generation
- Gaussian p-values for extreme significance
- Special focus on Scale 55 sqrt2 specialization

Previous results showed:
- Square Root of 2 appears to be the dominant organizing principle across scales
- Scale 55 shows extremely strong sqrt2 specialization in both datasets
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

# Get project root for robust path handling
project_root = os.path.abspath(os.path.dirname(__file__))

# Add project root to Python path
sys.path.insert(0, project_root)

# Configure timestamped logging to prevent overwrites
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"hierarchical_test_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import improved transfer entropy analysis functions
try:
    from cmb.improved_transfer_entropy import analyze_all_constants, create_summary_visualization
except ImportError as e:
    logger.error(f"Failed to import improved transfer entropy module: {e}")
    sys.exit(1)

def validate_data(data, name):
    """
    Validate the CMB power spectrum data
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data array to validate
    name : str
        Name of the dataset for error messages
    
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    # Check data shape and values
    if data.shape[1] < 2:
        logger.error(f"Invalid {name} data: requires at least 2 columns [l, Cl]")
        return False
    
    # Check for too many negative power spectrum values
    # Some negative values are expected in real CMB data due to noise and cosmic variance
    neg_count = np.sum(data[:, 1] < 0)
    if neg_count > 0:
        neg_percent = (neg_count / len(data)) * 100
        logger.warning(f"{name} data contains {neg_count} negative Cl values ({neg_percent:.1f}%)")
        
        # Only fail validation if a large percentage are negative (indicating a serious problem)
        if neg_percent > 30:
            logger.error(f"{name} data has too many negative values ({neg_percent:.1f}%). Check data processing.")
            return False
    
    # Check for sufficient multipoles
    if len(data) < 100:
        logger.warning(f"{name} data may be too small: {len(data)} multipoles. Recommended: >= 100")
        # Warning only, don't return False
    
    return True

def analyze_dataset(data, output_dir, dataset_type, n_surrogates, tolerance=0.05, focus_key_scales=True):
    """
    Run analysis for a single dataset with error handling
    
    Parameters:
    -----------
    data : numpy.ndarray
        Power spectrum data
    output_dir : str
        Directory to save results
    dataset_type : str
        Type of dataset (planck or wmap)
    n_surrogates : int
        Number of surrogate datasets
    tolerance : float
        Tolerance for mathematical constant matching
    focus_key_scales : bool
        Whether to focus on key scales (Fibonacci numbers and golden ratio related)
        
    Returns:
    --------
    dict
        Analysis results or empty dict if error occurred
    """
    try:
        results = analyze_all_constants(
            data, 
            output_dir, 
            dataset_type=dataset_type, 
            n_surrogates=n_surrogates,
            tolerance=tolerance,
            focus_key_scales=focus_key_scales,
            checkpoint_interval=100
        )
        
        # Save complete results
        results_file = os.path.join(output_dir, f"{dataset_type}_complete_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"{dataset_type.upper()} analysis completed and saved to {output_dir}")
        return results
    
    except Exception as e:
        logger.error(f"Error in {dataset_type} analysis: {e}")
        return {}

def extract_scale_55_results(results, constant="sqrt2"):
    """
    Extract information about Scale 55 relationships for a specific constant
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_all_constants
    constant : str
        Mathematical constant to focus on
        
    Returns:
    --------
    dict
        Scale 55 specific results
    """
    if not results or constant not in results:
        return {}
    
    # Initialize results container
    scale_55_results = {
        "pairs": [],
        "te_values": [],
        "z_scores": [],
        "p_values": []
    }
    
    # Extract scale pairs involving 55
    constant_results = results[constant]
    if "scale_pairs" in constant_results:
        for i, pair in enumerate(constant_results["scale_pairs"]):
            if 55 in pair:
                # Record this pair and its statistics
                scale_55_results["pairs"].append(pair)
                
                # Extract TE values if available
                if "te_values" in constant_results and i < len(constant_results["te_values"]):
                    scale_55_results["te_values"].append(constant_results["te_values"][i])
                
                # Extract z-scores if available
                if "z_scores" in constant_results and i < len(constant_results["z_scores"]):
                    scale_55_results["z_scores"].append(constant_results["z_scores"][i])
                
                # Extract p-values if available
                if "p_values" in constant_results and i < len(constant_results["p_values"]):
                    scale_55_results["p_values"].append(constant_results["p_values"][i])
    
    # Calculate some summary statistics
    if scale_55_results["z_scores"]:
        scale_55_results["max_z_score"] = max(scale_55_results["z_scores"])
        scale_55_results["mean_z_score"] = np.mean(scale_55_results["z_scores"])
    
    return scale_55_results

def run_hierarchical_test(planck_file, wmap_file, n_surrogates=10000, output_dir="results/hierarchical_architecture"):
    """
    Run Hierarchical Information Architecture Test on both datasets simultaneously
    
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
    logger.info(f"HIERARCHICAL INFORMATION ARCHITECTURE TEST - {timestamp}")
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
    
    planck_results = {}
    wmap_results = {}
    
    # Use ProcessPoolExecutor to run both analyses in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Submit both analyses
        planck_future = executor.submit(
            analyze_dataset, 
            planck_data, 
            planck_dir, 
            "planck", 
            n_surrogates
        )
        
        wmap_future = executor.submit(
            analyze_dataset, 
            wmap_data, 
            wmap_dir, 
            "wmap", 
            n_surrogates
        )
        
        # Get results (will wait for completion)
        try:
            planck_results = planck_future.result()
            wmap_results = wmap_future.result()
        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")
    
    # Special focus on Scale 55 sqrt2 specialization
    logger.info("\n" + "="*80)
    logger.info("ANALYZING SCALE 55 SQRT2 SPECIALIZATION")
    logger.info("="*80)
    
    planck_scale_55 = extract_scale_55_results(planck_results, "sqrt2")
    wmap_scale_55 = extract_scale_55_results(wmap_results, "sqrt2")
    
    # Log Scale 55 findings
    if planck_scale_55 and "max_z_score" in planck_scale_55:
        logger.info(f"Planck Scale 55 sqrt2 max z-score: {planck_scale_55['max_z_score']}")
    
    if wmap_scale_55 and "max_z_score" in wmap_scale_55:
        logger.info(f"WMAP Scale 55 sqrt2 max z-score: {wmap_scale_55['max_z_score']}")
    
    # Compare results across datasets
    try:
        logger.info("\n" + "="*80)
        logger.info("COMPARING RESULTS ACROSS DATASETS")
        logger.info("="*80)
        
        # Create enhanced comparison with Scale 55 focus
        comparison = {
            "planck": planck_results,
            "wmap": wmap_results,
            "scale_55_analysis": {
                "planck": planck_scale_55,
                "wmap": wmap_scale_55
            },
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
        
        # Create visualization comparing datasets
        try:
            create_summary_visualization(
                planck_results, 
                wmap_results, 
                output_dir, 
                title="Hierarchical Information Architecture Test Results"
            )
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
        
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
    parser = argparse.ArgumentParser(description="Run Hierarchical Information Architecture Test on CMB data")
    
    # Use absolute paths for defaults
    default_planck = os.path.join(project_root, 'data/processed/planck_power_spectrum.txt')
    default_wmap = os.path.join(project_root, 'data/processed/wmap_power_spectrum.txt')
    
    parser.add_argument('--planck', type=str, default=default_planck,
                       help='Path to processed Planck data file')
    parser.add_argument('--wmap', type=str, default=default_wmap,
                       help='Path to processed WMAP data file')
    parser.add_argument('--output', type=str, default='results/hierarchical_architecture',
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
    run_hierarchical_test(
        args.planck, 
        args.wmap, 
        n_surrogates=args.surrogates, 
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
