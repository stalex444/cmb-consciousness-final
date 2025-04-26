#!/usr/bin/env python
"""
Dual Dataset Analysis for CMB
============================

This script runs a comprehensive analysis on both WMAP and Planck datasets simultaneously,
ensuring consistent protocols and directly comparable results.

This approach addresses the requirement to analyze both datasets with identical parameters
and statistical methods to make valid scientific comparisons.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
import logging
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Import analysis functions
from cmb.transfer_entropy import (
    calculate_multiscale_entropy,
    phase_synchronization,
    golden_ratio_precision_analysis,
    cross_scale_information_flow,
    analyze_surrogate_parallel,
    generate_surrogate
)

# Configure logging with detailed formatting for progress tracking
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define enhanced JSON encoder to handle NumPy types and tuple keys at all nesting levels
class NumpyEncoder(json.JSONEncoder):
    """Enhanced encoder to handle NumPy types and tuple keys in nested structures"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


def convert_tuple_keys_recursive(obj):
    """Recursively convert tuple keys to strings in nested structures"""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Convert tuple keys to strings
            if isinstance(k, tuple):
                k = str(k)
            new_dict[k] = convert_tuple_keys_recursive(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_tuple_keys_recursive(item) for item in obj]
    else:
        return obj


def load_data(filepath):
    """Load CMB power spectrum data from file"""
    try:
        data = np.loadtxt(filepath)
        logger.info(f"Loaded data from {filepath} with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise


def run_parallel_analysis(data, analysis_type, n_surrogates, n_jobs):
    """Run analysis with real-time progress monitoring"""
    logger.info(f"Running {analysis_type} analysis on {n_surrogates} surrogates using {n_jobs} cores...")
    
    start_time = time.time()
    results = analyze_surrogate_parallel(
        data,
        analysis_type=analysis_type,
        n_surrogates=n_surrogates,
        n_jobs=n_jobs,
        scales=[2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"Completed {analysis_type} analysis in {elapsed_time:.2f} seconds")
    return results


def run_analysis_on_dataset(data, output_dir, dataset_name, n_surrogates, n_jobs):
    """Run all analyses on a single dataset with detailed progress monitoring"""
    logger.info(f"Starting comprehensive analysis on {dataset_name} dataset")
    logger.info(f"Will run {n_surrogates} surrogate simulations using {n_jobs} parallel jobs")
    
    # Create output directory
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(dataset_output_dir, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging to file for this dataset
    file_handler = logging.FileHandler(
        os.path.join(results_dir, f"{dataset_name}_analysis.log")
    )
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Record start time
    start_time = time.time()
    all_results = {}
    
    # 1. Multiscale Entropy Analysis
    logger.info("Starting multiscale entropy analysis...")
    mse_results = run_parallel_analysis(data, "multiscale_entropy", n_surrogates, n_jobs)
    all_results["multiscale_entropy"] = mse_results
    
    # Save interim results
    with open(os.path.join(results_dir, "multiscale_entropy_results.json"), "w") as f:
        json.dump(convert_tuple_keys_recursive(mse_results), f, cls=NumpyEncoder, indent=2)
    
    # 2. Transfer Entropy Analysis
    logger.info("Starting transfer entropy analysis...")
    te_results = run_parallel_analysis(data, "transfer_entropy", n_surrogates, n_jobs)
    all_results["transfer_entropy"] = te_results
    
    # Save interim results
    with open(os.path.join(results_dir, "transfer_entropy_results.json"), "w") as f:
        json.dump(convert_tuple_keys_recursive(te_results), f, cls=NumpyEncoder, indent=2)
    
    # 3. Phase Synchronization Analysis
    logger.info("Starting phase synchronization analysis...")
    ps_results = run_parallel_analysis(data, "phase_sync", n_surrogates, n_jobs)
    all_results["phase_synchronization"] = ps_results
    
    # Save interim results
    with open(os.path.join(results_dir, "phase_sync_results.json"), "w") as f:
        json.dump(convert_tuple_keys_recursive(ps_results), f, cls=NumpyEncoder, indent=2)
    
    # 4. Golden Ratio Precision Analysis
    logger.info("Starting golden ratio precision analysis...")
    golden_results = run_parallel_analysis(data, "golden_ratio", n_surrogates, n_jobs)
    all_results["golden_ratio"] = golden_results
    
    # Save interim results
    with open(os.path.join(results_dir, "golden_ratio_results.json"), "w") as f:
        json.dump(convert_tuple_keys_recursive(golden_results), f, cls=NumpyEncoder, indent=2)
    
    # 5. Cross-scale Information Flow Analysis
    logger.info("Starting cross-scale information flow analysis...")
    cross_scale_results = run_parallel_analysis(data, "cross_scale", n_surrogates, n_jobs)
    all_results["cross_scale"] = cross_scale_results
    
    # Save all results in a single file
    all_results_file = os.path.join(results_dir, "all_results.json")
    with open(all_results_file, "w") as f:
        json.dump(convert_tuple_keys_recursive(all_results), f, cls=NumpyEncoder, indent=2)
    
    # Calculate and log runtime
    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"Analysis complete. Results saved to {results_dir}")
    logger.info(f"Runtime: {runtime:.2f} seconds")
    
    # Return the path to the results directory and runtime
    return results_dir, runtime


def run_dual_dataset_analysis(planck_data_file, wmap_data_file, output_dir, n_surrogates, n_jobs):
    """Run simultaneous analysis on both datasets for direct comparison"""
    start_time = time.time()
    
    # Log the analysis start
    logger.info("=" * 70)
    logger.info("DUAL DATASET COMPREHENSIVE CMB ANALYSIS")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Analyzing both WMAP and Planck datasets simultaneously")
    logger.info(f"Using {n_surrogates} surrogate simulations")
    logger.info("=" * 70)
    
    # Create the main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up a log file for the overall analysis
    main_log_file = os.path.join(output_dir, "dual_analysis.log")
    file_handler = logging.FileHandler(main_log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Load both datasets
    logger.info("Loading Planck dataset...")
    planck_data = load_data(planck_data_file)
    
    logger.info("Loading WMAP dataset...")
    wmap_data = load_data(wmap_data_file)
    
    # Run analyses on both datasets in parallel for maximum performance
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit both datasets for parallel analysis
        planck_future = executor.submit(
            run_analysis_on_dataset, 
            planck_data, 
            output_dir, 
            "planck", 
            n_surrogates, 
            n_jobs
        )
        
        wmap_future = executor.submit(
            run_analysis_on_dataset, 
            wmap_data, 
            output_dir, 
            "wmap", 
            n_surrogates, 
            n_jobs
        )
        
        # Wait for both analyses to complete
        planck_results_dir, planck_runtime = planck_future.result()
        wmap_results_dir, wmap_runtime = wmap_future.result()
    
    # Calculate and log total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    
    # Save summary information
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "planck_data": planck_data_file,
        "wmap_data": wmap_data_file,
        "n_surrogates": n_surrogates,
        "planck_results_dir": planck_results_dir,
        "wmap_results_dir": wmap_results_dir,
        "planck_runtime": planck_runtime,
        "wmap_runtime": wmap_runtime,
        "total_runtime": total_runtime,
        "n_jobs": n_jobs
    }
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Log completion information
    hours, remainder = divmod(total_runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("=" * 70)
    logger.info("DUAL DATASET COMPREHENSIVE ANALYSIS COMPLETE")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("")
    logger.info("Both datasets were analyzed with identical parameters:")
    logger.info(f"- Number of surrogates: {n_surrogates}")
    logger.info(f"- Using enhanced comprehensive scales test")
    logger.info(f"- Parallel jobs: {n_jobs}")
    logger.info("=" * 70)
    
    return summary


def main():
    """Parse command line arguments and run the analysis"""
    parser = argparse.ArgumentParser(description="Run comprehensive dual dataset analysis")
    parser.add_argument("--planck", type=str, default="data/planck/planck_tt_spectrum_2018.txt",
                        help="Path to Planck CMB power spectrum data file")
    parser.add_argument("--wmap", type=str, default="data/wmap/wmap_tt_spectrum_9yr_v5.txt",
                        help="Path to WMAP CMB power spectrum data file")
    parser.add_argument("--output", type=str, default="results/dual_analysis",
                        help="Output directory for results")
    parser.add_argument("--surrogates", type=int, default=10000,
                        help="Number of surrogate datasets to generate")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Number of parallel jobs to use (-1 for all processors)")
    
    args = parser.parse_args()
    
    # Run the dual dataset analysis
    run_dual_dataset_analysis(
        args.planck,
        args.wmap,
        args.output,
        args.surrogates,
        args.jobs
    )
    
    # Print final message
    print(f"\nDual dataset analysis complete. Results in {args.output}")


if __name__ == "__main__":
    # For reliable multiprocessing on macOS
    multiprocessing.set_start_method("fork", force=True)
    main()
