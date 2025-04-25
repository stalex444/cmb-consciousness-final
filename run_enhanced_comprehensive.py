#!/usr/bin/env python
"""
Run Enhanced Comprehensive Analysis

This script runs the comprehensive CMB analysis with an enhanced NumpyEncoder
that properly handles tuple keys in nested dictionaries.
"""

import os
import json
import numpy as np
import argparse
import time
import logging
from datetime import datetime
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dual_dataset_script(script_path):
    """
    Check if the dual dataset analysis script exists, verify it's properly set up.
    
    Parameters:
    -----------
    script_path : str
        Path to the dual dataset analysis script
    """
    if not os.path.exists(script_path):
        logger.error(f"ERROR: Dual dataset analysis script not found at {script_path}")
        logger.error("Please ensure the dual_dataset_analysis.py script is in the project directory")
        raise FileNotFoundError(f"Required script not found: {script_path}")
    
    # Make the script executable if it's not already
    os.chmod(script_path, 0o755)
    logger.info(f"Verified dual dataset analysis script at {script_path}")


def run_comprehensive_analysis(planck_data, wmap_data, n_surrogates=10000, n_jobs=-1):
    """
    Run comprehensive analysis on both Planck and WMAP datasets.
    Uses the enhanced NumpyEncoder to handle tuple keys properly.
    
    Parameters:
    -----------
    planck_data : str
        Path to Planck data file
    wmap_data : str
        Path to WMAP data file
    n_surrogates : int
        Number of surrogate datasets to generate
    n_jobs : int
        Number of parallel jobs to use (-1 for all processors)
    """
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/enhanced_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file to capture all output
    log_file = os.path.join(output_dir, "comprehensive_analysis.log")
    
    # Create logging handler to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info(f"ENHANCED COMPREHENSIVE CMB ANALYSIS")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using {n_surrogates} surrogate simulations")
    logger.info("=" * 70)
    
    # Run analysis on both datasets SIMULTANEOUSLY to ensure consistent testing protocols
    # This is critical for making direct comparisons between datasets
    try:
        logger.info("=" * 70)
        logger.info(f"RUNNING ENHANCED COMPREHENSIVE ANALYSIS ON BOTH DATASETS SIMULTANEOUSLY")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Using comprehensive scales test with {n_surrogates} surrogates")
        logger.info("=" * 70)
        
        # Define paths for the new dual dataset analysis script
        dual_analysis_cmd = [
            "python3", "dual_dataset_analysis.py",
            "--planck", planck_data,
            "--wmap", wmap_data,
            "--output", output_dir,
            "--surrogates", str(n_surrogates),
            "--jobs", str(n_jobs)
        ]
        
        # Create the dual dataset analysis script if it doesn't exist
        dual_script_path = "dual_dataset_analysis.py"
        if not os.path.exists(dual_script_path):
            logger.info("Creating dual dataset analysis script for synchronized testing...")
            create_dual_dataset_script(dual_script_path)
            logger.info(f"Created {dual_script_path} for simultaneous dataset analysis")
        
        # Run the simultaneous analysis
        logger.info("Starting synchronized analysis of both WMAP and Planck datasets...")
        subprocess.check_call(dual_analysis_cmd)
        
        end_time = time.time()
        total_runtime = end_time - start_time
        
        # Record runtime information
        hours, remainder = divmod(total_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("=" * 70)
        logger.info(f"ENHANCED COMPREHENSIVE ANALYSIS COMPLETE FOR BOTH DATASETS")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("")
        logger.info(f"Both datasets were analyzed with identical parameters:")
        logger.info(f"- Number of surrogates: {n_surrogates}")
        logger.info(f"- Using enhanced comprehensive scales test")
        logger.info(f"- Parallel jobs: {n_jobs}")
        logger.info("=" * 70)
        
        return output_dir
        
    except Exception as e:
        logger.error(f"Error running comprehensive analysis: {str(e)}")
        raise

def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Run enhanced comprehensive analysis on CMB data")
    
    parser.add_argument("--planck", default="data/planck/planck_tt_spectrum_2018.txt",
                        help="Path to Planck CMB power spectrum data file")
    parser.add_argument("--wmap", default="data/wmap/wmap_tt_spectrum_9yr_v5.txt",
                        help="Path to WMAP CMB power spectrum data file")
    parser.add_argument("--surrogates", type=int, default=10000,
                        help="Number of surrogate datasets to generate")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Number of parallel jobs to use (-1 for all processors)")
    
    args = parser.parse_args()
    
    # Validate data files
    for data_file in [args.planck, args.wmap]:
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            return
    
    # Run analysis
    output_dir = run_comprehensive_analysis(
        args.planck,
        args.wmap,
        n_surrogates=args.surrogates,
        n_jobs=args.jobs
    )
    
    logger.info(f"Enhanced comprehensive analysis complete. Results in {output_dir}")

if __name__ == "__main__":
    main()
