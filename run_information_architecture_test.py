#!/usr/bin/env python3
"""
Information Architecture Test for CMB Data

This script runs the Information Architecture Test on both WMAP and Planck datasets
with 10,000 simulations without early stopping. The test analyzes how different 
mathematical constants (phi, sqrt2, sqrt3, ln2, e, pi) organize aspects of the 
hierarchical information structure in the Cosmic Microwave Background.

Previous results showed:
- WMAP: Statistical significance for Golden Ratio (φ): Score = 1.0203, p = 0.044838
- Planck: High significance for average layer specialization (p = 0.0000)
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

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import CMB analysis modules
from cmb.transfer_entropy import analyze_all_constants
from cmb.visualization import create_summary_visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("information_architecture_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_test(planck_file, wmap_file, n_surrogates=10000, output_dir="results/information_architecture"):
    """
    Run Information Architecture Test on both datasets
    
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
    """
    # Create timestamp for output
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
    logger.info(f"INFORMATION ARCHITECTURE TEST - {timestamp}")
    logger.info("="*80)
    logger.info(f"Planck data: {planck_file}")
    logger.info(f"WMAP data: {wmap_file}")
    logger.info(f"Number of surrogates: {n_surrogates}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)
    
    # Load data
    try:
        planck_data = np.loadtxt(planck_file)
        wmap_data = np.loadtxt(wmap_file)
        
        logger.info(f"Loaded Planck data: {planck_data.shape}")
        logger.info(f"Loaded WMAP data: {wmap_data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Run Analysis on Planck data
    logger.info("\n" + "="*80)
    logger.info("ANALYZING PLANCK DATASET")
    logger.info("="*80)
    
    try:
        planck_results = analyze_all_constants(
            planck_data, 
            planck_dir, 
            dataset_type="planck", 
            n_surrogates=n_surrogates,
            tolerance=0.05,
            focus_key_scales=True,
            checkpoint_interval=100
        )
        
        # Save complete results
        with open(os.path.join(planck_dir, "planck_complete_results.json"), 'w') as f:
            json.dump(planck_results, f, indent=2)
            
        logger.info(f"Planck analysis completed and saved to {planck_dir}")
    except Exception as e:
        logger.error(f"Error in Planck analysis: {e}")
    
    # Run Analysis on WMAP data
    logger.info("\n" + "="*80)
    logger.info("ANALYZING WMAP DATASET")
    logger.info("="*80)
    
    try:
        wmap_results = analyze_all_constants(
            wmap_data, 
            wmap_dir, 
            dataset_type="wmap", 
            n_surrogates=n_surrogates,
            tolerance=0.05,
            focus_key_scales=True,
            checkpoint_interval=100
        )
        
        # Save complete results
        with open(os.path.join(wmap_dir, "wmap_complete_results.json"), 'w') as f:
            json.dump(wmap_results, f, indent=2)
            
        logger.info(f"WMAP analysis completed and saved to {wmap_dir}")
    except Exception as e:
        logger.error(f"Error in WMAP analysis: {e}")
    
    # Compare results across datasets
    try:
        logger.info("\n" + "="*80)
        logger.info("COMPARING RESULTS ACROSS DATASETS")
        logger.info("="*80)
        
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
        
        # Create visualization comparing datasets
        create_summary_visualization(
            planck_results, 
            wmap_results, 
            output_dir, 
            title="Information Architecture Test Results"
        )
        
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
    parser = argparse.ArgumentParser(description="Run Information Architecture Test on CMB data")
    parser.add_argument('--planck', type=str, default='data/processed/planck_power_spectrum.txt',
                       help='Path to processed Planck data file')
    parser.add_argument('--wmap', type=str, default='data/processed/wmap_power_spectrum.txt',
                       help='Path to processed WMAP data file')
    parser.add_argument('--output', type=str, default='results/information_architecture',
                       help='Output directory')
    parser.add_argument('--surrogates', type=int, default=10000,
                       help='Number of surrogate datasets')
    
    args = parser.parse_args()
    
    # Run the test
    run_test(
        args.planck, 
        args.wmap, 
        n_surrogates=args.surrogates, 
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
