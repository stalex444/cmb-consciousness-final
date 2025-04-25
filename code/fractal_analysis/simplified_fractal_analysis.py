#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simplified_fractal_analysis.py

A simplified but methodologically sound approach to fractal analysis of CMB data.
This script analyzes both WMAP and Planck datasets sequentially using phase randomization
for surrogate generation and a proper lag selection for Hurst exponent calculation.
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from sklearn.linear_model import RANSACRegressor

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
PHI_CONJUGATE = 1 / PHI  # Golden ratio conjugate


def calculate_hurst_exponent(data, min_lag=5, max_lag=None, n_lags=15):
    """
    Calculate Hurst exponent using Rescaled Range (R/S) analysis
    
    Parameters:
    -----------
    data : array-like
        Time series data
    min_lag : int
        Minimum lag to consider
    max_lag : int, optional
        Maximum lag to consider (defaults to 1/3 of data length)
    n_lags : int
        Number of logarithmically spaced lags to use
        
    Returns:
    --------
    tuple
        (hurst_exponent, metrics_dict)
    """
    data = np.asarray(data)
    n = len(data)
    
    if max_lag is None:
        max_lag = n // 3  # Use up to 1/3 of data length for max lag
    
    # Generate logarithmically spaced lags
    lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), n_lags).astype(int))
    lags = lags[lags >= min_lag]  # Ensure minimum lag
    
    # Calculate R/S for different lags
    rs_values = []
    valid_lags = []
    
    for lag in lags:
        # Use overlapping windows for more robust estimation
        step_size = max(1, lag // 4)
        rs_lag = []
        
        for i in range(0, n - lag, step_size):
            x = data[i:i+lag]
            # Mean-adjusted series
            z = x - np.mean(x)
            # Cumulative deviation
            y = np.cumsum(z)
            # Range
            r = np.max(y) - np.min(y)
            # Standard deviation
            s = np.std(x)
            
            if s > 0:  # Avoid division by zero
                rs_lag.append(r/s)
        
        if len(rs_lag) >= 3:  # Require at least 3 valid R/S values
            rs_values.append(np.mean(rs_lag))
            valid_lags.append(lag)
    
    # Ensure we have enough points for regression
    if len(valid_lags) < 5:
        logger.warning(f"Not enough valid lags (only {len(valid_lags)}). Results may be unreliable.")
    
    # Log-log regression to find Hurst exponent
    log_lags = np.log10(valid_lags)
    log_rs = np.log10(rs_values)
    
    # Use RANSAC for robust regression against outliers
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(log_lags.reshape(-1, 1), log_rs)
    hurst = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    
    # Calculate R-squared
    y_pred = ransac.predict(log_lags.reshape(-1, 1))
    r_squared = np.corrcoef(log_rs, y_pred)[0, 1] ** 2
    std_err = np.sqrt(np.mean((log_rs - y_pred) ** 2))
    
    # Calculate proximity to golden ratio and its conjugate
    phi_proximity = abs(hurst - PHI)
    phi_conjugate_proximity = abs(hurst - PHI_CONJUGATE)
    
    # Compile metrics
    metrics = {
        'hurst': hurst,
        'intercept': intercept,
        'r_squared': r_squared,
        'std_error': std_err,
        'n_lags': len(valid_lags),
        'log_lags': log_lags.tolist(),
        'log_rs': log_rs.tolist(),
        'phi_proximity': phi_proximity,
        'phi_conjugate_proximity': phi_conjugate_proximity,
        'fractal_dimension': 2 - hurst
    }
    
    return hurst, metrics


def create_surrogate(data):
    """
    Create a surrogate by phase randomization that preserves power spectrum
    
    Parameters:
    -----------
    data : array-like
        Original time series
        
    Returns:
    --------
    array-like
        Surrogate time series
    """
    data = np.array(data)
    
    # FFT of the data
    fft_vals = np.fft.rfft(data)
    
    # Generate random phases
    random_phases = np.random.uniform(0, 2*np.pi, len(fft_vals))
    
    # Keep amplitude the same but randomize the phase
    fft_vals_new = np.abs(fft_vals) * np.exp(1j * random_phases)
    
    # Inverse FFT to get the surrogate time series
    surrogate = np.fft.irfft(fft_vals_new, n=len(data))
    
    return surrogate


def calculate_z_score(observed_value, surrogate_values):
    """
    Calculate z-score with minimum standard deviation threshold
    
    Parameters:
    -----------
    observed_value : float
        Observed value from real data
    surrogate_values : list/array
        Values from surrogate data
        
    Returns:
    --------
    float
        Z-score
    """
    surrogate_mean = np.mean(surrogate_values)
    surrogate_std = np.std(surrogate_values)
    
    # Use minimum threshold to prevent division by very small numbers
    min_std = 1e-10
    surrogate_std = max(surrogate_std, min_std)
    
    z_score = (observed_value - surrogate_mean) / surrogate_std
    return z_score


def analyze_dataset(data_file, dataset_name, n_surrogates=1000, output_dir="results/fractal_analysis"):
    """
    Analyze a single dataset for fractal properties
    
    Parameters:
    -----------
    data_file : str
        Path to data file
    dataset_name : str
        Name of dataset (e.g., 'planck', 'wmap')
    n_surrogates : int
        Number of surrogate datasets to generate
    output_dir : str
        Directory for output files
        
    Returns:
    --------
    dict
        Analysis results
    """
    # Create output directory
    dataset_dir = os.path.join(output_dir, dataset_name.lower())
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Setup log file
    log_file = os.path.join(dataset_dir, f"{dataset_name.lower()}_analysis.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log analysis start
    logger.info(f"Starting fractal analysis for {dataset_name} data")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Number of surrogates: {n_surrogates}")
    
    # Load data
    try:
        data = np.loadtxt(data_file)
        logger.info(f"Data loaded, shape: {data.shape}")
        
        # Extract power spectrum
        ell = data[:, 0]
        cl = data[:, 1]
        
        logger.info(f"Using {len(cl)} datapoints for analysis")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    # Calculate Hurst exponent for original data
    logger.info("Calculating Hurst exponent for original data")
    hurst, hurst_metrics = calculate_hurst_exponent(cl)
    logger.info(f"Original Hurst exponent: {hurst:.6f}")
    
    # Generate and analyze surrogate data
    logger.info(f"Generating {n_surrogates} surrogate datasets")
    surrogate_hurst = []
    
    start_time = time.time()
    for i in range(n_surrogates):
        # Progress reporting
        if (i+1) % 100 == 0 or i+1 == n_surrogates:
            elapsed = time.time() - start_time
            rate = (i+1) / elapsed if elapsed > 0 else 0
            remaining = (n_surrogates - (i+1)) / rate if rate > 0 else 0
            logger.info(f"Processed {i+1}/{n_surrogates} surrogates. "
                       f"Rate: {rate:.2f}/s. "
                       f"Est. remaining: {remaining/60:.1f} minutes.")
        
        # Generate surrogate
        surrogate = create_surrogate(cl)
        
        # Calculate Hurst exponent for surrogate
        h, _ = calculate_hurst_exponent(surrogate)
        surrogate_hurst.append(h)
    
    # Calculate statistics
    surrogate_mean = np.mean(surrogate_hurst)
    surrogate_std = np.std(surrogate_hurst)
    z_score = calculate_z_score(hurst, surrogate_hurst)
    
    # Calculate p-value (two-tailed)
    p_value = np.sum(np.abs(np.array(surrogate_hurst) - surrogate_mean) >= 
                    np.abs(hurst - surrogate_mean)) / n_surrogates
    
    # Ensure p-value is never exactly zero
    if p_value == 0:
        p_value = 1.0 / (n_surrogates + 1)
    
    # Calculate sigma (standard deviations from mean)
    sigma = stats.norm.ppf(1 - p_value/2) if p_value < 1 else 0.0
    
    # Calculate effect size (Cohen's d)
    effect_size = (hurst - surrogate_mean) / surrogate_std if surrogate_std > 0 else float('inf')
    
    # Create histogram visualization
    plt.figure(figsize=(10, 6))
    plt.hist(surrogate_hurst, bins=30, alpha=0.5, label='Surrogate data')
    plt.axvline(x=hurst, color='r', linestyle='--', 
               label=f'Original data (H={hurst:.4f}, {z_score:.2f} sigma)')
    plt.axvline(x=PHI_CONJUGATE, color='g', linestyle=':', 
               label=f'Golden ratio conjugate (1/φ={PHI_CONJUGATE:.4f})')
    plt.axvline(x=PHI, color='g', linestyle='-', 
               label=f'Golden ratio (φ={PHI:.4f})')
    plt.xlabel('Hurst Exponent (H)')
    plt.ylabel('Frequency')
    plt.title(f'Hurst Exponent Distribution - {dataset_name.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(dataset_dir, f"{dataset_name.lower()}_hurst_distribution.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    # Create R/S visualization
    plt.figure(figsize=(10, 6))
    log_lags = np.array(hurst_metrics['log_lags'])
    log_rs = np.array(hurst_metrics['log_rs'])
    intercept = hurst_metrics['intercept']
    
    plt.scatter(log_lags, log_rs, marker='o', label='Data points')
    plt.plot(log_lags, intercept + hurst * log_lags, 'r', 
            label=f'Fitted line (H={hurst:.4f})')
    plt.xlabel('Log10(lag)')
    plt.ylabel('Log10(R/S)')
    plt.title(f'R/S Analysis - {dataset_name.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save R/S plot
    rs_plot_file = os.path.join(dataset_dir, f"{dataset_name.lower()}_rs_analysis.png")
    plt.savefig(rs_plot_file, dpi=300)
    plt.close()
    
    # Compile results
    results = {
        'dataset': dataset_name,
        'file': data_file,
        'hurst': hurst,
        'fractal_dimension': hurst_metrics['fractal_dimension'],
        'r_squared': hurst_metrics['r_squared'],
        'surrogate_mean': surrogate_mean,
        'surrogate_std': surrogate_std,
        'z_score': z_score,
        'p_value': p_value,
        'sigma': sigma,
        'effect_size': effect_size,
        'phi_proximity': hurst_metrics['phi_proximity'],
        'phi_conjugate_proximity': hurst_metrics['phi_conjugate_proximity'],
        'n_surrogates': n_surrogates,
        'n_lags': hurst_metrics['n_lags'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save results to JSON
    import json
    class NumpyEncoder(json.JSONEncoder):
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
    
    results_file = os.path.join(dataset_dir, f"{dataset_name.lower()}_fractal_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    
    # Print summary
    print(f"\n{'-'*60}")
    print(f"Fractal Analysis Results for {dataset_name.upper()}")
    print(f"{'-'*60}")
    print(f"Hurst Exponent: {hurst:.6f}")
    print(f"Fractal Dimension: {results['fractal_dimension']:.6f}")
    print(f"R-squared: {results['r_squared']:.6f}")
    print(f"Z-score: {z_score:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Sigma: {sigma:.2f}")
    print(f"Proximity to φ: {results['phi_proximity']:.6f}")
    print(f"Proximity to 1/φ: {results['phi_conjugate_proximity']:.6f}")
    print(f"Effect Size: {effect_size:.4f}")
    print(f"{'-'*60}\n")
    
    # Log completion
    logger.info(f"Analysis complete for {dataset_name}")
    logger.info(f"Results saved to {results_file}")
    
    return results


def run_dual_analysis(planck_file, wmap_file, n_surrogates=1000, output_dir="results/fractal_analysis"):
    """
    Run fractal analysis on both Planck and WMAP datasets sequentially
    
    Parameters:
    -----------
    planck_file : str
        Path to Planck data file
    wmap_file : str
        Path to WMAP data file
    n_surrogates : int
        Number of surrogate datasets to generate
    output_dir : str
        Directory for output files
        
    Returns:
    --------
    dict
        Comparison results
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("SEQUENTIAL DUAL DATASET FRACTAL ANALYSIS")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Analyzing datasets with {n_surrogates} surrogate simulations each")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze Planck data
    logger.info("Analyzing Planck dataset...")
    planck_results = analyze_dataset(planck_file, 'planck', n_surrogates, output_dir)
    
    # Analyze WMAP data
    logger.info("Analyzing WMAP dataset...")
    wmap_results = analyze_dataset(wmap_file, 'wmap', n_surrogates, output_dir)
    
    # Calculate total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    hours, remainder = divmod(total_runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Create comparison
    comparison = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'planck': planck_results,
        'wmap': wmap_results,
        'comparison': {
            'hurst_difference': abs(planck_results['hurst'] - wmap_results['hurst']),
            'phi_proximity_difference': abs(planck_results['phi_proximity'] - wmap_results['phi_proximity']),
            'phi_conjugate_proximity_difference': abs(planck_results['phi_conjugate_proximity'] - wmap_results['phi_conjugate_proximity']),
            'z_score_difference': abs(planck_results['z_score'] - wmap_results['z_score']),
            'p_value_difference': abs(planck_results['p_value'] - wmap_results['p_value']),
            'hurst_ratio': planck_results['hurst'] / wmap_results['hurst'],
            'phi_proximity_ratio': planck_results['phi_proximity'] / wmap_results['phi_proximity'],
            'phi_conjugate_proximity_ratio': planck_results['phi_conjugate_proximity'] / wmap_results['phi_conjugate_proximity'],
        },
        'runtime_seconds': total_runtime,
        'parameters': {
            'n_surrogates': n_surrogates,
            'planck_data_file': planck_file,
            'wmap_data_file': wmap_file
        }
    }
    
    # Save comparison results
    import json
    class NumpyEncoder(json.JSONEncoder):
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
    
    # Save comparison
    comparison_file = os.path.join(output_dir, 'fractal_analysis_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, cls=NumpyEncoder, indent=2)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("DUAL DATASET FRACTAL ANALYSIS COMPARISON")
    print("=" * 60)
    print(f"Planck Hurst: {planck_results['hurst']:.6f}, p-value: {planck_results['p_value']:.6f}")
    print(f"WMAP Hurst: {wmap_results['hurst']:.6f}, p-value: {wmap_results['p_value']:.6f}")
    print(f"Planck 1/φ proximity: {planck_results['phi_conjugate_proximity']:.6f}")
    print(f"WMAP 1/φ proximity: {wmap_results['phi_conjugate_proximity']:.6f}")
    print(f"1/φ proximity ratio: {comparison['comparison']['phi_conjugate_proximity_ratio']:.6f}")
    print(f"φ proximity ratio: {comparison['comparison']['phi_proximity_ratio']:.6f}")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("=" * 60)
    
    # Log completion
    logger.info("=" * 60)
    logger.info("DUAL DATASET FRACTAL ANALYSIS COMPLETE")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info("=" * 60)
    
    return comparison


def main():
    """Main function to parse arguments and run analysis"""
    import argparse
    import concurrent.futures
    import multiprocessing
    
    parser = argparse.ArgumentParser(description='Run simplified fractal analysis on CMB datasets')
    parser.add_argument('--planck-data', type=str, default="data/planck/planck_tt_spectrum_2018.txt",
                       help='Path to Planck data file')
    parser.add_argument('--wmap-data', type=str, default="data/wmap/wmap_tt_spectrum_9yr_v5.txt",
                       help='Path to WMAP data file')
    parser.add_argument('--surrogates', type=int, default=1000,
                       help='Number of surrogate datasets (default: 1000)')
    parser.add_argument('--output-dir', type=str, default="results/fractal_analysis_simplified",
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Start time measurement
    start_time = time.time()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process both datasets simultaneously using separate processes
    multiprocessing.set_start_method('spawn', force=True)
    
    # Process both datasets simultaneously
    print("\n" + "=" * 60)
    print("SIMULTANEOUS DUAL DATASET FRACTAL ANALYSIS")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analyzing both datasets with {args.surrogates} surrogate simulations each")
    print("=" * 60)
    
    logger.info("=" * 60)
    logger.info("SIMULTANEOUS DUAL DATASET FRACTAL ANALYSIS")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Analyzing both datasets with {args.surrogates} surrogate simulations each")
    logger.info("=" * 60)
    
    # Create result storage
    results = {}
    
    # Run both analyses in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Submit both jobs
        future_wmap = executor.submit(analyze_dataset, args.wmap_data, 'wmap', args.surrogates, args.output_dir)
        future_planck = executor.submit(analyze_dataset, args.planck_data, 'planck', args.surrogates, args.output_dir)
        
        # Process results as they complete
        for future in concurrent.futures.as_completed([future_wmap, future_planck]):
            try:
                result = future.result()
                if result['dataset'].lower() == 'wmap':
                    results['wmap'] = result
                elif result['dataset'].lower() == 'planck':
                    results['planck'] = result
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
    
    # Create comparison if both results are available
    if 'wmap' in results and 'planck' in results:
        # Calculate comparison metrics
        comparison = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'planck': results['planck'],
            'wmap': results['wmap'],
            'comparison': {
                'hurst_difference': abs(results['planck']['hurst'] - results['wmap']['hurst']),
                'phi_proximity_difference': abs(results['planck']['phi_proximity'] - results['wmap']['phi_proximity']),
                'phi_conjugate_proximity_difference': abs(results['planck']['phi_conjugate_proximity'] - results['wmap']['phi_conjugate_proximity']),
                'z_score_difference': abs(results['planck']['z_score'] - results['wmap']['z_score']),
                'p_value_difference': abs(results['planck']['p_value'] - results['wmap']['p_value']),
                'hurst_ratio': results['planck']['hurst'] / results['wmap']['hurst'],
                'phi_proximity_ratio': results['planck']['phi_proximity'] / results['wmap']['phi_proximity'],
                'phi_conjugate_proximity_ratio': results['planck']['phi_conjugate_proximity'] / results['wmap']['phi_conjugate_proximity'],
            },
            'runtime_seconds': time.time() - start_time,
            'parameters': {
                'n_surrogates': args.surrogates,
                'planck_data_file': args.planck_data,
                'wmap_data_file': args.wmap_data
            }
        }
        
        # Save comparison results
        import json
        class NumpyEncoder(json.JSONEncoder):
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
        
        # Save comparison
        comparison_file = os.path.join(args.output_dir, 'fractal_analysis_comparison.json')
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, cls=NumpyEncoder, indent=2)
        
        # Print comparison
        print("\n" + "=" * 60)
        print("DUAL DATASET FRACTAL ANALYSIS COMPARISON")
        print("=" * 60)
        print(f"Planck Hurst: {results['planck']['hurst']:.6f}, p-value: {results['planck']['p_value']:.6f}")
        print(f"WMAP Hurst: {results['wmap']['hurst']:.6f}, p-value: {results['wmap']['p_value']:.6f}")
        print(f"Planck 1/φ proximity: {results['planck']['phi_conjugate_proximity']:.6f}")
        print(f"WMAP 1/φ proximity: {results['wmap']['phi_conjugate_proximity']:.6f}")
        print(f"1/φ proximity ratio: {comparison['comparison']['phi_conjugate_proximity_ratio']:.6f}")
        print(f"φ proximity ratio: {comparison['comparison']['phi_proximity_ratio']:.6f}")
        
        # Log completion
        hours, remainder = divmod(comparison['runtime_seconds'], 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print("=" * 60)
        
        logger.info("=" * 60)
        logger.info("DUAL DATASET FRACTAL ANALYSIS COMPLETE")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        logger.info("=" * 60)
    else:
        logger.error("Could not generate comparison due to missing results.")


if __name__ == "__main__":
    main()
