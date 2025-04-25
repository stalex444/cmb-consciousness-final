#!/usr/bin/env python3
"""
CMB Fractal Analysis Module

This module calculates the Hurst exponent and fractal dimension of CMB data
to quantify self-similarity and long-range persistence in the primordial radiation.
"""

import os
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from scipy import stats
import multiprocessing
import time
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
from cmb_statistical_testing import run_statistical_test, run_multiple_tests

def calculate_hurst_exponent(data, max_lag=None):
    """
    Calculate Hurst exponent using Rescaled Range (R/S) analysis
    
    Parameters:
    -----------
    data : array-like
        Time series data
    max_lag : int, optional
        Maximum lag to consider
    
    Returns:
    --------
    float
        Hurst exponent (0-1)
    dict
        Additional metrics
    """
    data = np.asarray(data)
    n = len(data)
    
    if max_lag is None:
        max_lag = n // 4
    
    # Determine logarithmically spaced lags
    lags = np.unique(np.logspace(0.7, np.log10(max_lag), 20).astype(int))
    
    # Calculate R/S for different lags
    rs_values = []
    
    for lag in lags:
        rs = []
        for i in range(0, n - lag, lag):
            x = data[i:i+lag]
            z = x - np.mean(x)
            r = np.max(np.cumsum(z)) - np.min(np.cumsum(z))  # Range
            s = np.std(x)  # Standard deviation
            if s > 0:
                rs.append(r/s)
        
        rs_values.append(np.mean(rs))
    
    # Calculate Hurst exponent through robust log-log regression using RANSAC
    from sklearn.linear_model import RANSACRegressor
    
    log_lags = np.log10(lags)
    log_rs = np.log10(rs_values)
    
    # Use RANSAC for robust regression that's less affected by outliers
    ransac = RANSACRegressor()
    ransac.fit(log_lags.reshape(-1, 1), log_rs)
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    
    # Calculate R-squared for the robust model
    y_pred = ransac.predict(log_lags.reshape(-1, 1))
    r_value = np.corrcoef(log_rs, y_pred)[0, 1]
    p_value = 0.0  # Not directly available from RANSAC
    std_err = np.sqrt(np.mean((log_rs - y_pred)**2))
    
    # Calculate proximity to golden ratio conjugate (1/phi)
    phi_conjugate = 0.618034
    phi_proximity = abs(slope - phi_conjugate)
    
    return slope, {
        'hurst': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_error': std_err,
        'log_lags': log_lags.tolist(),
        'log_rs': log_rs.tolist(),
        'phi_proximity': phi_proximity
    }

# Function for parallel surrogate processing
def process_surrogate(seed_and_data):
    """Process a single surrogate dataset with explicit seed control"""
    seed, cl = seed_and_data
    np.random.seed(seed)
    surrogate = create_surrogate(cl)
    h, _ = calculate_hurst_exponent(surrogate)
    return h

# Benjamini-Hochberg correction for multiple comparisons
def benjamini_hochberg_correction(p_values):
    """Apply Benjamini-Hochberg correction to p-values"""
    n = len(p_values)
    if n <= 1:
        return p_values
        
    # Sort p-values in ascending order
    indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[indices]
    
    # Calculate BH critical values
    rank = np.arange(1, n+1)
    critical_values = rank / float(n) * 0.05  # Using alpha=0.05
    
    # Find largest p-value that is <= its critical value
    below = sorted_p <= critical_values
    if not np.any(below):
        return np.ones_like(p_values)  # None significant
        
    largest_idx = np.where(below)[0].max()
    threshold = sorted_p[largest_idx]
    
    # Adjust p-values
    adjusted_p = np.ones_like(p_values, dtype=float)
    for i, p in enumerate(p_values):
        if p <= threshold:
            adjusted_p[i] = p * n / (np.where(sorted_p == p)[0][0] + 1)
        
    # Ensure adjusted p-values don't exceed 1
    adjusted_p = np.minimum(adjusted_p, 1.0)
    
    return adjusted_p

# Custom progress bar for real-time monitoring
def display_progress(i, total, start_time, surrogate_mean=None, surrogate_std=None, last_update=None, update_interval=1.0):
    """Display progress with time estimation"""
    now = time.time()
    
    # Only update at specified intervals to avoid too much output
    if last_update is None or (now - last_update) >= update_interval:
        percent = float(i) / total
        bar_length = 50
        filled_length = int(bar_length * percent)
        bar = '#' * filled_length + '.' * (bar_length - filled_length)
        
        # Calculate time metrics
        elapsed = now - start_time
        if i == 0:
            eta = '?'
            speed = 0
        else:
            speed = i / elapsed  # surrogates per second
            remaining = (total - i) / speed if speed > 0 else 0
            eta = str(timedelta(seconds=int(remaining)))
        
        # Format elapsed time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        # Print stats
        stats_str = ''
        if surrogate_mean is not None and surrogate_std is not None:
            stats_str = f', Mean: {surrogate_mean:.4f}, Std: {surrogate_std:.4f}'
        
        # Construct progress line
        progress_line = f"Progress: [{bar}] {percent*100:.1f}% ({i}/{total}) - Elapsed: {elapsed_str}, ETA: {eta}, Speed: {speed:.1f} surr/s{stats_str}\r"
        print(progress_line, end='', flush=True)
        
        return now  # Return the current time as the last update time
    
    return last_update

def analyze_cmb_fractal(data, output_dir, dataset_name, n_surrogates=1000):
    """
    Perform fractal analysis on CMB data with surrogate testing
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    output_dir : str
        Directory to save results
    dataset_name : str
        Name of dataset (e.g., 'planck', 'wmap')
    n_surrogates : int
        Number of surrogate datasets for statistical validation
    
    Returns:
    --------
    dict
        Dictionary of fractal analysis results
    """
    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'fractal_analysis.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log analysis start
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"\n{'='*50}")
    logging.info(f"Starting CMB Fractal Analysis for {dataset_name.upper()} at {current_time}")
    logging.info(f"Running {n_surrogates} surrogate simulations with parallel processing")
    logging.info(f"{'='*50}\n")
    
    # Print header
    print(f"\n{'#'*60}")
    print(f"## CMB FRACTAL ANALYSIS - {dataset_name.upper()}")
    print(f"## Starting analysis with {n_surrogates} surrogate tests")
    print(f"## Output directory: {os.path.abspath(output_dir)}")
    print(f"{'#'*60}\n")
    
    # Extract power spectrum
    ell = data[:, 0]
    cl = data[:, 1]
    
    print(f"Loading {dataset_name} data...")
    print(f"Data loaded: {len(cl)} datapoints\n")
    
    # Calculate Hurst exponent for original data
    print("Calculating Hurst exponent for original data...")
    hurst, hurst_metrics = calculate_hurst_exponent(cl)
    print(f"Original {dataset_name.upper()} data Hurst exponent: {hurst:.6f}\n")
    
    # Run surrogate testing with parallel processing
    print(f"Starting parallel surrogate testing with {n_surrogates} simulations...")
    
    # Determine number of CPU cores to use
    n_cores = multiprocessing.cpu_count()
    cores_to_use = min(n_cores - 1, 10)  # Leave one core free, max 10
    print(f"Using {cores_to_use} CPU cores for parallel processing\n")
    
    # Setup multiprocessing pool
    pool = multiprocessing.Pool(processes=cores_to_use)
    
    # Create a list of (seed, data) pairs for each process
    # Use explicit seeds for reproducibility
    seeds = np.arange(n_surrogates)  # Unique seed for each surrogate
    surrogate_inputs = [(int(seed), cl) for seed in seeds]
    
    # Start timing the surrogate testing
    start_time = time.time()
    last_update = None
    
    # Run surrogates in batches for better monitoring
    batch_size = 100
    surrogate_hurst = []
    
    for batch_start in range(0, n_surrogates, batch_size):
        batch_end = min(batch_start + batch_size, n_surrogates)
        batch_inputs = surrogate_inputs[batch_start:batch_end]
        
        # Run this batch
        results = []
        for i, _ in enumerate(pool.imap_unordered(process_surrogate, batch_inputs), 1):
            results.append(_)
            
            # Calculate running statistics for progress display
            current_mean = np.mean(surrogate_hurst + results) if (surrogate_hurst + results) else None
            current_std = np.std(surrogate_hurst + results) if len(surrogate_hurst + results) > 1 else None
            
            # Update progress bar
            progress_count = batch_start + i
            last_update = display_progress(progress_count, n_surrogates, start_time, 
                                          current_mean, current_std, last_update)
        
        # Add batch results to overall results
        surrogate_hurst.extend(results)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Print newline after progress bar
    print("\n\nSurrogate testing complete.")
    
    # Use standardized statistical testing framework for uniform analysis
    logger.info("Running statistical test with standardized framework")
    
    # Run the standardized statistical test
    test_results = run_statistical_test(
        observed_value=hurst,  # Use the calculated Hurst exponent
        surrogate_values=surrogate_hurst,
        test_name="hurst_exponent",
        direction="two-sided"
    )
    
    # Add golden ratio conjugate proximity
    phi_conjugate = 0.618034  # Golden ratio conjugate
    test_results['phi_proximity'] = abs(hurst - phi_conjugate)
    
    # Extract values for visualization and reporting
    surrogate_mean = test_results['surrogate_mean']
    surrogate_std = test_results['surrogate_std']
    z_score = test_results['z_score']
    p_value = test_results['p_value']
    sigma = test_results['sigma']
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.hist(surrogate_hurst, bins=30, alpha=0.5, label='Surrogate data')
    plt.axvline(x=hurst, color='r', linestyle='--', 
                label=f'Original data (H={hurst:.4f}, {z_score:.2f} sigma)')
    plt.xlabel('Hurst Exponent (H)')
    plt.ylabel('Frequency')
    plt.title(f'Fractal Analysis - {dataset_name.upper()}')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, f"{dataset_name}_hurst_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    # Create log-log plot
    plt.figure(figsize=(10, 6))
    plt.scatter(hurst_metrics['log_lags'], hurst_metrics['log_rs'], color='blue')
    plt.plot(hurst_metrics['log_lags'], 
             np.array(hurst_metrics['log_lags']) * hurst + hurst_metrics['intercept'], 
             'r-', label=f'Slope (H) = {hurst:.4f}')
    plt.xlabel('Log10(Lag)')
    plt.ylabel('Log10(R/S)')
    plt.title(f'Hurst Exponent Estimation - {dataset_name.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    log_fig_path = os.path.join(output_dir, f"{dataset_name}_hurst_loglog.png")
    plt.savefig(log_fig_path, dpi=300)
    plt.close()
    
    # Note: Statistical testing has already been performed by the standardized framework above
    # Just need to add additional metrics to the results
    
    # Calculate fractal dimension from Hurst exponent
    fractal_dim = 2 - hurst
    test_results['fractal_dimension'] = fractal_dim
    test_results['dataset'] = dataset_name
    test_results['hurst'] = hurst  # Original observed value
    
    # Apply BH correction (single test case - values won't change)
    corrected_results = [test_results]
    corrected_results = run_multiple_tests(corrected_results)
    
    # Use the corrected standardized results
    results = corrected_results[0]
    
    # Ensure standardized results are saved and returned
    results_final = {
        'hurst_exponent': hurst,
        'hurst_metrics': hurst_metrics,
        'fractal_dimension': 2 - hurst,
        'p_value': results['p_value'],
        'sigma': results['sigma'],
        'z_score': results['z_score'],
        'surrogate_mean': results['surrogate_mean'],
        'surrogate_std': results['surrogate_std'],
        'dataset': dataset_name,
        'random_surrogates': n_surrogates,
        'significance': 'significant' if results['p_value'] < 0.05 else 'not significant'
    }
    
    # Add adjusted values if they exist (for multiple tests)
    if 'adjusted_p_value' in results:
        results_final['adjusted_p_value'] = results['adjusted_p_value']
    if 'adjusted_sigma' in results:
        results_final['adjusted_sigma'] = results['adjusted_sigma']
    
    results_path = os.path.join(output_dir, f"{dataset_name}_fractal_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_final, f, indent=2)
        
    # Log completion
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    
    logging.info(f"Fractal analysis complete for {dataset_name.upper()}")
    logging.info(f"Hurst exponent: {hurst:.6f}")
    logging.info(f"Statistical significance: p={results_final['p_value']:.6f}")
    if 'adjusted_p_value' in results_final:
        logging.info(f"Adjusted p-value: {results_final['adjusted_p_value']:.6f}")
    logging.info(f"Proximity to golden ratio conjugate: {hurst_metrics['phi_proximity']:.6f}")
    logging.info(f"Analysis completed at {end_time} (elapsed: {elapsed_str})")
    logging.info(f"Results saved to {os.path.abspath(results_path)}")
    
    # Print final results
    print(f"\nAnalysis completed in {elapsed_str}")
    print(f"Hurst exponent: {hurst:.6f} (surrogate mean: {results_final['surrogate_mean']:.6f})")
    print(f"Statistical significance: p={results_final['p_value']:.6f}")
    if 'adjusted_p_value' in results_final:
        print(f"Adjusted p-value: {results_final['adjusted_p_value']:.6f}")
    print(f"Proximity to golden ratio conjugate (1/φ=0.618): {hurst_metrics['phi_proximity']:.6f}")
    print(f"Results saved to {os.path.basename(results_path)}")
    
    return results

def create_surrogate(data):
    """Create a surrogate by phase randomization"""
    # FFT
    fft = np.fft.fft(data)
    
    # Randomize phases but keep amplitudes
    magnitudes = np.abs(fft)
    phases = np.random.uniform(0, 2*np.pi, len(data))
    fft_surrogate = magnitudes * np.exp(1j * phases)
    
    # Ensure the result is real by making the spectrum symmetric
    fft_surrogate[0] = fft[0]  # Keep DC component
    if len(data) % 2 == 0:
        fft_surrogate[len(data)//2] = fft[len(data)//2]  # Keep Nyquist frequency
    
    # IFFT to get surrogate time series
    surrogate = np.real(np.fft.ifft(fft_surrogate))
    
    return surrogate

if __name__ == "__main__":
    # Handle multiprocessing on macOS
    multiprocessing.set_start_method('fork', force=True)
    
    parser = argparse.ArgumentParser(description='CMB Fractal Analysis')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to CMB data file')
    parser.add_argument('--output', type=str, default='fractal_results',
                        help='Output directory')
    parser.add_argument('--dataset', type=str, choices=['planck', 'wmap'], required=True,
                        help='Dataset name')
    parser.add_argument('--surrogates', type=int, default=1000,
                        help='Number of surrogate datasets')
    parser.add_argument('--cores', type=int, default=0,
                        help='Number of CPU cores to use (0=auto)')
    
    args = parser.parse_args()
    
    # Load data
    data = np.loadtxt(args.input)
    
    input_path = args.input
    
    # Print input file in header
    print(f"## Input file: {os.path.abspath(input_path)}")
    
    # Run analysis
    results = analyze_cmb_fractal(data, args.output, args.dataset, args.surrogates)
