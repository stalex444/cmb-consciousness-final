#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New fractal test for CMB data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_rs(data, lag):
    """Calculate R/S statistic for a given lag"""
    rs_values = []
    # Use overlapping windows with a step size of lag/4 for more data points
    step_size = max(1, lag // 4)
    
    for i in range(0, len(data) - lag, step_size):
        window = data[i:i+lag]
        # Skip windows with too few points
        if len(window) < lag:
            continue
            
        # Calculate mean-adjusted series
        mean = np.mean(window)
        z = window - mean
        
        # Calculate cumulative deviation
        y = np.cumsum(z)
        
        # Calculate range and standard deviation
        r = np.max(y) - np.min(y)
        s = np.std(window)
        
        # Skip if standard deviation is too small
        if s < 1e-10:
            continue
            
        rs_values.append(r/s)
    
    if not rs_values:
        return None
    
    return np.mean(rs_values)

def calculate_hurst_exponent(data, min_lag=5, max_lag=None, num_lags=50):
    """Calculate Hurst exponent using R/S analysis"""
    if max_lag is None:
        max_lag = len(data) // 2
    
    # Generate logarithmically spaced lags
    lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), num_lags).astype(int))
    
    # Calculate R/S for each lag
    rs_values = []
    valid_lags = []
    
    for lag in lags:
        rs = calculate_rs(data, lag)
        if rs is not None:
            rs_values.append(rs)
            valid_lags.append(lag)
    
    # Check if we have enough points for regression
    if len(valid_lags) < 5:
        logger.warning(f"Not enough valid lags ({len(valid_lags)}). Results may be unreliable.")
        return None, None
    
    # Perform log-log regression
    log_lags = np.log10(valid_lags)
    log_rs = np.log10(rs_values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_rs)
    
    return slope, {
        'intercept': intercept,
        'r_squared': r_value**2,
        'std_error': std_err,
        'log_lags': log_lags.tolist(),
        'log_rs': log_rs.tolist(),
        'n_lags': len(valid_lags)
    }

def create_surrogate(data):
    """Create phase-randomized surrogate"""
    # Get FFT
    fft = np.fft.rfft(data)
    
    # Get amplitudes and randomize phases
    amplitudes = np.abs(fft)
    phases = np.random.uniform(0, 2*np.pi, len(fft))
    
    # Create new FFT with same amplitudes but random phases
    fft_surrogate = amplitudes * np.exp(1j * phases)
    
    # Get IFFT
    surrogate = np.fft.irfft(fft_surrogate, n=len(data))
    
    return surrogate

def analyze_dataset(data, n_surrogates=1000, name="dataset"):
    """Analyze a dataset with surrogate testing"""
    logger.info(f"Starting fractal analysis for {name}. Data length: {len(data)}")
    
    # Calculate Hurst exponent for original data
    start_time = time.time()
    hurst, metrics = calculate_hurst_exponent(data)
    if hurst is None:
        logger.error(f"Could not calculate Hurst exponent for {name}")
        return None
        
    logger.info(f"Original Hurst exponent for {name}: {hurst:.6f}")
    logger.info(f"R-squared: {metrics['r_squared']:.6f}")
    logger.info(f"Used {metrics['n_lags']} lags for calculation")
    
    # Generate surrogates and calculate their Hurst exponents
    surrogate_hursts = []
    logger.info(f"Generating {n_surrogates} surrogate datasets...")
    
    for i in range(n_surrogates):
        if (i+1) % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Processed {i+1}/{n_surrogates} surrogates. Elapsed: {elapsed:.2f}s")
            
        # Generate surrogate
        surrogate = create_surrogate(data)
        
        # Calculate Hurst exponent
        surrogate_hurst, _ = calculate_hurst_exponent(surrogate)
        
        if surrogate_hurst is not None:
            surrogate_hursts.append(surrogate_hurst)
    
    # Calculate statistics
    surrogate_mean = np.mean(surrogate_hursts)
    surrogate_std = np.std(surrogate_hursts)
    
    # Ensure surrogate_std is not too small
    surrogate_std = max(surrogate_std, 1e-10)
    
    # Calculate z-score and p-value
    z_score = (hurst - surrogate_mean) / surrogate_std
    
    # Calculate empirical p-value
    p_value = np.mean(np.abs(np.array(surrogate_hursts) - surrogate_mean) >= 
                      np.abs(hurst - surrogate_mean))
    
    # Ensure p-value is never exactly zero
    if p_value == 0:
        p_value = 1.0 / (len(surrogate_hursts) + 1)
    
    # Calculate Gaussian p-value
    p_value_gaussian = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Calculate golden ratio metrics
    phi = (1 + np.sqrt(5)) / 2
    phi_conjugate = 1 / phi
    phi_proximity = abs(hurst - phi)
    phi_conjugate_proximity = abs(hurst - phi_conjugate)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    
    # Print results
    logger.info(f"Analysis for {name} completed in {total_time:.2f} seconds")
    logger.info(f"Hurst exponent: {hurst:.6f}")
    logger.info(f"Surrogate mean: {surrogate_mean:.6f}")
    logger.info(f"Surrogate std: {surrogate_std:.6f}")
    logger.info(f"Z-score: {z_score:.6f}")
    logger.info(f"Empirical p-value: {p_value:.6f}")
    logger.info(f"Gaussian p-value: {p_value_gaussian:.6f}")
    logger.info(f"Golden ratio proximity: {phi_proximity:.6f}")
    logger.info(f"Golden ratio conjugate proximity: {phi_conjugate_proximity:.6f}")
    
    # Compile results
    results = {
        'name': name,
        'hurst': hurst,
        'surrogate_mean': surrogate_mean,
        'surrogate_std': surrogate_std,
        'z_score': z_score,
        'p_value': p_value,
        'p_value_gaussian': p_value_gaussian,
        'phi_proximity': phi_proximity,
        'phi_conjugate_proximity': phi_conjugate_proximity,
        'n_surrogates': len(surrogate_hursts),
        'n_lags': metrics['n_lags'],
        'r_squared': metrics['r_squared'],
        'runtime_seconds': total_time
    }
    
    return results

def main():
    # Set up paths
    planck_file = "data/planck/planck_tt_spectrum_2018.txt"
    wmap_file = "data/wmap/wmap_tt_spectrum_9yr_v5.txt"
    output_dir = "results/fractal_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        planck_data = np.loadtxt(planck_file)
        wmap_data = np.loadtxt(wmap_file)
        
        # Extract power spectrum
        planck_cl = planck_data[:, 1]
        wmap_cl = wmap_data[:, 1]
        
        logger.info(f"Loaded Planck data: {len(planck_cl)} points")
        logger.info(f"Loaded WMAP data: {len(wmap_cl)} points")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Set surrogate count
    n_surrogates = 1000  # Start with fewer for testing
    
    # Analyze datasets
    planck_results = analyze_dataset(planck_cl, n_surrogates, name="Planck")
    wmap_results = analyze_dataset(wmap_cl, n_surrogates, name="WMAP")
    
    # Compare results
    logger.info("=" * 60)
    logger.info("COMPARISON OF RESULTS")
    logger.info("=" * 60)
    
    if planck_results is None or wmap_results is None:
        logger.error("Could not compare results due to calculation failures")
        return
        
    logger.info(f"Planck Hurst: {planck_results['hurst']:.6f}, p-value: {planck_results['p_value']:.6f}")
    logger.info(f"WMAP Hurst: {wmap_results['hurst']:.6f}, p-value: {wmap_results['p_value']:.6f}")
    logger.info(f"Planck phi_conjugate proximity: {planck_results['phi_conjugate_proximity']:.6f}")
    logger.info(f"WMAP phi_conjugate proximity: {wmap_results['phi_conjugate_proximity']:.6f}")
    logger.info(f"Ratio of proximities: {planck_results['phi_conjugate_proximity']/wmap_results['phi_conjugate_proximity']:.6f}")
    
    # Save results only if both analyses succeeded
    if planck_results is not None and wmap_results is not None:
        import json
        results = {
            'planck': planck_results,
            'wmap': wmap_results,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(output_dir, 'fractal_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {os.path.join(output_dir, 'fractal_results.json')}")
    else:
        logger.error("Could not save results due to calculation failures")

if __name__ == "__main__":
    main()
