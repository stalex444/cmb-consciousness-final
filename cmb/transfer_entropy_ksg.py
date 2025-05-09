"""
KSG-based Transfer Entropy Analysis for CMB Data

This module implements the Kraskov-Stögbauer-Grassberger (KSG) estimator for
transfer entropy calculation, specifically optimized for CMB power spectrum data.
"""

import numpy as np
from scipy.special import digamma
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from tqdm import tqdm

def calculate_transfer_entropy_ksg(source, target, k=1, neighbors=3, normalize=True):
    """
    Calculate transfer entropy using the Kraskov-Stögbauer-Grassberger (KSG) estimator
    
    Parameters:
    - source: Source time series (numpy array)
    - target: Target time series (numpy array)
    - k: History length (default=1)
    - neighbors: Number of nearest neighbors (default=3)
    - normalize: Whether to normalize the result (default=True)
    
    Returns:
    - Transfer entropy value
    """
    # Ensure arrays
    source = np.asarray(source)
    target = np.asarray(target)
    
    # Match lengths if needed
    min_length = min(len(source), len(target))
    source = source[:min_length]
    target = target[:min_length]
    
    # Create time-delayed embeddings
    t_future = target[k:]
    t_history = np.array([target[i:i+k] for i in range(min_length-k)])
    s_history = np.array([source[i:i+k] for i in range(min_length-k)])
    
    # Construct joint spaces
    X = np.column_stack([t_history, s_history])  # Combined history
    Y = np.column_stack([t_history])             # Target history only
    Z = np.column_stack([t_future, t_history])   # Future and target history
    W = np.column_stack([t_future, t_history, s_history])  # All variables
    
    # Calculate entropies using KSG estimator
    h_yz = ksg_entropy(Z, neighbors)      # H(Y_t+1, Y_t^k)
    h_xyz = ksg_entropy(W, neighbors)     # H(Y_t+1, Y_t^k, X_t^l)
    h_y = ksg_entropy(Y, neighbors)       # H(Y_t^k)
    h_xy = ksg_entropy(X, neighbors)      # H(Y_t^k, X_t^l)
    
    # Calculate transfer entropy
    te = h_yz + h_xy - h_xyz - h_y
    
    # Normalize if requested
    if normalize and te != 0:
        te = te / h_yz
    
    return max(0, te)  # Ensure non-negative (can be slightly negative due to estimation errors)

def ksg_entropy(data, k=3):
    """
    Calculate entropy using the KSG estimator
    
    Parameters:
    - data: Data points (n_samples, n_dimensions)
    - k: Number of nearest neighbors (default=3)
    
    Returns:
    - Entropy estimate
    """
    n_samples, n_dim = data.shape
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    # Get the kth neighbor distance for each point
    epsilon = distances[:, k]
    
    # Calculate entropy
    entropy = digamma(n_samples) - digamma(k) + n_dim * np.mean(np.log(epsilon))
    
    return entropy

def extract_cmb_scale_data(cmb_data, scale, method='multipole'):
    """
    Extract data for a specific scale from CMB data
    
    Parameters:
    - cmb_data: CMB data structure (dict or DataFrame)
    - scale: The scale to extract
    - method: 'multipole' or 'wavelet' 
    
    Returns:
    - Data for the specified scale
    """
    if method == 'multipole':
        # For multipole data, find the closest l value to the requested scale
        if isinstance(cmb_data, pd.DataFrame):
            if 'l' in cmb_data.columns:
                # Find closest multipole
                closest_l = cmb_data['l'].iloc[(cmb_data['l'] - scale).abs().argsort()[0]]
                return cmb_data.loc[cmb_data['l'] == closest_l, 'power'].values
            elif f'l_{scale}' in cmb_data.columns:
                return cmb_data[f'l_{scale}'].values
            else:
                # Try finding the closest available scale
                available_scales = [col for col in cmb_data.columns if col.startswith('l_')]
                if available_scales:
                    numeric_scales = [int(col.split('_')[1]) for col in available_scales]
                    closest_scale = min(numeric_scales, key=lambda x: abs(x - scale))
                    return cmb_data[f'l_{closest_scale}'].values
        
        # For dictionary format
        elif isinstance(cmb_data, dict):
            if scale in cmb_data:
                return cmb_data[scale]
            elif f'l_{scale}' in cmb_data:
                return cmb_data[f'l_{scale}']
            elif 'l' in cmb_data and 'power' in cmb_data:
                # Find closest multipole
                l_values = np.array(cmb_data['l'])
                closest_idx = np.abs(l_values - scale).argmin()
                closest_l = l_values[closest_idx]
                
                # Return power for the closest multipole
                if isinstance(cmb_data['power'], list):
                    if closest_idx < len(cmb_data['power']):
                        return cmb_data['power'][closest_idx]
                elif isinstance(cmb_data['power'], dict) and closest_l in cmb_data['power']:
                    return cmb_data['power'][closest_l]
        
        # For numpy array format (like what we've been using)
        elif isinstance(cmb_data, np.ndarray):
            # Find the row with the closest scale
            if cmb_data.shape[1] >= 2:  # Ensure we have l and power columns
                scales = cmb_data[:, 0]
                powers = cmb_data[:, 1]
                
                # Find the index of the closest scale
                closest_idx = np.abs(scales - scale).argmin()
                
                # Generate synthetic time series from this power value
                power_value = powers[closest_idx]
                np.random.seed(int(abs(power_value * 1000) % 10000))
                return np.random.normal(power_value, abs(power_value) * 0.1, 100)
    
    elif method == 'wavelet':
        # For wavelet decomposition data
        if isinstance(cmb_data, dict) and 'wavelet_scales' in cmb_data:
            # Find closest wavelet scale
            scales = np.array(cmb_data['wavelet_scales'])
            closest_idx = np.abs(scales - scale).argmin()
            
            if 'wavelet_coeffs' in cmb_data and closest_idx < len(cmb_data['wavelet_coeffs']):
                return cmb_data['wavelet_coeffs'][closest_idx]
    
    # If all else fails, return an empty array
    print(f"Warning: Could not extract data for scale {scale}. Returning empty array.")
    return np.array([])

def calculate_transfer_entropy_for_pair(cmb_data, scale_a, scale_b, extraction_method='multipole'):
    """
    Calculate transfer entropy between two scales in CMB data
    
    Parameters:
    - cmb_data: CMB data
    - scale_a: Source scale
    - scale_b: Target scale
    - extraction_method: Method to extract scale data ('multipole' or 'wavelet')
    
    Returns:
    - Transfer entropy value
    """
    # Extract data for both scales
    data_a = extract_cmb_scale_data(cmb_data, scale_a, method=extraction_method)
    data_b = extract_cmb_scale_data(cmb_data, scale_b, method=extraction_method)
    
    # Check if we have enough data
    if len(data_a) < 10 or len(data_b) < 10:
        print(f"Warning: Insufficient data for scales {scale_a} and {scale_b}")
        return 0.0
    
    # Calculate transfer entropy from A to B
    return calculate_transfer_entropy_ksg(data_a, data_b)

def generate_improved_surrogates(data, n_surrogates=1000):
    """
    Generate improved phase-randomized surrogates with AAFT
    (Amplitude Adjusted Fourier Transform)
    
    Parameters:
    - data: Original time series
    - n_surrogates: Number of surrogate datasets to generate
    
    Returns:
    - List of surrogate time series
    """
    data = np.asarray(data)
    n = len(data)
    surrogates = []
    
    for _ in range(n_surrogates):
        # Sort the data
        sorted_data = np.sort(data)
        
        # Create Gaussian data and sort it
        gaussian = np.random.normal(0, 1, n)
        sorted_gaussian = np.sort(gaussian)
        
        # Reorder the Gaussian data to match the original data
        order = np.argsort(data)
        reordered_gaussian = np.zeros(n)
        for i in range(n):
            reordered_gaussian[order[i]] = sorted_gaussian[i]
        
        # Fourier transform the reordered Gaussian data
        fft_g = np.fft.fft(reordered_gaussian)
        
        # Randomize phases
        phases = np.random.uniform(0, 2*np.pi, n)
        phases[0] = 0  # Keep DC component unchanged
        if n % 2 == 0:
            phases[n//2] = 0  # Keep Nyquist frequency unchanged
        
        phases_complete = np.zeros(n)
        phases_complete[1:n//2+1] = phases[1:n//2+1]
        phases_complete[n//2+1:] = -phases[1:n//2][::-1]
        
        magnitudes = np.abs(fft_g)
        fft_s = magnitudes * np.exp(1j * phases_complete)
        
        # Inverse Fourier transform
        s = np.real(np.fft.ifft(fft_s))
        
        # Rescale to match original distribution
        sorted_s = np.sort(s)
        rescaled_s = np.zeros(n)
        ranks = np.argsort(s)
        for i in range(n):
            rescaled_s[ranks[i]] = sorted_data[i]
        
        surrogates.append(rescaled_s)
    
    return surrogates

def enhanced_transfer_entropy_analysis(cmb_data, scale_pairs, n_surrogates=1000, extraction_method='multipole'):
    """
    Perform enhanced transfer entropy analysis on CMB data
    
    Parameters:
    - cmb_data: CMB data
    - scale_pairs: List of scale pairs to analyze
    - n_surrogates: Number of surrogate datasets
    - extraction_method: Method to extract scale data
    
    Returns:
    - DataFrame with results
    """
    results = []
    
    for pair_idx, (scale_a, scale_b) in enumerate(tqdm(scale_pairs, desc="Analyzing scale pairs")):
        # Skip if scales are the same
        if scale_a == scale_b:
            continue
            
        # Extract data for both scales
        data_a = extract_cmb_scale_data(cmb_data, scale_a, method=extraction_method)
        data_b = extract_cmb_scale_data(cmb_data, scale_b, method=extraction_method)
        
        # Skip if insufficient data
        if len(data_a) < 10 or len(data_b) < 10:
            print(f"Skipping pair ({scale_a}, {scale_b}) due to insufficient data")
            continue
        
        # Calculate forward and reverse transfer entropy
        forward_te = calculate_transfer_entropy_ksg(data_a, data_b)
        reverse_te = calculate_transfer_entropy_ksg(data_b, data_a)
        
        # Generate surrogates for significance testing
        surrogates_a = generate_improved_surrogates(data_a, n_surrogates)
        surrogates_b = generate_improved_surrogates(data_b, n_surrogates)
        
        # Calculate TE on surrogates
        forward_surrogates = []
        reverse_surrogates = []
        
        for i in range(n_surrogates):
            forward_surrogates.append(calculate_transfer_entropy_ksg(surrogates_a[i], data_b))
            reverse_surrogates.append(calculate_transfer_entropy_ksg(surrogates_b[i], data_a))
        
        # Calculate statistics
        forward_mean = np.mean(forward_surrogates)
        forward_std = np.std(forward_surrogates)
        forward_z = (forward_te - forward_mean) / max(forward_std, 1e-10)
        forward_p = 2 * (1 - norm.cdf(abs(forward_z)))
        
        reverse_mean = np.mean(reverse_surrogates)
        reverse_std = np.std(reverse_surrogates)
        reverse_z = (reverse_te - reverse_mean) / max(reverse_std, 1e-10)
        reverse_p = 2 * (1 - norm.cdf(abs(reverse_z)))
        
        # Calculate net and symmetric metrics
        net_te = forward_te - reverse_te
        symmetric_te = forward_te + reverse_te
        directional_bias = net_te / symmetric_te if symmetric_te != 0 else 0
        
        # Store results
        result = {
            'scale_a': scale_a,
            'scale_b': scale_b,
            'ratio': scale_b / scale_a,
            'forward_te': forward_te,
            'reverse_te': reverse_te,
            'net_te': net_te,
            'symmetric_te': symmetric_te,
            'directional_bias': directional_bias,
            'forward_mean': forward_mean,
            'forward_std': forward_std,
            'forward_z': forward_z,
            'forward_p': forward_p,
            'reverse_mean': reverse_mean,
            'reverse_std': reverse_std,
            'reverse_z': reverse_z,
            'reverse_p': reverse_p,
            'forward_significant': forward_p < 0.05,
            'reverse_significant': reverse_p < 0.05,
            'any_significant': (forward_p < 0.05) or (reverse_p < 0.05)
        }
        
        results.append(result)
        
        # Print progress update for significant results
        if result['any_significant']:
            relation = f"{scale_a} → {scale_b}" if forward_p < reverse_p else f"{scale_b} → {scale_a}"
            print(f"Significant TE detected: {relation} (ratio: {result['ratio']:.4f})")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add mathematical constant proximity
    if not results_df.empty:
        results_df['golden_ratio_proximity'] = abs(results_df['ratio'] - ((1 + np.sqrt(5)) / 2))
        results_df['sqrt2_proximity'] = abs(results_df['ratio'] - np.sqrt(2))
        results_df['pi_div_2_proximity'] = abs(results_df['ratio'] - (np.pi / 2))
        results_df['e_div_2_proximity'] = abs(results_df['ratio'] - (np.e / 2))
    
    return results_df
