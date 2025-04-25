#!/usr/bin/env python
"""
Statistical Utilities for CMB Analysis

This module provides common statistical functions used across CMB analysis modules,
including corrections for multiple comparisons and other statistical tests.
"""

import numpy as np
from scipy import stats

def apply_bh_correction(p_values):
    """Apply Benjamini-Hochberg correction to control false discovery rate
    
    Parameters:
    -----------
    p_values : array-like
        List or array of p-values to correct
    
    Returns:
    --------
    array-like
        Adjusted p-values, same shape as input
    """
    # Sort p-values and get original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]
    
    # Calculate BH critical values
    n = len(p_values)
    critical_values = np.arange(1, n+1) / n * 0.05  # Using 0.05 as alpha
    
    # Find largest p-value that passes BH criterion
    passing = sorted_p_values <= critical_values
    if not any(passing):
        return np.ones_like(p_values)  # No values pass
    
    # Get largest index that passed
    max_passing_idx = np.where(passing)[0].max()
    
    # Set adjusted p-values
    adjusted_p_values = np.ones_like(p_values)
    adjusted_p_values[sorted_indices[:max_passing_idx+1]] = sorted_p_values[:max_passing_idx+1] * n / (np.arange(1, max_passing_idx+2))
    
    return adjusted_p_values

def calculate_sigma_from_p(p_value):
    """Convert p-value to sigma (standard deviations from mean)
    
    Parameters:
    -----------
    p_value : float
        p-value (0-1)
    
    Returns:
    --------
    float
        Sigma value (standard deviations)
    """
    # For two-tailed test
    return stats.norm.ppf(1 - p_value/2) if p_value < 1 else 0

def apply_correction_to_results(results_dict, p_value_keys=None):
    """Apply BH correction to a dictionary of results
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing p-values to correct
    p_value_keys : list
        List of keys indicating p-values to correct
        
    Returns:
    --------
    dict
        Updated results with adjusted p-values and sigmas
    """
    if p_value_keys is None:
        p_value_keys = [k for k in results_dict.keys() if 'p_value' in k and 'adjusted' not in k]
    
    # Extract p-values
    p_values = [results_dict[k] for k in p_value_keys]
    
    # Apply correction
    adjusted_p_values = apply_bh_correction(p_values)
    
    # Update results with adjusted values
    for i, key in enumerate(p_value_keys):
        adjusted_key = key.replace('p_value', 'adjusted_p_value')
        results_dict[adjusted_key] = adjusted_p_values[i]
        
        # If there's a sigma value, calculate adjusted sigma
        sigma_key = key.replace('p_value', 'sigma')
        if sigma_key in results_dict:
            adjusted_sigma_key = sigma_key.replace('sigma', 'adjusted_sigma')
            results_dict[adjusted_sigma_key] = calculate_sigma_from_p(adjusted_p_values[i])
    
    return results_dict
