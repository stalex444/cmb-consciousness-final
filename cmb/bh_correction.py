"""
Benjamini-Hochberg Correction for Transfer Entropy Analysis

This module implements the Benjamini-Hochberg procedure for controlling 
the false discovery rate in multiple comparison tests, specifically for 
transfer entropy analysis of CMB data.
"""

import numpy as np
import pandas as pd

def apply_bh_correction(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction to p-values
    
    Parameters:
    - p_values: List or array of p-values
    - alpha: False discovery rate (default 0.05)
    
    Returns:
    - Array of booleans indicating which p-values are significant after correction
    """
    # Convert to numpy array if not already
    p_values = np.array(p_values)
    
    # Get number of tests
    m = len(p_values)
    
    # Get ranks (ascending order)
    ranks = np.argsort(p_values)
    
    # Calculate BH critical values
    critical_values = np.arange(1, m+1) / m * alpha
    
    # Find which p-values are significant
    significant = np.zeros_like(p_values, dtype=bool)
    
    # Sort p-values
    sorted_p = p_values[ranks]
    
    # Find the largest p-value that is less than its critical value
    max_significant_idx = -1
    for i in range(m):
        if sorted_p[i] <= critical_values[i]:
            max_significant_idx = i
    
    # Mark all p-values up to max_significant_idx as significant
    if max_significant_idx >= 0:
        significant[ranks[:max_significant_idx+1]] = True
    
    return significant

def apply_bh_to_te_results(results_df, alpha=0.05):
    """
    Apply BH correction to transfer entropy results
    
    Parameters:
    - results_df: DataFrame with transfer entropy results
    - alpha: False discovery rate (default 0.05)
    
    Returns:
    - Updated DataFrame with BH-corrected significance
    """
    # Make a copy to avoid modifying the original
    df = results_df.copy()
    
    # Get p-values (both forward and reverse)
    if 'forward_p' in df.columns and 'reverse_p' in df.columns:
        forward_p = df['forward_p'].values
        reverse_p = df['reverse_p'].values
    else:
        # Alternative column names
        forward_p = df['forward_p_value'].values if 'forward_p_value' in df.columns else df['p_value_forward'].values
        reverse_p = df['reverse_p_value'].values if 'reverse_p_value' in df.columns else df['p_value_reverse'].values
    
    # Apply BH correction separately to forward and reverse p-values
    forward_significant = apply_bh_correction(forward_p, alpha)
    reverse_significant = apply_bh_correction(reverse_p, alpha)
    
    # Update significance flags
    df['forward_significant_bh'] = forward_significant
    df['reverse_significant_bh'] = reverse_significant
    df['any_significant_bh'] = df['forward_significant_bh'] | df['reverse_significant_bh']
    
    return df

def analyze_bh_corrected_results(planck_results, wmap_results, alpha=0.05):
    """
    Apply BH correction and analyze results for both datasets
    
    Parameters:
    - planck_results: DataFrame with Planck transfer entropy results
    - wmap_results: DataFrame with WMAP transfer entropy results
    - alpha: False discovery rate (default 0.05)
    
    Returns:
    - Dictionary with analysis results
    """
    # Apply BH correction
    planck_bh = apply_bh_to_te_results(planck_results, alpha)
    wmap_bh = apply_bh_to_te_results(wmap_results, alpha)
    
    # Count significant pairs
    planck_sig_count = planck_bh['any_significant_bh'].sum()
    wmap_sig_count = wmap_bh['any_significant_bh'].sum()
    
    # Get significant pairs
    planck_sig_pairs = set([
        (row['scale_a'], row['scale_b']) 
        for _, row in planck_bh[planck_bh['any_significant_bh']].iterrows()
    ])
    wmap_sig_pairs = set([
        (row['scale_a'], row['scale_b']) 
        for _, row in wmap_bh[wmap_bh['any_significant_bh']].iterrows()
    ])
    
    # Find common significant pairs
    common_sig_pairs = planck_sig_pairs.intersection(wmap_sig_pairs)
    
    # Extract details of common significant pairs
    common_pairs_details = []
    for scale_a, scale_b in common_sig_pairs:
        planck_row = planck_bh[(planck_bh['scale_a'] == scale_a) & (planck_bh['scale_b'] == scale_b)].iloc[0]
        wmap_row = wmap_bh[(wmap_bh['scale_a'] == scale_a) & (wmap_bh['scale_b'] == scale_b)].iloc[0]
        
        common_pairs_details.append({
            'scale_a': scale_a,
            'scale_b': scale_b,
            'ratio': scale_b / scale_a,
            'planck_forward_z': planck_row.get('forward_z', planck_row.get('z_score_forward')),
            'planck_reverse_z': planck_row.get('reverse_z', planck_row.get('z_score_reverse')),
            'wmap_forward_z': wmap_row.get('forward_z', wmap_row.get('z_score_forward')),
            'wmap_reverse_z': wmap_row.get('reverse_z', wmap_row.get('z_score_reverse')),
        })
    
    # Get significant pairs with golden ratio relationships
    golden_ratio = (1 + np.sqrt(5)) / 2  # ~1.618
    sqrt2 = np.sqrt(2)  # ~1.414
    
    planck_golden_ratio_pairs = [
        (row['scale_a'], row['scale_b'], row['scale_b']/row['scale_a'])
        for _, row in planck_bh[planck_bh['any_significant_bh']].iterrows()
        if abs(row['scale_b']/row['scale_a'] - golden_ratio) < 0.01
    ]
    
    planck_sqrt2_pairs = [
        (row['scale_a'], row['scale_b'], row['scale_b']/row['scale_a'])
        for _, row in planck_bh[planck_bh['any_significant_bh']].iterrows()
        if abs(row['scale_b']/row['scale_a'] - sqrt2) < 0.01
    ]
    
    wmap_golden_ratio_pairs = [
        (row['scale_a'], row['scale_b'], row['scale_b']/row['scale_a'])
        for _, row in wmap_bh[wmap_bh['any_significant_bh']].iterrows()
        if abs(row['scale_b']/row['scale_a'] - golden_ratio) < 0.01
    ]
    
    wmap_sqrt2_pairs = [
        (row['scale_a'], row['scale_b'], row['scale_b']/row['scale_a'])
        for _, row in wmap_bh[wmap_bh['any_significant_bh']].iterrows()
        if abs(row['scale_b']/row['scale_a'] - sqrt2) < 0.01
    ]
    
    # Return results
    results = {
        'planck_total_significant': planck_sig_count,
        'planck_percent_significant': planck_sig_count / len(planck_bh) * 100,
        'wmap_total_significant': wmap_sig_count,
        'wmap_percent_significant': wmap_sig_count / len(wmap_bh) * 100,
        'common_significant_count': len(common_sig_pairs),
        'common_significant_details': common_pairs_details,
        'planck_golden_ratio_pairs': planck_golden_ratio_pairs,
        'planck_sqrt2_pairs': planck_sqrt2_pairs,
        'wmap_golden_ratio_pairs': wmap_golden_ratio_pairs,
        'wmap_sqrt2_pairs': wmap_sqrt2_pairs,
        'planck_df': planck_bh,
        'wmap_df': wmap_bh
    }
    
    return results
