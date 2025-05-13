#!/usr/bin/env python3
"""
Apply Benjamini-Hochberg Correction to Existing CMB Analysis Results

This script takes the existing results from the CMB sequence analysis and applies
the Benjamini-Hochberg correction to control the false discovery rate across multiple tests.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg procedure for controlling false discovery rate
    
    Parameters:
    - p_values: Array of p-values
    - alpha: False discovery rate threshold (default: 0.05)
    
    Returns:
    - Boolean array indicating which p-values are significant after correction
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)
    
    if n_tests == 0:
        return np.array([])
    
    # Create array of indices to return in original order
    order = np.argsort(p_values)
    rank = np.empty_like(order)
    rank[order] = np.arange(n_tests)
    
    # Calculate Benjamini-Hochberg critical values
    critical_values = (rank + 1) / n_tests * alpha
    
    # Find which p-values are significant
    significant = p_values <= critical_values
    
    # Find the largest significant p-value and make all smaller p-values significant
    if significant.any():
        significant = p_values <= p_values[order][significant[order][-1]]
    
    return significant

def apply_bh_to_cmb_results(results_dir, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction to results from CMB sequence analysis
    
    Parameters:
    - results_dir: Directory containing CMB sequence analysis results
    - alpha: False discovery rate threshold (default: 0.05)
    
    Returns:
    - Dictionary with corrected results
    """
    # Find the WMAP and Planck results files
    wmap_results_file = os.path.join(results_dir, "extended_wmap_sequence_analysis", "results.json")
    planck_results_file = os.path.join(results_dir, "extended_planck_sequence_analysis", "results.json")
    
    if not os.path.exists(wmap_results_file) or not os.path.exists(planck_results_file):
        print(f"Results files not found in {results_dir}")
        print(f"Looking for: {wmap_results_file} and {planck_results_file}")
        return None
    
    # Load results
    with open(wmap_results_file, 'r') as f:
        wmap_results = json.load(f)
    
    with open(planck_results_file, 'r') as f:
        planck_results = json.load(f)
    
    # Extract p-values and test information
    all_tests = []
    
    # Process WMAP results
    for seq_name, seq_data in wmap_results['sequence_analysis'].items():
        for const_name, test_data in seq_data.items():
            if 'p_value' in test_data:
                all_tests.append({
                    'dataset': 'WMAP',
                    'sequence': seq_name,
                    'constant': const_name,
                    'p_value': test_data['p_value'],
                    'z_score': test_data.get('z_score', 0),
                    'real_count': test_data.get('real_count', 0),
                    'surrogate_mean': test_data.get('surrogate_mean', 0),
                    'significant': test_data.get('significant', False)
                })
    
    # Process Planck results
    for seq_name, seq_data in planck_results['sequence_analysis'].items():
        for const_name, test_data in seq_data.items():
            if 'p_value' in test_data:
                all_tests.append({
                    'dataset': 'Planck',
                    'sequence': seq_name,
                    'constant': const_name,
                    'p_value': test_data['p_value'],
                    'z_score': test_data.get('z_score', 0),
                    'real_count': test_data.get('real_count', 0),
                    'surrogate_mean': test_data.get('surrogate_mean', 0),
                    'significant': test_data.get('significant', False)
                })
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(all_tests)
    
    if len(df) == 0:
        print("No tests found in results files.")
        return None
    
    # Apply BH correction separately to each dataset
    df['significant_bh'] = False  # Initialize column
    
    for dataset in ['WMAP', 'Planck']:
        # Get indices for this dataset
        dataset_indices = df.index[df['dataset'] == dataset].tolist()
        
        if len(dataset_indices) > 0:
            # Get p-values for this dataset
            p_values = df.loc[dataset_indices, 'p_value'].values
            
            # Apply BH correction
            significant_bh = benjamini_hochberg_correction(p_values, alpha)
            
            # Update dataframe at specific indices
            for i, idx in enumerate(dataset_indices):
                if i < len(significant_bh):  # Safety check
                    df.at[idx, 'significant_bh'] = bool(significant_bh[i])
    
    # Count significant results
    sig_before = df['significant'].sum()
    sig_after = df['significant_bh'].sum()
    
    print(f"Total tests: {len(df)}")
    print(f"Significant results before correction: {sig_before}/{len(df)} ({sig_before/len(df)*100:.1f}%)")
    print(f"Significant results after correction: {sig_after}/{len(df)} ({sig_after/len(df)*100:.1f}%)")
    
    # Print significant findings by dataset
    for dataset in ['WMAP', 'Planck']:
        dataset_df = df[df['dataset'] == dataset]
        sig_count = dataset_df['significant_bh'].sum()
        print(f"\n{dataset} significant after BH correction: {sig_count}/{len(dataset_df)} ({sig_count/len(dataset_df)*100:.1f}%)")
        
        if sig_count > 0:
            sig_df = dataset_df[dataset_df['significant_bh']]
            print(f"\nSignificant {dataset} findings:")
            
            # Group by sequence type
            for sequence in sig_df['sequence'].unique():
                seq_df = sig_df[sig_df['sequence'] == sequence]
                print(f"\n  {sequence} sequence:")
                
                for _, row in seq_df.iterrows():
                    print(f"    {row['constant']}: p={row['p_value']:.6f}, z={row['z_score']:.2f}, "
                          f"real={row['real_count']}, surr={row['surrogate_mean']:.2f}")
    
    # Write results to CSV
    output_dir = os.path.join(results_dir, f"bh_corrected_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(os.path.join(output_dir, "bh_corrected_all_tests.csv"), index=False)
    
    # Write significant findings to CSV
    sig_df = df[df['significant_bh']]
    if len(sig_df) > 0:
        sig_df.to_csv(os.path.join(output_dir, "bh_corrected_significant.csv"), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot WMAP results
    wmap_df = df[df['dataset'] == 'WMAP']
    if len(wmap_df) > 0:
        plt.scatter(wmap_df['p_value'], wmap_df['z_score'], label='WMAP (Not Significant)', 
                   alpha=0.6, s=50, color='blue', marker='o')
        
        wmap_sig = wmap_df[wmap_df['significant_bh']]
        if len(wmap_sig) > 0:
            plt.scatter(wmap_sig['p_value'], wmap_sig['z_score'], label='WMAP (BH Significant)', 
                       alpha=1.0, s=80, color='darkblue', marker='*')
    
    # Plot Planck results
    planck_df = df[df['dataset'] == 'Planck']
    if len(planck_df) > 0:
        plt.scatter(planck_df['p_value'], planck_df['z_score'], label='Planck (Not Significant)', 
                   alpha=0.6, s=50, color='red', marker='^')
        
        planck_sig = planck_df[planck_df['significant_bh']]
        if len(planck_sig) > 0:
            plt.scatter(planck_sig['p_value'], planck_sig['z_score'], label='Planck (BH Significant)', 
                       alpha=1.0, s=80, color='darkred', marker='*')
    
    # Set scale and labels
    plt.xscale('log')
    plt.xlabel('p-value (log scale)')
    plt.ylabel('z-score')
    plt.title(f'CMB Sequence Analysis Results with BH Correction (α={alpha})')
    plt.axhline(y=1.96, color='green', linestyle='--', alpha=0.5, label='p=0.05 (z=1.96)')
    plt.axvline(x=0.05, color='green', linestyle='--', alpha=0.5)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bh_corrected_results.png"), dpi=300)
    
    # Create a more detailed HTML report
    html_path = os.path.join(output_dir, "bh_corrected_results.html")
    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>CMB Sequence Analysis with BH Correction</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .significant {{ background-color: #e8f7f2; font-weight: bold; }}
        .header {{ background-color: #3498db; color: white; padding: 10px; }}
        .section {{ margin-top: 20px; }}
    </style>
</head>
<body>
    <h1>CMB Sequence Analysis with Benjamini-Hochberg Correction</h1>
    <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>BH Correction with α:</strong> {alpha}</p>
    
    <div class="section">
        <h2>Statistical Summary</h2>
        <p>Total tests: {len(df)}</p>
        <p>Significant before correction: {sig_before}/{len(df)} ({sig_before/len(df)*100:.1f}%)</p>
        <p>Significant after correction: {sig_after}/{len(df)} ({sig_after/len(df)*100:.1f}%)</p>
        
        <h3>By Dataset</h3>
        <table>
            <tr>
                <th>Dataset</th>
                <th>Total Tests</th>
                <th>Significant Before BH</th>
                <th>Significant After BH</th>
            </tr>
""")
        
        for dataset in ['WMAP', 'Planck']:
            dataset_df = df[df['dataset'] == dataset]
            sig_before_ds = dataset_df['significant'].sum()
            sig_after_ds = dataset_df['significant_bh'].sum()
            
            f.write(f"""
            <tr>
                <td>{dataset}</td>
                <td>{len(dataset_df)}</td>
                <td>{sig_before_ds} ({sig_before_ds/len(dataset_df)*100:.1f}%)</td>
                <td>{sig_after_ds} ({sig_after_ds/len(dataset_df)*100:.1f}%)</td>
            </tr>
            """)
        
        f.write("""
        </table>
    </div>
    
    <div class="section">
        <h2>Significant Results After BH Correction</h2>
        <table>
            <tr>
                <th>Dataset</th>
                <th>Sequence</th>
                <th>Constant</th>
                <th>p-value</th>
                <th>z-score</th>
                <th>Real Count</th>
                <th>Surrogate Mean</th>
            </tr>
        """)
        
        for _, row in sig_df.sort_values(['dataset', 'sequence', 'p_value']).iterrows():
            f.write(f"""
            <tr class="significant">
                <td>{row['dataset']}</td>
                <td>{row['sequence']}</td>
                <td>{row['constant']}</td>
                <td>{row['p_value']:.6f}</td>
                <td>{row['z_score']:.2f}</td>
                <td>{row['real_count']}</td>
                <td>{row['surrogate_mean']:.2f}</td>
            </tr>
            """)
        
        if len(sig_df) == 0:
            f.write("""
            <tr>
                <td colspan="7" style="text-align: center;">No significant results after BH correction</td>
            </tr>
            """)
        
        f.write("""
        </table>
    </div>
    
    <div class="section">
        <h2>Visualization</h2>
        <img src="bh_corrected_results.png" alt="BH Corrected Results" style="max-width: 100%;">
    </div>
    
    <div class="section">
        <h2>Conclusion</h2>
        <p>
            The Benjamini-Hochberg correction helps control the false discovery rate in multiple testing scenarios.
            It reduces the number of significant findings but increases confidence that the remaining findings are genuine.
        </p>
    </div>
</body>
</html>
        """)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"HTML report: {html_path}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply BH correction to CMB sequence analysis results")
    parser.add_argument("--results_dir", default="results", help="Directory containing CMB sequence analysis results")
    parser.add_argument("--alpha", type=float, default=0.05, help="False discovery rate for BH correction")
    
    args = parser.parse_args()
    
    apply_bh_to_cmb_results(args.results_dir, args.alpha)
