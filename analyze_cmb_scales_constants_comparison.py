#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMAP vs Planck CMB Scale Constants Analysis

This script performs a comprehensive analysis of mathematical patterns in 
CMB data from both WMAP and Planck datasets, focusing on the relationship
between multipole sequences (Fibonacci, Primes, Powers of 2) and mathematical
constants (phi, pi, e, etc.). 

It properly loads and processes each dataset separately, runs surrogate testing
with statistical validation, applies Benjamini-Hochberg correction, and provides
diagnostic visualizations to explain any discrepancies between datasets.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from statsmodels.stats.multitest import multipletests

# ========== SEQUENCE GENERATION FUNCTIONS ==========

def generate_fibonacci(max_l):
    """Generate Fibonacci numbers starting from 2 up to max_l."""
    fib = [2, 3]
    while True:
        next_fib = fib[-1] + fib[-2]
        if next_fib > max_l:
            break
        fib.append(next_fib)
    return fib

def generate_primes(max_l):
    """Generate prime numbers up to max_l using Sieve of Eratosthenes."""
    sieve = [True] * (max_l + 1)
    sieve[0:2] = [False, False]
    for p in range(2, int(np.sqrt(max_l)) + 1):
        if sieve[p]:
            for multiple in range(p*p, max_l + 1, p):
                sieve[multiple] = False
    return [p for p in range(2, max_l + 1) if sieve[p]]

def generate_powers_of_two(max_l):
    """Generate powers of 2 up to max_l."""
    powers = []
    p = 2
    while p <= max_l:
        powers.append(p)
        p *= 2
    return powers

# ========== RATIO ANALYSIS FUNCTIONS ==========

def compute_consecutive_ratios(seq):
    """Compute ratios of consecutive terms: seq[i+1] / seq[i]."""
    if len(seq) < 2:
        return []
    return [seq[i+1] / seq[i] for i in range(len(seq)-1)]

def count_close_ratios(ratios, constant, tolerance):
    """Count how many ratios are within tolerance of a constant."""
    return sum(1 for r in ratios if abs(r - constant) <= tolerance * constant)

# ========== SURROGATE TESTING FUNCTIONS ==========

def generate_surrogate_sequence(all_l, seq_length):
    """Randomly select and sort seq_length distinct multipoles from all_l."""
    # Ensure we don't try to sample more elements than available
    if seq_length > len(all_l):
        print(f"Warning: Requested {seq_length} elements but only {len(all_l)} available.")
        seq_length = min(seq_length, len(all_l))
    
    return sorted(np.random.choice(all_l, seq_length, replace=False))

def worker(args):
    """Worker function for parallel surrogate testing."""
    worker_id, n_surrogates, all_l, seq_length, constant, tolerance, log_frequency = args
    surrogate_counts = []
    
    # Process surrogates assigned to this worker
    for i in range(n_surrogates):
        surrogate_seq = generate_surrogate_sequence(all_l, seq_length)
        surrogate_ratios = compute_consecutive_ratios(surrogate_seq)
        count = count_close_ratios(surrogate_ratios, constant, tolerance)
        surrogate_counts.append(count)
        
        # Log progress periodically
        if (i+1) % log_frequency == 0:
            print(f"Worker {worker_id}: Completed {i+1}/{n_surrogates} surrogates")
    
    return surrogate_counts

def perform_parallel_surrogate_test(all_l, seq_length, constant, tolerance, n_surrogates, n_processes=None):
    """Perform surrogate testing in parallel."""
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Limit to a reasonable number of processes
    n_processes = min(n_processes, 10)
    
    # Prepare arguments for each worker
    surrogates_per_worker = n_surrogates // n_processes
    remainder = n_surrogates % n_processes
    log_frequency = max(1, surrogates_per_worker // 10)  # Log approximately 10 times per worker
    
    args_list = [
        (i, surrogates_per_worker + (1 if i < remainder else 0),
         all_l, seq_length, constant, tolerance, log_frequency)
        for i in range(n_processes)
    ]
    
    # Verify total surrogates
    total_surrogates = sum(args[1] for args in args_list)
    print(f"Distributing {total_surrogates} surrogates across {n_processes} processes")
    assert total_surrogates == n_surrogates, f"Expected {n_surrogates} but got {total_surrogates}"
    
    # Run workers in parallel
    start_time = time.time()
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(worker, args_list), total=n_processes, desc="Processing"))
    
    # Combine results from all workers
    surrogate_counts = []
    for result in results:
        surrogate_counts.extend(result)
    
    # Verify count
    print(f"Total surrogates computed: {len(surrogate_counts)}")
    assert len(surrogate_counts) == n_surrogates, \
        f"Expected {n_surrogates} but got {len(surrogate_counts)}"
    
    end_time = time.time()
    duration = end_time - start_time
    processing_rate = n_surrogates / duration if duration > 0 else 0
    
    print(f"Completed {n_surrogates} surrogate tests in {duration:.1f}s")
    print(f"Processing rate: {processing_rate:.1f} surrogates/second")
    
    return surrogate_counts

# ========== VISUALIZATION FUNCTIONS ==========

def plot_fibonacci_cl(wmap_data, planck_data, fib_multipoles, output_dir):
    """Plot Cl values at Fibonacci multipoles for WMAP and Planck."""
    # Extract the multipoles and Cl values from the datasets
    wmap_l = wmap_data[:, 0].astype(int)
    wmap_cl = wmap_data[:, 1]
    planck_l = planck_data[:, 0].astype(int)
    planck_cl = planck_data[:, 1]
    
    # Find Fibonacci multipoles present in each dataset
    wmap_fib_indices = [np.where(wmap_l == l)[0][0] for l in fib_multipoles if l in wmap_l]
    planck_fib_indices = [np.where(planck_l == l)[0][0] for l in fib_multipoles if l in planck_l]
    
    wmap_fib_l = [wmap_l[i] for i in wmap_fib_indices]
    wmap_fib_cl = [wmap_cl[i] for i in wmap_fib_indices]
    planck_fib_l = [planck_l[i] for i in planck_fib_indices]
    planck_fib_cl = [planck_cl[i] for i in planck_fib_indices]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(wmap_fib_l, wmap_fib_cl, 'bo-', label='WMAP')
    plt.plot(planck_fib_l, planck_fib_cl, 'rs-', label='Planck')
    
    # Add labels for each point
    for i, l in enumerate(wmap_fib_l):
        plt.annotate(str(l), (wmap_fib_l[i], wmap_fib_cl[i]), xytext=(5, 5), 
                     textcoords='offset points', color='blue')
    for i, l in enumerate(planck_fib_l):
        plt.annotate(str(l), (planck_fib_l[i], planck_fib_cl[i]), xytext=(5, -10), 
                     textcoords='offset points', color='red')
    
    plt.xlabel('Multipole (l)', fontsize=12)
    plt.ylabel('Power Spectrum (Cl)', fontsize=12)
    plt.title('CMB Power Spectrum at Fibonacci Multipoles', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fibonacci_cl_comparison.png'), dpi=300)
    plt.close()
    
    # Also create a ratio plot
    if len(wmap_fib_l) > 1 and len(planck_fib_l) > 1:
        plt.figure(figsize=(12, 8))
        wmap_ratios = compute_consecutive_ratios(wmap_fib_l)
        planck_ratios = compute_consecutive_ratios(planck_fib_l)
        
        plt.plot(wmap_fib_l[1:], wmap_ratios, 'bo-', label='WMAP')
        plt.plot(planck_fib_l[1:], planck_ratios, 'rs-', label='Planck')
        
        plt.axhline(y=(1 + np.sqrt(5)) / 2, color='g', linestyle='--', label='φ (Golden Ratio)')
        
        plt.xlabel('Multipole (l)', fontsize=12)
        plt.ylabel('Ratio of Consecutive Fibonacci Multipoles', fontsize=12)
        plt.title('Consecutive Multipole Ratios at Fibonacci Points', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.savefig(os.path.join(output_dir, 'fibonacci_ratio_comparison.png'), dpi=300)
        plt.close()

def plot_sequences_and_constants(sequences, constants, dataset_name, output_dir):
    """Create visualization of sequences and their ratios relative to constants."""
    n_sequences = len(sequences)
    fig, axes = plt.subplots(n_sequences, 1, figsize=(12, 5 * n_sequences))
    
    if n_sequences == 1:
        axes = [axes]
    
    # Create a color map for constants
    constant_colors = plt.cm.tab10(np.linspace(0, 1, len(constants)))
    
    for i, (seq_name, seq) in enumerate(sequences.items()):
        if len(seq) < 2:
            continue
            
        ratios = compute_consecutive_ratios(seq)
        ax = axes[i]
        
        # Plot the ratios
        ax.plot(seq[1:], ratios, 'o-', label=f'{seq_name} Ratios')
        
        # Add lines for each constant
        for j, (const_name, constant) in enumerate(constants.items()):
            color = constant_colors[j]
            ax.axhline(y=constant, color=color, linestyle='--', 
                      label=f'{const_name}={constant:.6f}')
            
        ax.set_title(f"Consecutive Ratios in {seq_name} Sequence - {dataset_name}", fontsize=14)
        ax.set_xlabel("Multipole (l)", fontsize=12)
        ax.set_ylabel("Ratio (l+1/l)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Only show legend for the first subplot to avoid clutter
        if i == 0:
            ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_sequence_ratios.png'), dpi=300)
    plt.close()

# ========== MAIN ANALYSIS FUNCTIONS ==========

def load_and_preprocess_data(wmap_path, planck_path):
    """Load and preprocess WMAP and Planck data."""
    print("Loading and preprocessing data...")
    
    # Check if files exist
    if not os.path.exists(wmap_path):
        print(f"Error: WMAP data file {wmap_path} not found.")
        # Generate synthetic data for testing if file not found
        print("Generating synthetic WMAP data for testing...")
        wmap_data = np.column_stack((
            np.arange(2, 1200),  # Multipoles
            np.random.gamma(2, 0.5, 1198)  # Synthetic Cl values
        ))
    else:
        wmap_data = np.loadtxt(wmap_path)
        print(f"Loaded WMAP data: {wmap_data.shape[0]} multipoles")
    
    if not os.path.exists(planck_path):
        print(f"Error: Planck data file {planck_path} not found.")
        # Generate synthetic data for testing if file not found
        print("Generating synthetic Planck data for testing...")
        planck_data = np.column_stack((
            np.arange(2, 2500),  # Multipoles
            np.random.gamma(2, 0.5, 2498)  # Synthetic Cl values
        ))
    else:
        planck_data = np.loadtxt(planck_path)
        print(f"Loaded Planck data: {planck_data.shape[0]} multipoles")
    
    # Print the ranges of multipoles in each dataset
    wmap_l_min, wmap_l_max = int(min(wmap_data[:, 0])), int(max(wmap_data[:, 0]))
    planck_l_min, planck_l_max = int(min(planck_data[:, 0])), int(max(planck_data[:, 0]))
    
    print(f"WMAP multipole range: {wmap_l_min} to {wmap_l_max}")
    print(f"Planck multipole range: {planck_l_min} to {planck_l_max}")
    
    return wmap_data, planck_data

def analyze_dataset(data, dataset_name, constants, tolerance, n_surrogates, alpha, n_processes, output_dir):
    """Main analysis for a single dataset."""
    all_l = data[:, 0].astype(int)
    max_l = max(all_l)
    
    print(f"\n{'-'*50}")
    print(f"ANALYZING {dataset_name.upper()} DATASET")
    print(f"{'-'*50}")
    print(f"Maximum multipole: {max_l}")
    
    # Generate mathematical sequences
    sequences = {
        'Fibonacci': generate_fibonacci(max_l),
        'Primes': generate_primes(max_l),
        'Powers of 2': generate_powers_of_two(max_l)
    }
    
    # Count multipoles in each sequence and check if they exist in the dataset
    for seq_name, seq in sequences.items():
        found_multipoles = [l for l in seq if l in all_l]
        print(f"{seq_name}: {len(seq)} multipoles, {len(found_multipoles)} found in dataset")
        if len(found_multipoles) != len(seq):
            print(f"  Missing multipoles: {set(seq) - set(found_multipoles)}")
    
    # Create visualizations of sequences and their ratios
    print("\nGenerating sequence visualizations...")
    plot_sequences_and_constants(sequences, constants, dataset_name, output_dir)
    
    # Run analysis for each sequence and constant
    results = {}
    p_values = []
    test_labels = []
    
    print("\nRunning analysis for each sequence and constant...")
    for seq_name, seq in sequences.items():
        if len(seq) < 2:
            print(f"Sequence {seq_name} has less than 2 elements. Skipping.")
            continue
        
        # Compute consecutive ratios for the sequence
        ratios = compute_consecutive_ratios(seq)
        results[seq_name] = {}
        
        print(f"\nAnalysis for {seq_name} (length={len(seq)}, ratios={len(ratios)}):")
        for const_name, constant in constants.items():
            print(f"\nAnalyzing {const_name} ({constant:.4f})...")
            
            # Count ratios close to the constant in the real sequence
            real_count = count_close_ratios(ratios, constant, tolerance)
            print(f"  {seq_name}: {len(seq)} of {len(seq)} multipoles found in data")
            print(f"  Real count of close ratios: {real_count}")
            
            # Run surrogate testing
            print(f"  Starting {n_surrogates} surrogate tests using {n_processes} processes...")
            surrogate_counts = perform_parallel_surrogate_test(
                all_l, len(seq), constant, tolerance, n_surrogates, n_processes
            )
            
            # Compute statistics
            surrogate_mean = np.mean(surrogate_counts)
            surrogate_std = np.std(surrogate_counts)
            
            # Handle zero standard deviation case
            if surrogate_std > 0:
                z_score = (real_count - surrogate_mean) / surrogate_std
            else:
                z_score = 0 if real_count == surrogate_mean else float('inf') * np.sign(real_count - surrogate_mean)
            
            # Compute p-value
            p_value = sum(1 for x in surrogate_counts if x >= real_count) / n_surrogates
            
            # Ensure minimum p-value is not zero
            if p_value == 0 and real_count > 0:
                p_value = 1.0 / n_surrogates  # Minimum possible p-value
            
            # Store results
            results[seq_name][const_name] = {
                'real_count': real_count,
                'surrogate_mean': surrogate_mean,
                'surrogate_std': surrogate_std,
                'p_value': p_value,
                'z_score': z_score,
                'n_surrogates': n_surrogates
            }
            
            p_values.append(p_value)
            test_labels.append(f"{seq_name} - {const_name}")
            
            print(f"  {const_name}: real={real_count}, surr={surrogate_mean:.2f}±{surrogate_std:.2f}, p={p_value:.6f}, z={z_score:.2f}")
            
            # Create histogram plot for this result
            plt.figure(figsize=(10, 6))
            plt.hist(surrogate_counts, bins=max(10, int(max(surrogate_counts)+1)), alpha=0.7)
            plt.axvline(x=real_count, color='red', linestyle='--', label=f'Real count: {real_count}')
            plt.axvline(x=surrogate_mean, color='green', linestyle='-', label=f'Mean: {surrogate_mean:.2f}')
            plt.xlabel('Count of close ratios')
            plt.ylabel('Frequency')
            
            sig_marker = " ***" if p_value < 0.05 else ""
            plt.title(f'{dataset_name} - {seq_name} - {const_name} (p={p_value:.6f}{sig_marker})')
            plt.legend()
            plt.tight_layout()
            
            # Create plots directory if it doesn't exist
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Use a safe filename
            safe_const_name = const_name.replace('/', '_div_').replace(' ', '_')
            plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{seq_name}_{safe_const_name}_histogram.png"))
            plt.close()
    
    # Apply Benjamini-Hochberg correction
    print("\nApplying Benjamini-Hochberg correction to all tests...")
    start_time = time.time()
    
    if p_values:
        # Count significant tests before correction
        significant_before = sum(1 for p in p_values if p < alpha)
        
        # Apply BH correction
        is_significant, adjusted_p_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        
        # Count significant tests after correction
        significant_after = sum(is_significant)
        
        # Store adjusted p-values in results
        for i, (label, p_adj, sig) in enumerate(zip(test_labels, adjusted_p_values, is_significant)):
            seq_name, const_name = label.split(' - ')
            results[seq_name][const_name]['adjusted_p_value'] = p_adj
            results[seq_name][const_name]['significant'] = sig
        
        end_time = time.time()
        bh_time = end_time - start_time
        
        print(f"BH correction completed in {bh_time:.2f} seconds")
        print(f"Before BH correction: {significant_before}/{len(p_values)} tests significant ({significant_before/len(p_values)*100:.2f}%)")
        print(f"After BH correction: {significant_after}/{len(p_values)} tests significant ({significant_after/len(p_values)*100:.2f}%)")
        
        # Display significant results
        if significant_after > 0:
            print("\nSignificant tests after BH correction:")
            for label, p_adj, sig in zip(test_labels, adjusted_p_values, is_significant):
                if sig:
                    seq_name, const_name = label.split(' - ')
                    z = results[seq_name][const_name]['z_score']
                    print(f"  {label}: p={p_adj:.6f} (z={z:.2f})")
    else:
        print("No p-values to correct.")
    
    # Save results to CSV
    results_df = []
    for seq_name in results:
        for const_name in results[seq_name]:
            row = {
                'Dataset': dataset_name,
                'Sequence': seq_name,
                'Constant': const_name,
                'Constant_Value': constants[const_name],
                'Real_Count': results[seq_name][const_name]['real_count'],
                'Surrogate_Mean': results[seq_name][const_name]['surrogate_mean'],
                'Surrogate_Std': results[seq_name][const_name]['surrogate_std'],
                'P_Value': results[seq_name][const_name]['p_value'],
                'Z_Score': results[seq_name][const_name]['z_score']
            }
            if 'adjusted_p_value' in results[seq_name][const_name]:
                row['Adjusted_P_Value'] = results[seq_name][const_name]['adjusted_p_value']
                row['Significant'] = results[seq_name][const_name]['significant']
            results_df.append(row)
    
    results_df = pd.DataFrame(results_df)
    results_csv_path = os.path.join(output_dir, f"{dataset_name}_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to {results_csv_path}")
    
    return results, sequences, results_df

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Analyze mathematical patterns in CMB data')
    parser.add_argument('--wmap', type=str, default='data/wmap_tt_spectrum_9yr_v5.txt',
                        help='Path to WMAP power spectrum data')
    parser.add_argument('--planck', type=str, default='data/planck_tt_spectrum_2018.txt',
                        help='Path to Planck power spectrum data')
    parser.add_argument('--surrogates', type=int, default=1000,
                        help='Number of surrogate tests to run (default: 1000)')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of parallel processes (default: # of CPU cores)')
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='Tolerance for ratio matching (default: 0.01 = 1%)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for multiple testing correction (default: 0.05)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: results_YYYYMMDD_HHMMSS)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_cmb_scales_comparison_{timestamp}"
    else:
        output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Define mathematical constants
    constants = {
        'phi': (1 + np.sqrt(5)) / 2,  # Golden ratio ≈ 1.618
        'sqrt2': np.sqrt(2),          # Square root of 2 ≈ 1.414
        'sqrt3': np.sqrt(3),          # Square root of 3 ≈ 1.732
        'sqrt5': np.sqrt(5),          # Square root of 5 ≈ 2.236
        'pi': np.pi,                  # Pi ≈ 3.142
        'e': np.exp(1),               # Euler's number ≈ 2.718
        '2': 2.0,                     # 2
        'ln2': np.log(2),             # Natural log of 2 ≈ 0.693
        '1/phi': 2 / (1 + np.sqrt(5)),# Reciprocal of Golden Ratio ≈ 0.618
        'gamma': 0.5772156649015329   # Euler-Mascheroni constant ≈ 0.577
    }
    
    # Print configuration
    print("\nCMB SCALES AND CONSTANTS COMPARISON ANALYSIS")
    print("=" * 50)
    print(f"WMAP data: {args.wmap}")
    print(f"Planck data: {args.planck}")
    print(f"Number of surrogates: {args.surrogates}")
    print(f"Processes: {'Auto' if args.processes is None else args.processes}")
    print(f"Tolerance: {args.tolerance:.1%}")
    print(f"Alpha (significance level): {args.alpha}")
    print("Constants being analyzed:")
    for name, value in constants.items():
        print(f"  {name}: {value:.6f}")
    print("=" * 50)
    
    # Load data
    wmap_data, planck_data = load_and_preprocess_data(args.wmap, args.planck)
    
    # Plot Fibonacci Cl comparison
    fib_multipoles = generate_fibonacci(max(max(wmap_data[:, 0]), max(planck_data[:, 0])))
    plot_fibonacci_cl(wmap_data, planck_data, fib_multipoles, output_dir)
    
    # Analyze WMAP
    wmap_results, wmap_sequences, wmap_df = analyze_dataset(
        wmap_data, 'WMAP', constants, args.tolerance, args.surrogates,
        args.alpha, args.processes, output_dir
    )
    
    # Analyze Planck
    planck_results, planck_sequences, planck_df = analyze_dataset(
        planck_data, 'Planck', constants, args.tolerance, args.surrogates,
        args.alpha, args.processes, output_dir
    )
    
    # Create combined results CSV
    combined_df = pd.concat([wmap_df, planck_df])
    combined_csv_path = os.path.join(output_dir, "combined_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\nCombined results saved to {combined_csv_path}")
    
    # Generate comparison summary
    significant_wmap = wmap_df[wmap_df['Significant'] == True] if 'Significant' in wmap_df.columns else pd.DataFrame()
    significant_planck = planck_df[planck_df['Significant'] == True] if 'Significant' in planck_df.columns else pd.DataFrame()
    
    print("\nCOMPARISON SUMMARY")
    print("=" * 50)
    print(f"WMAP: {len(significant_wmap)} significant results")
    for _, row in significant_wmap.iterrows():
        print(f"  {row['Sequence']} - {row['Constant']}: p={row['Adjusted_P_Value']:.6f}, z={row['Z_Score']:.2f}")
    
    print(f"\nPlanck: {len(significant_planck)} significant results")
    for _, row in significant_planck.iterrows():
        print(f"  {row['Sequence']} - {row['Constant']}: p={row['Adjusted_P_Value']:.6f}, z={row['Z_Score']:.2f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
