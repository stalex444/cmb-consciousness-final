import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import os
import multiprocessing
from functools import partial
import time
import json
from datetime import datetime

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Define mathematical constants at the module level
constants = {
    'phi': (1 + np.sqrt(5)) / 2,  # Golden ratio
    'sqrt2': np.sqrt(2),
    'pi': np.pi,
    'e': np.e,
    '2': 2.0
}

# Function to generate Fibonacci sequence up to max_l
def generate_fibonacci(max_l):
    """Generate Fibonacci numbers starting from 2 up to max_l."""
    fib = [2, 3]
    while True:
        next_fib = fib[-1] + fib[-2]
        if next_fib > max_l:
            break
        fib.append(next_fib)
    return fib

# Function to generate prime numbers up to max_l using Sieve of Eratosthenes
def generate_primes(max_l):
    """Generate prime numbers up to max_l."""
    sieve = [True] * (max_l + 1)
    sieve[0:2] = [False, False]
    for p in range(2, int(np.sqrt(max_l)) + 1):
        if sieve[p]:
            for multiple in range(p*p, max_l + 1, p):
                sieve[multiple] = False
    return [p for p in range(2, max_l + 1) if sieve[p]]

# Function to generate powers of 2 up to max_l
def generate_powers_of_two(max_l):
    """Generate powers of 2 up to max_l."""
    powers = []
    p = 2
    while p <= max_l:
        powers.append(p)
        p *= 2
    return powers

# Function to compute consecutive ratios in a sequence
def compute_consecutive_ratios(seq):
    """Compute ratios of consecutive terms: seq[i+1] / seq[i]."""
    return [seq[i+1] / seq[i] for i in range(len(seq)-1)]

# Function to count ratios close to a constant within tolerance
def count_close_ratios(ratios, constant, tolerance):
    """Count how many ratios are within tolerance of a constant."""
    return sum(1 for r in ratios if abs(r - constant) / constant < tolerance)

# Function to generate a surrogate sequence
def generate_surrogate_sequence(all_l, seq_length):
    """Randomly select and sort seq_length distinct multipoles from all_l."""
    return sorted(np.random.choice(all_l, seq_length, replace=False))

# Function to perform surrogate test
def perform_surrogate_test(all_l, seq_length, constant, tolerance, n_surrogates):
    """Perform surrogate test by generating random sequences and counting close ratios."""
    surrogate_counts = []
    for _ in range(n_surrogates):
        surrogate_seq = generate_surrogate_sequence(all_l, seq_length)
        surrogate_ratios = compute_consecutive_ratios(surrogate_seq)
        count = count_close_ratios(surrogate_ratios, constant, tolerance)
        surrogate_counts.append(count)
    return surrogate_counts

# Function to analyze a dataset
def analyze_dataset(file_path, dataset_name, n_surrogates=10000, tolerance=0.01):
    """Analyze a single CMB dataset."""
    print(f"\n{'=' * 80}")
    print(f"Analyzing {dataset_name} dataset with {n_surrogates} surrogate simulations")
    print(f"{'=' * 80}")
    
    # Load the data
    try:
        data = np.loadtxt(file_path)
        all_l = data[:, 0].astype(int)  # Extract multipoles
        max_l = max(all_l)
        print(f"Loaded {len(all_l)} multipoles from {dataset_name}, maximum l = {max_l}")
    except Exception as e:
        print(f"Error loading {dataset_name} data: {e}")
        return None

    # Generate sequences
    fib_seq = generate_fibonacci(max_l)
    prime_seq = generate_primes(max_l)
    power2_seq = generate_powers_of_two(max_l)
    sequences = {
        'Fibonacci': fib_seq,
        'Primes': prime_seq,
        'Powers of 2': power2_seq
    }

    # Print out the sequences for this dataset
    print(f"\nMathematical sequences in {dataset_name}:")
    for seq_name, seq in sequences.items():
        print(f"{seq_name}: {seq[:10]}... (total: {len(seq)} multipoles)")
    
    # Store results
    results = {
        'dataset': dataset_name,
        'max_l': int(max_l),
        'n_surrogates': n_surrogates,
        'tolerance': tolerance,
        'sequence_lengths': {seq_name: len(seq) for seq_name, seq in sequences.items()},
        'sequence_analysis': {}
    }
    
    # Analyze each sequence
    for seq_name, seq in sequences.items():
        if len(seq) < 2:
            print(f"Sequence {seq_name} has less than 2 elements. Skipping.")
            continue
            
        ratios = compute_consecutive_ratios(seq)
        print(f"\nAnalysis for {seq_name} in {dataset_name} (length={len(seq)}):")
        
        seq_results = {}
        for const_name, constant in constants.items():
            print(f"  Analyzing {const_name} ({constant:.4f})... ", end='', flush=True)
            start_time = time.time()
            
            # Count real occurrences
            real_count = count_close_ratios(ratios, constant, tolerance)
            
            # Run surrogate tests (this is the time-consuming part)
            surrogate_counts = perform_surrogate_test(all_l, len(seq), constant, tolerance, n_surrogates)
            
            # Calculate statistics
            mean_surrogate = np.mean(surrogate_counts)
            std_surrogate = np.std(surrogate_counts)
            if std_surrogate > 0:
                z_score = (real_count - mean_surrogate) / std_surrogate
            else:
                z_score = 0 if real_count == mean_surrogate else float('inf')
            p_value = np.sum(np.array(surrogate_counts) >= real_count) / n_surrogates
            p_value = max(p_value, 1.0 / n_surrogates)  # Minimum p-value based on number of surrogates
            
            # Create significance marker for visual identification
            sig_marker = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
            
            elapsed = time.time() - start_time
            print(f"done in {elapsed:.1f}s")
            print(f"    {const_name}: real={real_count}, surr={mean_surrogate:.2f}, p={p_value:.6f}{sig_marker}, z={z_score:.2f}")
            
            seq_results[const_name] = {
                'real_count': int(real_count),
                'surrogate_mean': float(mean_surrogate),
                'surrogate_std': float(std_surrogate),
                'p_value': float(p_value),
                'z_score': float(z_score),
                'significant': p_value < 0.05
            }
        
        results['sequence_analysis'][seq_name] = seq_results
    
    # Generate visualizations for this dataset
    fig, axes = plt.subplots(len(sequences), 1, figsize=(10, 5 * len(sequences)))
    if len(sequences) == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot
        
    for i, (seq_name, seq) in enumerate(sequences.items()):
        if len(seq) < 2:
            continue
        ratios = compute_consecutive_ratios(seq)
        ax = axes[i]
        ax.plot(range(len(ratios)), ratios, 'o-', label=seq_name)
        
        # Plot constant lines with different colors
        colors = ['r', 'g', 'b', 'm', 'c']
        for j, (const_name, constant) in enumerate(constants.items()):
            color = colors[j % len(colors)]
            ax.axhline(y=constant, color=color, linestyle='--',
                       label=f"{const_name}={constant:.4f}" if i == 0 else "")
                       
        ax.set_title(f"{dataset_name}: Consecutive Ratios for {seq_name}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Ratio")
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_ratios_plot.png')
    plt.close()
    
    # Apply K-means clustering for this dataset
    all_ratios = []
    for seq_name, seq in sequences.items():
        if len(seq) < 2:
            continue
        ratios = compute_consecutive_ratios(seq)
        all_ratios.extend([(r, seq_name) for r in ratios])

    # Add clustering results to the output
    results['clustering'] = {}
    if all_ratios:
        ratios_array = np.array([r[0] for r in all_ratios]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(ratios_array)
        centers = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_
        print(f"\nKMeans Clustering Results for {dataset_name}:")
        
        for cluster_id in range(3):
            cluster_ratios = [all_ratios[i] for i in range(len(all_ratios)) if labels[i] == cluster_id]
            seq_counts = Counter([seq for _, seq in cluster_ratios])
            print(f"Cluster {cluster_id}: center={centers[cluster_id]:.4f}, composition={seq_counts}")
            
            # Check if cluster center is close to any fundamental constant
            closest_constant = None
            min_diff_pct = float('inf')
            for const_name, constant in constants.items():
                diff_pct = abs(centers[cluster_id] - constant) / constant
                if diff_pct < min_diff_pct:
                    min_diff_pct = diff_pct
                    closest_constant = const_name
            
            proximity_msg = ""
            if min_diff_pct < 0.05:  # Within 5% of a constant
                const_val = constants[closest_constant]
                proximity_msg = f"Close to {closest_constant}={const_val:.4f} (within {min_diff_pct*100:.2f}%)"
                print(f"  {proximity_msg}")
            
            results['clustering'][f'cluster_{cluster_id}'] = {
                'center': float(centers[cluster_id]),
                'size': len(cluster_ratios),
                'composition': {k: v for k, v in seq_counts.items()},
                'closest_constant': closest_constant,
                'proximity': float(min_diff_pct)
            }
    else:
        print("No ratios available for clustering.")
    
    # Save detailed results to file
    output_dir = f"results/{dataset_name.lower()}_sequence_analysis"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nDetailed results saved to {output_dir}/results.json")
    return results

# Main analysis
if __name__ == "__main__":
    # Set up paths to the datasets
    wmap_path = '/Users/stephaniealexander/CascadeProjects/cmb-consciousness/data/wmap_power_spectrum.txt'
    planck_path = '/Users/stephaniealexander/CascadeProjects/cmb-consciousness/data/planck_power_spectrum.txt'
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Set analysis parameters
    n_surrogates = 10000  # 10,000 surrogate simulations
    tolerance = 0.01      # 1% tolerance
    
    print(f"Starting CMB Sequence Analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running {n_surrogates} surrogate simulations with {tolerance*100:.1f}% tolerance")
    print(f"Analyzing mathematical constants: phi, sqrt2, pi, e, 2")
    print(f"Looking for patterns in Fibonacci, Prime, and Powers of 2 sequences")
    start_time = time.time()

    # Analyze both datasets
    wmap_results = analyze_dataset(wmap_path, "WMAP", n_surrogates, tolerance)
    planck_results = analyze_dataset(planck_path, "Planck", n_surrogates, tolerance)
    
    # Create a combined visualization comparing both datasets
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Define sequences to compare across both datasets
    seq_names = ['Fibonacci', 'Primes', 'Powers of 2']
    dataset_names = ['WMAP', 'Planck']
    
    # Generate side-by-side comparison plots
    for i, seq_name in enumerate(seq_names):
        for j, dataset_name in enumerate(dataset_names):
            file_path = wmap_path if dataset_name == 'WMAP' else planck_path
            try:
                data = np.loadtxt(file_path)
                all_l = data[:, 0].astype(int)
                max_l = max(all_l)
                
                # Generate the sequence for this dataset
                if seq_name == 'Fibonacci':
                    seq = generate_fibonacci(max_l)
                elif seq_name == 'Primes':
                    seq = generate_primes(max_l)
                else:  # Powers of 2
                    seq = generate_powers_of_two(max_l)
                
                if len(seq) < 2:
                    axes[i, j].text(0.5, 0.5, f"No {seq_name} sequence\nfor {dataset_name}", 
                                     ha='center', va='center', transform=axes[i, j].transAxes)
                    continue
                    
                # Compute ratios and plot
                ratios = compute_consecutive_ratios(seq)
                axes[i, j].plot(range(len(ratios)), ratios, 'o-', label=f"{dataset_name} {seq_name}")
                
                # Add horizontal lines for constants
                colors = ['r', 'g', 'b', 'm', 'c']
                for k, (const_name, constant) in enumerate(constants.items()):
                    color = colors[k % len(colors)]
                    axes[i, j].axhline(y=constant, color=color, linestyle='--',
                                 label=f"{const_name}={constant:.4f}" if i == 0 and j == 0 else "")
                
                # Set plot properties
                axes[i, j].set_title(f"{dataset_name}: {seq_name} Ratios")
                axes[i, j].set_xlabel("Index")
                axes[i, j].set_ylabel("Ratio")
                if i == 0 and j == 0:
                    axes[i, j].legend(loc='upper right')
                    
            except Exception as e:
                axes[i, j].text(0.5, 0.5, f"Error processing {dataset_name}\n{str(e)}", 
                                 ha='center', va='center', transform=axes[i, j].transAxes)
    
    plt.tight_layout()
    plt.savefig('combined_cmb_analysis.png', dpi=300)
    
    # Generate a comparison HTML report
    with open('cmb_sequence_analysis_report.html', 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>CMB Sequence Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ display: flex; justify-content: space-between; }}
        .dataset {{ width: 48%; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .significant {{ color: red; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin-top: 20px; }}
    </style>
</head>
<body>
    <h1>CMB Sequence Analysis Report</h1>
    <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Parameters:</strong> {n_surrogates:,} surrogate simulations, {tolerance*100:.1f}% tolerance</p>
    
    <h2>Summary of Findings</h2>
    <div class="container">
    """)
        
        # Add WMAP summary
        f.write("""<div class="dataset">
            <h3>WMAP Dataset</h3>
            <table>
                <tr><th>Sequence</th><th>Constant</th><th>Real Count</th><th>Expected</th><th>p-value</th><th>z-score</th></tr>
        """)
        
        # Add WMAP results to table
        for seq_name in seq_names:
            if seq_name not in wmap_results['sequence_analysis']:
                continue
            for const_name in constants.keys():
                result = wmap_results['sequence_analysis'][seq_name][const_name]
                sig_class = "significant" if result['p_value'] < 0.05 else ""
                f.write(f"""<tr class="{sig_class}">""")
                f.write(f"""<td>{seq_name}</td><td>{const_name}</td><td>{result['real_count']}</td>""")
                f.write(f"""<td>{result['surrogate_mean']:.2f}</td><td>{result['p_value']:.6f}</td><td>{result['z_score']:.2f}</td>""")
                f.write("""</tr>""")
        
        f.write("""</table></div>""")
        
        # Add Planck summary
        f.write("""<div class="dataset">
            <h3>Planck Dataset</h3>
            <table>
                <tr><th>Sequence</th><th>Constant</th><th>Real Count</th><th>Expected</th><th>p-value</th><th>z-score</th></tr>
        """)
        
        # Add Planck results to table
        for seq_name in seq_names:
            if seq_name not in planck_results['sequence_analysis']:
                continue
            for const_name in constants.keys():
                result = planck_results['sequence_analysis'][seq_name][const_name]
                sig_class = "significant" if result['p_value'] < 0.05 else ""
                f.write(f"""<tr class="{sig_class}">""")
                f.write(f"""<td>{seq_name}</td><td>{const_name}</td><td>{result['real_count']}</td>""")
                f.write(f"""<td>{result['surrogate_mean']:.2f}</td><td>{result['p_value']:.6f}</td><td>{result['z_score']:.2f}</td>""")
                f.write("""</tr>""")
        
        f.write("""</table></div></div>""")
        
        # Add images
        f.write("""<h2>Visualizations</h2>
            <h3>Combined Comparison</h3>
            <img src="combined_cmb_analysis.png" alt="Combined Analysis Plot">
            
            <div class="container">
                <div class="dataset">
                    <h3>WMAP Detailed Plot</h3>
                    <img src="WMAP_ratios_plot.png" alt="WMAP Analysis Plot">
                </div>
                <div class="dataset">
                    <h3>Planck Detailed Plot</h3>
                    <img src="Planck_ratios_plot.png" alt="Planck Analysis Plot">
                </div>
            </div>
            
            <h2>Conclusions</h2>
            <p>This analysis examined mathematical patterns in the CMB power spectrum multipoles from both the WMAP and Planck datasets.</p>
            <p>The most striking findings are:</p>
            <ul>
        """)
        
        # Add significant findings
        for dataset_name, results in [('WMAP', wmap_results), ('Planck', planck_results)]:
            for seq_name in seq_names:
                if seq_name not in results['sequence_analysis']:
                    continue
                for const_name, result in results['sequence_analysis'][seq_name].items():
                    if result['significant']:
                        f.write(f"""<li><strong>{dataset_name}:</strong> {seq_name} sequence shows {result['real_count']} ratios close to {const_name} """)
                        f.write(f"""(expected {result['surrogate_mean']:.2f}, p={result['p_value']:.6f}, z={result['z_score']:.2f})</li>""")
        
        f.write("""</ul>
        </body>
        </html>""")
    
    # Print completion message
    elapsed = time.time() - start_time
    print(f"\nCMB Sequence Analysis completed in {elapsed/60:.2f} minutes")
    print(f"Results saved to:")
    print(f"  - results/wmap_sequence_analysis/results.json")
    print(f"  - results/planck_sequence_analysis/results.json")
    print(f"  - cmb_sequence_analysis_report.html")
    print(f"  - combined_cmb_analysis.png")
    print(f"  - Analyzed with {n_surrogates:,} simulations as requested")
    print(f"  - WMAP and Planck datasets analyzed simultaneously with {n_surrogates:,} simulations")
