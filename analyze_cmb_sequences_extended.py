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
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Define mathematical constants at the module level for use throughout the script
constants = {
    'phi': (1 + np.sqrt(5)) / 2,  # Golden ratio ≈ 1.618
    'sqrt2': np.sqrt(2),          # Square root of 2 ≈ 1.414
    'sqrt3': np.sqrt(3),          # Square root of 3 ≈ 1.732
    'sqrt5': np.sqrt(5),          # Square root of 5 ≈ 2.236
    'pi': np.pi,                  # Pi ≈ 3.142
    'e': np.e,                    # Euler's number ≈ 2.718
    '2': 2.0,                     # 2
    'ln2': np.log(2),             # Natural logarithm of 2 ≈ 0.693
    '1/phi': 2/(1 + np.sqrt(5)),  # Reciprocal of Golden ratio ≈ 0.618
    'gamma': 0.57721566490153286  # Euler-Mascheroni constant ≈ 0.577
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

    # Initialize results dictionary
    results = {
        'dataset': dataset_name,
        'max_l': max_l,
        'n_surrogates': n_surrogates,
        'tolerance': tolerance,
        'sequence_lengths': {seq_name: len(seq) for seq_name, seq in sequences.items()},
        'sequence_analysis': {},
        'clustering': {}
    }

    # Analyze each sequence
    for seq_name, seq in sequences.items():
        results['sequence_analysis'][seq_name] = {}
        
        if len(seq) < 2:
            print(f"Sequence {seq_name} has less than 2 elements. Skipping.")
            continue
            
        ratios = compute_consecutive_ratios(seq)
        print(f"\nAnalysis for {seq_name} in {dataset_name} (length={len(seq)}):")
        
        for const_name, constant in constants.items():
            print(f"  Analyzing {const_name} ({constant:.4f})...", end=" ", flush=True)
            start_time = time.time()
            
            real_count = count_close_ratios(ratios, constant, tolerance)
            surrogate_counts = perform_surrogate_test(all_l, len(seq), constant, tolerance, n_surrogates)
            mean_surrogate = np.mean(surrogate_counts)
            std_surrogate = np.std(surrogate_counts)
            
            if std_surrogate > 0:
                z_score = (real_count - mean_surrogate) / std_surrogate
            else:
                z_score = 0 if real_count == mean_surrogate else float('inf')
                
            p_value = np.sum(np.array(surrogate_counts) >= real_count) / n_surrogates
            significant = p_value < 0.05
            
            elapsed = time.time() - start_time
            print(f"done in {elapsed:.1f}s")
            
            # Add significance asterisks for visual identification
            sig_marker = "***" if p_value <= 0.001 else "**" if p_value <= 0.01 else "*" if p_value <= 0.05 else ""
            print(f"    {const_name}: real={real_count}, surr={mean_surrogate:.2f}, p={p_value:.6f}{sig_marker}, z={z_score:.2f}")
            
            # Store results
            results['sequence_analysis'][seq_name][const_name] = {
                'real_count': real_count,
                'surrogate_mean': mean_surrogate,
                'surrogate_std': std_surrogate,
                'p_value': p_value,
                'z_score': z_score,
                'significant': significant
            }

    # Collect ratios for clustering
    all_ratios = []
    ratio_sources = {}  # Track which sequence each ratio comes from
    
    for seq_name, seq in sequences.items():
        if len(seq) < 2:
            continue
        ratios = compute_consecutive_ratios(seq)
        for i, r in enumerate(ratios):
            all_ratios.append(r)
            ratio_sources[len(all_ratios) - 1] = (seq_name, seq[i], seq[i+1])

    # Apply K-means clustering
    if all_ratios:
        ratios_array = np.array(all_ratios).reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(ratios_array)
        centers = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_

        print(f"\nKMeans Clustering Results for {dataset_name}:")
        for cluster_id in range(3):
            # Get indices of ratios in this cluster
            cluster_indices = [i for i in range(len(all_ratios)) if labels[i] == cluster_id]
            
            # Count occurrences of each sequence in this cluster
            seq_counts = Counter([ratio_sources[i][0] for i in cluster_indices])
            
            # Find which constant this cluster is closest to
            center = centers[cluster_id]
            min_diff = float('inf')
            closest_constant = None
            
            for const_name, const_val in constants.items():
                diff = abs(center - const_val)
                if diff < min_diff:
                    min_diff = diff
                    closest_constant = const_name
            
            min_diff_pct = min_diff / constants[closest_constant]
            
            # Print cluster information
            print(f"Cluster {cluster_id}: center={center:.4f}, composition={seq_counts}")
            
            # Note if the cluster center is close to a mathematical constant
            proximity_msg = ""
            if min_diff_pct < 0.05:  # Within 5% of a constant
                const_val = constants[closest_constant]
                proximity_msg = f"Close to {closest_constant}={const_val:.4f} (within {min_diff_pct*100:.2f}%)"
                print(f"  {proximity_msg}")
            
            results['clustering'][f'cluster_{cluster_id}'] = {
                'center': center,
                'size': len(cluster_indices),
                'composition': seq_counts,
                'closest_constant': closest_constant,
                'proximity': min_diff_pct
            }

    # Save detailed results to file
    output_dir = f"results/extended_{dataset_name.lower()}_sequence_analysis"
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
    
    print(f"Starting Extended CMB Sequence Analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running {n_surrogates} surrogate simulations with {tolerance*100:.1f}% tolerance")
    print(f"Analyzing mathematical constants: {', '.join(constants.keys())}")
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
            data = np.loadtxt(file_path)
            all_l = data[:, 0].astype(int)
            max_l = max(all_l)
            
            # Generate sequence
            if seq_name == 'Fibonacci':
                seq = generate_fibonacci(max_l)
            elif seq_name == 'Primes':
                seq = generate_primes(max_l)
            else:  # Powers of 2
                seq = generate_powers_of_two(max_l)
                
            if len(seq) < 2:
                continue
                
            # Calculate ratios
            ratios = compute_consecutive_ratios(seq)
            
            # Plot ratios
            ax = axes[i, j]
            ax.plot(range(len(ratios)), ratios, 'o-', label=seq_name, markersize=4)
            
            # Add horizontal lines for constants
            for k, (const_name, constant) in enumerate(constants.items()):
                if 0.5 <= constant <= 3.5:  # Only plot constants in this range for clarity
                    color = plt.cm.tab10(k % 10)
                    ax.axhline(y=constant, color=color, linestyle='--',
                              label=f"{const_name}={constant:.3f}")
            
            ax.set_title(f"{seq_name} Ratios - {dataset_name}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Ratio")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('combined_extended_cmb_analysis.png', dpi=300)
    
    # Generate HTML report with detailed results
    with open("extended_cmb_sequence_analysis_report.html", "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Extended CMB Sequence Analysis Report</title>
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
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>Extended CMB Sequence Analysis Report</h1>
    <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Parameters:</strong> {n_surrogates:,} surrogate simulations, {tolerance*100:.1f}% tolerance</p>
    
    <div class="section">
        <h2>Mathematical Constants Analyzed</h2>
        <table>
            <tr>
                <th>Constant</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
""")
        
        # Add constants to the report
        for const_name, const_val in constants.items():
            description = {
                'phi': 'Golden Ratio',
                'sqrt2': 'Square Root of 2',
                'sqrt3': 'Square Root of 3',
                'sqrt5': 'Square Root of 5',
                'pi': 'Pi',
                'e': 'Euler\'s Number',
                '2': 'Number 2',
                'ln2': 'Natural Logarithm of 2',
                '1/phi': 'Reciprocal of Golden Ratio',
                'gamma': 'Euler-Mascheroni Constant'
            }.get(const_name, '')
            
            f.write(f"""<tr>
                <td>{const_name}</td>
                <td>{const_val:.6f}</td>
                <td>{description}</td>
            </tr>""")
        
        f.write("""
        </table>
    </div>
        
    <div class="section">
        <h2>WMAP Dataset Results</h2>
        <h3>Sequence Analysis</h3>
        <table>
            <tr>
                <th>Sequence</th>
                <th>Constant</th>
                <th>Real Count</th>
                <th>Surrogate Mean</th>
                <th>P-value</th>
                <th>Z-score</th>
            </tr>
""")
        
        # Add WMAP results to the report
        seq_names = ['Fibonacci', 'Primes', 'Powers of 2']
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
        
        f.write("""
        </table>
        
        <h3>Clustering Analysis</h3>
        <table>
            <tr>
                <th>Cluster</th>
                <th>Center</th>
                <th>Size</th>
                <th>Composition</th>
                <th>Closest Constant</th>
                <th>Proximity</th>
            </tr>
""")
        
        # Add WMAP clustering results
        for cluster_id in range(3):
            cluster_key = f'cluster_{cluster_id}'
            if cluster_key in wmap_results['clustering']:
                cluster = wmap_results['clustering'][cluster_key]
                composition = ", ".join([f"{seq}: {count}" for seq, count in cluster['composition'].items()])
                f.write(f"""<tr>
                    <td>Cluster {cluster_id}</td>
                    <td>{cluster['center']:.4f}</td>
                    <td>{cluster['size']}</td>
                    <td>{composition}</td>
                    <td>{cluster['closest_constant']} ({constants[cluster['closest_constant']]:.4f})</td>
                    <td>{cluster['proximity']*100:.2f}%</td>
                </tr>""")
        
        f.write("""
        </table>
    </div>
    
    <div class="section">
        <h2>Planck Dataset Results</h2>
        <h3>Sequence Analysis</h3>
        <table>
            <tr>
                <th>Sequence</th>
                <th>Constant</th>
                <th>Real Count</th>
                <th>Surrogate Mean</th>
                <th>P-value</th>
                <th>Z-score</th>
            </tr>
""")
        
        # Add Planck results to the report
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
        
        f.write("""
        </table>
        
        <h3>Clustering Analysis</h3>
        <table>
            <tr>
                <th>Cluster</th>
                <th>Center</th>
                <th>Size</th>
                <th>Composition</th>
                <th>Closest Constant</th>
                <th>Proximity</th>
            </tr>
""")
        
        # Add Planck clustering results
        for cluster_id in range(3):
            cluster_key = f'cluster_{cluster_id}'
            if cluster_key in planck_results['clustering']:
                cluster = planck_results['clustering'][cluster_key]
                composition = ", ".join([f"{seq}: {count}" for seq, count in cluster['composition'].items()])
                f.write(f"""<tr>
                    <td>Cluster {cluster_id}</td>
                    <td>{cluster['center']:.4f}</td>
                    <td>{cluster['size']}</td>
                    <td>{composition}</td>
                    <td>{cluster['closest_constant']} ({constants[cluster['closest_constant']]:.4f})</td>
                    <td>{cluster['proximity']*100:.2f}%</td>
                </tr>""")
        
        f.write("""
        </table>
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        <img src="combined_extended_cmb_analysis.png" alt="CMB Sequence Analysis Visualization">
    </div>
</body>
</html>""")

    # Print completion message
    elapsed = time.time() - start_time
    print(f"\nExtended CMB Sequence Analysis completed in {elapsed/60:.2f} minutes")
    print(f"Results saved to:")
    print(f"  - results/extended_wmap_sequence_analysis/results.json")
    print(f"  - results/extended_planck_sequence_analysis/results.json")
    print(f"  - extended_cmb_sequence_analysis_report.html")
    print(f"  - combined_extended_cmb_analysis.png")
    print(f"  - Analyzed with {n_surrogates:,} simulations as requested")
    print(f"  - WMAP and Planck datasets analyzed simultaneously with {n_surrogates:,} simulations")
    print(f"  - Extended analysis with {len(constants)} mathematical constants")
