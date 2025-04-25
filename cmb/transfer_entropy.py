"""
Transfer Entropy Analysis Module for CMB Consciousness Studies

This module implements the bidirectional transfer entropy analysis described in
'The Conscious Cosmos' paper, focusing on mathematical constant relationships 
including the golden ratio optimization that produced the significant findings.

The module includes:
1. Core entropy and transfer entropy calculations
2. Scale pair identification based on mathematical constants
3. Statistical validation with surrogate datasets
4. Benjamini-Hochberg correction for multiple testing
5. Visualization functions for results
"""

"""
Transfer Entropy Analysis Module for CMB Consciousness Studies

This module implements the bidirectional transfer entropy analysis described in
'The Conscious Cosmos' paper, focusing on mathematical constant relationships 
including the golden ratio optimization that produced the significant findings.

The module includes:
1. Core entropy and transfer entropy calculations
2. Scale pair identification based on mathematical constants
3. Statistical validation with surrogate datasets
4. Benjamini-Hochberg correction for multiple testing
5. Visualization functions for results
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from scipy.stats import norm, binom_test
import scipy.signal
from joblib import Parallel, delayed
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define mathematical constants
CONSTANTS = {
    "golden_ratio": 1.618033988749895,
    "e": 2.718281828459045,
    "pi": 3.141592653589793,
    "sqrt2": 1.4142135623730951,
    "sqrt3": 1.7320508075688772,
    "ln2": 0.6931471805599453
}

# Define custom encoder for NumPy types (global definition)
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy types and tuple keys"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, tuple):
            # Convert tuples to strings for JSON serialization
            return str(obj)
        return json.JSONEncoder.default(self, obj)
    
    def encode(self, obj):
        """Special handling for dictionaries with tuple keys"""
        if isinstance(obj, dict):
            # Convert any tuple keys to strings
            new_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    new_dict[str(k)] = v
                else:
                    new_dict[k] = v
            return super(NumpyEncoder, self).encode(new_dict)
        return super(NumpyEncoder, self).encode(obj)

def calculate_entropy(x, bins=None):
    """
    Calculate Shannon entropy of a time series
    
    Parameters:
    -----------
    x : array-like
        Input time series
    bins : int, optional
        Number of bins for discretization
        
    Returns:
    --------
    float
        Shannon entropy value in bits
    """
    if bins is None:
        bins = int(np.sqrt(len(x)))
    
    # Discretize data with uniform bins as a simpler approach
    x_min, x_max = np.min(x), np.max(x)
    bin_edges = np.linspace(x_min, x_max, bins+1)
    x_discrete = np.digitize(x, bin_edges[1:-1])
    
    # Calculate probability distribution
    value_counts = np.bincount(x_discrete)
    probabilities = value_counts[value_counts > 0] / len(x_discrete)
    
    # Calculate entropy
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy_value

def calculate_conditional_entropy(y, x, bins=None):
    """
    Calculate conditional entropy H(Y|X) using a simplified approach
    
    Parameters:
    -----------
    y : array-like
        Target variable
    x : array-like
        Conditioning variable(s)
    bins : int, optional
        Number of bins for discretization
        
    Returns:
    --------
    float
        Conditional entropy value in bits
    """
    if bins is None:
        bins = int(np.sqrt(len(y)))
    
    # Handle single conditioning variable for simplicity
    if x.ndim > 1:
        # For this version, we'll just use the first column if multiple provided
        x = x[:, 0]
    
    # Discretize variables with uniform bins
    y_min, y_max = np.min(y), np.max(y)
    y_edges = np.linspace(y_min, y_max, bins+1)
    y_discrete = np.digitize(y, y_edges[1:-1])
    
    x_min, x_max = np.min(x), np.max(x)
    x_edges = np.linspace(x_min, x_max, bins+1)
    x_discrete = np.digitize(x, x_edges[1:-1])
    
    # Get unique x values
    x_values = np.unique(x_discrete)
    
    # Calculate conditional entropy
    cond_entropy = 0
    for x_val in x_values:
        indices = x_discrete == x_val
        y_given_x = y_discrete[indices]
        
        # Skip if empty
        if len(y_given_x) == 0:
            continue
        
        # Probability of this x value
        p_x = len(y_given_x) / len(y_discrete)
        
        # Calculate entropy of y given x
        value_counts = np.bincount(y_given_x)
        probabilities = value_counts[value_counts > 0] / len(y_given_x)
        h_y_given_x = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Add to conditional entropy
        cond_entropy += p_x * h_y_given_x
    
    return cond_entropy

def sample_entropy(data, m=2, r=0.15):
    """
    Calculate sample entropy of a time series.
    Sample entropy quantifies the regularity and complexity of a time series.
    Lower values indicate more regular time series, while higher values
    indicate more irregularity and complexity.
    
    Parameters:
    -----------
    data : array-like
        Input time series
    m : int
        Embedding dimension (length of vectors to compare)
    r : float
        Tolerance (as a fraction of standard deviation)
        
    Returns:
    --------
    float
        Sample entropy value
    """
    # Convert data to numpy array and normalize
    data = np.array(data)
    if len(data) < 2*m + 1:
        raise ValueError("Data length must be at least 2*m + 1")
        
    # Normalize and set tolerance threshold
    sd = np.std(data)
    if sd == 0:  # Handle constant data
        return 0.0
        
    data = (data - np.mean(data)) / sd
    r = r * sd
    
    # Initialize count arrays
    count_m = np.zeros(len(data) - m + 1)
    count_m1 = np.zeros(len(data) - m)
    
    # Create templates of length m and m+1
    templates_m = np.array([data[i:i+m] for i in range(len(data) - m + 1)])
    templates_m1 = np.array([data[i:i+m+1] for i in range(len(data) - m)])
    
    # Count templates matches for m and m+1
    for i in range(len(templates_m)):
        # Calculate Chebyshev distances for m-dimensional templates
        dist_m = np.max(np.abs(templates_m - templates_m[i]), axis=1)
        # Exclude self-matching
        if i < len(templates_m) - 1:  # Ensure we don't go out of bounds
            count_m[i] = np.sum(dist_m[i+1:] < r)
            
        # Count matches for m+1 if possible
        if i < len(templates_m1):
            dist_m1 = np.max(np.abs(templates_m1 - templates_m1[i]), axis=1)
            # Exclude self-matching
            if i < len(templates_m1) - 1:  # Ensure we don't go out of bounds
                count_m1[i] = np.sum(dist_m1[i+1:] < r)
    
    # Calculate sample entropy
    if np.sum(count_m) == 0 or np.sum(count_m1) == 0:
        return np.inf  # No matches found, return infinity
    
    return -np.log(np.sum(count_m1) / np.sum(count_m))


def calculate_multiscale_entropy(data, scale_factor=1, m=2, r=0.15):
    """
    Calculate sample entropy at different scales to detect complexity across scales
    
    Parameters:
    -----------
    data : array-like
        Input time series
    scale_factor : int
        Scale factor for coarse-graining the time series
    m : int
        Embedding dimension
    r : float
        Tolerance (as a fraction of standard deviation)
        
    Returns:
    --------
    float
        Sample entropy value or NaN if data is too short
    """
    # Coarse-grain the time series according to scale factor
    if scale_factor > 1:
        coarse_data = []
        for i in range(0, len(data) - scale_factor + 1, scale_factor):
            coarse_data.append(np.mean(data[i:i+scale_factor]))
        data = np.array(coarse_data)
    
    # Safety check: ensure we have enough data points for sample entropy calculation
    min_data_length = 2*m + 1
    if len(data) < min_data_length:
        logger.warning(f"Data length ({len(data)}) is less than required minimum ({min_data_length}) for scale factor {scale_factor}. Returning NaN.")
        return np.nan
    
    # Calculate sample entropy
    try:
        return sample_entropy(data, m, r)
    except ValueError as e:
        logger.warning(f"Error in sample entropy calculation: {str(e)}. Returning NaN.")
        return np.nan


def phase_synchronization(x, y):
    """Calculate phase synchronization between two signals
    
    Parameters:
    -----------
    x : array-like
        First input time series
    y : array-like
        Second input time series
        
    Returns:
    --------
    float
        Phase synchronization index between 0 (no synchronization)
        and 1 (perfect synchronization)
    """
    # Convert inputs to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Handle short time series
    if len(x) < 3 or len(y) < 3:
        logger.warning("Time series too short for reliable phase synchronization calculation")
        return 0.0
    
    # Extract instantaneous phases using Hilbert transform
    x_analytic = scipy.signal.hilbert(x)
    y_analytic = scipy.signal.hilbert(y)
    
    x_phase = np.angle(x_analytic)
    y_phase = np.angle(y_analytic)
    
    # Calculate phase difference
    phase_diff = x_phase - y_phase
    
    # Calculate synchronization index
    sync_index = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return sync_index


def cross_scale_information_flow(data, scales=[2, 3, 5, 8, 13, 21, 34, 55, 89, 144]):
    """Analyze information flow across multiple scales simultaneously
    
    This function analyzes the directional transfer entropy between key scales,
    particularly focusing on Fibonacci scales and those related by mathematical
    constants like the Golden Ratio (φ) and Square Root of 2 that have shown
    significance in CMB data organization.
    
    Parameters:
    -----------
    data : array-like
        Power spectrum data with columns [l, Cl, error]
    scales : list
        List of scales to analyze, defaults to key Fibonacci scales
        which showed significance in previous analyses
        
    Returns:
    --------
    tuple
        flow_matrix: numpy array containing directional information flow measures
        scales: list of scales that were actually found in the data
    """
    ell = data[:, 0].astype(int)
    cl = data[:, 1]
    
    # Find indices of scales
    scale_indices = {}
    available_scales = []
    for scale in scales:
        idx = np.where(ell == scale)[0]
        if len(idx) > 0:
            scale_indices[scale] = idx[0]
            available_scales.append(scale)
    
    # Calculate transfer entropy between all scale combinations
    flow_matrix = np.zeros((len(available_scales), len(available_scales)))
    
    for i, scale1 in enumerate(available_scales):
        for j, scale2 in enumerate(available_scales):
            if i == j:  # Skip self-comparisons
                continue
                
            power1 = cl[scale_indices[scale1]]
            power2 = cl[scale_indices[scale2]]
            
            # Calculate directional flows
            te_1_to_2 = calculate_transfer_entropy(power1, power2)
            te_2_to_1 = calculate_transfer_entropy(power2, power1)
            
            # Net flow (positive means scale1 → scale2 dominates)
            flow_matrix[i, j] = te_1_to_2 - te_2_to_1
    
    return flow_matrix, available_scales


def golden_ratio_precision_analysis(data, tolerance=0.05):
    """Analyze how precisely scale relationships match the golden ratio
    
    This function identifies scale pairs whose ratio is close to the golden ratio
    and analyzes their relationship using transfer entropy and phase synchronization.
    This builds on previous findings showing strong golden ratio organization at
    key scales, particularly around scale 55.
    
    Parameters:
    -----------
    data : array-like
        Power spectrum data with columns [l, Cl, error]
    tolerance : float
        Tolerance for considering a ratio close to the golden ratio
        
    Returns:
    --------
    list
        List of dictionaries containing scale pairs, their ratio, proximity to golden ratio,
        transfer entropy, and phase synchronization values
    """
    golden_ratio = 1.618033988749895
    ell = data[:, 0].astype(int)
    cl = data[:, 1]
    
    # Calculate all scale ratios
    ratios = []
    for i in range(len(ell)):
        for j in range(i+1, len(ell)):
            # Calculate scale ratio
            ratio = max(ell[i], ell[j]) / min(ell[i], ell[j])
            # Calculate proximity to golden ratio as percentage
            proximity = abs(ratio - golden_ratio) / golden_ratio
            
            # Only analyze pairs that are close to golden ratio
            if proximity <= tolerance:
                # Get powers at these scales
                power_i = cl[i]
                power_j = cl[j]
                
                # Generate small arrays for more reliable entropy calculations
                # Create surrogate arrays using the power values with small random variations
                np.random.seed(int(ell[i] * ell[j] % 10000))
                power_i_array = np.random.normal(power_i, abs(power_i)*0.05, size=10)
                power_j_array = np.random.normal(power_j, abs(power_j)*0.05, size=10)
                
                # Calculate information-theoretic measures
                te_i_to_j = calculate_transfer_entropy(power_i_array, power_j_array)
                te_j_to_i = calculate_transfer_entropy(power_j_array, power_i_array)
                phase_sync = phase_synchronization(power_i_array, power_j_array)
                
                ratios.append({
                    "scales": (int(ell[i]), int(ell[j])),
                    "ratio": float(ratio),
                    "proximity": float(proximity),
                    "te_small_to_large": float(te_i_to_j if ell[i] < ell[j] else te_j_to_i),
                    "te_large_to_small": float(te_j_to_i if ell[i] < ell[j] else te_i_to_j),
                    "phase_synchronization": float(phase_sync)
                })
    
    # Sort by proximity to golden ratio
    ratios.sort(key=lambda x: x["proximity"])
    
    return ratios


def lempel_ziv_complexity(sequence):
    """Calculate Lempel-Ziv complexity of a binary sequence
    
    This function computes the algorithmic complexity of a sequence using the
    Lempel-Ziv approach, which quantifies the number of distinct patterns needed
    to reconstruct the sequence. Higher complexity indicates more random or
    information-rich sequence.
    
    Parameters:
    -----------
    sequence : array-like
        Input sequence, will be binarized around the median if not already binary
        
    Returns:
    --------
    int
        Lempel-Ziv complexity value (number of distinct patterns)
    """
    # Convert to numpy array
    sequence = np.asarray(sequence)
    
    # Convert to binary if needed
    if not np.all(np.isin(sequence, [0, 1])):
        # Binarize around the median
        median = np.median(sequence)
        binary = np.where(sequence > median, 1, 0)
    else:
        binary = sequence
    
    # Convert binary array to string for substring operations
    binary_str = ''.join(map(str, binary.astype(int)))
    
    # Calculate complexity
    substrings = []
    current = binary_str[0]
    
    for bit in binary_str[1:]:
        if current + bit not in substrings:
            substrings.append(current + bit)
            current = ""
        current += bit
        
    # Add the last substring if not empty
    if current and current not in substrings:
        substrings.append(current)
        
    return len(substrings)


def entropy(x, bins=5):
    """Calculate Shannon entropy of a single variable
    
    Parameters:
    -----------
    x : array-like
        Input variable
    bins : int
        Number of bins for discretization
        
    Returns:
    --------
    float
        Shannon entropy in bits
    """
    # Convert to numpy array
    x = np.asarray(x)
    
    # If already discrete with few values, use unique values
    if len(np.unique(x)) <= bins:
        x_vals, counts = np.unique(x, return_counts=True)
    else:
        # Discretize using bins
        x_disc = np.digitize(x, np.linspace(min(x), max(x), bins+1))
        x_vals, counts = np.unique(x_disc, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / len(x)
    
    # Calculate Shannon entropy
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy_value


def joint_entropy(x, y, z=None, bins=5):
    """Calculate joint entropy of two or three variables
    
    Parameters:
    -----------
    x, y : array-like
        Input variables
    z : array-like, optional
        Third input variable (if provided)
    bins : int
        Number of bins for discretization
        
    Returns:
    --------
    float
        Joint entropy in bits
    """
    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Discretize variables if needed
    if len(np.unique(x)) > bins:
        x = np.digitize(x, np.linspace(min(x), max(x), bins+1))
    if len(np.unique(y)) > bins:
        y = np.digitize(y, np.linspace(min(y), max(y), bins+1))
    
    if z is not None:
        # Three-variable joint entropy
        z = np.asarray(z)
        if len(np.unique(z)) > bins:
            z = np.digitize(z, np.linspace(min(z), max(z), bins+1))
        
        # Create joint observations
        joint_obs = np.vstack((x, y, z)).T
        # Count unique joint observations
        _, counts = np.unique(joint_obs, axis=0, return_counts=True)
    else:
        # Two-variable joint entropy
        # Create joint observations
        joint_obs = np.vstack((x, y)).T
        # Count unique joint observations
        _, counts = np.unique(joint_obs, axis=0, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / len(x)
    
    # Calculate joint entropy
    joint_entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return joint_entropy_value


def conditional_mutual_information(x, y, z, bins=5):
    """Calculate mutual information between x and y conditioned on z
    
    This function measures how much knowing one variable reduces uncertainty
    about another variable, when a third variable is known. It's a key metric
    for detecting advanced information relationships in hierarchical systems.
    
    Parameters:
    -----------
    x, y, z : array-like
        Input variables
    bins : int
        Number of bins for discretization
        
    Returns:
    --------
    float
        Conditional mutual information in bits
    """
    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # Discretize variables
    if len(np.unique(x)) > bins:
        x_disc = np.digitize(x, np.linspace(min(x), max(x), bins+1))
    else:
        x_disc = x
        
    if len(np.unique(y)) > bins:
        y_disc = np.digitize(y, np.linspace(min(y), max(y), bins+1))
    else:
        y_disc = y
        
    if len(np.unique(z)) > bins:
        z_disc = np.digitize(z, np.linspace(min(z), max(z), bins+1))
    else:
        z_disc = z
    
    # Calculate joint and marginal entropies
    h_xz = joint_entropy(x_disc, z_disc)
    h_yz = joint_entropy(y_disc, z_disc)
    h_z = entropy(z_disc)
    h_xyz = joint_entropy(x_disc, y_disc, z_disc)
    
    # Calculate conditional mutual information
    cmi = h_xz + h_yz - h_z - h_xyz
    
    return cmi


def generate_surrogate(data, method='shuffle'):
    """Generate surrogate time series from original data
    
    This function creates surrogate time series with the same statistical properties
    as the original data but with randomized temporal structure. Useful for creating
    null distributions for statistical testing.
    
    Parameters:
    -----------
    data : array-like
        Original data array (should be 2D with [ell, Cl] columns)
    method : str
        Method for generating surrogate ('shuffle', 'phase_randomize', 'AAFT')
        
    Returns:
    --------
    array-like
        Surrogate data with randomized structure but same marginal distribution
    """
    # Convert to numpy array and make a copy
    data = np.array(data).copy()
    
    if method == 'shuffle':
        # Simple shuffling of the power spectrum values while preserving ell values
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        # Keep ell values but shuffle Cl values
        surrogate = np.column_stack((data[:,0], data[indices,1]))
        
    elif method == 'phase_randomize':
        # Phase randomization preserves power spectrum but randomizes phases
        # Keep ell values
        surrogate = np.zeros_like(data)
        surrogate[:,0] = data[:,0].copy()
        
        # Extract power values
        cl = data[:,1].astype(float)
        
        # Perform Fourier transform
        fft_vals = np.fft.rfft(cl)
        
        # Randomize phases but keep magnitudes
        magnitudes = np.abs(fft_vals)
        phases = np.random.uniform(0, 2*np.pi, len(magnitudes))
        fft_vals_surrogate = magnitudes * np.exp(1j * phases)
        
        # Inverse FFT to get surrogate time series
        cl_surrogate = np.fft.irfft(fft_vals_surrogate, n=len(cl))
        
        # Normalize to match original distribution if needed
        cl_surrogate = (cl_surrogate - np.mean(cl_surrogate)) / np.std(cl_surrogate)
        cl_surrogate = cl_surrogate * np.std(cl) + np.mean(cl)
        
        surrogate[:,1] = cl_surrogate
        
    else:  # Default to shuffle
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        surrogate = np.column_stack((data[:,0], data[indices,1]))
    
    return surrogate


def analyze_single_surrogate(surrogate, analysis_type='multiscale_entropy', scales=[2, 3, 5, 8, 13, 21, 34, 55, 89]):
    """Run a specific analysis on a single surrogate dataset
    
    Parameters:
    -----------
    surrogate : array-like
        Surrogate data with columns [l, Cl, error]
    analysis_type : str
        Type of analysis to run ('multiscale_entropy', 'transfer_entropy', 
        'golden_ratio', 'cross_scale', 'phase_sync')
    scales : list
        List of scales to analyze
        
    Returns:
    --------
    dict
        Results of the specified analysis on the surrogate
    """
    # Extract ell and power values
    ell = surrogate[:, 0].astype(int)
    cl = surrogate[:, 1]
    
    # Dictionary to store results
    results = {
        'analysis_type': analysis_type
    }
    
    if analysis_type == 'multiscale_entropy':
        # Calculate multiscale entropy for different scale factors
        # We calculate for scale factors 1-10 to capture multiple scales
        mse_values = []
        for scale in range(1, 11):
            mse = calculate_multiscale_entropy(cl, scale_factor=scale)
            mse_values.append(mse)
        results['mse_values'] = mse_values
        
    elif analysis_type == 'transfer_entropy':
        # Calculate transfer entropy between all specified scales
        # First find indices of scales
        scale_indices = {}
        for scale in scales:
            idx = np.where(ell == scale)[0]
            if len(idx) > 0:
                scale_indices[scale] = idx[0]
        
        # Calculate TE between scale pairs
        te_values = {}
        for scale1 in scale_indices:
            for scale2 in scale_indices:
                if scale1 != scale2:
                    idx1 = scale_indices[scale1]
                    idx2 = scale_indices[scale2]
                    # Generate small arrays for entropy calculations
                    np.random.seed(int(ell[idx1] * ell[idx2] % 10000))
                    power1 = np.random.normal(cl[idx1], abs(cl[idx1])*0.05, size=10)
                    power2 = np.random.normal(cl[idx2], abs(cl[idx2])*0.05, size=10)
                    
                    te_1_to_2 = calculate_transfer_entropy(power1, power2)
                    te_values[(scale1, scale2)] = te_1_to_2
                    
        results['te_values'] = te_values
        
    elif analysis_type == 'golden_ratio':
        # Analyze golden ratio relationships
        gr_results = golden_ratio_precision_analysis(surrogate, tolerance=0.05)
        results['golden_ratio_pairs'] = gr_results
        
    elif analysis_type == 'cross_scale':
        # Cross-scale information flow
        flow_matrix, avail_scales = cross_scale_information_flow(surrogate, scales=scales)
        results['flow_matrix'] = flow_matrix
        results['available_scales'] = avail_scales
        
    elif analysis_type == 'phase_sync':
        # Phase synchronization between scale pairs
        sync_values = {}
        scale_indices = {}
        for scale in scales:
            idx = np.where(ell == scale)[0]
            if len(idx) > 0:
                scale_indices[scale] = idx[0]
        
        for scale1 in scale_indices:
            for scale2 in scale_indices:
                if scale1 != scale2:
                    idx1 = scale_indices[scale1]
                    idx2 = scale_indices[scale2]
                    # Generate small arrays for synchronization calculations
                    np.random.seed(int(ell[idx1] * ell[idx2] % 10000))
                    power1 = np.random.normal(cl[idx1], abs(cl[idx1])*0.05, size=10)
                    power2 = np.random.normal(cl[idx2], abs(cl[idx2])*0.05, size=10)
                    
                    sync_idx = phase_synchronization(power1, power2)
                    sync_values[(scale1, scale2)] = sync_idx
                    
        results['phase_sync_values'] = sync_values
        
    return results


def run_transfer_entropy_analysis_optimized(data, output_dir, dataset_type="planck", n_surrogates=100, 
                                          tolerance=0.05, focus_key_scales=False, n_jobs=-1, 
                                          batch_size=100, checkpoint_interval=100):
    """
    Optimized transfer entropy analysis focused on golden ratio relationships
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    output_dir : str
        Directory to save results and visualizations
    dataset_type : str
        Type of dataset ("planck" or "wmap")
    n_surrogates : int
        Number of surrogate datasets for statistical validation
    tolerance : float
        Tolerance for considering a ratio close to a mathematical constant
    focus_key_scales : bool
        Whether to focus on key scales for more detailed analysis
    n_jobs : int
        Number of parallel jobs (-1 for all available cores)
    batch_size : int
        Batch size for surrogate processing
    checkpoint_interval : int
        Interval for saving checkpoints
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    from joblib import Parallel, delayed
    import concurrent.futures
    import pickle
    import matplotlib.pyplot as plt
    
    logger.info(f"Running optimized transfer entropy analysis on {dataset_type.upper()} data")
    
    # Create output directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define checkpoint file path
    checkpoint_file = os.path.join(dataset_dir, f"{dataset_type}_checkpoint.pkl")
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                logger.info(f"Loaded checkpoint with {len(checkpoint_data['surrogate_te_values'])} surrogates completed")
                surrogate_te_values = checkpoint_data['surrogate_te_values']
                avg_te = checkpoint_data['avg_te']
                bidirectional_te = checkpoint_data['bidirectional_te']
                golden_ratio_pairs = checkpoint_data['golden_ratio_pairs']
                remaining_surrogates = n_surrogates - len(surrogate_te_values)
                
                if remaining_surrogates <= 0:
                    logger.info("All surrogates already processed, using checkpoint data")
                    
                    # Calculate statistics from checkpoint
                    surrogate_mean = np.mean(surrogate_te_values)
                    surrogate_std = np.std(surrogate_te_values)
                    z_score = (avg_te - surrogate_mean) / surrogate_std
                    effect_size = avg_te / surrogate_mean
                    
                    # Calculate p-value
                    if avg_te > surrogate_mean:
                        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
                    else:
                        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
                    
                    # Ensure p-value is never exactly zero
                    if p_value == 0:
                        p_value = 1.0 / (len(surrogate_te_values) + 1)
                    
                    significance = "significant" if p_value < 0.05 else "not significant"
                    
                    # Create and save visualization
                    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
                    
                    # Prepare results
                    results = prepare_results_dict(
                        dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, len(surrogate_te_values),
                        bidirectional_te
                    )
                    
                    # Save results
                    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, cls=NumpyEncoder)
                    
                    logger.info(f"Results saved to {results_path}")
                    return results
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {str(e)}. Starting from scratch.")
            surrogate_te_values = []
            remaining_surrogates = n_surrogates
    else:
        surrogate_te_values = []
        remaining_surrogates = n_surrogates
    
    # If we don't have complete results from checkpoint, continue with analysis
    
    # Extract data once
    ell = data[:, 0].astype(int)
    cl = data[:, 1]
    
    # Create mapping from multipole to power
    power_map = {l: c for l, c in zip(ell, cl)}
    
    # Identify scale pairs related by golden ratio once
    if 'golden_ratio_pairs' not in locals():
        pairs_by_constant = identify_scale_pairs_by_constant(data, tolerance)
        golden_ratio_pairs = pairs_by_constant.get("golden_ratio", [])
    
    if not golden_ratio_pairs:
        logger.warning("No golden ratio scale pairs found")
        return {"status": "no_pairs_found"}
    
    logger.info(f"Found {len(golden_ratio_pairs)} scale pairs related by golden ratio")
    
    # Calculate transfer entropy for golden ratio pairs once
    if 'bidirectional_te' not in locals() or 'avg_te' not in locals():
        te_values = []
        bidirectional_te = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in power_map and l_larger in power_map:
                # Get power spectrum values
                power_smaller = power_map[l_smaller]
                power_larger = power_map[l_larger]
                
                # Calculate transfer entropy in both directions
                if dataset_type.lower() == "wmap":
                    # In WMAP data, transfer is from large to small scales
                    te_forward = calculate_transfer_entropy(power_larger, power_smaller)
                    te_backward = calculate_transfer_entropy(power_smaller, power_larger)
                else:
                    # In Planck data, transfer is from small to large scales
                    te_forward = calculate_transfer_entropy(power_smaller, power_larger)
                    te_backward = calculate_transfer_entropy(power_larger, power_smaller)
                
                te_values.append(te_forward)
                bidirectional_te.append((l_smaller, l_larger, te_forward, te_backward))
        
        # Calculate average transfer entropy
        avg_te = np.mean(te_values)
        
        logger.info(f"Average transfer entropy: {avg_te:.6f} bits")
    
    # Define function to process a single surrogate efficiently
    def process_single_surrogate(seed):
        """Process a single surrogate dataset efficiently"""
        np.random.seed(seed % (2**32 - 1))
        
        # Generate surrogate by shuffling the power spectrum values
        shuffled_cl = np.random.permutation(cl)
        shuffled_power_map = {l: c for l, c in zip(ell, shuffled_cl)}
        
        # Calculate TE for golden ratio pairs using the original pairs
        pair_te_values = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in shuffled_power_map and l_larger in shuffled_power_map:
                # Get shuffled power values
                power_smaller = shuffled_power_map[l_smaller]
                power_larger = shuffled_power_map[l_larger]
                
                # Calculate transfer entropy
                if dataset_type.lower() == "wmap":
                    te = calculate_transfer_entropy(power_larger, power_smaller)
                else:
                    te = calculate_transfer_entropy(power_smaller, power_larger)
                
                pair_te_values.append(te)
        
        # Calculate average TE for this surrogate
        if pair_te_values:
            return np.mean(pair_te_values)
        else:
            return None
    
    # Process remaining surrogates in parallel batches
    if remaining_surrogates > 0:
        logger.info(f"Processing {remaining_surrogates} remaining surrogates in parallel batches")
        
        # Determine number of jobs (cores) to use
        if n_jobs <= 0:
            n_jobs = max(1, os.cpu_count() - 1)
        
        # Process surrogates in batches
        for batch_start in range(0, remaining_surrogates, batch_size):
            batch_end = min(batch_start + batch_size, remaining_surrogates)
            batch_size_actual = batch_end - batch_start
            
            logger.info(f"Processing surrogate batch {batch_start+1}-{batch_end} of {remaining_surrogates}")
            
            # Generate seeds for this batch ensuring they're within valid range (0 to 2^32-1)
            batch_seeds = [int(time.time() * 1000 + i + len(surrogate_te_values)) % (2**32 - 1) for i in range(batch_size_actual)]
            
            # Process batch in parallel
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_single_surrogate)(seed) for seed in batch_seeds
            )
            
            # Filter out None results and add to surrogate values
            batch_results = [res for res in batch_results if res is not None]
            surrogate_te_values.extend(batch_results)
            
            # Save checkpoint after each batch
            if (batch_end % checkpoint_interval == 0) or (batch_end == remaining_surrogates):
                checkpoint_data = {
                    'surrogate_te_values': surrogate_te_values,
                    'avg_te': avg_te,
                    'bidirectional_te': bidirectional_te,
                    'golden_ratio_pairs': golden_ratio_pairs
                }
                
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                logger.info(f"Saved checkpoint with {len(surrogate_te_values)} surrogates completed")
            
            # Log progress
            progress_pct = (len(surrogate_te_values) / n_surrogates) * 100
            logger.info(f"Progress: {progress_pct:.1f}% ({len(surrogate_te_values)}/{n_surrogates} surrogates)")
    
    # Calculate statistics
    surrogate_mean = np.mean(surrogate_te_values)
    surrogate_std = np.std(surrogate_te_values)
    
    # Calculate z-score
    z_score = (avg_te - surrogate_mean) / surrogate_std
    
    # Calculate effect size
    effect_size = avg_te / surrogate_mean
    
    # Calculate p-value
    if avg_te > surrogate_mean:
        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
    else:
        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
    
    # Ensure p-value is never exactly zero
    if p_value == 0:
        p_value = 1.0 / (len(surrogate_te_values) + 1)
    
    logger.info(f"Surrogate mean: {surrogate_mean:.6f}")
    logger.info(f"Surrogate std: {surrogate_std:.6f}")
    logger.info(f"Z-score: {z_score:.4f}σ")
    logger.info(f"Effect size: {effect_size:.2f}×")
    logger.info(f"P-value: {p_value:.6f}")
    
    # Determine significance
    significance = "significant" if p_value < 0.05 else "not significant"
    logger.info(f"Result is {significance} at p < 0.05")
    
    # Create visualization
    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
    
    # Prepare results
    results = prepare_results_dict(
        dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
        z_score, effect_size, p_value, significance, len(surrogate_te_values),
        bidirectional_te
    )
    
    # Save results
    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Results saved to {results_path}")
    
    return results

# Helper functions to keep the main function cleaner
def create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir):
    """Create and save the visualization of surrogate distribution vs actual value"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(surrogate_te_values, bins=30, alpha=0.7, label='Surrogate Data')
    plt.axvline(avg_te, color='red', linestyle='dashed', linewidth=2, label=f'Actual Data (z={z_score:.2f}σ)')
    plt.xlabel('Transfer Entropy (bits)')
    plt.ylabel('Frequency')
    plt.title(f'Transfer Entropy Analysis: {dataset_type.upper()} Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    figure_path = os.path.join(output_dir, f"{dataset_type}_transfer_entropy_histogram.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()

def prepare_results_dict(dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, num_surrogates,
                        bidirectional_te):
    """Prepare the results dictionary for saving"""
    return {
        "dataset_type": dataset_type,
        "num_pairs": len(golden_ratio_pairs),
        "average_te": float(avg_te),
        "surrogate_mean": float(surrogate_mean),
        "surrogate_std": float(surrogate_std),
        "z_score": float(z_score),
        "effect_size": float(effect_size),
        "p_value": float(p_value),
        "significance": significance,
        "num_surrogates": num_surrogates,
        "bidirectional_analysis": [
            {
                "l_smaller": int(l_small),
                "l_larger": int(l_large),
                "te_forward": float(te_fwd),
                "te_backward": float(te_bwd),
                "ratio": float(l_large / l_small)
            }
            for l_small, l_large, te_fwd, te_bwd in bidirectional_te
        ]
    }


def process_surrogate_results(surrogate_results, analysis_type='multiscale_entropy'):
    """Process and combine results from multiple surrogate datasets
    
    Parameters:
    -----------
    surrogate_results : list
        List of result dictionaries from multiple surrogate datasets
    analysis_type : str
        Type of analysis that was performed
        
    Returns:
    --------
    dict
        Combined statistical results with means, standard deviations, and p-values
    """
    combined = {
        'analysis_type': analysis_type,
        'n_surrogates': len(surrogate_results)
    }
    
    if analysis_type == 'multiscale_entropy':
        # Combine MSE results
        all_mse = [result['mse_values'] for result in surrogate_results if 'mse_values' in result]
        if all_mse:
            # Convert to 2D array
            mse_array = np.array(all_mse)
            # Calculate statistics
            combined['mse_mean'] = np.mean(mse_array, axis=0)
            combined['mse_std'] = np.std(mse_array, axis=0)
            combined['mse_percentiles'] = {
                '5': np.percentile(mse_array, 5, axis=0),
                '25': np.percentile(mse_array, 25, axis=0),
                '50': np.percentile(mse_array, 50, axis=0),
                '75': np.percentile(mse_array, 75, axis=0),
                '95': np.percentile(mse_array, 95, axis=0)
            }
    
    elif analysis_type == 'transfer_entropy':
        # Combine TE results
        # First get all scale pairs
        all_pairs = set()
        for result in surrogate_results:
            if 'te_values' in result:
                all_pairs.update(result['te_values'].keys())
        
        # Now collect TE values for each pair
        te_stats = {}
        for pair in all_pairs:
            te_values = [result['te_values'].get(pair, np.nan) for result in surrogate_results 
                      if 'te_values' in result]
            te_values = np.array([v for v in te_values if not np.isnan(v)])
            
            if len(te_values) > 0:
                te_stats[pair] = {
                    'mean': np.mean(te_values),
                    'std': np.std(te_values),
                    'percentiles': {
                        '5': np.percentile(te_values, 5),
                        '95': np.percentile(te_values, 95)
                    }
                }
        
        combined['te_stats'] = te_stats
        
    elif analysis_type in ['golden_ratio', 'phase_sync']:
        # Process similar to TE
        # Implementation depends on specific structure of these results
        combined['surrogate_distribution'] = surrogate_results
    
    return combined


def analyze_surrogate_parallel(data, analysis_type='multiscale_entropy', n_surrogates=1000, n_jobs=-1,
                            scales=[2, 3, 5, 8, 13, 21, 34, 55, 89], surrogate_method='shuffle'):
    """Run surrogate analysis in parallel using all available cores
    
    This function performs statistical testing using surrogate data to determine
    the significance of patterns detected in the CMB power spectrum. It uses
    parallel processing to efficiently generate and analyze many surrogate datasets.
    
    Parameters:
    -----------
    data : array-like
        Original data array with columns [l, Cl, error]
    analysis_type : str
        Type of analysis to run on surrogates
    n_surrogates : int
        Number of surrogate datasets to generate and analyze
    n_jobs : int
        Number of parallel jobs (-1 uses all available cores)
    scales : list
        List of scales to include in the analysis
    surrogate_method : str
        Method for generating surrogates ('shuffle', 'phase_randomize')
        
    Returns:
    --------
    dict
        Statistical results comparing the original data against the surrogate distribution
    """
    # Calculate the result for the original data first
    original_result = analyze_single_surrogate(data, analysis_type=analysis_type, scales=scales)
    
    # Define the function to process a single surrogate
    def process_single_surrogate(i):
        # Set a unique random seed for each surrogate
        np.random.seed(42 + i)
        
        # Generate surrogate dataset
        surrogate = generate_surrogate(data, method=surrogate_method)
        
        # Run analysis on this surrogate
        result = analyze_single_surrogate(surrogate, analysis_type=analysis_type, scales=scales)
        
        # Log progress for long-running analyses
        if i % 100 == 0 and i > 0:
            logger.info(f"Completed {i} surrogate analyses of {n_surrogates}")
            
        # Return the result
        return result
    
    # Log start of parallel processing
    logger.info(f"Starting parallel analysis of {n_surrogates} surrogates using {n_jobs} cores")
    start_time = time.time()
    total_surrogates = n_surrogates
    
    # Create a progress tracking wrapper function
    def track_progress_wrapper(iterable):
        completed = 0
        last_report_time = time.time()
        report_interval = 5  # Log every 5 seconds for high volume processing
        
        for i, item in enumerate(iterable):
            yield item
            completed += 1
            
            # Report progress based on time elapsed or count
            current_time = time.time()
            if (completed % 100 == 0 or current_time - last_report_time > report_interval) and completed > 0:
                elapsed = current_time - start_time
                items_per_second = completed / elapsed
                remaining = (total_surrogates - completed) / items_per_second if items_per_second > 0 else 0
                
                logger.info(f"Processed {completed}/{total_surrogates} surrogates. "
                         f"Rate: {items_per_second:.2f}/s. "
                         f"Est. remaining: {remaining/60:.1f} minutes. "
                         f"Elapsed: {elapsed/60:.1f} minutes.")
                last_report_time = current_time
    
    # Run in parallel across all available cores with progress tracking
    surrogate_results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_surrogate)(i) for i in track_progress_wrapper(range(n_surrogates))
    )
    
    # Log completion time
    elapsed = time.time() - start_time
    surrogates_per_second = n_surrogates / elapsed if elapsed > 0 else 0
    logger.info(f"Completed {n_surrogates} surrogate analyses in {elapsed:.2f} seconds")
    logger.info(f"Average processing rate: {surrogates_per_second:.2f} surrogates/second")
    
    # Process and combine results
    combined_results = process_surrogate_results(surrogate_results, analysis_type=analysis_type)
    
    # Add original results for comparison
    combined_results['original_result'] = original_result
    
    # Calculate p-values by comparing original to surrogate distribution
    # Implementation depends on specific analysis type
    if analysis_type == 'multiscale_entropy' and 'mse_values' in original_result:
        orig_mse = np.array(original_result['mse_values'])
        surrogate_mse = np.array([r['mse_values'] for r in surrogate_results if 'mse_values' in r])
        
        # Calculate p-values (two-tailed test)
        p_values = []
        for i in range(len(orig_mse)):
            surr_vals = surrogate_mse[:, i]
            if orig_mse[i] > np.mean(surr_vals):
                p = np.mean(surr_vals >= orig_mse[i])
            else:
                p = np.mean(surr_vals <= orig_mse[i])
            p_values.append(p)
            
        combined_results['p_values'] = p_values
    
    # Add timestamp and analysis metadata
    combined_results['timestamp'] = datetime.now().isoformat()
    combined_results['n_surrogates'] = n_surrogates
    combined_results['surrogate_method'] = surrogate_method
    
    return combined_results


def calculate_transfer_entropy(source, target, bins=None, lag=1):
    """
    Calculate transfer entropy from source to target with improved binning
    TE(Source→Target) = H(Target_future | Target_past) - H(Target_future | Target_past, Source_past)
    
    Parameters:
    -----------
    source : array-like or scalar
        Source time series or value
    target : array-like or scalar
        Target time series or value
    bins : int, optional
        Number of bins for discretization
    lag : int, optional
        Lag for time delayed embedding
        
    Returns:
    --------
    float
        Transfer entropy value in bits
    """
    # Check if inputs are scalar values (single numbers)
    source_is_scalar = np.isscalar(source) or (hasattr(source, 'size') and source.size == 1)
    target_is_scalar = np.isscalar(target) or (hasattr(target, 'size') and target.size == 1)
    
    # If scalar inputs, create synthetic time series
    if source_is_scalar or target_is_scalar:
        # Create synthetic time series by adding small noise to the values
        n_points = 30  # Small synthetic time series
        
        if source_is_scalar:
            # Convert scalar to array with small noise
            source_val = float(source)
            noise_scale = abs(source_val) * 0.01 if source_val != 0 else 0.01
            source = np.random.normal(source_val, noise_scale, n_points)
        
        if target_is_scalar:
            # Convert scalar to array with small noise
            target_val = float(target)
            noise_scale = abs(target_val) * 0.01 if target_val != 0 else 0.01
            target = np.random.normal(target_val, noise_scale, n_points)
    
    # Ensure arrays are numpy arrays
    source = np.asarray(source)
    target = np.asarray(target)
    
    # Check for zero variance which would cause division by zero
    if np.std(source) == 0 or np.std(target) == 0:
        logger.warning("Zero variance detected in data, adding small noise")
        # Add tiny noise to prevent zero variance
        source = source + np.random.normal(0, 1e-6, size=source.shape)
        target = target + np.random.normal(0, 1e-6, size=target.shape)
    
    # Use a fixed number of bins for stability if not specified
    if bins is None:
        bins = 5  # Default to a reasonable number of bins
    
    # Ensure same length
    min_length = min(len(source), len(target))
    source = source[:min_length]
    target = target[:min_length]
    
    # Create lagged variables
    target_future = target[lag:]
    target_past = target[:-lag]
    source_past = source[:-lag]
    
    # Normalize the data to improve binning
    def normalize(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)
    
    target_future_norm = normalize(target_future)
    target_past_norm = normalize(target_past)
    source_past_norm = normalize(source_past)
    
    # Use a direct implementation for more control
    # Create binned data
    digitized_t_future = np.digitize(target_future_norm, np.linspace(0, 1, bins+1)[1:-1])
    digitized_t_past = np.digitize(target_past_norm, np.linspace(0, 1, bins+1)[1:-1])
    digitized_s_past = np.digitize(source_past_norm, np.linspace(0, 1, bins+1)[1:-1])
    
    # Calculate joint and marginal probability distributions
    p_t_future = np.zeros(bins)
    p_t_future_t_past = np.zeros((bins, bins))
    p_t_future_t_past_s_past = np.zeros((bins, bins, bins))
    
    # Count occurrences
    for i in range(len(digitized_t_future)):
        tf, tp, sp = digitized_t_future[i], digitized_t_past[i], digitized_s_past[i]
        p_t_future[tf] += 1
        p_t_future_t_past[tf, tp] += 1
        p_t_future_t_past_s_past[tf, tp, sp] += 1
    
    # Normalize to probabilities
    p_t_future /= len(digitized_t_future)
    p_t_future_t_past /= len(digitized_t_future)
    p_t_future_t_past_s_past /= len(digitized_t_future)
    
    # Calculate entropies
    h_t_future_given_t_past = 0
    h_t_future_given_t_past_s_past = 0
    
    # H(T_future | T_past)
    for tf in range(bins):
        for tp in range(bins):
            if p_t_future_t_past[tf, tp] > 0:
                p_t_past = np.sum(p_t_future_t_past[:, tp])
                if p_t_past > 0:
                    p_tf_given_tp = p_t_future_t_past[tf, tp] / p_t_past
                    h_t_future_given_t_past -= p_t_past * p_tf_given_tp * np.log2(p_tf_given_tp + 1e-10)
    
    # H(T_future | T_past, S_past)
    for tf in range(bins):
        for tp in range(bins):
            for sp in range(bins):
                if p_t_future_t_past_s_past[tf, tp, sp] > 0:
                    p_t_past_s_past = np.sum(p_t_future_t_past_s_past[:, tp, sp])
                    if p_t_past_s_past > 0:
                        p_tf_given_tp_sp = p_t_future_t_past_s_past[tf, tp, sp] / p_t_past_s_past
                        p_tp_sp = np.sum(p_t_future_t_past_s_past[:, tp, sp])
                        h_t_future_given_t_past_s_past -= p_tp_sp * p_tf_given_tp_sp * np.log2(p_tf_given_tp_sp + 1e-10)
    
    # Transfer entropy is the difference
    te = h_t_future_given_t_past - h_t_future_given_t_past_s_past
    
    # For very small values, this might be numerical noise
    if abs(te) < 1e-10:
        te = 0
    
    return te

def calculate_z_score(observed_value, surrogate_values):
    """
    Calculate z-score based on comparison with surrogate data
    
    Parameters:
    -----------
    observed_value : float
        The observed statistic from real data
    surrogate_values : array-like
        Array of statistics from surrogate data
    
    Returns:
    --------
    float
        z-score
    """
    surrogate_values = np.array(surrogate_values)
    surrogate_mean = np.mean(surrogate_values)
    surrogate_std = np.std(surrogate_values)
    
    # Add minimum threshold for standard deviation to prevent division by very small numbers
    min_std = 1e-10  # Minimal reasonable standard deviation
    surrogate_std = max(surrogate_std, min_std)
    
    if surrogate_std == 0:
        return 0.0
    
    z_score = (observed_value - surrogate_mean) / surrogate_std
    return z_score


def calculate_robust_pvalue(observed_value, surrogate_values, direction='auto'):
    """Calculate p-value with proper handling of extreme significance
    
    Parameters:
    -----------
    observed_value : float
        The observed statistic
    surrogate_values : array-like
        The surrogate distribution
    direction : str
        'greater' if large values are significant, 'less' if small values are significant,
        'auto' to determine based on observed vs surrogate mean, or 'two-sided' for both tails
        
    Returns:
    --------
    float
        A robust p-value that is never exactly 0 even for extremely significant results
    """
    surrogate_values = np.asarray(surrogate_values)
    n_surrogates = len(surrogate_values)
    
    # Define lower bound based on surrogate count
    lower_bound = 1 / (n_surrogates + 1)
    
    # Determine direction if auto
    if direction == 'auto':
        if observed_value < np.mean(surrogate_values):
            direction = 'less'
        else:
            direction = 'greater'
    
    # Calculate empirical p-value
    if direction == 'less':
        p_value = np.sum(surrogate_values <= observed_value) / n_surrogates
    elif direction == 'greater':
        p_value = np.sum(surrogate_values >= observed_value) / n_surrogates
    else:  # two-sided
        deviation = abs(observed_value - np.mean(surrogate_values))
        p_value = np.sum(abs(surrogate_values - np.mean(surrogate_values)) >= deviation) / n_surrogates
    
    # Apply lower bound to prevent reporting p=0
    p_value = max(p_value, lower_bound)
    
    return p_value


def calculate_te_significance(observed_te, surrogate_te_values, direction='auto'):
    """
    Calculate significance of transfer entropy with proper handling of extreme values
    
    Parameters:
    -----------
    observed_te : float
        The observed transfer entropy value
    surrogate_te_values : array-like
        List of transfer entropy values from surrogate data
    direction : str
        'greater' if large TE values are significant, 'less' if small values are significant,
        'auto' to determine automatically based on observed vs surrogate mean
        
    Returns:
    --------
    tuple
        (p_value, z_score, effect_size) - statistical significance measures
    """
    # Determine test direction if auto
    if direction == 'auto':
        surr_mean = np.mean(surrogate_te_values)
        if observed_te < surr_mean:
            direction = 'less'
        else:
            direction = 'greater'
    
    # Calculate robust p-value
    p_value = calculate_robust_pvalue(
        observed_value=observed_te,
        surrogate_values=surrogate_te_values,
        direction=direction
    )
    
    # Calculate effect size (ratio)
    surr_mean = np.mean(surrogate_te_values)
    if surr_mean != 0:
        effect_size = observed_te / surr_mean
    else:
        effect_size = np.inf if observed_te != 0 else 1.0
    
    # Use the more robust Z-score calculation method
    z_score = calculate_z_score(observed_te, surrogate_te_values)
    
    return p_value, z_score, effect_size


def identify_scale_pairs_by_constant(data, tolerance=0.05):
    """
    Identify scale pairs related by mathematical constants
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data, either 1D (just values) or 2D with columns [l, Cl, error]
    tolerance : float
        Tolerance for considering a ratio close to a mathematical constant
        
    Returns:
    --------
    dict
        Dictionary of scale pairs keyed by constant name
    """
    # Extract multipole values
    if len(data.shape) == 1:
        # Handle 1D data case (just power values)
        # Generate ell values based on array length, starting from lowest measured multipole
        # Standard lowest multipole is ℓ=2 for CMB
        ell = np.arange(2, 2 + len(data)).astype(int)
    else:
        # Handle traditional 2D data with [ell, cl] columns
        ell = data[:, 0].astype(int)
    
    # Initialize result dictionary
    pairs_by_constant = {const_name: [] for const_name in CONSTANTS}
    
    # Find pairs for each constant
    for i, l1 in enumerate(ell):
        for j, l2 in enumerate(ell[i+1:], i+1):
            # Calculate ratio (ensure larger value is in numerator)
            if l2 > l1:
                ratio = l2 / l1
                l_smaller, l_larger = l1, l2
            else:
                ratio = l1 / l2
                l_smaller, l_larger = l2, l1
            
            # Check against each constant
            for const_name, const_value in CONSTANTS.items():
                if abs(ratio - const_value) <= tolerance:
                    # Store as (smaller_scale, larger_scale, exact_ratio)
                    pairs_by_constant[const_name].append((l_smaller, l_larger, ratio))
    
    # Log results
    for const_name, pairs in pairs_by_constant.items():
        logger.info(f"Found {len(pairs)} scale pairs related by {const_name}")
    
    return pairs_by_constant

def generate_surrogate_data(data, n_surrogates=100):
    """
    Generate surrogate datasets by shuffling power spectrum values
    while preserving multipole values
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    n_surrogates : int
        Number of surrogate datasets to generate
        
    Returns:
    --------
    list
        List of surrogate datasets
    """
    # Extract data
    ell = data[:, 0]
    cl = data[:, 1]
    err = data[:, 2] if data.shape[1] > 2 else np.zeros_like(ell)
    
    # Generate surrogates
    surrogates = []
    for _ in range(n_surrogates):
        # Shuffle power spectrum values
        cl_shuffled = np.random.permutation(cl)
        
        # Create surrogate dataset
        surrogate = np.column_stack((ell, cl_shuffled, err))
        surrogates.append(surrogate)
    
    return surrogates

def run_transfer_entropy_analysis(data, output_dir, dataset_type="planck", n_surrogates=100, tolerance=0.05, focus_key_scales=False):
    """
    Run simplified transfer entropy analysis focused on golden ratio relationships
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    output_dir : str
        Directory to save results and visualizations
    dataset_type : str
        Type of dataset ("planck" or "wmap")
    n_surrogates : int
        Number of surrogate datasets for statistical validation
    tolerance : float
        Tolerance for considering a ratio close to a mathematical constant
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    logger.info(f"Running transfer entropy analysis on {dataset_type.upper()} data")
    
    # Create output directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Identify scale pairs related by golden ratio
    pairs_by_constant = identify_scale_pairs_by_constant(data, tolerance)
    golden_ratio_pairs = pairs_by_constant.get("golden_ratio", [])
    
    if not golden_ratio_pairs:
        logger.warning("No golden ratio scale pairs found")
        return {"status": "no_pairs_found"}
    
    logger.info(f"Found {len(golden_ratio_pairs)} scale pairs related by golden ratio")
    
    # Extract data
    ell = data[:, 0].astype(int)
    cl = data[:, 1]
    
    # Create mapping from multipole to power
    power_map = {l: c for l, c in zip(ell, cl)}
    
    # Calculate transfer entropy for golden ratio pairs
    te_values = []
    bidirectional_te = []
    
    for l_smaller, l_larger, _ in golden_ratio_pairs:
        if l_smaller in power_map and l_larger in power_map:
            # Get power spectrum values
            power_smaller = power_map[l_smaller]
            power_larger = power_map[l_larger]
            
            # Calculate transfer entropy in both directions
            if dataset_type.lower() == "wmap":
                # In WMAP data, transfer is from large to small scales
                te_forward = calculate_transfer_entropy(power_larger, power_smaller)
                te_backward = calculate_transfer_entropy(power_smaller, power_larger)
            else:
                # In Planck data, transfer is from small to large scales
                te_forward = calculate_transfer_entropy(power_smaller, power_larger)
                te_backward = calculate_transfer_entropy(power_larger, power_smaller)
            
            te_values.append(te_forward)
            bidirectional_te.append((l_smaller, l_larger, te_forward, te_backward))
    
    # Calculate average transfer entropy
    avg_te = np.mean(te_values)
    
    logger.info(f"Average transfer entropy: {avg_te:.6f} bits")
    
    # Generate surrogate datasets
    logger.info(f"Generating {n_surrogates} surrogate datasets")
    surrogate_datasets = generate_surrogate_data(data, n_surrogates)
    
    # Calculate transfer entropy for surrogates
    surrogate_te_values = []
    
    for i, surrogate in enumerate(surrogate_datasets):
        # Identify scale pairs in surrogate
        surrogate_pairs = identify_scale_pairs_by_constant(surrogate, tolerance)
        golden_ratio_pairs_surrogate = surrogate_pairs.get("golden_ratio", [])
        
        # Skip if no pairs
        if not golden_ratio_pairs_surrogate:
            continue
        
        # Create mapping for surrogate
        surrogate_ell = surrogate[:, 0].astype(int)
        surrogate_cl = surrogate[:, 1]
        surrogate_power_map = {l: c for l, c in zip(surrogate_ell, surrogate_cl)}
        
        # Calculate TE for each pair
        pair_te_values = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs_surrogate:
            if l_smaller in surrogate_power_map and l_larger in surrogate_power_map:
                # Get power spectrum values
                power_smaller = surrogate_power_map[l_smaller]
                power_larger = surrogate_power_map[l_larger]
                
                # Calculate transfer entropy
                if dataset_type.lower() == "wmap":
                    te = calculate_transfer_entropy(power_larger, power_smaller)
                else:
                    te = calculate_transfer_entropy(power_smaller, power_larger)
                
                pair_te_values.append(te)
        
        # Calculate average TE for this surrogate
        if pair_te_values:
            surrogate_te_values.append(np.mean(pair_te_values))
    
    # Calculate statistical significance
    surrogate_mean = np.mean(surrogate_te_values)
    surrogate_std = np.std(surrogate_te_values)
    
    # Calculate z-score
    z_score = (avg_te - surrogate_mean) / surrogate_std
    
    # Calculate effect size
    effect_size = avg_te / surrogate_mean
    
    # Calculate p-value
    if avg_te > surrogate_mean:
        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
    else:
        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
    
    # Ensure p-value is never exactly zero
    if p_value == 0:
        p_value = 1.0 / (len(surrogate_te_values) + 1)
    
    logger.info(f"Surrogate mean: {surrogate_mean:.6f}")
    logger.info(f"Surrogate std: {surrogate_std:.6f}")
    logger.info(f"Z-score: {z_score:.4f}σ")
    logger.info(f"Effect size: {effect_size:.2f}×")
    logger.info(f"P-value: {p_value:.6f}")
    
    # Determine significance
    significance = "significant" if p_value < 0.05 else "not significant"
    logger.info(f"Result is {significance} at p < 0.05")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.hist(surrogate_te_values, bins=30, alpha=0.7, label='Surrogate Data')
    plt.axvline(avg_te, color='red', linestyle='dashed', linewidth=2, label=f'Actual Data (z={z_score:.2f}σ)')
    plt.xlabel('Transfer Entropy (bits)')
    plt.ylabel('Frequency')
    plt.title(f'Transfer Entropy Analysis: {dataset_type.upper()} Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    figure_path = os.path.join(output_dir, f"{dataset_type}_transfer_entropy_histogram.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    
    # Save results
    results = {
        "dataset_type": dataset_type,
        "num_pairs": len(golden_ratio_pairs),
        "average_te": float(avg_te),
        "surrogate_mean": float(surrogate_mean),
        "surrogate_std": float(surrogate_std),
        "z_score": float(z_score),
        "effect_size": float(effect_size),
        "p_value": float(p_value),
        "significance": significance,
        "num_surrogates": len(surrogate_te_values),
        "bidirectional_analysis": [
            {
                "l_smaller": int(l_small),
                "l_larger": int(l_large),
                "te_forward": float(te_fwd),
                "te_backward": float(te_bwd),
                "ratio": float(l_large / l_small)
            }
            for l_small, l_large, te_fwd, te_bwd in bidirectional_te
        ]
    }
    
    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Results saved to {results_path}")
    
    return results


def run_transfer_entropy_analysis_optimized(data, output_dir, dataset_type="planck", n_surrogates=100, 
                                          tolerance=0.05, focus_key_scales=False, n_jobs=-1, 
                                          batch_size=100, checkpoint_interval=100):
    """
    Optimized transfer entropy analysis focused on golden ratio relationships
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    output_dir : str
        Directory to save results and visualizations
    dataset_type : str
        Type of dataset ("planck" or "wmap")
    n_surrogates : int
        Number of surrogate datasets for statistical validation
    tolerance : float
        Tolerance for considering a ratio close to a mathematical constant
    focus_key_scales : bool
        Whether to focus on key scales for more detailed analysis
    n_jobs : int
        Number of parallel jobs (-1 for all available cores)
    batch_size : int
        Batch size for surrogate processing
    checkpoint_interval : int
        Interval for saving checkpoints
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    from joblib import Parallel, delayed
    import concurrent.futures
    import pickle
    import matplotlib.pyplot as plt
    
    logger.info(f"Running optimized transfer entropy analysis on {dataset_type.upper()} data")
    
    # Create output directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define checkpoint file path
    checkpoint_file = os.path.join(dataset_dir, f"{dataset_type}_checkpoint.pkl")
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                logger.info(f"Loaded checkpoint with {len(checkpoint_data['surrogate_te_values'])} surrogates completed")
                surrogate_te_values = checkpoint_data['surrogate_te_values']
                avg_te = checkpoint_data['avg_te']
                bidirectional_te = checkpoint_data['bidirectional_te']
                golden_ratio_pairs = checkpoint_data['golden_ratio_pairs']
                remaining_surrogates = n_surrogates - len(surrogate_te_values)
                
                if remaining_surrogates <= 0:
                    logger.info("All surrogates already processed, using checkpoint data")
                    
                    # Calculate statistics from checkpoint
                    surrogate_mean = np.mean(surrogate_te_values)
                    surrogate_std = np.std(surrogate_te_values)
                    z_score = (avg_te - surrogate_mean) / surrogate_std
                    effect_size = avg_te / surrogate_mean
                    
                    # Calculate p-value
                    if avg_te > surrogate_mean:
                        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
                    else:
                        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
                    
                    # Ensure p-value is never exactly zero
                    if p_value == 0:
                        p_value = 1.0 / (len(surrogate_te_values) + 1)
                    
                    significance = "significant" if p_value < 0.05 else "not significant"
                    
                    # Create and save visualization
                    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
                    
                    # Prepare results
                    results = prepare_results_dict(
                        dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, len(surrogate_te_values),
                        bidirectional_te
                    )
                    
                    # Save results
                    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, cls=NumpyEncoder)
                    
                    logger.info(f"Results saved to {results_path}")
                    return results
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {str(e)}. Starting from scratch.")
            surrogate_te_values = []
            remaining_surrogates = n_surrogates
    else:
        surrogate_te_values = []
        remaining_surrogates = n_surrogates
    
    # If we don't have complete results from checkpoint, continue with analysis
    
    # Extract data once
    ell = data[:, 0].astype(int)
    cl = data[:, 1]
    
    # Create mapping from multipole to power
    power_map = {l: c for l, c in zip(ell, cl)}
    
    # Identify scale pairs related by golden ratio once
    if 'golden_ratio_pairs' not in locals():
        pairs_by_constant = identify_scale_pairs_by_constant(data, tolerance)
        golden_ratio_pairs = pairs_by_constant.get("golden_ratio", [])
    
    if not golden_ratio_pairs:
        logger.warning("No golden ratio scale pairs found")
        return {"status": "no_pairs_found"}
    
    logger.info(f"Found {len(golden_ratio_pairs)} scale pairs related by golden ratio")
    
    # Calculate transfer entropy for golden ratio pairs once
    if 'bidirectional_te' not in locals() or 'avg_te' not in locals():
        te_values = []
        bidirectional_te = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in power_map and l_larger in power_map:
                # Get power spectrum values
                power_smaller = power_map[l_smaller]
                power_larger = power_map[l_larger]
                
                # Calculate transfer entropy in both directions
                if dataset_type.lower() == "wmap":
                    # In WMAP data, transfer is from large to small scales
                    te_forward = calculate_transfer_entropy(power_larger, power_smaller)
                    te_backward = calculate_transfer_entropy(power_smaller, power_larger)
                else:
                    # In Planck data, transfer is from small to large scales
                    te_forward = calculate_transfer_entropy(power_smaller, power_larger)
                    te_backward = calculate_transfer_entropy(power_larger, power_smaller)
                
                te_values.append(te_forward)
                bidirectional_te.append((l_smaller, l_larger, te_forward, te_backward))
        
        # Calculate average transfer entropy
        avg_te = np.mean(te_values)
        
        logger.info(f"Average transfer entropy: {avg_te:.6f} bits")
    
    # Define function to process a single surrogate efficiently
    def process_single_surrogate(seed):
        """Process a single surrogate dataset efficiently"""
        np.random.seed(seed % (2**32 - 1))
        
        # Generate surrogate by shuffling the power spectrum values
        shuffled_cl = np.random.permutation(cl)
        shuffled_power_map = {l: c for l, c in zip(ell, shuffled_cl)}
        
        # Calculate TE for golden ratio pairs using the original pairs
        pair_te_values = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in shuffled_power_map and l_larger in shuffled_power_map:
                # Get shuffled power values
                power_smaller = shuffled_power_map[l_smaller]
                power_larger = shuffled_power_map[l_larger]
                
                # Calculate transfer entropy
                if dataset_type.lower() == "wmap":
                    te = calculate_transfer_entropy(power_larger, power_smaller)
                else:
                    te = calculate_transfer_entropy(power_smaller, power_larger)
                
                pair_te_values.append(te)
        
        # Calculate average TE for this surrogate
        if pair_te_values:
            return np.mean(pair_te_values)
        else:
            return None
    
    # Process remaining surrogates in parallel batches
    if remaining_surrogates > 0:
        logger.info(f"Processing {remaining_surrogates} remaining surrogates in parallel batches")
        
        # Determine number of jobs (cores) to use
        if n_jobs <= 0:
            n_jobs = max(1, os.cpu_count() - 1)
        
        # Process surrogates in batches
        for batch_start in range(0, remaining_surrogates, batch_size):
            batch_end = min(batch_start + batch_size, remaining_surrogates)
            batch_size_actual = batch_end - batch_start
            
            logger.info(f"Processing surrogate batch {batch_start+1}-{batch_end} of {remaining_surrogates}")
            
            # Generate seeds for this batch ensuring they're within valid range (0 to 2^32-1)
            batch_seeds = [int(time.time() * 1000 + i + len(surrogate_te_values)) % (2**32 - 1) for i in range(batch_size_actual)]
            
            # Process batch in parallel
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_single_surrogate)(seed) for seed in batch_seeds
            )
            
            # Filter out None results and add to surrogate values
            batch_results = [res for res in batch_results if res is not None]
            surrogate_te_values.extend(batch_results)
            
            # Save checkpoint after each batch
            if (batch_end % checkpoint_interval == 0) or (batch_end == remaining_surrogates):
                checkpoint_data = {
                    'surrogate_te_values': surrogate_te_values,
                    'avg_te': avg_te,
                    'bidirectional_te': bidirectional_te,
                    'golden_ratio_pairs': golden_ratio_pairs
                }
                
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                logger.info(f"Saved checkpoint with {len(surrogate_te_values)} surrogates completed")
            
            # Log progress
            progress_pct = (len(surrogate_te_values) / n_surrogates) * 100
            logger.info(f"Progress: {progress_pct:.1f}% ({len(surrogate_te_values)}/{n_surrogates} surrogates)")
    
    # Calculate statistics
    surrogate_mean = np.mean(surrogate_te_values)
    surrogate_std = np.std(surrogate_te_values)
    
    # Calculate z-score
    z_score = (avg_te - surrogate_mean) / surrogate_std
    
    # Calculate effect size
    effect_size = avg_te / surrogate_mean
    
    # Calculate p-value
    if avg_te > surrogate_mean:
        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
    else:
        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
    
    # Ensure p-value is never exactly zero
    if p_value == 0:
        p_value = 1.0 / (len(surrogate_te_values) + 1)
    
    logger.info(f"Surrogate mean: {surrogate_mean:.6f}")
    logger.info(f"Surrogate std: {surrogate_std:.6f}")
    logger.info(f"Z-score: {z_score:.4f}σ")
    logger.info(f"Effect size: {effect_size:.2f}×")
    logger.info(f"P-value: {p_value:.6f}")
    
    # Determine significance
    significance = "significant" if p_value < 0.05 else "not significant"
    logger.info(f"Result is {significance} at p < 0.05")
    
    # Create visualization
    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
    
    # Prepare results
    results = prepare_results_dict(
        dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
        z_score, effect_size, p_value, significance, len(surrogate_te_values),
        bidirectional_te
    )
    
    # Save results
    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Results saved to {results_path}")
    
    return results

# Helper functions to keep the main function cleaner
def create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir):
    """Create and save the visualization of surrogate distribution vs actual value"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(surrogate_te_values, bins=30, alpha=0.7, label='Surrogate Data')
    plt.axvline(avg_te, color='red', linestyle='dashed', linewidth=2, label=f'Actual Data (z={z_score:.2f}σ)')
    plt.xlabel('Transfer Entropy (bits)')
    plt.ylabel('Frequency')
    plt.title(f'Transfer Entropy Analysis: {dataset_type.upper()} Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    figure_path = os.path.join(output_dir, f"{dataset_type}_transfer_entropy_histogram.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()

def prepare_results_dict(dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, num_surrogates,
                        bidirectional_te):
    """Prepare the results dictionary for saving"""
    return {
        "dataset_type": dataset_type,
        "num_pairs": len(golden_ratio_pairs),
        "average_te": float(avg_te),
        "surrogate_mean": float(surrogate_mean),
        "surrogate_std": float(surrogate_std),
        "z_score": float(z_score),
        "effect_size": float(effect_size),
        "p_value": float(p_value),
        "significance": significance,
        "num_surrogates": num_surrogates,
        "bidirectional_analysis": [
            {
                "l_smaller": int(l_small),
                "l_larger": int(l_large),
                "te_forward": float(te_fwd),
                "te_backward": float(te_bwd),
                "ratio": float(l_large / l_small)
            }
            for l_small, l_large, te_fwd, te_bwd in bidirectional_te
        ]
    }

def apply_benjamini_hochberg_correction(p_values):
    """
    Apply Benjamini-Hochberg correction to control the false discovery rate
    
    Parameters:
    -----------
    p_values : dict
        Dictionary of p-values keyed by test name
        
    Returns:
    --------
    dict
        Dictionary of corrected p-values
    """
    # Extract p-values and corresponding keys
    keys = list(p_values.keys())
    values = [p_values[k] for k in keys]
    
    # Sort p-values
    sorted_indices = np.argsort(values)
    sorted_p_values = [values[i] for i in sorted_indices]
    sorted_keys = [keys[i] for i in sorted_indices]
    
    # Calculate BH critical values
    n = len(values)
    bh_values = {}
    
    # Apply BH procedure
    for i, (key, p_value) in enumerate(zip(sorted_keys, sorted_p_values)):
        # BH critical value
        bh_critical = (i + 1) / n * 0.05
        
        # Adjusted p-value
        if i > 0:
            adj_p_value = min(sorted_p_values[i] * n / (i + 1), bh_values[sorted_keys[i-1]])
        else:
            adj_p_value = sorted_p_values[i] * n / (i + 1)
        
        # Store adjusted p-value
        bh_values[key] = adj_p_value
    
    return bh_values

def analyze_specific_scales(data, scales_of_interest=[55, 89, 30], focus_key_scales=False):
    """
    Enhanced analysis of specific scales that previously showed significance
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    scales_of_interest : list
        List of scales to analyze in detail (default: [55, 89, 30])
        
    Returns:
    --------
    dict
        Dictionary containing detailed analysis results for specific scales
    """
    # Extract data
    ell = data[:, 0].astype(int)
    cl = data[:, 1]
    
    results = {}
    
    for scale in scales_of_interest:
        # Check if scale exists in the data
        if scale not in ell:
            scale_idx = np.argmin(np.abs(ell - scale))
            actual_scale = ell[scale_idx]
            logger.warning(f"Scale {scale} not found, using closest scale {actual_scale}")
            scale = actual_scale
        else:
            scale_idx = np.where(ell == scale)[0][0]
        
        # Get power at this scale
        power = cl[scale_idx]
        
        # Calculate relationship with other scales
        scale_results = {}
        
        # Focus on golden ratio and sqrt2 relationships
        for const_name, const_value in [("golden_ratio", 1.618033988749895), ("sqrt2", 1.4142135623730951)]:
            related_scales = []
            
            # Find scales related by this constant
            for other_scale in ell:
                ratio = max(scale, other_scale) / min(scale, other_scale)
                if abs(ratio - const_value) <= 0.05:
                    related_scales.append(other_scale)
            
            if related_scales:
                # Calculate transfer entropy for these related scales
                te_values = []
                
                for related_scale in related_scales:
                    related_idx = np.where(ell == related_scale)[0][0]
                    related_power = cl[related_idx]
                    
                    # Calculate in both directions
                    if focus_key_scales:
                        bins = 10  # Use more bins for key scale analysis
                    else:
                        bins = 5
                    te_forward = calculate_transfer_entropy(power, related_power, bins=bins)
                    te_backward = calculate_transfer_entropy(related_power, power, bins=bins)
                    te_backward = calculate_transfer_entropy(related_power, power, bins=5)
                    
                    te_values.append(max(te_forward, te_backward))
                
                avg_te = np.mean(te_values) if te_values else 0
                
                scale_results[const_name] = {
                    "related_scales": related_scales,
                    "avg_transfer_entropy": avg_te,
                    "num_related_scales": len(related_scales)
                }
        
        results[scale] = scale_results
    
    return results


def run_transfer_entropy_analysis_optimized(data, output_dir, dataset_type="planck", n_surrogates=100, 
                                          tolerance=0.05, focus_key_scales=False, n_jobs=-1, 
                                          batch_size=100, checkpoint_interval=100):
    """
    Optimized transfer entropy analysis focused on golden ratio relationships
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    output_dir : str
        Directory to save results and visualizations
    dataset_type : str
        Type of dataset ("planck" or "wmap")
    n_surrogates : int
        Number of surrogate datasets for statistical validation
    tolerance : float
        Tolerance for considering a ratio close to a mathematical constant
    focus_key_scales : bool
        Whether to focus on key scales for more detailed analysis
    n_jobs : int
        Number of parallel jobs (-1 for all available cores)
    batch_size : int
        Batch size for surrogate processing
    checkpoint_interval : int
        Interval for saving checkpoints
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    from joblib import Parallel, delayed
    import concurrent.futures
    import pickle
    import matplotlib.pyplot as plt
    
    logger.info(f"Running optimized transfer entropy analysis on {dataset_type.upper()} data")
    
    # Create output directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define checkpoint file path
    checkpoint_file = os.path.join(dataset_dir, f"{dataset_type}_checkpoint.pkl")
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                logger.info(f"Loaded checkpoint with {len(checkpoint_data['surrogate_te_values'])} surrogates completed")
                surrogate_te_values = checkpoint_data['surrogate_te_values']
                avg_te = checkpoint_data['avg_te']
                bidirectional_te = checkpoint_data['bidirectional_te']
                golden_ratio_pairs = checkpoint_data['golden_ratio_pairs']
                remaining_surrogates = n_surrogates - len(surrogate_te_values)
                
                if remaining_surrogates <= 0:
                    logger.info("All surrogates already processed, using checkpoint data")
                    
                    # Calculate statistics from checkpoint
                    surrogate_mean = np.mean(surrogate_te_values)
                    surrogate_std = np.std(surrogate_te_values)
                    z_score = (avg_te - surrogate_mean) / surrogate_std
                    effect_size = avg_te / surrogate_mean
                    
                    # Calculate p-value
                    if avg_te > surrogate_mean:
                        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
                    else:
                        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
                    
                    # Ensure p-value is never exactly zero
                    if p_value == 0:
                        p_value = 1.0 / (len(surrogate_te_values) + 1)
                    
                    significance = "significant" if p_value < 0.05 else "not significant"
                    
                    # Create and save visualization
                    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
                    
                    # Prepare results
                    results = prepare_results_dict(
                        dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, len(surrogate_te_values),
                        bidirectional_te
                    )
                    
                    # Save results
                    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, cls=NumpyEncoder)
                    
                    logger.info(f"Results saved to {results_path}")
                    return results
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {str(e)}. Starting from scratch.")
            surrogate_te_values = []
            remaining_surrogates = n_surrogates
    else:
        surrogate_te_values = []
        remaining_surrogates = n_surrogates
    
    # If we don't have complete results from checkpoint, continue with analysis
    
    # Extract data once
    ell = data[:, 0].astype(int)
    cl = data[:, 1]
    
    # Create mapping from multipole to power
    power_map = {l: c for l, c in zip(ell, cl)}
    
    # Identify scale pairs related by golden ratio once
    if 'golden_ratio_pairs' not in locals():
        pairs_by_constant = identify_scale_pairs_by_constant(data, tolerance)
        golden_ratio_pairs = pairs_by_constant.get("golden_ratio", [])
    
    if not golden_ratio_pairs:
        logger.warning("No golden ratio scale pairs found")
        return {"status": "no_pairs_found"}
    
    logger.info(f"Found {len(golden_ratio_pairs)} scale pairs related by golden ratio")
    
    # Calculate transfer entropy for golden ratio pairs once
    if 'bidirectional_te' not in locals() or 'avg_te' not in locals():
        te_values = []
        bidirectional_te = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in power_map and l_larger in power_map:
                # Get power spectrum values
                power_smaller = power_map[l_smaller]
                power_larger = power_map[l_larger]
                
                # Calculate transfer entropy in both directions
                if dataset_type.lower() == "wmap":
                    # In WMAP data, transfer is from large to small scales
                    te_forward = calculate_transfer_entropy(power_larger, power_smaller)
                    te_backward = calculate_transfer_entropy(power_smaller, power_larger)
                else:
                    # In Planck data, transfer is from small to large scales
                    te_forward = calculate_transfer_entropy(power_smaller, power_larger)
                    te_backward = calculate_transfer_entropy(power_larger, power_smaller)
                
                te_values.append(te_forward)
                bidirectional_te.append((l_smaller, l_larger, te_forward, te_backward))
        
        # Calculate average transfer entropy
        avg_te = np.mean(te_values)
        
        logger.info(f"Average transfer entropy: {avg_te:.6f} bits")
    
    # Define function to process a single surrogate efficiently
    def process_single_surrogate(seed):
        """Process a single surrogate dataset efficiently"""
        np.random.seed(seed % (2**32 - 1))
        
        # Generate surrogate by shuffling the power spectrum values
        shuffled_cl = np.random.permutation(cl)
        shuffled_power_map = {l: c for l, c in zip(ell, shuffled_cl)}
        
        # Calculate TE for golden ratio pairs using the original pairs
        pair_te_values = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in shuffled_power_map and l_larger in shuffled_power_map:
                # Get shuffled power values
                power_smaller = shuffled_power_map[l_smaller]
                power_larger = shuffled_power_map[l_larger]
                
                # Calculate transfer entropy
                if dataset_type.lower() == "wmap":
                    te = calculate_transfer_entropy(power_larger, power_smaller)
                else:
                    te = calculate_transfer_entropy(power_smaller, power_larger)
                
                pair_te_values.append(te)
        
        # Calculate average TE for this surrogate
        if pair_te_values:
            return np.mean(pair_te_values)
        else:
            return None
    
    # Process remaining surrogates in parallel batches
    if remaining_surrogates > 0:
        logger.info(f"Processing {remaining_surrogates} remaining surrogates in parallel batches")
        
        # Determine number of jobs (cores) to use
        if n_jobs <= 0:
            n_jobs = max(1, os.cpu_count() - 1)
        
        # Process surrogates in batches
        for batch_start in range(0, remaining_surrogates, batch_size):
            batch_end = min(batch_start + batch_size, remaining_surrogates)
            batch_size_actual = batch_end - batch_start
            
            logger.info(f"Processing surrogate batch {batch_start+1}-{batch_end} of {remaining_surrogates}")
            
            # Generate seeds for this batch ensuring they're within valid range (0 to 2^32-1)
            batch_seeds = [int(time.time() * 1000 + i + len(surrogate_te_values)) % (2**32 - 1) for i in range(batch_size_actual)]
            
            # Process batch in parallel
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_single_surrogate)(seed) for seed in batch_seeds
            )
            
            # Filter out None results and add to surrogate values
            batch_results = [res for res in batch_results if res is not None]
            surrogate_te_values.extend(batch_results)
            
            # Save checkpoint after each batch
            if (batch_end % checkpoint_interval == 0) or (batch_end == remaining_surrogates):
                checkpoint_data = {
                    'surrogate_te_values': surrogate_te_values,
                    'avg_te': avg_te,
                    'bidirectional_te': bidirectional_te,
                    'golden_ratio_pairs': golden_ratio_pairs
                }
                
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                logger.info(f"Saved checkpoint with {len(surrogate_te_values)} surrogates completed")
            
            # Log progress
            progress_pct = (len(surrogate_te_values) / n_surrogates) * 100
            logger.info(f"Progress: {progress_pct:.1f}% ({len(surrogate_te_values)}/{n_surrogates} surrogates)")
    
    # Calculate statistics
    surrogate_mean = np.mean(surrogate_te_values)
    surrogate_std = np.std(surrogate_te_values)
    
    # Calculate z-score
    z_score = (avg_te - surrogate_mean) / surrogate_std
    
    # Calculate effect size
    effect_size = avg_te / surrogate_mean
    
    # Calculate p-value
    if avg_te > surrogate_mean:
        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
    else:
        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
    
    # Ensure p-value is never exactly zero
    if p_value == 0:
        p_value = 1.0 / (len(surrogate_te_values) + 1)
    
    logger.info(f"Surrogate mean: {surrogate_mean:.6f}")
    logger.info(f"Surrogate std: {surrogate_std:.6f}")
    logger.info(f"Z-score: {z_score:.4f}σ")
    logger.info(f"Effect size: {effect_size:.2f}×")
    logger.info(f"P-value: {p_value:.6f}")
    
    # Determine significance
    significance = "significant" if p_value < 0.05 else "not significant"
    logger.info(f"Result is {significance} at p < 0.05")
    
    # Create visualization
    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
    
    # Prepare results
    results = prepare_results_dict(
        dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
        z_score, effect_size, p_value, significance, len(surrogate_te_values),
        bidirectional_te
    )
    
    # Save results
    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Results saved to {results_path}")
    
    return results

# Helper functions to keep the main function cleaner
def create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir):
    """Create and save the visualization of surrogate distribution vs actual value"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(surrogate_te_values, bins=30, alpha=0.7, label='Surrogate Data')
    plt.axvline(avg_te, color='red', linestyle='dashed', linewidth=2, label=f'Actual Data (z={z_score:.2f}σ)')
    plt.xlabel('Transfer Entropy (bits)')
    plt.ylabel('Frequency')
    plt.title(f'Transfer Entropy Analysis: {dataset_type.upper()} Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    figure_path = os.path.join(output_dir, f"{dataset_type}_transfer_entropy_histogram.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()

def prepare_results_dict(dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, num_surrogates,
                        bidirectional_te):
    """Prepare the results dictionary for saving"""
    return {
        "dataset_type": dataset_type,
        "num_pairs": len(golden_ratio_pairs),
        "average_te": float(avg_te),
        "surrogate_mean": float(surrogate_mean),
        "surrogate_std": float(surrogate_std),
        "z_score": float(z_score),
        "effect_size": float(effect_size),
        "p_value": float(p_value),
        "significance": significance,
        "num_surrogates": num_surrogates,
        "bidirectional_analysis": [
            {
                "l_smaller": int(l_small),
                "l_larger": int(l_large),
                "te_forward": float(te_fwd),
                "te_backward": float(te_bwd),
                "ratio": float(l_large / l_small)
            }
            for l_small, l_large, te_fwd, te_bwd in bidirectional_te
        ]
    }

def analyze_all_constants(data, output_dir, dataset_type="planck", n_surrogates=100, tolerance=0.05, focus_key_scales=False, checkpoint_interval=100):
    """
    Run transfer entropy analysis for all mathematical constants with checkpointing and real-time progress monitoring.
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    output_dir : str
        Directory to save results and visualizations
    dataset_type : str
        Type of dataset ("planck" or "wmap")
    n_surrogates : int
        Number of surrogate datasets for statistical validation
    tolerance : float
        Tolerance for considering a ratio close to a mathematical constant
    focus_key_scales : bool
        Whether to focus analysis on Fibonacci and golden ratio related scales
    checkpoint_interval : int
        Interval for saving checkpoint data (default: 100 surrogates)
        
    Returns:
    --------
    dict
        Dictionary containing all analysis results with BH correction
    """
    start_time = time.time()
    logger.info(f"Running analyze_all_constants for {dataset_type} data with {n_surrogates} surrogates")
    
    # Create checkpoint and progress monitoring directories
    checkpoint_dir = os.path.join(output_dir, f"{dataset_type}_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    progress_file = os.path.join(checkpoint_dir, "progress.json")
    checkpoint_file = os.path.join(checkpoint_dir, "latest_checkpoint.json")
    
    # If focusing on key scales, log specialized analysis
    if focus_key_scales:
        logger.info(f"Focusing on key scales (Fibonacci numbers and golden ratio related) in {dataset_type.upper()} data")
        # This can adjust how we analyze the data with enhanced settings
    
    # Extract multipoles and power spectrum values
    if len(data.shape) == 1:
        # Handle 1D data case (just power values)
        cl = data
        # Generate ell values based on array length, starting from lowest measured multipole
        # Standard lowest multipole is ℓ=2 for CMB
        ell = np.arange(2, 2 + len(cl)).astype(int)
    else:
        # Handle traditional 2D data with [ell, cl] columns
        ell = data[:, 0].astype(int)
        cl = data[:, 1]
    
    # Identify scale pairs related by mathematical constants
    pairs_by_constant = identify_scale_pairs_by_constant(data, tolerance)
    
    # Initialize or load previous checkpoint
    checkpoint_loaded = False
    all_constants = [name for name, pairs in pairs_by_constant.items() if len(pairs) > 0]
    current_constant_idx = 0
    current_surrogate = 0
    results = {}
    p_values = {}
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', {})
                current_constant_idx = checkpoint_data.get('current_constant_idx', 0)
                current_surrogate = checkpoint_data.get('current_surrogate', 0)
                logger.info(f"Loaded checkpoint: processing constant {current_constant_idx+1}/{len(all_constants)}, surrogate {current_surrogate}/{n_surrogates}")
                checkpoint_loaded = True
                
                # Extract existing p-values from completed constants
                for const_name, const_data in results.items():
                    if 'completed' in const_data and const_data['completed']:
                        p_values[const_name] = const_data.get('p_value', 1.0)
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {str(e)}. Starting from beginning.")
    
    # Create mapping from multipole to power
    power_map = {l: c for l, c in zip(ell, cl)}
    
    # Count total constants to analyze
    total_constants = sum(1 for _, pairs in pairs_by_constant.items() if pairs)
    constants_completed = sum(1 for const_name, data in results.items() 
                            if 'completed' in data and data['completed'])
    
    # Initialize progress tracking
    start_time = time.time() if not checkpoint_loaded else time.time() - 3600  # Assume 1 hour of previous work if checkpoint loaded
    
    # Update initial progress
    progress_data = {
        'total_constants': total_constants,
        'constants_completed': constants_completed,
        'current_constant_idx': current_constant_idx,
        'current_constant': all_constants[current_constant_idx] if current_constant_idx < len(all_constants) else 'None',
        'total_surrogates': n_surrogates,
        'current_surrogate': current_surrogate,
        'percent_complete': (constants_completed / total_constants * 100) if total_constants > 0 else 0,
        'elapsed_time': 0,
        'estimated_time_remaining': 'Calculating...',
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2, cls=NumpyEncoder)
    
    # Start or resume analysis from checkpoint
    constant_names = [name for name, pairs in pairs_by_constant.items() if pairs]
    
    for idx, const_name in enumerate(constant_names):
        # Skip constants before the current index if resuming
        if checkpoint_loaded and idx < current_constant_idx:
            logger.info(f"Skipping already completed constant: {const_name}")
            continue
            
        pairs = pairs_by_constant[const_name]
        if not pairs:
            continue
        
        # Check if this constant was already fully processed
        if const_name in results and 'completed' in results[const_name] and results[const_name]['completed']:
            logger.info(f"Skipping already completed constant: {const_name}")
            continue
        
        logger.info(f"Analyzing {const_name} with {len(pairs)} pairs")
        
        # Modify the analyze_constant function to support checkpointing
        const_result = analyze_constant_with_checkpointing(
            data, pairs, const_name, dataset_type, power_map, n_surrogates,
            checkpoint_file=checkpoint_file, progress_file=progress_file,
            start_time=start_time, constant_idx=idx, total_constants=total_constants,
            checkpoint_interval=checkpoint_interval, current_surrogate=current_surrogate if idx == current_constant_idx else 0
        )
        
        # Store results
        results[const_name] = const_result
        
        # Store p-value for BH correction
        p_values[const_name] = const_result.get("p_value", 1.0)
        
        # Reset current_surrogate for next constant
        current_surrogate = 0
        
        # Update progress
        constants_completed += 1
        progress_data = {
            'total_constants': total_constants,
            'constants_completed': constants_completed,
            'current_constant_idx': idx + 1,
            'current_constant': constant_names[idx + 1] if idx + 1 < len(constant_names) else 'Complete',
            'total_surrogates': n_surrogates,
            'current_surrogate': 0,
            'percent_complete': (constants_completed / total_constants * 100),
            'elapsed_time': time.time() - start_time,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2, cls=NumpyEncoder)
    
    # Apply BH correction
    corrected_p_values = apply_benjamini_hochberg_correction(p_values)
    
    # Update results with corrected p-values
    for const_name in results:
        results[const_name]["bh_corrected_p_value"] = corrected_p_values.get(const_name, 1.0)
        
        # Recalculate significance based on corrected p-value
        p_corrected = results[const_name]["bh_corrected_p_value"]
        if p_corrected < 0.05:
            significance = "significant"
        else:
            significance = "not significant"
            
        results[const_name]["significance_after_correction"] = significance
        
        logger.info(f"{const_name} - Original p: {p_values[const_name]:.6f}, "
                   f"BH corrected p: {p_corrected:.6f}, "
                   f"Significance: {significance}")
    
    # Save combined results
    # Ensure the dataset directory exists
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    results_path = os.path.join(dataset_dir, f"{dataset_type}_all_constants_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
        
    logger.info(f"All constants analysis with BH correction saved to {results_path}")
    
    return results


def run_transfer_entropy_analysis_optimized(data, output_dir, dataset_type="planck", n_surrogates=100, 
                                          tolerance=0.05, focus_key_scales=False, n_jobs=-1, 
                                          batch_size=100, checkpoint_interval=100):
    """
    Optimized transfer entropy analysis focused on golden ratio relationships
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    output_dir : str
        Directory to save results and visualizations
    dataset_type : str
        Type of dataset ("planck" or "wmap")
    n_surrogates : int
        Number of surrogate datasets for statistical validation
    tolerance : float
        Tolerance for considering a ratio close to a mathematical constant
    focus_key_scales : bool
        Whether to focus on key scales for more detailed analysis
    n_jobs : int
        Number of parallel jobs (-1 for all available cores)
    batch_size : int
        Batch size for surrogate processing
    checkpoint_interval : int
        Interval for saving checkpoints
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    from joblib import Parallel, delayed
    import concurrent.futures
    import pickle
    import matplotlib.pyplot as plt
    
    logger.info(f"Running optimized transfer entropy analysis on {dataset_type.upper()} data")
    
    # Create output directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define checkpoint file path
    checkpoint_file = os.path.join(dataset_dir, f"{dataset_type}_checkpoint.pkl")
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                logger.info(f"Loaded checkpoint with {len(checkpoint_data['surrogate_te_values'])} surrogates completed")
                surrogate_te_values = checkpoint_data['surrogate_te_values']
                avg_te = checkpoint_data['avg_te']
                bidirectional_te = checkpoint_data['bidirectional_te']
                golden_ratio_pairs = checkpoint_data['golden_ratio_pairs']
                remaining_surrogates = n_surrogates - len(surrogate_te_values)
                
                if remaining_surrogates <= 0:
                    logger.info("All surrogates already processed, using checkpoint data")
                    
                    # Calculate statistics from checkpoint
                    surrogate_mean = np.mean(surrogate_te_values)
                    surrogate_std = np.std(surrogate_te_values)
                    z_score = (avg_te - surrogate_mean) / surrogate_std
                    effect_size = avg_te / surrogate_mean
                    
                    # Calculate p-value
                    if avg_te > surrogate_mean:
                        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
                    else:
                        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
                    
                    # Ensure p-value is never exactly zero
                    if p_value == 0:
                        p_value = 1.0 / (len(surrogate_te_values) + 1)
                    
                    significance = "significant" if p_value < 0.05 else "not significant"
                    
                    # Create and save visualization
                    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
                    
                    # Prepare results
                    results = prepare_results_dict(
                        dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, len(surrogate_te_values),
                        bidirectional_te
                    )
                    
                    # Save results
                    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, cls=NumpyEncoder)
                    
                    logger.info(f"Results saved to {results_path}")
                    return results
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {str(e)}. Starting from scratch.")
            surrogate_te_values = []
            remaining_surrogates = n_surrogates
    else:
        surrogate_te_values = []
        remaining_surrogates = n_surrogates
    
    # If we don't have complete results from checkpoint, continue with analysis
    
    # Extract data once
    ell = data[:, 0].astype(int)
    cl = data[:, 1]
    
    # Create mapping from multipole to power
    power_map = {l: c for l, c in zip(ell, cl)}
    
    # Identify scale pairs related by golden ratio once
    if 'golden_ratio_pairs' not in locals():
        pairs_by_constant = identify_scale_pairs_by_constant(data, tolerance)
        golden_ratio_pairs = pairs_by_constant.get("golden_ratio", [])
    
    if not golden_ratio_pairs:
        logger.warning("No golden ratio scale pairs found")
        return {"status": "no_pairs_found"}
    
    logger.info(f"Found {len(golden_ratio_pairs)} scale pairs related by golden ratio")
    
    # Calculate transfer entropy for golden ratio pairs once
    if 'bidirectional_te' not in locals() or 'avg_te' not in locals():
        te_values = []
        bidirectional_te = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in power_map and l_larger in power_map:
                # Get power spectrum values
                power_smaller = power_map[l_smaller]
                power_larger = power_map[l_larger]
                
                # Calculate transfer entropy in both directions
                if dataset_type.lower() == "wmap":
                    # In WMAP data, transfer is from large to small scales
                    te_forward = calculate_transfer_entropy(power_larger, power_smaller)
                    te_backward = calculate_transfer_entropy(power_smaller, power_larger)
                else:
                    # In Planck data, transfer is from small to large scales
                    te_forward = calculate_transfer_entropy(power_smaller, power_larger)
                    te_backward = calculate_transfer_entropy(power_larger, power_smaller)
                
                te_values.append(te_forward)
                bidirectional_te.append((l_smaller, l_larger, te_forward, te_backward))
        
        # Calculate average transfer entropy
        avg_te = np.mean(te_values)
        
        logger.info(f"Average transfer entropy: {avg_te:.6f} bits")
    
    # Define function to process a single surrogate efficiently
    def process_single_surrogate(seed):
        """Process a single surrogate dataset efficiently"""
        np.random.seed(seed % (2**32 - 1))
        
        # Generate surrogate by shuffling the power spectrum values
        shuffled_cl = np.random.permutation(cl)
        shuffled_power_map = {l: c for l, c in zip(ell, shuffled_cl)}
        
        # Calculate TE for golden ratio pairs using the original pairs
        pair_te_values = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in shuffled_power_map and l_larger in shuffled_power_map:
                # Get shuffled power values
                power_smaller = shuffled_power_map[l_smaller]
                power_larger = shuffled_power_map[l_larger]
                
                # Calculate transfer entropy
                if dataset_type.lower() == "wmap":
                    te = calculate_transfer_entropy(power_larger, power_smaller)
                else:
                    te = calculate_transfer_entropy(power_smaller, power_larger)
                
                pair_te_values.append(te)
        
        # Calculate average TE for this surrogate
        if pair_te_values:
            return np.mean(pair_te_values)
        else:
            return None
    
    # Process remaining surrogates in parallel batches
    if remaining_surrogates > 0:
        logger.info(f"Processing {remaining_surrogates} remaining surrogates in parallel batches")
        
        # Determine number of jobs (cores) to use
        if n_jobs <= 0:
            n_jobs = max(1, os.cpu_count() - 1)
        
        # Process surrogates in batches
        for batch_start in range(0, remaining_surrogates, batch_size):
            batch_end = min(batch_start + batch_size, remaining_surrogates)
            batch_size_actual = batch_end - batch_start
            
            logger.info(f"Processing surrogate batch {batch_start+1}-{batch_end} of {remaining_surrogates}")
            
            # Generate seeds for this batch ensuring they're within valid range (0 to 2^32-1)
            batch_seeds = [int(time.time() * 1000 + i + len(surrogate_te_values)) % (2**32 - 1) for i in range(batch_size_actual)]
            
            # Process batch in parallel
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_single_surrogate)(seed) for seed in batch_seeds
            )
            
            # Filter out None results and add to surrogate values
            batch_results = [res for res in batch_results if res is not None]
            surrogate_te_values.extend(batch_results)
            
            # Save checkpoint after each batch
            if (batch_end % checkpoint_interval == 0) or (batch_end == remaining_surrogates):
                checkpoint_data = {
                    'surrogate_te_values': surrogate_te_values,
                    'avg_te': avg_te,
                    'bidirectional_te': bidirectional_te,
                    'golden_ratio_pairs': golden_ratio_pairs
                }
                
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                logger.info(f"Saved checkpoint with {len(surrogate_te_values)} surrogates completed")
            
            # Log progress
            progress_pct = (len(surrogate_te_values) / n_surrogates) * 100
            logger.info(f"Progress: {progress_pct:.1f}% ({len(surrogate_te_values)}/{n_surrogates} surrogates)")
    
    # Calculate statistics
    surrogate_mean = np.mean(surrogate_te_values)
    surrogate_std = np.std(surrogate_te_values)
    
    # Calculate z-score
    z_score = (avg_te - surrogate_mean) / surrogate_std
    
    # Calculate effect size
    effect_size = avg_te / surrogate_mean
    
    # Calculate p-value
    if avg_te > surrogate_mean:
        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
    else:
        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
    
    # Ensure p-value is never exactly zero
    if p_value == 0:
        p_value = 1.0 / (len(surrogate_te_values) + 1)
    
    logger.info(f"Surrogate mean: {surrogate_mean:.6f}")
    logger.info(f"Surrogate std: {surrogate_std:.6f}")
    logger.info(f"Z-score: {z_score:.4f}σ")
    logger.info(f"Effect size: {effect_size:.2f}×")
    logger.info(f"P-value: {p_value:.6f}")
    
    # Determine significance
    significance = "significant" if p_value < 0.05 else "not significant"
    logger.info(f"Result is {significance} at p < 0.05")
    
    # Create visualization
    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
    
    # Prepare results
    results = prepare_results_dict(
        dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
        z_score, effect_size, p_value, significance, len(surrogate_te_values),
        bidirectional_te
    )
    
    # Save results
    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Results saved to {results_path}")
    
    return results

# Helper functions to keep the main function cleaner
def create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir):
    """Create and save the visualization of surrogate distribution vs actual value"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(surrogate_te_values, bins=30, alpha=0.7, label='Surrogate Data')
    plt.axvline(avg_te, color='red', linestyle='dashed', linewidth=2, label=f'Actual Data (z={z_score:.2f}σ)')
    plt.xlabel('Transfer Entropy (bits)')
    plt.ylabel('Frequency')
    plt.title(f'Transfer Entropy Analysis: {dataset_type.upper()} Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    figure_path = os.path.join(output_dir, f"{dataset_type}_transfer_entropy_histogram.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()

def prepare_results_dict(dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, num_surrogates,
                        bidirectional_te):
    """Prepare the results dictionary for saving"""
    return {
        "dataset_type": dataset_type,
        "num_pairs": len(golden_ratio_pairs),
        "average_te": float(avg_te),
        "surrogate_mean": float(surrogate_mean),
        "surrogate_std": float(surrogate_std),
        "z_score": float(z_score),
        "effect_size": float(effect_size),
        "p_value": float(p_value),
        "significance": significance,
        "num_surrogates": num_surrogates,
        "bidirectional_analysis": [
            {
                "l_smaller": int(l_small),
                "l_larger": int(l_large),
                "te_forward": float(te_fwd),
                "te_backward": float(te_bwd),
                "ratio": float(l_large / l_small)
            }
            for l_small, l_large, te_fwd, te_bwd in bidirectional_te
        ]
    }


def analyze_constant_with_checkpointing(data, pairs, const_name, dataset_type, power_map, n_surrogates,
                               checkpoint_file, progress_file, start_time, constant_idx, total_constants,
                               checkpoint_interval=100, current_surrogate=0):
    """Enhanced analyze_constant function with checkpointing and progress tracking.
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data
    pairs : list
        List of scale pairs related by the constant
    const_name : str
        Name of the constant being analyzed
    dataset_type : str
        Type of dataset ("planck" or "wmap")
    power_map : dict
        Dictionary mapping scale to power value
    n_surrogates : int
        Total number of surrogate datasets to generate
    checkpoint_file : str
        Path to checkpoint file
    progress_file : str
        Path to progress tracking file
    start_time : float
        Start time of the analysis (for timing)
    constant_idx : int
        Index of current constant being processed
    total_constants : int
        Total number of constants to analyze
    checkpoint_interval : int
        Interval for saving checkpoints
    current_surrogate : int
        Starting surrogate index if resuming
    
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Calculate transfer entropy for pairs
    te_values = []
    
    for l_smaller, l_larger, _ in pairs:
        if l_smaller in power_map and l_larger in power_map:
            # Get power spectrum values
            val_smaller = power_map[l_smaller]
            val_larger = power_map[l_larger]
            
            # Calculate transfer entropy in both directions
            te_small_to_large = calculate_transfer_entropy(val_smaller, val_larger, bins=5)
            te_large_to_small = calculate_transfer_entropy(val_larger, val_smaller, bins=5)
            
            # Store results
            te_values.append({
                "scale_smaller": int(l_smaller),
                "scale_larger": int(l_larger),
                "te_small_to_large": float(te_small_to_large),
                "te_large_to_small": float(te_large_to_small),
                "te_differential": float(te_small_to_large - te_large_to_small)
            })
    
    # Calculate mean transfer entropy values
    if te_values:
        te_small_to_large_vals = [te["te_small_to_large"] for te in te_values]
        te_large_to_small_vals = [te["te_large_to_small"] for te in te_values]
        te_differential_vals = [te["te_differential"] for te in te_values]
        
        mean_te_small_to_large = np.mean(te_small_to_large_vals)
        mean_te_large_to_small = np.mean(te_large_to_small_vals)
        mean_te_differential = np.mean(te_differential_vals)
    else:
        mean_te_small_to_large = 0
        mean_te_large_to_small = 0
        mean_te_differential = 0
    
    # Check for existing surrogate data in checkpoint
    surrogate_te_small_to_large = []
    surrogate_te_large_to_small = []
    surrogate_te_differential = []
    
    # Load existing checkpoint data if available
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                if 'surrogate_data' in checkpoint_data and const_name in checkpoint_data['surrogate_data']:
                    existing_surrogates = checkpoint_data['surrogate_data'][const_name]
                    surrogate_te_small_to_large = existing_surrogates.get('small_to_large', [])
                    surrogate_te_large_to_small = existing_surrogates.get('large_to_small', [])
                    surrogate_te_differential = existing_surrogates.get('differential', [])
                    current_surrogate = len(surrogate_te_small_to_large)
                    logger.info(f"Loaded {current_surrogate} existing surrogates for {const_name}")
        except Exception as e:
            logger.warning(f"Error loading surrogate data: {str(e)}. Starting from surrogate {current_surrogate}.")
    
    # Generate surrogate datasets for statistical validation, continuing from where we left off
    for i in range(current_surrogate, n_surrogates):
        # Update progress every 10 surrogates
        if i % 10 == 0 or i == n_surrogates - 1:
            elapsed = time.time() - start_time
            constants_progress = constant_idx / total_constants
            surrogate_progress = (i / n_surrogates) / total_constants
            overall_progress = (constants_progress + surrogate_progress) * 100
            
            # Estimate time remaining
            if overall_progress > 0:
                total_time_estimate = elapsed / (overall_progress / 100)
                remaining = total_time_estimate - elapsed
                hours, remainder = divmod(remaining, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_remaining = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            else:
                time_remaining = "Calculating..."
            
            # Update progress file
            progress_data = {
                'total_constants': total_constants,
                'current_constant_idx': constant_idx,
                'current_constant': const_name,
                'total_surrogates': n_surrogates,
                'current_surrogate': i,
                'percent_complete': overall_progress,
                'elapsed_time': elapsed,
                'estimated_time_remaining': time_remaining,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2, cls=NumpyEncoder)
        
        # Generate surrogate by shuffling the original data
        surrogate_data = np.random.permutation(list(power_map.values()))
        surrogate_map = {l: surrogate_data[i % len(surrogate_data)] for i, l in enumerate(power_map.keys())}
        
        # Calculate TE for surrogate pairs
        surrogate_te_values = []
        
        for l_smaller, l_larger, _ in pairs:
            if l_smaller in surrogate_map and l_larger in surrogate_map:
                val_smaller = surrogate_map[l_smaller]
                val_larger = surrogate_map[l_larger]
                
                te_small_to_large = calculate_transfer_entropy(val_smaller, val_larger, bins=5)
                te_large_to_small = calculate_transfer_entropy(val_larger, val_smaller, bins=5)
                
                surrogate_te_values.append({
                    "te_small_to_large": float(te_small_to_large),
                    "te_large_to_small": float(te_large_to_small),
                    "te_differential": float(te_small_to_large - te_large_to_small)
                })
        
        # Calculate mean TE for this surrogate
        if surrogate_te_values:
            mean_surr_small_to_large = np.mean([te["te_small_to_large"] for te in surrogate_te_values])
            mean_surr_large_to_small = np.mean([te["te_large_to_small"] for te in surrogate_te_values])
            mean_surr_differential = np.mean([te["te_differential"] for te in surrogate_te_values])
            
            surrogate_te_small_to_large.append(mean_surr_small_to_large)
            surrogate_te_large_to_small.append(mean_surr_large_to_small)
            surrogate_te_differential.append(mean_surr_differential)
            
        # Save checkpoint periodically
        if (i + 1) % checkpoint_interval == 0 or i == n_surrogates - 1:
            # Load existing checkpoint to update it
            checkpoint_data = {}
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)
                except Exception:
                    pass
            
            # Initialize surrogate_data if needed
            if 'surrogate_data' not in checkpoint_data:
                checkpoint_data['surrogate_data'] = {}
            
            # Update surrogate data for this constant
            checkpoint_data['surrogate_data'][const_name] = {
                'small_to_large': surrogate_te_small_to_large,
                'large_to_small': surrogate_te_large_to_small,
                'differential': surrogate_te_differential
            }
            
            # Update current progress
            checkpoint_data['current_constant_idx'] = constant_idx
            checkpoint_data['current_surrogate'] = i + 1
            checkpoint_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Saved checkpoint at surrogate {i+1}/{n_surrogates} for {const_name}")

    # Calculate statistics from surrogate distributions
    if surrogate_te_small_to_large:
        # Calculate mean and std of surrogate distributions
        mean_surr_small_to_large = np.mean(surrogate_te_small_to_large)
        std_surr_small_to_large = np.std(surrogate_te_small_to_large) if len(surrogate_te_small_to_large) > 1 else 1e-10
        
        mean_surr_large_to_small = np.mean(surrogate_te_large_to_small)
        std_surr_large_to_small = np.std(surrogate_te_large_to_small) if len(surrogate_te_large_to_small) > 1 else 1e-10
        
        mean_surr_differential = np.mean(surrogate_te_differential)
        std_surr_differential = np.std(surrogate_te_differential) if len(surrogate_te_differential) > 1 else 1e-10
        
        # Calculate z-scores
        z_small_to_large = (mean_te_small_to_large - mean_surr_small_to_large) / std_surr_small_to_large
        z_large_to_small = (mean_te_large_to_small - mean_surr_large_to_small) / std_surr_large_to_small
        z_differential = (mean_te_differential - mean_surr_differential) / std_surr_differential
        
        # Calculate p-values (one-tailed test)
        p_small_to_large = 1 - norm.cdf(z_small_to_large)
        p_large_to_small = 1 - norm.cdf(z_large_to_small)
        p_differential = 1 - norm.cdf(z_differential) if mean_te_differential > mean_surr_differential else norm.cdf(z_differential)
        
        # Calculate effect sizes (ratio of real to surrogate mean)
        effect_small_to_large = mean_te_small_to_large / mean_surr_small_to_large if mean_surr_small_to_large > 0 else 0
        effect_large_to_small = mean_te_large_to_small / mean_surr_large_to_small if mean_surr_large_to_small > 0 else 0
        effect_differential = mean_te_differential / mean_surr_differential if mean_surr_differential > 0 else 0
    else:
        # Default values if no surrogates were generated
        z_small_to_large = 0
        z_large_to_small = 0
        z_differential = 0
        
        p_small_to_large = 1.0
        p_large_to_small = 1.0
        p_differential = 1.0
        
        effect_small_to_large = 0
        effect_large_to_small = 0
        effect_differential = 0
        
        mean_surr_small_to_large = 0
        std_surr_small_to_large = 0
        mean_surr_large_to_small = 0
        std_surr_large_to_small = 0
        mean_surr_differential = 0
        std_surr_differential = 0
    
    # Prepare and return results
    result = {
        # Real data TE values
        "mean_te_small_to_large": float(mean_te_small_to_large),
        "mean_te_large_to_small": float(mean_te_large_to_small),
        "mean_te_differential": float(mean_te_differential),
        
        # Surrogate statistics
        "surrogate_mean_small_to_large": float(mean_surr_small_to_large),
        "surrogate_std_small_to_large": float(std_surr_small_to_large),
        "surrogate_mean_large_to_small": float(mean_surr_large_to_small),
        "surrogate_std_large_to_small": float(std_surr_large_to_small),
        "surrogate_mean_differential": float(mean_surr_differential),
        "surrogate_std_differential": float(std_surr_differential),
        
        # Statistical measures
        "z_score_small_to_large": float(z_small_to_large),
        "z_score_large_to_small": float(z_large_to_small),
        "z_score_differential": float(z_differential),
        
        "p_value_small_to_large": float(p_small_to_large),
        "p_value_large_to_small": float(p_large_to_small),
        "p_value_differential": float(p_differential),
        
        "effect_size_small_to_large": float(effect_small_to_large),
        "effect_size_large_to_small": float(effect_large_to_small),
        "effect_size_differential": float(effect_differential),
        
        # Use the differential p-value as the primary p-value for BH correction
        "p_value": float(p_differential),
        
        # Information about number of surrogates used
        "n_surrogates": len(surrogate_te_small_to_large),
        
        # Detailed information about scale pairs
        "scale_pairs": pairs,
        "num_pairs": len(pairs),
        "te_values": te_values,
        
        # Mark as completed
        "completed": True
    }
    
    return result


def analyze_constant(data, pairs, const_name, dataset_type, power_map, n_surrogates):
    """Helper function to analyze a specific constant"""
    # Calculate transfer entropy for pairs
    te_values = []
    
    for l_smaller, l_larger, _ in pairs:
        if l_smaller in power_map and l_larger in power_map:
            # Get power spectrum values
            val_smaller = power_map[l_smaller]
            val_larger = power_map[l_larger]
            
            # Calculate transfer entropy in both directions
            te_small_to_large = calculate_transfer_entropy(val_smaller, val_larger, bins=5)
            te_large_to_small = calculate_transfer_entropy(val_larger, val_smaller, bins=5)
            
            # Store results
            te_values.append({
                "scale_smaller": int(l_smaller),
                "scale_larger": int(l_larger),
                "te_small_to_large": float(te_small_to_large),
                "te_large_to_small": float(te_large_to_small),
                "te_differential": float(te_small_to_large - te_large_to_small)
            })
    
    if not te_values:
        return {"status": "no_valid_calculations"}
    
    # Calculate mean transfer entropy values
    te_small_to_large_vals = [te["te_small_to_large"] for te in te_values]
    te_large_to_small_vals = [te["te_large_to_small"] for te in te_values]
    te_differential_vals = [te["te_differential"] for te in te_values]
    
    mean_te_small_to_large = np.mean(te_small_to_large_vals)
    mean_te_large_to_small = np.mean(te_large_to_small_vals)
    mean_te_differential = np.mean(te_differential_vals)
    
    # Generate surrogate datasets for statistical validation
    surrogate_te_small_to_large = []
    surrogate_te_large_to_small = []
    surrogate_te_differential = []
    
    for i in range(n_surrogates):
        # Generate surrogate by shuffling the original data
        surrogate_data = np.random.permutation(list(power_map.values()))
        surrogate_map = {l: surrogate_data[i % len(surrogate_data)] for i, l in enumerate(power_map.keys())}
        
        # Calculate TE for surrogate pairs
        surrogate_te_values = []
        
        for l_smaller, l_larger, _ in pairs:
            if l_smaller in surrogate_map and l_larger in surrogate_map:
                val_smaller = surrogate_map[l_smaller]
                val_larger = surrogate_map[l_larger]
                
                te_small_to_large = calculate_transfer_entropy(val_smaller, val_larger, bins=5)
                te_large_to_small = calculate_transfer_entropy(val_larger, val_smaller, bins=5)
                
                surrogate_te_values.append({
                    "te_small_to_large": float(te_small_to_large),
                    "te_large_to_small": float(te_large_to_small),
                    "te_differential": float(te_small_to_large - te_large_to_small)
                })
        
        # Calculate mean TE for this surrogate
        if surrogate_te_values:
            mean_surr_small_to_large = np.mean([te["te_small_to_large"] for te in surrogate_te_values])
            mean_surr_large_to_small = np.mean([te["te_large_to_small"] for te in surrogate_te_values])
            mean_surr_differential = np.mean([te["te_differential"] for te in surrogate_te_values])
            
            surrogate_te_small_to_large.append(mean_surr_small_to_large)
            surrogate_te_large_to_small.append(mean_surr_large_to_small)
            surrogate_te_differential.append(mean_surr_differential)
    
    # Calculate statistics from surrogate distributions
    if surrogate_te_small_to_large:
        # Calculate mean and std of surrogate distributions
        mean_surr_small_to_large = np.mean(surrogate_te_small_to_large)
        std_surr_small_to_large = np.std(surrogate_te_small_to_large) if len(surrogate_te_small_to_large) > 1 else 1e-10
        
        mean_surr_large_to_small = np.mean(surrogate_te_large_to_small)
        std_surr_large_to_small = np.std(surrogate_te_large_to_small) if len(surrogate_te_large_to_small) > 1 else 1e-10
        
        mean_surr_differential = np.mean(surrogate_te_differential)
        std_surr_differential = np.std(surrogate_te_differential) if len(surrogate_te_differential) > 1 else 1e-10
        
        # Calculate z-scores
        z_small_to_large = (mean_te_small_to_large - mean_surr_small_to_large) / std_surr_small_to_large
        z_large_to_small = (mean_te_large_to_small - mean_surr_large_to_small) / std_surr_large_to_small
        z_differential = (mean_te_differential - mean_surr_differential) / std_surr_differential
        
        # Calculate p-values (one-tailed test)
        p_small_to_large = 1 - norm.cdf(z_small_to_large)
        p_large_to_small = 1 - norm.cdf(z_large_to_small)
        p_differential = 1 - norm.cdf(z_differential) if mean_te_differential > mean_surr_differential else norm.cdf(z_differential)
        
        # Calculate effect sizes (ratio of real to surrogate mean)
        effect_small_to_large = mean_te_small_to_large / mean_surr_small_to_large if mean_surr_small_to_large > 0 else 0
        effect_large_to_small = mean_te_large_to_small / mean_surr_large_to_small if mean_surr_large_to_small > 0 else 0
        effect_differential = mean_te_differential / mean_surr_differential if mean_surr_differential > 0 else 0
    else:
        # Default values if no surrogates were generated
        z_small_to_large = 0
        z_large_to_small = 0
        z_differential = 0
        
        p_small_to_large = 1.0
        p_large_to_small = 1.0
        p_differential = 1.0
        
        effect_small_to_large = 0
        effect_large_to_small = 0
        effect_differential = 0
        
        mean_surr_small_to_large = 0
        std_surr_small_to_large = 0
        mean_surr_large_to_small = 0
        std_surr_large_to_small = 0
        mean_surr_differential = 0
        std_surr_differential = 0
    
    # Prepare and return results
    result = {
        # Real data TE values
        "mean_te_small_to_large": float(mean_te_small_to_large),
        "mean_te_large_to_small": float(mean_te_large_to_small),
        "mean_te_differential": float(mean_te_differential),
        
        # Surrogate statistics
        "surrogate_mean_small_to_large": float(mean_surr_small_to_large),
        "surrogate_std_small_to_large": float(std_surr_small_to_large),
        "surrogate_mean_large_to_small": float(mean_surr_large_to_small),
        "surrogate_std_large_to_small": float(std_surr_large_to_small),
        "surrogate_mean_differential": float(mean_surr_differential),
        "surrogate_std_differential": float(std_surr_differential),
        
        # Statistical measures
        "z_score_small_to_large": float(z_small_to_large),
        "z_score_large_to_small": float(z_large_to_small),
        "z_score_differential": float(z_differential),
        
        "p_value_small_to_large": float(p_small_to_large),
        "p_value_large_to_small": float(p_large_to_small),
        "p_value_differential": float(p_differential),
        
        "effect_size_small_to_large": float(effect_small_to_large),
        "effect_size_large_to_small": float(effect_large_to_small),
        "effect_size_differential": float(effect_differential),
        
        # Use the differential p-value as the primary p-value for BH correction
        "p_value": float(p_differential),
        
        # Detailed information about scale pairs
        "scale_pairs": pairs,
        "num_pairs": len(pairs),
        "te_values": te_values
    }
    
    return result
    
    # Calculate average transfer entropy
    avg_te = np.mean(te_values)
    
    # Generate surrogate datasets
    surrogate_datasets = generate_surrogate_data(data, n_surrogates)
    
    # Calculate transfer entropy for surrogates
    surrogate_te_values = []
    
    for surrogate in surrogate_datasets:
        # Identify scale pairs in surrogate
        surrogate_pairs = identify_scale_pairs_by_constant(surrogate, 0.05)
        
        # Get pairs for this constant
        const_pairs = surrogate_pairs.get(const_name, [])
        
        # Skip if no pairs
        if not const_pairs:
            continue
        
        # Create mapping for surrogate
        surrogate_ell = surrogate[:, 0].astype(int)
        surrogate_cl = surrogate[:, 1]
        surrogate_power_map = {l: c for l, c in zip(surrogate_ell, surrogate_cl)}
        
        # Calculate TE for each pair
        pair_te_values = []
        
        for l_smaller, l_larger, _ in const_pairs:
            if l_smaller in surrogate_power_map and l_larger in surrogate_power_map:
                # Get power spectrum values
                power_smaller = surrogate_power_map[l_smaller]
                power_larger = surrogate_power_map[l_larger]
                
                # Calculate transfer entropy
                if dataset_type.lower() == "wmap":
                    te = calculate_transfer_entropy(power_larger, power_smaller)
                else:
                    te = calculate_transfer_entropy(power_smaller, power_larger)
                
                pair_te_values.append(te)
        
        # Calculate average TE for this surrogate
        if pair_te_values:
            surrogate_te_values.append(np.mean(pair_te_values))
    
    # Calculate statistical significance
    surrogate_mean = np.mean(surrogate_te_values) if surrogate_te_values else 0
    surrogate_std = np.std(surrogate_te_values) if surrogate_te_values else 1
    
    # Calculate z-score
    if surrogate_std == 0:
        z_score = 0
    else:
        z_score = (avg_te - surrogate_mean) / surrogate_std
    
    # Calculate effect size
    if surrogate_mean == 0:
        effect_size = np.inf if avg_te > 0 else -np.inf
    else:
        effect_size = avg_te / surrogate_mean
    
    # Calculate p-value
    if avg_te > surrogate_mean:
        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
    else:
        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
    
    # Ensure p-value is never exactly zero
    if p_value == 0:
        p_value = 1.0 / (len(surrogate_te_values) + 1)
    
    return {
        "constant": const_name,
        "average_te": avg_te,
        "surrogate_mean": surrogate_mean,
        "surrogate_std": surrogate_std,
        "z_score": z_score,
        "effect_size": effect_size,
        "p_value": p_value,
        "num_pairs": len(pairs),
        "num_valid_pairs": len(te_values),
        "num_surrogates": len(surrogate_te_values)
    }


def analyze_key_scales(data, output_dir, dataset_type="planck", n_surrogates=100, tolerance=0.05):
    """
    Perform enhanced analysis on key scales related to Fibonacci numbers and the golden ratio.
    
    Parameters
    ----------
    data : numpy.ndarray
        Array containing CMB power spectrum data with columns [l, Cl, error].
    output_dir : str
        Output directory for results.
    dataset_type : str
        Type of dataset ("planck" or "wmap").
    n_surrogates : int
        Number of surrogate datasets to generate.
    tolerance : float
        Tolerance for scale pair relationships.
        
    Returns
    -------
    dict
        Dictionary containing analysis results for key scales.
    """
    logger.info(f"Running enhanced analysis on key Fibonacci and golden ratio related scales for {dataset_type.upper()} data")
    
    # Create output directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract ell values (multipole scales) from the first column of data
    ell_values = data[:, 0].astype(int)
    
    # Define key Fibonacci scales to analyze
    fibonacci_scales = [21, 34, 55, 89, 144, 233]
    
    # Define other key scales related to golden ratio (phi) and multiples
    phi = (1 + 5**0.5) / 2
    phi_related_scales = [
        int(phi * s) for s in range(20, 200) if int(phi * s) < max(ell_values)
    ]
    
    # Combine all key scales
    key_scales = sorted(list(set(fibonacci_scales + phi_related_scales)))
    
    # Filter to ensure all scales are in the data range
    key_scales = [s for s in key_scales if s in ell_values]
    
    logger.info(f"Analyzing {len(key_scales)} key scales: {key_scales}")
    
    # Analysis results
    results = {
        "key_scales": key_scales,
        "fibonacci_scales": fibonacci_scales,
        "phi_related_scales": phi_related_scales,
        "scale_results": {}
    }
    
    # Create a mapping of ell values to power spectrum values (Cl) from the second column
    power_map = {int(row[0]): row[1] for row in data}
    
    # Analyze transfer entropy for each key scale as the source
    for source_scale in key_scales:
        # Find index of source scale in data
        source_idx = np.where(ell_values == source_scale)[0][0]
        source_data = data[source_idx, 1]  # Power spectrum value (Cl) is in the second column
        
        # Analyze transfer to other key scales
        scale_results = {}
        for target_scale in key_scales:
            # Skip self-transfers
            if target_scale == source_scale:
                continue
                
            # Find index of target scale in data
            target_idx = np.where(ell_values == target_scale)[0][0]
            target_data = data[target_idx, 1]  # Power spectrum value (Cl) is in the second column
            
            # Calculate the transfer entropy
            te = calculate_transfer_entropy(source_data, target_data)
            
            # Calculate transfer entropy for surrogate data
            surrogate_te = []
            for _ in range(n_surrogates):
                # Generate surrogate data - for scalar data, we need to create small arrays
                # Generate small surrogate arrays from random normal distributions with same mean and std
                np.random.seed(int(_ * source_data * target_data) % 10000)  # Reproducible but varying seed
                surrogate_source_array = np.random.normal(source_data, abs(source_data)*0.1, size=10) 
                surrogate_target_array = np.random.normal(target_data, abs(target_data)*0.1, size=10)
                
                # Calculate transfer entropy for surrogate data
                surrogate_te.append(calculate_transfer_entropy(surrogate_source_array, surrogate_target_array))
                
            # Calculate statistics
            surrogate_mean = np.mean(surrogate_te)
            surrogate_std = np.std(surrogate_te)
            z_score = (te - surrogate_mean) / surrogate_std if surrogate_std > 0 else 0
            p_value = np.mean([1 if x >= te else 0 for x in surrogate_te])
            
            # Save results
            scale_results[target_scale] = {
                "transfer_entropy": float(te),
                "surrogate_mean": float(surrogate_mean),
                "surrogate_std": float(surrogate_std),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }
            
        # Save results for this source scale
        results["scale_results"][source_scale] = scale_results
        
        # Log significant findings
        significant_targets = [t for t, r in scale_results.items() if r["significant"]]
        if significant_targets:
            logger.info(f"Scale {source_scale} has significant transfer entropy to scales: {significant_targets}")
    
    # Check for special relationships between Fibonacci scales
    fibonacci_pairs = []
    for i, s1 in enumerate(fibonacci_scales):
        for j, s2 in enumerate(fibonacci_scales):
            if i != j and s1 in key_scales and s2 in key_scales:
                # Check if s1 is in the results and s2 is in the scale_results for s1
                if s1 in results["scale_results"] and s2 in results["scale_results"][s1]:
                    # Get the result for this Fibonacci pair
                    result = results["scale_results"][s1][s2]
                    if result["significant"]:
                        fibonacci_pairs.append((s1, s2, result["z_score"], result["p_value"]))
    
    # Sort Fibonacci pairs by significance (lowest p-value first)
    fibonacci_pairs.sort(key=lambda x: x[3])
    
    # Add Fibonacci pair results
    results["fibonacci_pairs"] = [
        {
            "source": int(s1),
            "target": int(s2),
            "z_score": float(z),
            "p_value": float(p)
        }
        for s1, s2, z, p in fibonacci_pairs
    ]
    
    # Log the number of significant Fibonacci pairs
    logger.info(f"Found {len(fibonacci_pairs)} significant Fibonacci pairs")
    
    # Special analysis for scale 55 (known to be significant from previous research)
    if 55 in key_scales:
        logger.info("Performing detailed analysis on scale 55")
        scale_55_results = {}
        source_idx = np.where(ell_values == 55)[0][0]
        source_data = data[source_idx, 1]  # Power spectrum value (Cl) is in the second column
        
        # Analyze transfer to all available scales
        for target_scale in ell_values:
            if target_scale == 55:
                continue
                
            target_idx = np.where(ell_values == target_scale)[0][0]
            target_data = data[target_idx, 1]  # Power spectrum value (Cl) is in the second column
            
            # Calculate the transfer entropy
            te = calculate_transfer_entropy(source_data, target_data)
            
            # Calculate transfer entropy for surrogate data
            surrogate_te = []
            for _ in range(min(20, n_surrogates)):  # Use fewer surrogates for comprehensive analysis
                # Generate surrogate data - for scalar data, we need to create small arrays
                # Generate small surrogate arrays from random normal distributions with same mean and std
                np.random.seed(int(_ * source_data * target_data) % 10000)  # Reproducible but varying seed
                surrogate_source_array = np.random.normal(source_data, abs(source_data)*0.1, size=10) 
                surrogate_target_array = np.random.normal(target_data, abs(target_data)*0.1, size=10)
                
                # Calculate transfer entropy for surrogate data
                surrogate_te.append(calculate_transfer_entropy(surrogate_source_array, surrogate_target_array))
                
            # Calculate statistics
            surrogate_mean = np.mean(surrogate_te)
            surrogate_std = np.std(surrogate_te)
            z_score = (te - surrogate_mean) / surrogate_std if surrogate_std > 0 else 0
            p_value = np.mean([1 if x >= te else 0 for x in surrogate_te])
            
            # Save results
            scale_55_results[target_scale] = {
                "transfer_entropy": float(te),
                "surrogate_mean": float(surrogate_mean),
                "surrogate_std": float(surrogate_std),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }
        
        # Find significant targets
        significant_targets = [(t, r["z_score"], r["p_value"]) for t, r in scale_55_results.items() if r["significant"]]
        significant_targets.sort(key=lambda x: x[2])  # Sort by p-value
        
        # Check if any of these significant targets are related to phi
        phi_related_significant = []
        for scale, z_score, p_value in significant_targets:
            ratio = scale / 55 if scale > 55 else 55 / scale
            if abs(ratio - phi) < tolerance:
                phi_related_significant.append((scale, z_score, p_value, "phi"))
            # Also check for sqrt2 relationship (which has shown importance in previous findings)
            elif abs(ratio - 2**0.5) < tolerance:
                phi_related_significant.append((scale, z_score, p_value, "sqrt2"))
        
        # Add special scale 55 results
        results["scale_55_detailed"] = {
            "all_significant": [
                {"scale": int(t), "z_score": float(z), "p_value": float(p)}
                for t, z, p in significant_targets
            ],
            "phi_related_significant": [
                {"scale": int(t), "z_score": float(z), "p_value": float(p), "relation": r}
                for t, z, p, r in phi_related_significant
            ]
        }
        
        logger.info(f"Scale 55 has {len(significant_targets)} significant connections")
        logger.info(f"Of these, {len(phi_related_significant)} are related to phi or sqrt2")
    
    # Save results to file
    results_path = os.path.join(output_dir, f"{dataset_type}_key_scales_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Key scales analysis results saved to {results_path}")
    
    return results


def run_transfer_entropy_analysis_optimized(data, output_dir, dataset_type="planck", n_surrogates=100, 
                                          tolerance=0.05, focus_key_scales=False, n_jobs=-1, 
                                          batch_size=100, checkpoint_interval=100):
    """
    Optimized transfer entropy analysis focused on golden ratio relationships
    
    Parameters:
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    output_dir : str
        Directory to save results and visualizations
    dataset_type : str
        Type of dataset ("planck" or "wmap")
    n_surrogates : int
        Number of surrogate datasets for statistical validation
    tolerance : float
        Tolerance for considering a ratio close to a mathematical constant
    focus_key_scales : bool
        Whether to focus on key scales for more detailed analysis
    n_jobs : int
        Number of parallel jobs (-1 for all available cores)
    batch_size : int
        Batch size for surrogate processing
    checkpoint_interval : int
        Interval for saving checkpoints
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    from joblib import Parallel, delayed
    import concurrent.futures
    import pickle
    import matplotlib.pyplot as plt
    
    logger.info(f"Running optimized transfer entropy analysis on {dataset_type.upper()} data")
    
    # Create output directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define checkpoint file path
    checkpoint_file = os.path.join(dataset_dir, f"{dataset_type}_checkpoint.pkl")
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                logger.info(f"Loaded checkpoint with {len(checkpoint_data['surrogate_te_values'])} surrogates completed")
                surrogate_te_values = checkpoint_data['surrogate_te_values']
                avg_te = checkpoint_data['avg_te']
                bidirectional_te = checkpoint_data['bidirectional_te']
                golden_ratio_pairs = checkpoint_data['golden_ratio_pairs']
                remaining_surrogates = n_surrogates - len(surrogate_te_values)
                
                if remaining_surrogates <= 0:
                    logger.info("All surrogates already processed, using checkpoint data")
                    
                    # Calculate statistics from checkpoint
                    surrogate_mean = np.mean(surrogate_te_values)
                    surrogate_std = np.std(surrogate_te_values)
                    z_score = (avg_te - surrogate_mean) / surrogate_std
                    effect_size = avg_te / surrogate_mean
                    
                    # Calculate p-value
                    if avg_te > surrogate_mean:
                        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
                    else:
                        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
                    
                    # Ensure p-value is never exactly zero
                    if p_value == 0:
                        p_value = 1.0 / (len(surrogate_te_values) + 1)
                    
                    significance = "significant" if p_value < 0.05 else "not significant"
                    
                    # Create and save visualization
                    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
                    
                    # Prepare results
                    results = prepare_results_dict(
                        dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, len(surrogate_te_values),
                        bidirectional_te
                    )
                    
                    # Save results
                    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, cls=NumpyEncoder)
                    
                    logger.info(f"Results saved to {results_path}")
                    return results
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {str(e)}. Starting from scratch.")
            surrogate_te_values = []
            remaining_surrogates = n_surrogates
    else:
        surrogate_te_values = []
        remaining_surrogates = n_surrogates
    
    # If we don't have complete results from checkpoint, continue with analysis
    
    # Extract data once
    ell = data[:, 0].astype(int)
    cl = data[:, 1]
    
    # Create mapping from multipole to power
    power_map = {l: c for l, c in zip(ell, cl)}
    
    # Identify scale pairs related by golden ratio once
    if 'golden_ratio_pairs' not in locals():
        pairs_by_constant = identify_scale_pairs_by_constant(data, tolerance)
        golden_ratio_pairs = pairs_by_constant.get("golden_ratio", [])
    
    if not golden_ratio_pairs:
        logger.warning("No golden ratio scale pairs found")
        return {"status": "no_pairs_found"}
    
    logger.info(f"Found {len(golden_ratio_pairs)} scale pairs related by golden ratio")
    
    # Calculate transfer entropy for golden ratio pairs once
    if 'bidirectional_te' not in locals() or 'avg_te' not in locals():
        te_values = []
        bidirectional_te = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in power_map and l_larger in power_map:
                # Get power spectrum values
                power_smaller = power_map[l_smaller]
                power_larger = power_map[l_larger]
                
                # Calculate transfer entropy in both directions
                if dataset_type.lower() == "wmap":
                    # In WMAP data, transfer is from large to small scales
                    te_forward = calculate_transfer_entropy(power_larger, power_smaller)
                    te_backward = calculate_transfer_entropy(power_smaller, power_larger)
                else:
                    # In Planck data, transfer is from small to large scales
                    te_forward = calculate_transfer_entropy(power_smaller, power_larger)
                    te_backward = calculate_transfer_entropy(power_larger, power_smaller)
                
                te_values.append(te_forward)
                bidirectional_te.append((l_smaller, l_larger, te_forward, te_backward))
        
        # Calculate average transfer entropy
        avg_te = np.mean(te_values)
        
        logger.info(f"Average transfer entropy: {avg_te:.6f} bits")
    
    # Define function to process a single surrogate efficiently
    def process_single_surrogate(seed):
        """Process a single surrogate dataset efficiently"""
        np.random.seed(seed % (2**32 - 1))
        
        # Generate surrogate by shuffling the power spectrum values
        shuffled_cl = np.random.permutation(cl)
        shuffled_power_map = {l: c for l, c in zip(ell, shuffled_cl)}
        
        # Calculate TE for golden ratio pairs using the original pairs
        pair_te_values = []
        
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in shuffled_power_map and l_larger in shuffled_power_map:
                # Get shuffled power values
                power_smaller = shuffled_power_map[l_smaller]
                power_larger = shuffled_power_map[l_larger]
                
                # Calculate transfer entropy
                if dataset_type.lower() == "wmap":
                    te = calculate_transfer_entropy(power_larger, power_smaller)
                else:
                    te = calculate_transfer_entropy(power_smaller, power_larger)
                
                pair_te_values.append(te)
        
        # Calculate average TE for this surrogate
        if pair_te_values:
            return np.mean(pair_te_values)
        else:
            return None
    
    # Process remaining surrogates in parallel batches
    if remaining_surrogates > 0:
        logger.info(f"Processing {remaining_surrogates} remaining surrogates in parallel batches")
        
        # Determine number of jobs (cores) to use
        if n_jobs <= 0:
            n_jobs = max(1, os.cpu_count() - 1)
        
        # Process surrogates in batches
        for batch_start in range(0, remaining_surrogates, batch_size):
            batch_end = min(batch_start + batch_size, remaining_surrogates)
            batch_size_actual = batch_end - batch_start
            
            logger.info(f"Processing surrogate batch {batch_start+1}-{batch_end} of {remaining_surrogates}")
            
            # Generate seeds for this batch ensuring they're within valid range (0 to 2^32-1)
            batch_seeds = [int(time.time() * 1000 + i + len(surrogate_te_values)) % (2**32 - 1) for i in range(batch_size_actual)]
            
            # Process batch in parallel
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_single_surrogate)(seed) for seed in batch_seeds
            )
            
            # Filter out None results and add to surrogate values
            batch_results = [res for res in batch_results if res is not None]
            surrogate_te_values.extend(batch_results)
            
            # Save checkpoint after each batch
            if (batch_end % checkpoint_interval == 0) or (batch_end == remaining_surrogates):
                checkpoint_data = {
                    'surrogate_te_values': surrogate_te_values,
                    'avg_te': avg_te,
                    'bidirectional_te': bidirectional_te,
                    'golden_ratio_pairs': golden_ratio_pairs
                }
                
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                logger.info(f"Saved checkpoint with {len(surrogate_te_values)} surrogates completed")
            
            # Log progress
            progress_pct = (len(surrogate_te_values) / n_surrogates) * 100
            logger.info(f"Progress: {progress_pct:.1f}% ({len(surrogate_te_values)}/{n_surrogates} surrogates)")
    
    # Calculate statistics
    surrogate_mean = np.mean(surrogate_te_values)
    surrogate_std = np.std(surrogate_te_values)
    
    # Calculate z-score
    z_score = (avg_te - surrogate_mean) / surrogate_std
    
    # Calculate effect size
    effect_size = avg_te / surrogate_mean
    
    # Calculate p-value
    if avg_te > surrogate_mean:
        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
    else:
        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
    
    # Ensure p-value is never exactly zero
    if p_value == 0:
        p_value = 1.0 / (len(surrogate_te_values) + 1)
    
    logger.info(f"Surrogate mean: {surrogate_mean:.6f}")
    logger.info(f"Surrogate std: {surrogate_std:.6f}")
    logger.info(f"Z-score: {z_score:.4f}σ")
    logger.info(f"Effect size: {effect_size:.2f}×")
    logger.info(f"P-value: {p_value:.6f}")
    
    # Determine significance
    significance = "significant" if p_value < 0.05 else "not significant"
    logger.info(f"Result is {significance} at p < 0.05")
    
    # Create visualization
    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
    
    # Prepare results
    results = prepare_results_dict(
        dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
        z_score, effect_size, p_value, significance, len(surrogate_te_values),
        bidirectional_te
    )
    
    # Save results
    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Results saved to {results_path}")
    
    return results

# Helper functions to keep the main function cleaner
def create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir):
    """Create and save the visualization of surrogate distribution vs actual value"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(surrogate_te_values, bins=30, alpha=0.7, label='Surrogate Data')
    plt.axvline(avg_te, color='red', linestyle='dashed', linewidth=2, label=f'Actual Data (z={z_score:.2f}σ)')
    plt.xlabel('Transfer Entropy (bits)')
    plt.ylabel('Frequency')
    plt.title(f'Transfer Entropy Analysis: {dataset_type.upper()} Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    figure_path = os.path.join(output_dir, f"{dataset_type}_transfer_entropy_histogram.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()

def prepare_results_dict(dataset_type, golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, num_surrogates,
                        bidirectional_te):
    """Prepare the results dictionary for saving"""
    return {
        "dataset_type": dataset_type,
        "num_pairs": len(golden_ratio_pairs),
        "average_te": float(avg_te),
        "surrogate_mean": float(surrogate_mean),
        "surrogate_std": float(surrogate_std),
        "z_score": float(z_score),
        "effect_size": float(effect_size),
        "p_value": float(p_value),
        "significance": significance,
        "num_surrogates": num_surrogates,
        "bidirectional_analysis": [
            {
                "l_smaller": int(l_small),
                "l_larger": int(l_large),
                "te_forward": float(te_fwd),
                "te_backward": float(te_bwd),
                "ratio": float(l_large / l_small)
            }
            for l_small, l_large, te_fwd, te_bwd in bidirectional_te
        ]
    }

def run_transfer_entropy_analysis_balanced(data, output_dir, dataset_type="planck", n_surrogates=100, 
                                        tolerance=0.05, focus_key_scales=False, n_jobs=-1, 
                                        batch_size=100, checkpoint_interval=100, sampling_strategy="stratified"):
    """
    Run a balanced transfer entropy analysis focused on golden ratio relationships,
    using stratified sampling for computational efficiency
    
    Parameters
    -----------
    data : np.ndarray
        Power spectrum data with columns [l, Cl, error]
    output_dir : str
        Directory to save results and visualizations
    dataset_type : str
        Type of dataset ("planck" or "wmap")
    n_surrogates : int
        Number of surrogate datasets for statistical validation
    tolerance : float
        Tolerance for considering a ratio close to a mathematical constant
    focus_key_scales : bool
        Whether to focus on key scales for more detailed analysis
    n_jobs : int
        Number of parallel jobs (-1 for all available cores)
    batch_size : int
        Batch size for surrogate processing
    checkpoint_interval : int
        Interval for saving checkpoints
    sampling_strategy : str
        Strategy for sampling scale pairs - "stratified", "fibonacci_focused", or "all"
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    from joblib import Parallel, delayed
    import concurrent.futures
    import pickle
    import matplotlib.pyplot as plt
    
    logger.info(f"Running balanced transfer entropy analysis on {dataset_type.upper()} data")
    logger.info(f"Using sampling strategy: {sampling_strategy}")
    
    # Create output directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define checkpoint file path
    checkpoint_file = os.path.join(dataset_dir, f"{dataset_type}_checkpoint.pkl")
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                logger.info(f"Loaded checkpoint with {len(checkpoint_data['surrogate_te_values'])} surrogates completed")
                surrogate_te_values = checkpoint_data['surrogate_te_values']
                avg_te = checkpoint_data['avg_te']
                bidirectional_te = checkpoint_data['bidirectional_te']
                golden_ratio_pairs = checkpoint_data['golden_ratio_pairs']
                used_scale_pairs = checkpoint_data.get('used_scale_pairs', [])
                remaining_surrogates = n_surrogates - len(surrogate_te_values)
                
                if remaining_surrogates <= 0:
                    logger.info("All surrogates already processed, using checkpoint data")
                    
                    # Calculate statistics from checkpoint
                    surrogate_mean = np.mean(surrogate_te_values)
                    surrogate_std = np.std(surrogate_te_values)
                    z_score = (avg_te - surrogate_mean) / surrogate_std
                    effect_size = avg_te / surrogate_mean
                    
                    # Calculate p-value
                    if avg_te > surrogate_mean:
                        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
                    else:
                        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
                    
                    # Ensure p-value is never exactly zero
                    if p_value == 0:
                        p_value = 1.0 / (len(surrogate_te_values) + 1)
                    
                    significance = "significant" if p_value < 0.05 else "not significant"
                    
                    # Create and save visualization
                    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
                    
                    # Prepare results
                    results = prepare_results_dict(
                        dataset_type, used_scale_pairs if used_scale_pairs else golden_ratio_pairs, avg_te, surrogate_mean, surrogate_std,
                        z_score, effect_size, p_value, significance, len(surrogate_te_values),
                        bidirectional_te
                    )
                    
                    # Save results
                    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, cls=NumpyEncoder)
                    
                    logger.info(f"Results saved to {results_path}")
                    return results
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {str(e)}. Starting from scratch.")
            surrogate_te_values = []
            remaining_surrogates = n_surrogates
    else:
        surrogate_te_values = []
        remaining_surrogates = n_surrogates
    
    # If we don't have complete results from checkpoint, continue with analysis
    
    # Extract data once
    ell = data[:, 0].astype(int)
    cl = data[:, 1]
    
    # Create mapping from multipole to power
    power_map = {l: c for l, c in zip(ell, cl)}
    
    # Identify scale pairs related by golden ratio once
    if 'golden_ratio_pairs' not in locals():
        pairs_by_constant = identify_scale_pairs_by_constant(data, tolerance)
        golden_ratio_pairs = pairs_by_constant.get("golden_ratio", [])
        
        # Get all pairs by mathematical constants
        all_constant_pairs = []
        for constant, pairs in pairs_by_constant.items():
            all_constant_pairs.extend([(l_smaller, l_larger, constant) for l_smaller, l_larger, _ in pairs])
        
        # For stratified sampling, we'll also get the Fibonacci scale pairs
        fibonacci_scales = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        fibonacci_related_pairs = []
        for l_smaller, l_larger, constant in all_constant_pairs:
            if l_smaller in fibonacci_scales or l_larger in fibonacci_scales:
                fibonacci_related_pairs.append((l_smaller, l_larger, constant))
    
    if not golden_ratio_pairs:
        logger.warning("No golden ratio scale pairs found")
        return {"status": "no_pairs_found"}
    
    logger.info(f"Found {len(golden_ratio_pairs)} scale pairs related by golden ratio")
    
    # Select scale pairs based on sampling strategy
    if sampling_strategy == "all":
        # Use all golden ratio pairs
        used_scale_pairs = golden_ratio_pairs
        logger.info(f"Using all {len(used_scale_pairs)} golden ratio scale pairs")
    
    elif sampling_strategy == "fibonacci_focused":
        # Focus on golden ratio pairs that involve Fibonacci scales
        fibonacci_scales = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        used_scale_pairs = []
        for l_smaller, l_larger, _ in golden_ratio_pairs:
            if l_smaller in fibonacci_scales or l_larger in fibonacci_scales:
                used_scale_pairs.append((l_smaller, l_larger, _))
        
        # If too few pairs, add some non-Fibonacci pairs
        if len(used_scale_pairs) < 100:
            remaining_pairs = [p for p in golden_ratio_pairs if p not in used_scale_pairs]
            used_scale_pairs.extend(remaining_pairs[:min(100, len(remaining_pairs))])
            
        logger.info(f"Using {len(used_scale_pairs)} scale pairs focused on Fibonacci scales")
    
    elif sampling_strategy == "stratified":
        # Use stratified sampling - mix of Fibonacci related and random pairs
        # First, identify pairs involving Fibonacci scales
        fibonacci_scales = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        fibonacci_pairs = []
        other_pairs = []
        
        for pair in golden_ratio_pairs:
            l_smaller, l_larger, _ = pair
            if l_smaller in fibonacci_scales or l_larger in fibonacci_scales:
                fibonacci_pairs.append(pair)
            else:
                other_pairs.append(pair)
        
        # Take all Fibonacci pairs and a random sample of other pairs
        # Aim for ~1000 total pairs or 10% of all pairs, whichever is smaller
        target_count = min(1000, int(0.1 * len(golden_ratio_pairs)))
        remaining_count = max(0, target_count - len(fibonacci_pairs))
        
        if remaining_count > 0 and other_pairs:
            # Random sample without replacement
            np.random.seed(42)  # For reproducibility
            random_indices = np.random.choice(len(other_pairs), 
                                             size=min(remaining_count, len(other_pairs)), 
                                             replace=False)
            random_pairs = [other_pairs[i] for i in random_indices]
            used_scale_pairs = fibonacci_pairs + random_pairs
        else:
            used_scale_pairs = fibonacci_pairs
        
        logger.info(f"Using stratified sampling with {len(fibonacci_pairs)} Fibonacci-related pairs")
        logger.info(f"and {len(used_scale_pairs) - len(fibonacci_pairs)} randomly selected pairs")
        logger.info(f"Total: {len(used_scale_pairs)} scale pairs ({len(used_scale_pairs)/len(golden_ratio_pairs):.1%} of all golden ratio pairs)")
    
    else:
        logger.warning(f"Unknown sampling strategy: {sampling_strategy}. Using all golden ratio pairs.")
        used_scale_pairs = golden_ratio_pairs
    
    # Calculate transfer entropy for selected scale pairs once
    if 'bidirectional_te' not in locals() or 'avg_te' not in locals():
        te_values = []
        bidirectional_te = []
        
        for l_smaller, l_larger, _ in used_scale_pairs:
            if l_smaller in power_map and l_larger in power_map:
                # Get power spectrum values
                power_smaller = power_map[l_smaller]
                power_larger = power_map[l_larger]
                
                # Calculate transfer entropy in both directions
                if dataset_type.lower() == "wmap":
                    # In WMAP data, transfer is from large to small scales
                    te_forward = calculate_transfer_entropy(power_larger, power_smaller)
                    te_backward = calculate_transfer_entropy(power_smaller, power_larger)
                else:
                    # In Planck data, transfer is from small to large scales
                    te_forward = calculate_transfer_entropy(power_smaller, power_larger)
                    te_backward = calculate_transfer_entropy(power_larger, power_smaller)
                
                te_values.append(te_forward)
                bidirectional_te.append((l_smaller, l_larger, te_forward, te_backward))
        
        # Calculate average transfer entropy
        avg_te = np.mean(te_values)
        
        logger.info(f"Average transfer entropy: {avg_te:.6f} bits")
    
    # Define function to process a single surrogate efficiently
    def process_single_surrogate(seed):
        """Process a single surrogate dataset efficiently"""
        np.random.seed(seed % (2**32 - 1))
        
        # Generate surrogate by shuffling the power spectrum values
        shuffled_cl = np.random.permutation(cl)
        shuffled_power_map = {l: c for l, c in zip(ell, shuffled_cl)}
        
        # Calculate TE for selected scale pairs using the original pairs list
        pair_te_values = []
        
        for l_smaller, l_larger, _ in used_scale_pairs:
            if l_smaller in shuffled_power_map and l_larger in shuffled_power_map:
                # Get shuffled power values
                power_smaller = shuffled_power_map[l_smaller]
                power_larger = shuffled_power_map[l_larger]
                
                # Calculate transfer entropy
                if dataset_type.lower() == "wmap":
                    te = calculate_transfer_entropy(power_larger, power_smaller)
                else:
                    te = calculate_transfer_entropy(power_smaller, power_larger)
                
                pair_te_values.append(te)
        
        # Calculate average TE for this surrogate
        if pair_te_values:
            return np.mean(pair_te_values)
        else:
            return None
    
    # Early stopping variables
    z_scores = []
    early_stop = False
    required_stable_iterations = 5  # Number of consecutive batches where z-score is stable
    stable_count = 0
    
    # Process remaining surrogates in parallel batches
    if remaining_surrogates > 0:
        logger.info(f"Processing {remaining_surrogates} remaining surrogates in parallel batches")
        
        # Determine number of jobs (cores) to use
        if n_jobs <= 0:
            n_jobs = max(1, os.cpu_count() - 1)
        
        # Process surrogates in batches
        for batch_start in range(0, remaining_surrogates, batch_size):
            batch_end = min(batch_start + batch_size, remaining_surrogates)
            batch_size_actual = batch_end - batch_start
            
            logger.info(f"Processing surrogate batch {batch_start+1}-{batch_end} of {remaining_surrogates}")
            
            # Generate seeds for this batch ensuring they're within valid range (0 to 2^32-1)
            batch_seeds = [int(time.time() * 1000 + i + len(surrogate_te_values)) % (2**32 - 1) for i in range(batch_size_actual)]
            
            # Process batch in parallel
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_single_surrogate)(seed) for seed in batch_seeds
            )
            
            # Filter out None results and add to surrogate values
            batch_results = [res for res in batch_results if res is not None]
            surrogate_te_values.extend(batch_results)
            
            # Save checkpoint after each batch
            if (batch_end % checkpoint_interval == 0) or (batch_end == remaining_surrogates):
                checkpoint_data = {
                    "surrogate_te_values": surrogate_te_values,
                    "avg_te": avg_te,
                    "bidirectional_te": bidirectional_te,
                    "golden_ratio_pairs": golden_ratio_pairs,
                    "used_scale_pairs": used_scale_pairs
                }
                
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                logger.info(f"Saved checkpoint with {len(surrogate_te_values)} surrogates completed")
                
                # Calculate current z-score
                if len(surrogate_te_values) >= 100:
                    surrogate_mean = np.mean(surrogate_te_values)
                    surrogate_std = np.std(surrogate_te_values)
                    current_z = (avg_te - surrogate_mean) / surrogate_std
                    z_scores.append(current_z)
                    logger.info(f"Current Z-score after {len(surrogate_te_values)} surrogates: {current_z:.4f}σ")
                    
                    # Check if z-score has stabilized (only if we have enough scores)
                    if len(z_scores) >= 3:
                        last_z = z_scores[-1]
                        prev_z = z_scores[-2]
                        z_diff = abs(last_z - prev_z)
                        
                        # If z-score change is very small (< 0.01), count as stable
                        if z_diff < 0.01:
                            stable_count += 1
                            logger.info(f"Z-score stabilizing: {stable_count}/{required_stable_iterations} stable iterations")
                        else:
                            stable_count = 0
                        
                        # If we've had enough stable iterations and at least 2000 surrogates, stop early
                        if stable_count >= required_stable_iterations and len(surrogate_te_values) >= 2000:
                            logger.info(f"Z-score has stabilized at {last_z:.4f}σ after {len(surrogate_te_values)} surrogates")
                            logger.info("Early stopping criteria met")
                            early_stop = True
                            break
            
            # If early stopping criteria met, break the batch loop
            if early_stop:
                break
    
    # Calculate statistics from final surrogate values
    surrogate_mean = np.mean(surrogate_te_values)
    surrogate_std = np.std(surrogate_te_values)
    z_score = (avg_te - surrogate_mean) / surrogate_std
    effect_size = avg_te / surrogate_mean
    
    # Calculate p-value
    if avg_te > surrogate_mean:
        p_value = np.sum(np.array(surrogate_te_values) >= avg_te) / len(surrogate_te_values)
    else:
        p_value = np.sum(np.array(surrogate_te_values) <= avg_te) / len(surrogate_te_values)
    
    # Ensure p-value is never exactly zero
    if p_value == 0:
        p_value = 1.0 / (len(surrogate_te_values) + 1)
    
    significance = "significant" if p_value < 0.05 else "not significant"
    
    # Create and save visualization
    create_and_save_visualization(surrogate_te_values, avg_te, z_score, dataset_type, output_dir)
    
    # Prepare results
    results = prepare_results_dict(
        dataset_type, used_scale_pairs, avg_te, surrogate_mean, surrogate_std,
        z_score, effect_size, p_value, significance, len(surrogate_te_values),
        bidirectional_te
    )
    
    # Save results
    results_path = os.path.join(output_dir, f"{dataset_type}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Results saved to {results_path}")
    return results
