"""
Benjamini-Hochberg Correction for CMB Scale Analysis

This module implements the Benjamini-Hochberg procedure for controlling 
the false discovery rate in multiple comparison tests, specifically for 
CMB scale analysis and mathematical constant tests.
"""

import numpy as np
import pandas as pd
import os
import json
from scipy import stats

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

def infer_relationship_type(ratio):
    """
    Infer the type of mathematical relationship based on the ratio
    
    Parameters:
    - ratio: The ratio between scales
    
    Returns:
    - String indicating the inferred relationship type
    """
    # Define mathematical constants and their tolerances
    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # ~1.618
    SQRT2 = np.sqrt(2)  # ~1.414
    PI_DIV_2 = np.pi / 2  # ~1.571
    E_DIV_2 = np.e / 2  # ~1.359
    
    # Define tolerances for each constant
    tolerances = {
        'golden_ratio': 0.01,
        'sqrt2': 0.01,
        'pi_div_2': 0.01,
        'e_div_2': 0.01,
        'integer': 0.01
    }
    
    # Check proximity to each constant
    if abs(ratio - GOLDEN_RATIO) <= tolerances['golden_ratio']:
        return 'golden_ratio'
    if abs(ratio - SQRT2) <= tolerances['sqrt2']:
        return 'sqrt2'
    if abs(ratio - PI_DIV_2) <= tolerances['pi_div_2']:
        return 'pi_div_2'
    if abs(ratio - E_DIV_2) <= tolerances['e_div_2']:
        return 'e_div_2'
    
    # Check if it's close to an integer or simple fraction
    for i in range(1, 6):
        if abs(ratio - i) <= tolerances['integer']:
            return f'integer_{i}'
        for j in range(1, 6):
            if j > i and abs(ratio - (i/j)) <= tolerances['integer']:
                return f'fraction_{i}_{j}'
            if i > j and abs(ratio - (i/j)) <= tolerances['integer']:
                return f'fraction_{i}_{j}'
    
    return 'other'

def apply_bh_correction_to_scales_analysis(results_file, output_file=None, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction to CMB scales analysis results
    
    Parameters:
    - results_file: Path to CSV file with scale analysis results
    - output_file: Path to save BH-corrected results (default: adds '_bh' to input filename)
    - alpha: False discovery rate threshold (default: 0.05)
    
    Returns:
    - DataFrame with BH-corrected results
    """
    # Default output file name if not provided
    if output_file is None:
        base, ext = os.path.splitext(results_file)
        output_file = f"{base}_bh{ext}"
    
    # Load results
    print(f"Loading scale analysis results from {results_file}")
    df = pd.read_csv(results_file)
    
    # If p_value isn't directly found, look for alternative column names
    p_value_alternatives = ['p_value', 'p-value', 'pvalue', 'p', 'significance', 'prob']
    p_col = None
    for col in p_value_alternatives:
        if col in df.columns:
            p_col = col
            break
    
    if p_col is None:
        raise ValueError(f"Could not find p-value column. Please ensure results contain one of: {p_value_alternatives}")
    
    # Extract p-values
    p_values = df[p_col].values
    
    # Apply BH correction
    print(f"Applying Benjamini-Hochberg correction to {len(p_values)} tests with alpha={alpha}")
    significant = benjamini_hochberg_correction(p_values, alpha)
    
    # Add BH-corrected significance to results
    df['significant_bh'] = significant
    
    # Count significant results before and after correction
    sig_before = (p_values < 0.05).sum()
    sig_after = significant.sum()
    print(f"Significant results before correction: {sig_before}/{len(p_values)} ({sig_before/len(p_values)*100:.1f}%)")
    print(f"Significant results after correction: {sig_after}/{len(p_values)} ({sig_after/len(p_values)*100:.1f}%)")
    
    # Save BH-corrected results
    df.to_csv(output_file, index=False)
    print(f"BH-corrected results saved to {output_file}")
    
    # Generate summary of significant findings after correction
    if sig_after > 0:
        print("\nSignificant findings after BH correction:")
        sig_df = df[df['significant_bh']]
        
        # If we have columns identifying mathematical relationships, group by them
        if 'relationship_type' in df.columns:
            relationship_counts = sig_df.groupby('relationship_type').size()
            print("By relationship type:")
            for rel_type, count in relationship_counts.items():
                print(f"  {rel_type}: {count}")
            
            # For each relationship type, show the most significant examples
            for rel_type in relationship_counts.index:
                type_df = sig_df[sig_df['relationship_type'] == rel_type].sort_values(p_col)
                print(f"\nTop {min(3, len(type_df))} {rel_type} relationships:")
                for _, row in type_df.head(3).iterrows():
                    scale_pair = str(row.get('scale_pair', ''))
                    if not scale_pair and 'scale_a' in row and 'scale_b' in row:
                        scale_pair = f"{row['scale_a']} - {row['scale_b']}"
                    print(f"  {scale_pair}: ratio={row.get('ratio', 'N/A'):.6f}, p={row[p_col]:.6e}")
        else:
            # Try to infer relationship types based on ratio proximity to constants
            sig_df['inferred_type'] = sig_df['ratio'].apply(infer_relationship_type)
            relationship_counts = sig_df.groupby('inferred_type').size()
            print("By inferred relationship type:")
            for rel_type, count in relationship_counts.items():
                print(f"  {rel_type}: {count}")
            
            # Show the most significant examples by inferred type
            for rel_type in relationship_counts.index:
                type_df = sig_df[sig_df['inferred_type'] == rel_type].sort_values(p_col)
                print(f"\nTop {min(3, len(type_df))} {rel_type} relationships:")
                for _, row in type_df.head(3).iterrows():
                    scale_pair = str(row.get('scale_pair', ''))
                    if not scale_pair and 'scale_a' in row and 'scale_b' in row:
                        scale_pair = f"{row['scale_a']} - {row['scale_b']}"
                    print(f"  {scale_pair}: ratio={row.get('ratio', 'N/A'):.6f}, p={row[p_col]:.6e}")
        
        # Print top 5 most significant findings overall
        print("\nTop 5 most significant findings overall:")
        for _, row in sig_df.sort_values(p_col).head(5).iterrows():
            scale_pair = str(row.get('scale_pair', ''))
            if not scale_pair and 'scale_a' in row and 'scale_b' in row:
                scale_pair = f"{row['scale_a']} - {row['scale_b']}"
            print(f"  {scale_pair}: ratio={row.get('ratio', 'N/A'):.6f}, p={row[p_col]:.6e}")
    
    return df

def apply_bh_to_fibonacci_sequence_analysis(results_file, output_file=None, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction specifically to Fibonacci sequence analysis results
    
    Parameters:
    - results_file: Path to JSON or CSV file with Fibonacci sequence analysis results
    - output_file: Path to save BH-corrected results
    - alpha: False discovery rate threshold (default: 0.05)
    
    Returns:
    - Corrected results data structure
    """
    # Check file extension
    ext = os.path.splitext(results_file)[1].lower()
    
    if ext == '.json':
        # Load JSON data
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Apply BH correction to each set of p-values
        # This is highly dependent on the structure of your JSON results
        if 'fibonacci' in data:
            p_values = []
            p_value_locations = []
            
            # Collect all p-values and their locations
            for const in ['phi', 'sqrt2', 'pi', 'e', '2']:
                if const in data['fibonacci']:
                    p_values.append(data['fibonacci'][const]['p_value'])
                    p_value_locations.append(('fibonacci', const))
            
            if 'powers_of_2' in data:
                for const in ['phi', 'sqrt2', 'pi', 'e', '2']:
                    if const in data['powers_of_2']:
                        p_values.append(data['powers_of_2'][const]['p_value'])
                        p_value_locations.append(('powers_of_2', const))
            
            # Apply BH correction
            significant = benjamini_hochberg_correction(p_values, alpha)
            
            # Update data with corrected significance
            for i, (sequence, const) in enumerate(p_value_locations):
                data[sequence][const]['significant_bh'] = bool(significant[i])
            
            # Save corrected results
            if output_file is None:
                base, ext = os.path.splitext(results_file)
                output_file = f"{base}_bh{ext}"
                
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Print summary
            sig_before = sum(1 for p in p_values if p < 0.05)
            sig_after = significant.sum()
            print(f"Significant results before correction: {sig_before}/{len(p_values)} ({sig_before/len(p_values)*100:.1f}%)")
            print(f"Significant results after correction: {sig_after}/{len(p_values)} ({sig_after/len(p_values)*100:.1f}%)")
            
            # Print significant findings
            if sig_after > 0:
                print("\nSignificant findings after BH correction:")
                for i, (sequence, const) in enumerate(p_value_locations):
                    if significant[i]:
                        print(f"  {sequence} - {const}: p={p_values[i]:.6e}")
            
            return data
        else:
            print("JSON structure not recognized. Please modify the script to match your data format.")
            return data
    
    elif ext == '.csv':
        # For CSV files, use the general scales analysis function
        return apply_bh_correction_to_scales_analysis(results_file, output_file, alpha)
    
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Please provide a JSON or CSV file.")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python apply_bh_correction.py <results_file> [output_file] [alpha]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    
    # Infer which function to use based on the filename
    if 'fibonacci' in results_file.lower() or 'sequence' in results_file.lower():
        apply_bh_to_fibonacci_sequence_analysis(results_file, output_file, alpha)
    else:
        apply_bh_correction_to_scales_analysis(results_file, output_file, alpha)
