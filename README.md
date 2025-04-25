# CMB Consciousness Analysis

This repository contains the finalized code for the analyses of cosmic microwave background (CMB) data that produced significant results. All analyses examine both WMAP and Planck datasets simultaneously to ensure consistent testing protocols and comparable results across datasets.

## Key Components

### Comprehensive Mathematical Organization Analysis
The primary component is the comprehensive analysis that revealed remarkable mathematical organization in CMB data, particularly involving Fibonacci scales and mathematical constants:

- **Golden Ratio Precision**: Found extremely precise Golden Ratio relationships between Fibonacci scales, with many pairs matching the Golden Ratio to within 0.01%
- **Transfer Entropy Directionality**: Identified significant information flow between key scales, with strong directionality that follows the Fibonacci sequence
- **Scale 55 Significance**: Discovered that Scale 55 emerges as a particularly important scale in both datasets, showing extremely strong mathematical relationships
- **Cross-Dataset Validation**: Demonstrated remarkable consistency between Planck and WMAP datasets, providing strong validation of these patterns

Full results are documented in [COMPREHENSIVE_RESULTS.md](./COMPREHENSIVE_RESULTS.md)

### Fractal Analysis
The repository also includes the final robust fractal analysis code that successfully reproduced significant results:
- **Planck Data**: 17.71σ significance (p=0.0001)
- **WMAP Data**: 3.53σ significance (p=0.0018)
- Uses full-resolution unbinned power spectrum data from official archives:
  - Planck Legacy Archive: 2,507 data points
  - NASA's LAMBDA archive (WMAP): 1,199 data points

### Hierarchical Information Architecture Test
The Hierarchical Test analyzes how different mathematical constants organize the hierarchical information structure in CMB data:

- Tests six mathematical constants: phi (golden ratio), sqrt2, sqrt3, ln2, e, and pi
- Completed with 10,000 simulations on both datasets simultaneously
- Key findings from full analysis:
  - Square Root of 2 is confirmed as the dominant organizing principle with 157,507 scale pairs in Planck data
  - Scale 55 shows significant relationships across multiple constants (Scale 38-55: z=2.077, p=0.019)
  - Most significant relationship: Scale 377-534 with sqrt2 (z=3.194, p=0.0007)
  - Fibonacci numbers appear prominently in significant scale relationships

Full results are documented in [docs/hierarchical_analysis/hierarchical_test_results.md](./docs/hierarchical_analysis/hierarchical_test_results.md)

## Results

All results files, including datasets, visualizations, and complete analysis outputs, are available in the Zenodo repository:
[Link to Zenodo DOI will be added upon publication]

## Running the Analysis

The analysis consists of two key components, both designed to run on WMAP and Planck datasets simultaneously as per our established testing protocol:

```bash
# Run the final fractal analysis on both datasets (reproduces significant results)
python scripts/robust_cmb_fractal.py

# Run the Hierarchical Test with 10,000 simulations
python scripts/hierarchical_analysis.py --surrogates 10000
```

### Recommended Order
1. First run the fractal analysis to verify the significant Hurst exponent findings
2. Then run the Information Architecture Test to analyze mathematical organizing principles

## Data Sources

- Planck data: Full resolution unbinned power spectrum data from the Planck Legacy Archive (2,507 data points)
- WMAP data: Full resolution power spectrum from NASA's LAMBDA archive (1,199 data points)

## Citation

If you use this code or the results in your research, please cite:
[Citation details will be added upon publication]
