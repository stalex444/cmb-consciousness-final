# CMB Consciousness Analysis

This repository contains the code for cosmic microwave background (CMB) data analysis focusing on potential mathematical organization and fractal properties. The analysis examines both WMAP and Planck datasets simultaneously to ensure consistent testing protocols and comparable results.

## Key Components

### Fractal Analysis
- Reproduces significant fractal analysis results with high statistical significance
- WMAP (3.53σ) and Planck (17.71σ) datasets both show significant fractal properties
- Uses full-resolution unbinned power spectrum data from official archives (WMAP: 1,199 data points, Planck: 2,507 data points)

### Information Architecture Tests
- Analyzes how different mathematical constants (phi, sqrt2, sqrt3, ln2, e, pi) organize aspects of the hierarchical information structure in CMB data
- Identifies significant patterns across both datasets
- Key findings include the dominant role of the Square Root of 2 as an organizing principle across scales

## Results

All results files, including datasets, visualizations, and complete analysis outputs, are available in the Zenodo repository:
[Link to Zenodo DOI will be added upon publication]

## Running the Analysis

All analysis code is designed to be run on both datasets simultaneously to ensure comparable results:

```bash
# Run the fractal analysis on both datasets
python robust_cmb_fractal.py

# Run the Information Architecture Test with 10,000 simulations
python run_information_architecture_test.py --surrogates 10000
```

## Data Sources

- Planck data: Full resolution unbinned power spectrum data from the Planck Legacy Archive (2,507 data points)
- WMAP data: Full resolution power spectrum from NASA's LAMBDA archive (1,199 data points)

## Citation

If you use this code or the results in your research, please cite:
[Citation details will be added upon publication]
