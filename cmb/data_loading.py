"""
CMB Data Loading module - Handles loading and initial processing of WMAP and Planck data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import logging

logger = logging.getLogger(__name__)

def load_power_spectrum(file_path, dataset_type="planck"):
    """
    Load CMB power spectrum data from file
    """
    logger.info(f"Loading {dataset_type.upper()} power spectrum from: {file_path}")
    
    try:
        # First try the standard format
        data = np.loadtxt(file_path)
        if data.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns (l, Cl), found {data.shape[1]}")
    except Exception as e:
        logger.warning(f"Standard loading failed: {e}. Trying alternative formats...")
        
        # Try different formats based on dataset type
        if dataset_type.lower() == "planck":
            try:
                # Try Planck format from PLA
                data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None).values
            except:
                # Try FITS format for Planck
                with fits.open(file_path) as hdul:
                    data = hdul[1].data
                    # Extract l and Cl columns
                    ell = data['ell']
                    cl = data['d_ell']
                    error = data.get('error', np.sqrt(cl))  # Use sqrt(Cl) as error if not available
                    data = np.column_stack((ell, cl, error))
        else:
            # Try WMAP format
            try:
                data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None).values
            except:
                raise ValueError(f"Could not load {dataset_type} data from {file_path}")
    
    # Validate the loaded data
    if data.shape[0] < 30:
        logger.warning(f"Loaded data has only {data.shape[0]} data points, which is fewer than expected")
    
    logger.info(f"Successfully loaded {dataset_type.upper()} power spectrum: {data.shape[0]} data points")
    
    # Ensure we have l, Cl, and error columns
    if data.shape[1] == 2:
        # Add error column (sqrt(Cl) as a simple approximation if not available)
        error_col = np.sqrt(np.abs(data[:, 1]))
        data = np.column_stack((data, error_col))
    
    return data
