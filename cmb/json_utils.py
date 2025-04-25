"""
JSON utilities for handling special types in CMB analysis results
"""
import json
import numpy as np

# Helper function to convert tuple keys to strings in nested dictionaries
def convert_tuple_keys(obj):
    """Convert tuple keys in dictionaries to strings for JSON serialization"""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                new_key = str(k)
            else:
                new_key = k
            new_dict[new_key] = convert_tuple_keys(v)  # Process nested values recursively
        return new_dict
    elif isinstance(obj, list):
        return [convert_tuple_keys(item) for item in obj]  # Process lists recursively
    else:
        return obj  # Return non-dict/list objects as is

class EnhancedNumpyEncoder(json.JSONEncoder):
    """Enhanced encoder to handle NumPy types and tuple keys"""
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
            return super(EnhancedNumpyEncoder, self).encode(convert_tuple_keys(obj))
        return super(EnhancedNumpyEncoder, self).encode(obj)
