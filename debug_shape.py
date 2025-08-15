#!/usr/bin/env python3
"""Debug shape creation issue."""

import sys
import os
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ['ACOUSTO_GEN_FORCE_MOCK'] = '1'

def debug_shape():
    try:
        # Import mock backend first
        from acousto_gen.mock_backend import setup_mock_dependencies
        setup_mock_dependencies()
        
        import numpy as np
        
        print("Testing numpy operations directly:")
        
        # Test linspace
        x = np.linspace(-0.1, 0.1, 10)
        print(f"linspace result: {type(x)}, data: {x.data[:5] if hasattr(x, 'data') else 'no data attr'}")
        
        # Test meshgrid
        y = np.linspace(-0.1, 0.1, 10) 
        z = np.linspace(0, 0.2, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        print(f"meshgrid results: X shape={getattr(X, 'shape', 'no shape')}, Y shape={getattr(Y, 'shape', 'no shape')}, Z shape={getattr(Z, 'shape', 'no shape')}")
        
        # Test math operations
        r_squared = ((X - 0)**2 + (Y - 0)**2 + (Z - 0.1)**2)
        print(f"r_squared shape: {getattr(r_squared, 'shape', 'no shape')}")
        
        # Test exp
        field_data = 3000 * np.exp(-r_squared / (2 * 0.005**2))
        print(f"field_data shape: {getattr(field_data, 'shape', 'no shape')}")
        
        # Test astype
        complex_data = field_data.astype(complex)
        print(f"complex_data shape: {getattr(complex_data, 'shape', 'no shape')}")
        print(f"complex_data data length: {len(complex_data.data) if hasattr(complex_data, 'data') else 'no data'}")
        
        # Check intermediate results
        print(f"field_data data length: {len(field_data.data) if hasattr(field_data, 'data') else 'no data'}")
        print(f"r_squared data length: {len(r_squared.data) if hasattr(r_squared, 'data') else 'no data'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_shape()