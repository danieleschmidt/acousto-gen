#!/usr/bin/env python3
"""Debug test to isolate the complex number issue."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ['ACOUSTO_GEN_FORCE_MOCK'] = '1'

def debug_test():
    try:
        import acousto_gen
        from acousto_gen.core import AcousticHologram
        import numpy as np
        
        print("✅ Basic imports successful")
        
        # Create mock transducer
        class MockTransducer:
            def __init__(self):
                self.positions = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
        
        transducer = MockTransducer()
        print("✅ Mock transducer created")
        
        # Try creating hologram
        print("🔍 Creating AcousticHologram...")
        hologram = AcousticHologram(
            transducer=transducer,
            frequency=40000,
            medium="air"
        )
        print("✅ AcousticHologram created successfully")
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_test()