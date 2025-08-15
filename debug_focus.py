#!/usr/bin/env python3
"""Debug focus creation issue."""

import sys
import os
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ['ACOUSTO_GEN_FORCE_MOCK'] = '1'

def debug_focus():
    try:
        import acousto_gen
        from acousto_gen.core import AcousticHologram
        import numpy as np
        
        # Create mock transducer
        class MockTransducer:
            def __init__(self):
                self.positions = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
        
        transducer = MockTransducer()
        hologram = AcousticHologram(
            transducer=transducer,
            frequency=40000,
            medium="air"
        )
        
        print("✅ Hologram created, now testing focus creation...")
        
        # Try to create a focus point
        target = hologram.create_focus_point(
            position=[0, 0, 0.1],
            pressure=3000
        )
        
        print("✅ Focus point created successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_focus()