#!/usr/bin/env python3
"""Debug optimization issue."""

import sys
import os
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ['ACOUSTO_GEN_FORCE_MOCK'] = '1'

def debug_optimization():
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
        
        print(f"Transducer positions type: {type(transducer.positions)}")
        print(f"Transducer positions shape: {getattr(transducer.positions, 'shape', 'no shape')}")
        print(f"Length of positions: {len(transducer.positions)}")
        print(f"First position: {transducer.positions[0]}")
        print(f"First position type: {type(transducer.positions[0])}")
        if hasattr(transducer.positions[0], '__getitem__'):
            print(f"First position [0]: {transducer.positions[0][0]}")
        
        # Test the positions iteration
        print("Testing positions iteration:")
        for i in range(len(transducer.positions)):
            pos = transducer.positions[i]
            print(f"  Position {i}: type={type(pos)}, data={pos}")
            if hasattr(pos, '__getitem__'):
                try:
                    print(f"    pos[0] = {pos[0]}")
                except Exception as e:
                    print(f"    Error accessing pos[0]: {e}")
        
        # Create target and optimize (minimal iterations for speed)
        target = hologram.create_focus_point(
            position=[0, 0, 0.1],
            pressure=3000
        )
        
        print("✅ Target created, starting optimization...")
        
        result = hologram.optimize(
            target=target,
            iterations=2,  # Minimal iterations for debugging
            method="adam"
        )
        
        print("✅ Optimization successful")
        print(f"   Final loss: {result['final_loss']:.6f}")
        print(f"   Phases shape: {result['phases'].shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_optimization()