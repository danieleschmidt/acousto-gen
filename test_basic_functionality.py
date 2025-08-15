#!/usr/bin/env python3
"""
Basic functionality test for Acousto-Gen Generation 1.
Tests core system components without full dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Force mock mode to avoid PyTorch dependency issues
os.environ['ACOUSTO_GEN_FORCE_MOCK'] = '1'

def test_basic_imports():
    """Test basic package imports."""
    try:
        import acousto_gen
        print("✅ Package import successful")
        print(f"   Version: {acousto_gen.__version__}")
        
        # Test system check
        checks = acousto_gen.system_check()
        print(f"✅ System check completed: {sum(checks.values())}/{len(checks)} checks passed")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_transducer_creation():
    """Test basic transducer array creation."""
    try:
        # Test with mock backend if real dependencies unavailable
        import acousto_gen
        from acousto_gen.core import AcousticHologram
        
        # Create a simple transducer configuration
        import numpy as np
        
        # Create a mock transducer array structure
        class MockTransducer:
            def __init__(self):
                self.positions = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
        
        transducer = MockTransducer()
        
        # Test hologram creation
        hologram = AcousticHologram(
            transducer=transducer,
            frequency=40000,
            medium="air"
        )
        
        print("✅ Basic transducer and hologram creation successful")
        print(f"   Frequency: {hologram.frequency} Hz")
        print(f"   Medium: {hologram.medium_name}")
        
        return True
    except Exception as e:
        print(f"❌ Transducer creation failed: {e}")
        return False

def test_focus_point_creation():
    """Test target focus point creation."""
    try:
        import acousto_gen
        from acousto_gen.core import AcousticHologram
        import numpy as np
        
        class MockTransducer:
            def __init__(self):
                self.positions = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
        
        transducer = MockTransducer()
        hologram = AcousticHologram(
            transducer=transducer,
            frequency=40000,
            medium="air"
        )
        
        # Create a focus point
        target = hologram.create_focus_point(
            position=[0, 0, 0.1],
            pressure=3000
        )
        
        print("✅ Focus point creation successful")
        print(f"   Target shape: {target.data.shape}")
        print(f"   Max pressure: {np.max(np.abs(target.data)):.1f}")
        
        return True
    except Exception as e:
        print(f"❌ Focus point creation failed: {e}")
        return False

def test_basic_optimization():
    """Test basic optimization functionality."""
    try:
        import acousto_gen
        from acousto_gen.core import AcousticHologram
        import numpy as np
        
        class MockTransducer:
            def __init__(self):
                self.positions = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
        
        transducer = MockTransducer()
        hologram = AcousticHologram(
            transducer=transducer,
            frequency=40000,
            medium="air"
        )
        
        # Create target and optimize (minimal iterations for speed)
        target = hologram.create_focus_point(
            position=[0, 0, 0.1],
            pressure=3000
        )
        
        result = hologram.optimize(
            target=target,
            iterations=2,  # Minimal iterations for speed
            method="adam"
        )
        
        print("✅ Basic optimization successful")
        print(f"   Final loss: {result['final_loss']:.6f}")
        print(f"   Phases shape: {result['phases'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False

def main():
    """Run basic functionality tests."""
    print("🧪 Testing Acousto-Gen Basic Functionality (Generation 1)")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Transducer Creation", test_transducer_creation), 
        ("Focus Point Creation", test_focus_point_creation),
        ("Basic Optimization", test_basic_optimization)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n🔍 Testing {name}...")
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All basic functionality tests PASSED! Generation 1 complete.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)