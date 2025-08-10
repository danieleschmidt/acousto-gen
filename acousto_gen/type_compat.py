"""
Type compatibility module for Acousto-Gen.
Provides type aliases that work with both real and mock backends.
"""

from typing import Union, Any

# Import check for mock mode
try:
    from .mock_backend import check_and_setup, MockTensor, MockArray
    MOCK_MODE = not check_and_setup()
except ImportError:
    MOCK_MODE = False
    MockTensor = Any
    MockArray = Any

# Import with fallback
if not MOCK_MODE:
    try:
        import torch
        import numpy as np
        TensorType = torch.Tensor
        ArrayType = np.ndarray
    except ImportError:
        # Fallback to mock
        from .mock_backend import setup_mock_dependencies
        setup_mock_dependencies()
        import torch
        import numpy as np
        TensorType = Union[torch.Tensor, MockTensor, Any]
        ArrayType = Union[np.ndarray, MockArray, Any]
else:
    # In mock mode
    import torch
    import numpy as np
    TensorType = Union[torch.Tensor, MockTensor, Any]
    ArrayType = Union[np.ndarray, MockArray, Any]

# Generic type aliases that work in both modes
Tensor = TensorType
Array = ArrayType

# Compatibility aliases for backward compatibility
NumpyArray = ArrayType
TorchTensor = TensorType

# Export the types
__all__ = [
    'Tensor',
    'Array',
    'NumpyArray', 
    'TorchTensor',
    'TensorType', 
    'ArrayType',
    'MOCK_MODE'
]