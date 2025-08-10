"""
Mock backend for Acousto-Gen when dependencies are not available.
Provides graceful degradation and demonstration capabilities.
"""

import sys
import warnings
from typing import Any, List, Tuple, Dict, Optional, Union
import math
import random


class MockArray:
    """Mock numpy array for demonstration purposes."""
    
    def __init__(self, data, dtype=None):
        if isinstance(data, (list, tuple)):
            self.data = list(data)
        elif isinstance(data, (int, float)):
            self.data = [data]
        else:
            self.data = [0]
        
        self.dtype = dtype
        self.shape = (len(self.data),) if isinstance(self.data, list) else (1,)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value
    
    def __add__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a + b for a, b in zip(self.data, other.data)])
        return MockArray([x + other for x in self.data])
    
    def __mul__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a * b for a, b in zip(self.data, other.data)])
        return MockArray([x * other for x in self.data])
    
    def __sub__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a - b for a, b in zip(self.data, other.data)])
        return MockArray([x - other for x in self.data])
    
    def copy(self):
        return MockArray(self.data.copy())
    
    def tolist(self):
        return self.data
    
    def astype(self, dtype):
        return MockArray(self.data, dtype=dtype)
    
    @property
    def size(self):
        return len(self.data)


class MockNumpy:
    """Mock numpy module for demonstration."""
    
    @staticmethod
    def array(data, dtype=None):
        return MockArray(data, dtype)
    
    @staticmethod
    def zeros(shape, dtype=None):
        if isinstance(shape, int):
            size = shape
        else:
            size = shape[0] if len(shape) == 1 else shape[0] * shape[1] * shape[2]
        return MockArray([0] * size, dtype)
    
    @staticmethod
    def ones(shape, dtype=None):
        if isinstance(shape, int):
            size = shape
        else:
            size = shape[0] if len(shape) == 1 else shape[0] * shape[1] * shape[2]
        return MockArray([1] * size, dtype)
    
    @staticmethod
    def random_uniform(low=0, high=1, size=10):
        return MockArray([random.uniform(low, high) for _ in range(size)])
    
    @staticmethod
    def linspace(start, stop, num):
        if num <= 1:
            return MockArray([start])
        step = (stop - start) / (num - 1)
        return MockArray([start + i * step for i in range(num)])
    
    @staticmethod
    def meshgrid(x, y, z, indexing='ij'):
        # Simplified meshgrid
        return MockArray([0] * 10), MockArray([0] * 10), MockArray([0] * 10)
    
    @staticmethod
    def exp(x):
        if hasattr(x, 'data'):
            return MockArray([math.exp(val) for val in x.data])
        return math.exp(x)
    
    @staticmethod
    def sin(x):
        if hasattr(x, 'data'):
            return MockArray([math.sin(val) for val in x.data])
        return math.sin(x)
    
    @staticmethod
    def cos(x):
        if hasattr(x, 'data'):
            return MockArray([math.cos(val) for val in x.data])
        return math.cos(x)
    
    @staticmethod
    def sqrt(x):
        if hasattr(x, 'data'):
            return MockArray([math.sqrt(val) for val in x.data])
        return math.sqrt(x)
    
    @staticmethod
    def abs(x):
        if hasattr(x, 'data'):
            return MockArray([abs(val) for val in x.data])
        return abs(x)
    
    @staticmethod
    def angle(x):
        if hasattr(x, 'data'):
            return MockArray([0] * len(x.data))  # Simplified
        return 0
    
    @staticmethod
    def linalg_norm(x):
        if hasattr(x, 'data'):
            return math.sqrt(sum(val**2 for val in x.data))
        return abs(x)
    
    @staticmethod
    def maximum(a, b):
        if hasattr(a, 'data') and hasattr(b, 'data'):
            return MockArray([max(x, y) for x, y in zip(a.data, b.data)])
        return max(a, b)
    
    @staticmethod
    def clip(a, min_val, max_val):
        if hasattr(a, 'data'):
            return MockArray([max(min_val, min(max_val, x)) for x in a.data])
        return max(min_val, min(max_val, a))
    
    @staticmethod
    def mean(a):
        if hasattr(a, 'data'):
            return sum(a.data) / len(a.data) if a.data else 0
        return a
    
    @staticmethod
    def max(a):
        if hasattr(a, 'data'):
            return max(a.data) if a.data else 0
        return a
    
    @staticmethod
    def min(a):
        if hasattr(a, 'data'):
            return min(a.data) if a.data else 0
        return a
    
    @staticmethod
    def sum(a):
        if hasattr(a, 'data'):
            return sum(a.data)
        return a
    
    @staticmethod
    def argmin(a):
        if hasattr(a, 'data'):
            return a.data.index(min(a.data)) if a.data else 0
        return 0
    
    @staticmethod
    def argmax(a):
        if hasattr(a, 'data'):
            return a.data.index(max(a.data)) if a.data else 0
        return 0
    
    @staticmethod
    def real(x):
        return x  # Simplified for demo
    
    @staticmethod
    def imag(x):
        return MockArray([0] * (len(x.data) if hasattr(x, 'data') else 1))
    
    @staticmethod
    def remainder(x, divisor):
        if hasattr(x, 'data'):
            return MockArray([val % divisor for val in x.data])
        return x % divisor
    
    @staticmethod
    def where(condition):
        # Simplified implementation
        return ([0, 1, 2],)
    
    @staticmethod
    def iscomplexobj(x):
        return False
    
    # Mathematical constants
    pi = math.pi
    
    # Types for type hints
    ndarray = MockArray
    complex64 = 'complex64'
    float32 = 'float32'
    
    # Linalg submodule
    class linalg:
        @staticmethod
        def norm(x):
            return MockNumpy.linalg_norm(x)


class MockTorch:
    """Mock torch module for demonstration."""
    
    @staticmethod
    def tensor(data, dtype=None, device='cpu'):
        return MockTensor(data, dtype, device)
    
    @staticmethod
    def zeros(size, dtype=None, device='cpu'):
        if isinstance(size, (list, tuple)):
            # Calculate total size from shape
            total_size = 1
            for dim in size:
                total_size *= dim
            size = total_size
        return MockTensor([0] * size, dtype, device)
    
    @staticmethod
    def ones(size, dtype=None, device='cpu'):
        if isinstance(size, (list, tuple)):
            # Calculate total size from shape
            total_size = 1
            for dim in size:
                total_size *= dim
            size = total_size
        return MockTensor([1] * size, dtype, device)
    
    @staticmethod
    def randn(size, requires_grad=False, device='cpu'):
        if isinstance(size, (list, tuple)):
            # Calculate total size from shape
            total_size = 1
            for dim in size:
                total_size *= dim
            size = total_size
        data = [random.gauss(0, 1) for _ in range(size)]
        tensor = MockTensor(data, device=device)
        tensor.requires_grad = requires_grad
        return tensor
    
    @staticmethod
    def device(name):
        return f"device({name})"
    
    @staticmethod
    def cuda_is_available():
        return False
    
    # CUDA module mock
    class cuda:
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def device_count():
            return 0
        
        @staticmethod
        def empty_cache():
            pass
    
    @staticmethod
    def diff(input_tensor, n=1, dim=-1):
        # Simplified diff implementation
        return input_tensor
    
    @staticmethod
    def mean(input_tensor, dim=None):
        if hasattr(input_tensor, 'data'):
            return MockTensor([sum(input_tensor.data) / len(input_tensor.data)])
        return input_tensor
    
    @staticmethod
    def sum(input_tensor, dim=None):
        if hasattr(input_tensor, 'data'):
            return MockTensor([sum(input_tensor.data)])
        return input_tensor
    
    @staticmethod
    def abs(input_tensor):
        if hasattr(input_tensor, 'data'):
            return MockTensor([abs(x) for x in input_tensor.data])
        return abs(input_tensor)
    
    @staticmethod
    def sqrt(input_tensor):
        if hasattr(input_tensor, 'data'):
            return MockTensor([math.sqrt(abs(x)) for x in input_tensor.data])
        return math.sqrt(abs(input_tensor))
    
    @staticmethod
    def exp(input_tensor):
        if hasattr(input_tensor, 'data'):
            return MockTensor([math.exp(x) for x in input_tensor.data])
        return math.exp(input_tensor)
    
    @staticmethod
    def sin(input_tensor):
        if hasattr(input_tensor, 'data'):
            return MockTensor([math.sin(x) for x in input_tensor.data])
        return math.sin(input_tensor)
    
    @staticmethod
    def cos(input_tensor):
        if hasattr(input_tensor, 'data'):
            return MockTensor([math.cos(x) for x in input_tensor.data])
        return math.cos(input_tensor)
    
    @staticmethod
    def maximum(a, b):
        if hasattr(a, 'data') and hasattr(b, 'data'):
            return MockTensor([max(x, y) for x, y in zip(a.data, b.data)])
        elif hasattr(a, 'data'):
            return MockTensor([max(x, b) for x in a.data])
        elif hasattr(b, 'data'):
            return MockTensor([max(a, x) for x in b.data])
        return max(a, b)
    
    # Dtypes
    float32 = 'float32'
    complex64 = 'complex64'
    
    # Add dtype alias for compatibility
    dtype = float32
    
    # Type alias for compatibility
    Tensor = 'MockTensor'
    
    # Neural network module
    class nn:
        class Module:
            def __init__(self):
                pass
            
            def to(self, device):
                return self
            
            def train(self):
                pass
            
            def eval(self):
                pass
            
            def parameters(self):
                return []
        
        class functional:
            @staticmethod
            def pad(input_tensor, pad_config, mode='replicate'):
                return input_tensor
            
            @staticmethod
            def mse_loss(pred, target):
                return MockTensor([0.5])
        
        class Parameter:
            def __init__(self, data):
                self.data = data
                self.requires_grad = True
            
            def detach(self):
                return self
            
            def cpu(self):
                return self
            
            def numpy(self):
                return self.data
        
        @staticmethod
        def Linear(in_features, out_features):
            return MockModule()
        
        @staticmethod
        def ReLU():
            return MockModule()
        
        @staticmethod
        def MSELoss():
            return MockLoss()
        
        @staticmethod
        def BatchNorm1d(num_features):
            return MockModule()
        
        @staticmethod
        def Dropout(p):
            return MockModule()
        
        @staticmethod
        def Tanh():
            return MockModule()
        
        @staticmethod
        def LeakyReLU(negative_slope):
            return MockModule()
        
        @staticmethod
        def ELU():
            return MockModule()
        
        @staticmethod
        def Sequential(*modules):
            return MockModule()
    
    # Optimizer module  
    class optim:
        class Adam:
            def __init__(self, params, lr=0.001):
                self.params = params
                self.lr = lr
            
            def zero_grad(self):
                pass
            
            def step(self, closure=None):
                if closure:
                    return closure()
                return 0
        
        class SGD:
            def __init__(self, params, lr=0.01, momentum=0):
                self.params = params
                self.lr = lr
                self.momentum = momentum
            
            def zero_grad(self):
                pass
            
            def step(self):
                pass
        
        class LBFGS:
            def __init__(self, params, lr=1):
                self.params = params
                self.lr = lr
            
            def zero_grad(self):
                pass
            
            def step(self, closure):
                return closure()
        
        class lr_scheduler:
            class ReduceLROnPlateau:
                def __init__(self, optimizer, patience=10, factor=0.5):
                    pass
                
                def step(self, metrics):
                    pass
    
    # FFT module
    class fft:
        @staticmethod
        def fft2(input_tensor):
            return input_tensor
        
        @staticmethod
        def ifft2(input_tensor):
            return input_tensor
        
        @staticmethod
        def fftfreq(n, d=1.0):
            return MockArray(list(range(n)))
    
    # Utils module
    class utils:
        class data:
            class Dataset:
                """Base dataset class for mock PyTorch."""
                def __init__(self):
                    pass
                
                def __len__(self):
                    return 0
                
                def __getitem__(self, idx):
                    return None
            
            class TensorDataset(Dataset):
                def __init__(self, *datasets):
                    super().__init__()
                    self.datasets = datasets
                
                def __len__(self):
                    return len(self.datasets[0].data) if self.datasets else 0
                
                def __getitem__(self, idx):
                    return tuple(d.data[idx] for d in self.datasets)
            
            class DataLoader:
                def __init__(self, dataset, batch_size=1, shuffle=False):
                    self.dataset = dataset
                    self.batch_size = batch_size
                
                def __iter__(self):
                    for i in range(0, len(self.dataset), self.batch_size):
                        yield self.dataset[i:i+self.batch_size]


class MockTensor:
    """Mock torch tensor."""
    
    def __init__(self, data, dtype=None, device='cpu'):
        self.data = MockArray(data)
        self.dtype = dtype
        self.device = device
        self.requires_grad = False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def detach(self):
        return self
    
    def cpu(self):
        return self
    
    def numpy(self):
        return self.data
    
    def item(self):
        return self.data[0] if self.data else 0
    
    def backward(self):
        pass
    
    def squeeze(self):
        return self
    
    def unsqueeze(self, dim):
        return self
    
    def view(self, *shape):
        return self
    
    @property
    def shape(self):
        return self.data.shape


class MockModule:
    """Mock neural network module."""
    
    def __call__(self, x):
        return x
    
    def to(self, device):
        return self
    
    def parameters(self):
        return []


class MockLoss:
    """Mock loss function."""
    
    def __call__(self, pred, target):
        return MockTensor([0.5])  # Dummy loss


def setup_mock_dependencies():
    """Setup mock dependencies when real ones aren't available."""
    
    # Mock numpy
    if 'numpy' not in sys.modules:
        sys.modules['numpy'] = MockNumpy()
        sys.modules['np'] = MockNumpy()
    
    # Mock torch
    if 'torch' not in sys.modules:
        mock_torch = MockTorch()
        sys.modules['torch'] = mock_torch
        sys.modules['torch.nn'] = mock_torch.nn
        sys.modules['torch.nn.functional'] = mock_torch.nn.functional
        sys.modules['torch.optim'] = mock_torch.optim
        sys.modules['torch.fft'] = mock_torch.fft
        sys.modules['torch.utils'] = mock_torch.utils
        sys.modules['torch.utils.data'] = mock_torch.utils.data
    
    # Mock scipy (minimal)
    if 'scipy' not in sys.modules:
        class MockScipy:
            class interpolate:
                class RegularGridInterpolator:
                    def __init__(self, *args, **kwargs):
                        pass
                    
                    def __call__(self, points):
                        return MockArray([0] * len(points))
            
            class ndimage:
                @staticmethod
                def maximum_filter(arr, size):
                    return arr
            
            class spatial:
                class distance:
                    @staticmethod
                    def cdist(a, b):
                        return MockArray([[0]])
            
            class optimize:
                @staticmethod
                def minimize(*args, **kwargs):
                    class Result:
                        x = MockArray([0])
                        success = True
                    return Result()
        
        sys.modules['scipy'] = MockScipy()
        sys.modules['scipy.interpolate'] = MockScipy.interpolate
        sys.modules['scipy.ndimage'] = MockScipy.ndimage
        sys.modules['scipy.spatial'] = MockScipy.spatial
        sys.modules['scipy.optimize'] = MockScipy.optimize
    
    # Issue warning
    warnings.warn(
        "🚧 Running in DEMO MODE - Key dependencies (NumPy, PyTorch, SciPy) not found. "
        "Mock implementations are being used for demonstration purposes only. "
        "Install dependencies with: pip install numpy torch scipy",
        UserWarning
    )
    
    return True


def check_and_setup():
    """Check dependencies and setup mocks if needed."""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    
    try:
        import torch
    except ImportError:
        missing.append('torch')
    
    try:
        import scipy
    except ImportError:
        missing.append('scipy')
    
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print("🔧 Setting up mock backend for demonstration...")
        setup_mock_dependencies()
        return False  # Running in mock mode
    else:
        print("✅ All dependencies available")
        return True  # Running with real dependencies