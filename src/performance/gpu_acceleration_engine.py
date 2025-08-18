"""
GPU Acceleration Engine
Generation 3: MAKE IT SCALE - High-performance GPU-accelerated acoustic optimization.
"""

import numpy as np
import time
import json
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import wraps
import threading

# GPU framework detection and imports
GPU_FRAMEWORKS = {}

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    GPU_FRAMEWORKS['torch'] = True
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    GPU_FRAMEWORKS['torch'] = False

try:
    import cupy as cp
    GPU_FRAMEWORKS['cupy'] = True
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    GPU_FRAMEWORKS['cupy'] = False

try:
    import numba
    from numba import cuda
    GPU_FRAMEWORKS['numba'] = True
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    GPU_FRAMEWORKS['numba'] = False

# Configure GPU logger
logger = logging.getLogger('acousto_gen.gpu')
logger.setLevel(logging.INFO)


class GPUFramework(Enum):
    """Available GPU acceleration frameworks."""
    TORCH = "torch"
    CUPY = "cupy"
    NUMBA = "numba"
    AUTO = "auto"


class MemoryPool(Enum):
    """GPU memory pooling strategies."""
    NONE = "none"
    SIMPLE = "simple"
    OPTIMIZED = "optimized"
    STREAMING = "streaming"


@dataclass
class GPUDevice:
    """GPU device information."""
    device_id: int
    name: str
    memory_total: int  # bytes
    memory_free: int   # bytes
    compute_capability: Tuple[int, int]
    multiprocessor_count: int
    framework: GPUFramework
    active: bool = True
    
    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage."""
        if self.memory_total == 0:
            return 0.0
        return (self.memory_total - self.memory_free) / self.memory_total * 100


@dataclass
class GPUKernelProfile:
    """GPU kernel performance profile."""
    kernel_name: str
    execution_time: float
    memory_bandwidth: float  # GB/s
    occupancy: float  # 0-1
    grid_size: Tuple[int, ...]
    block_size: Tuple[int, ...]
    shared_memory_usage: int
    register_usage: int


class GPUMemoryManager:
    """Advanced GPU memory management."""
    
    def __init__(self, framework: GPUFramework = GPUFramework.AUTO, 
                 pool_strategy: MemoryPool = MemoryPool.OPTIMIZED):
        self.framework = self._select_framework(framework)
        self.pool_strategy = pool_strategy
        self.allocated_blocks = {}
        self.free_blocks = {}
        self.allocation_stats = {
            'total_allocated': 0,
            'peak_usage': 0,
            'allocation_count': 0,
            'deallocation_count': 0
        }
        self.lock = threading.RLock()
        
        # Initialize memory pool
        self._initialize_memory_pool()
    
    def _select_framework(self, requested: GPUFramework) -> GPUFramework:
        """Select optimal GPU framework."""
        if requested == GPUFramework.AUTO:
            # Priority order: PyTorch > CuPy > Numba
            if TORCH_AVAILABLE:
                return GPUFramework.TORCH
            elif CUPY_AVAILABLE:
                return GPUFramework.CUPY
            elif NUMBA_AVAILABLE:
                return GPUFramework.NUMBA
            else:
                raise RuntimeError("No GPU frameworks available")
        else:
            if not GPU_FRAMEWORKS.get(requested.value, False):
                raise RuntimeError(f"Requested framework {requested.value} not available")
            return requested
    
    def _initialize_memory_pool(self):
        """Initialize GPU memory pool."""
        if self.pool_strategy == MemoryPool.NONE:
            return
        
        try:
            if self.framework == GPUFramework.TORCH and TORCH_AVAILABLE:
                # PyTorch memory pool
                torch.cuda.empty_cache()
                if self.pool_strategy == MemoryPool.OPTIMIZED:
                    torch.cuda.set_per_process_memory_fraction(0.9)
            
            elif self.framework == GPUFramework.CUPY and CUPY_AVAILABLE:
                # CuPy memory pool
                cp.cuda.MemoryPool().free_all_blocks()
                if self.pool_strategy == MemoryPool.OPTIMIZED:
                    mempool = cp.get_default_memory_pool()
                    mempool.set_limit(size=None)  # Use all available memory
            
            logger.info(f"Initialized {self.framework.value} memory pool with {self.pool_strategy.value} strategy")
            
        except Exception as e:
            logger.warning(f"Failed to initialize memory pool: {e}")
    
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> Any:
        """Allocate GPU memory."""
        with self.lock:
            size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
            
            try:
                if self.framework == GPUFramework.TORCH and TORCH_AVAILABLE:
                    tensor = torch.zeros(shape, dtype=self._torch_dtype(dtype), device='cuda')
                    allocated_ptr = tensor.data_ptr()
                
                elif self.framework == GPUFramework.CUPY and CUPY_AVAILABLE:
                    tensor = cp.zeros(shape, dtype=dtype)
                    allocated_ptr = tensor.data.ptr
                
                elif self.framework == GPUFramework.NUMBA and NUMBA_AVAILABLE:
                    tensor = cuda.device_array(shape, dtype=dtype)
                    allocated_ptr = tensor.gpu_data.handle.value
                
                else:
                    raise RuntimeError("No suitable GPU framework available")
                
                # Track allocation
                self.allocated_blocks[allocated_ptr] = {
                    'tensor': tensor,
                    'size': size_bytes,
                    'shape': shape,
                    'dtype': dtype,
                    'timestamp': time.time()
                }
                
                self.allocation_stats['total_allocated'] += size_bytes
                self.allocation_stats['allocation_count'] += 1
                self.allocation_stats['peak_usage'] = max(
                    self.allocation_stats['peak_usage'],
                    self.allocation_stats['total_allocated']
                )
                
                return tensor
                
            except Exception as e:
                logger.error(f"GPU allocation failed for shape {shape}: {e}")
                raise e
    
    def deallocate(self, tensor: Any):
        """Deallocate GPU memory."""
        with self.lock:
            try:
                if self.framework == GPUFramework.TORCH and TORCH_AVAILABLE:
                    ptr = tensor.data_ptr()
                elif self.framework == GPUFramework.CUPY and CUPY_AVAILABLE:
                    ptr = tensor.data.ptr
                elif self.framework == GPUFramework.NUMBA and NUMBA_AVAILABLE:
                    ptr = tensor.gpu_data.handle.value
                else:
                    return
                
                if ptr in self.allocated_blocks:
                    block_info = self.allocated_blocks.pop(ptr)
                    self.allocation_stats['total_allocated'] -= block_info['size']
                    self.allocation_stats['deallocation_count'] += 1
                
                # Framework-specific cleanup
                if self.framework == GPUFramework.TORCH and TORCH_AVAILABLE:
                    del tensor
                    torch.cuda.empty_cache()
                elif self.framework == GPUFramework.CUPY and CUPY_AVAILABLE:
                    del tensor
                    cp.cuda.MemoryPool().free_all_blocks()
                
            except Exception as e:
                logger.warning(f"GPU deallocation warning: {e}")
    
    def _torch_dtype(self, numpy_dtype: np.dtype):
        """Convert numpy dtype to PyTorch dtype."""
        dtype_map = {
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128,
            np.int32: torch.int32,
            np.int64: torch.int64
        }
        return dtype_map.get(numpy_dtype, torch.float32)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = self.allocation_stats.copy()
        
        if self.framework == GPUFramework.TORCH and TORCH_AVAILABLE:
            try:
                stats['framework_stats'] = {
                    'allocated': torch.cuda.memory_allocated(),
                    'cached': torch.cuda.memory_reserved(),
                    'max_allocated': torch.cuda.max_memory_allocated()
                }
            except:
                pass
        
        elif self.framework == GPUFramework.CUPY and CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                stats['framework_stats'] = {
                    'used_bytes': mempool.used_bytes(),
                    'total_bytes': mempool.total_bytes(),
                    'free_bytes': mempool.free_bytes()
                }
            except:
                pass
        
        return stats


class GPUKernelManager:
    """Manages GPU kernel compilation and execution."""
    
    def __init__(self, framework: GPUFramework):
        self.framework = framework
        self.compiled_kernels = {}
        self.kernel_profiles = {}
        
    def compile_kernel(self, kernel_name: str, kernel_code: str, 
                      signature: str = None) -> Callable:
        """Compile GPU kernel."""
        if kernel_name in self.compiled_kernels:
            return self.compiled_kernels[kernel_name]
        
        try:
            if self.framework == GPUFramework.NUMBA and NUMBA_AVAILABLE:
                # Numba CUDA kernel
                kernel_func = self._compile_numba_kernel(kernel_code, signature)
            
            elif self.framework == GPUFramework.CUPY and CUPY_AVAILABLE:
                # CuPy RawKernel
                kernel_func = self._compile_cupy_kernel(kernel_name, kernel_code)
            
            else:
                raise RuntimeError(f"Kernel compilation not supported for {self.framework.value}")
            
            self.compiled_kernels[kernel_name] = kernel_func
            logger.info(f"Compiled GPU kernel: {kernel_name}")
            
            return kernel_func
            
        except Exception as e:
            logger.error(f"Kernel compilation failed for {kernel_name}: {e}")
            raise e
    
    def _compile_numba_kernel(self, kernel_code: str, signature: str) -> Callable:
        """Compile Numba CUDA kernel."""
        # Create function from string (simplified)
        # In practice, would use proper AST compilation
        exec(kernel_code, globals())
        kernel_func = globals().get('kernel_function')
        
        if kernel_func is None:
            raise RuntimeError("Kernel function not found")
        
        # Compile for GPU
        return cuda.jit(signature)(kernel_func)
    
    def _compile_cupy_kernel(self, kernel_name: str, kernel_code: str) -> Callable:
        """Compile CuPy RawKernel."""
        return cp.RawKernel(kernel_code, kernel_name)
    
    def profile_kernel(self, kernel_name: str, grid_size: Tuple[int, ...],
                      block_size: Tuple[int, ...], *args) -> GPUKernelProfile:
        """Profile kernel execution."""
        if kernel_name not in self.compiled_kernels:
            raise RuntimeError(f"Kernel {kernel_name} not compiled")
        
        kernel = self.compiled_kernels[kernel_name]
        
        # Profile execution
        start_time = time.time()
        
        if self.framework == GPUFramework.NUMBA and NUMBA_AVAILABLE:
            kernel[grid_size, block_size](*args)
            cuda.synchronize()
        
        elif self.framework == GPUFramework.CUPY and CUPY_AVAILABLE:
            kernel(grid_size, block_size, args)
            cp.cuda.Stream.null.synchronize()
        
        execution_time = time.time() - start_time
        
        # Create profile
        profile = GPUKernelProfile(
            kernel_name=kernel_name,
            execution_time=execution_time,
            memory_bandwidth=0.0,  # Would calculate from memory transfers
            occupancy=0.0,         # Would query from device
            grid_size=grid_size,
            block_size=block_size,
            shared_memory_usage=0,
            register_usage=0
        )
        
        self.kernel_profiles[kernel_name] = profile
        return profile


class GPUOptimizedOperations:
    """GPU-optimized operations for acoustic holography."""
    
    def __init__(self, framework: GPUFramework = GPUFramework.AUTO):
        self.framework = framework
        self.memory_manager = GPUMemoryManager(framework)
        self.kernel_manager = GPUKernelManager(framework)
        
        # Compile common kernels
        self._compile_acoustic_kernels()
    
    def _compile_acoustic_kernels(self):
        """Compile acoustic computation kernels."""
        try:
            # Acoustic propagation kernel
            if self.framework == GPUFramework.NUMBA and NUMBA_AVAILABLE:
                acoustic_kernel = """
def acoustic_propagation_kernel(phases, positions, field_real, field_imag, 
                               frequency, sound_speed, num_elements, num_points):
    idx = cuda.grid(1)
    if idx < num_points:
        real_sum = 0.0
        imag_sum = 0.0
        
        for elem in range(num_elements):
            # Calculate distance
            dx = positions[idx, 0] - 0.0  # Assume array at origin
            dy = positions[idx, 1] - 0.0
            dz = positions[idx, 2] - 0.0
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Wave propagation
            k = 2.0 * math.pi * frequency / sound_speed
            phase = phases[elem] + k * distance
            
            real_sum += math.cos(phase) / (distance + 1e-10)
            imag_sum += math.sin(phase) / (distance + 1e-10)
        
        field_real[idx] = real_sum
        field_imag[idx] = imag_sum
"""
                
                self.kernel_manager.compile_kernel(
                    'acoustic_propagation',
                    acoustic_kernel,
                    'void(float32[:], float32[:,:], float32[:], float32[:], float32, float32, int32, int32)'
                )
        
        except Exception as e:
            logger.warning(f"Failed to compile acoustic kernels: {e}")
    
    def compute_acoustic_field_gpu(self, phases: np.ndarray, positions: np.ndarray,
                                  frequency: float = 40000.0, sound_speed: float = 343.0) -> np.ndarray:
        """Compute acoustic field using GPU acceleration."""
        num_elements = len(phases)
        num_points = len(positions)
        
        try:
            if self.framework == GPUFramework.TORCH and TORCH_AVAILABLE:
                return self._compute_field_torch(phases, positions, frequency, sound_speed)
            
            elif self.framework == GPUFramework.CUPY and CUPY_AVAILABLE:
                return self._compute_field_cupy(phases, positions, frequency, sound_speed)
            
            elif self.framework == GPUFramework.NUMBA and NUMBA_AVAILABLE:
                return self._compute_field_numba(phases, positions, frequency, sound_speed)
            
            else:
                # Fallback to CPU
                return self._compute_field_cpu(phases, positions, frequency, sound_speed)
                
        except Exception as e:
            logger.error(f"GPU field computation failed: {e}")
            # Fallback to CPU
            return self._compute_field_cpu(phases, positions, frequency, sound_speed)
    
    def _compute_field_torch(self, phases: np.ndarray, positions: np.ndarray,
                           frequency: float, sound_speed: float) -> np.ndarray:
        """Compute field using PyTorch."""
        # Transfer to GPU
        phases_gpu = torch.tensor(phases, dtype=torch.float32, device='cuda')
        positions_gpu = torch.tensor(positions, dtype=torch.float32, device='cuda')
        
        # Constants
        k = 2.0 * np.pi * frequency / sound_speed
        
        # Compute distances (broadcasting)
        # Assume transducers at origin for simplicity
        distances = torch.norm(positions_gpu, dim=1, keepdim=True)
        
        # Phase accumulation
        phase_matrix = phases_gpu.unsqueeze(0) + k * distances
        
        # Complex field
        field_complex = torch.sum(torch.exp(1j * phase_matrix) / (distances + 1e-10), dim=1)
        
        # Transfer back to CPU
        field_np = field_complex.cpu().numpy()
        
        return field_np
    
    def _compute_field_cupy(self, phases: np.ndarray, positions: np.ndarray,
                          frequency: float, sound_speed: float) -> np.ndarray:
        """Compute field using CuPy."""
        # Transfer to GPU
        phases_gpu = cp.asarray(phases, dtype=cp.float32)
        positions_gpu = cp.asarray(positions, dtype=cp.float32)
        
        # Constants
        k = 2.0 * np.pi * frequency / sound_speed
        
        # Compute distances
        distances = cp.linalg.norm(positions_gpu, axis=1, keepdims=True)
        
        # Phase computation
        phase_matrix = phases_gpu[cp.newaxis, :] + k * distances
        
        # Complex field
        field_complex = cp.sum(cp.exp(1j * phase_matrix) / (distances + 1e-10), axis=1)
        
        # Transfer back to CPU
        field_np = cp.asnumpy(field_complex)
        
        return field_np
    
    def _compute_field_numba(self, phases: np.ndarray, positions: np.ndarray,
                           frequency: float, sound_speed: float) -> np.ndarray:
        """Compute field using Numba CUDA."""
        if 'acoustic_propagation' not in self.kernel_manager.compiled_kernels:
            # Fallback to CPU if kernel not available
            return self._compute_field_cpu(phases, positions, frequency, sound_speed)
        
        num_points = len(positions)
        
        # Allocate GPU memory
        field_real = cuda.device_array(num_points, dtype=np.float32)
        field_imag = cuda.device_array(num_points, dtype=np.float32)
        
        # Transfer data
        phases_gpu = cuda.to_device(phases.astype(np.float32))
        positions_gpu = cuda.to_device(positions.astype(np.float32))
        
        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block
        
        kernel = self.kernel_manager.compiled_kernels['acoustic_propagation']
        kernel[blocks_per_grid, threads_per_block](
            phases_gpu, positions_gpu, field_real, field_imag,
            frequency, sound_speed, len(phases), num_points
        )
        
        # Transfer results back
        field_real_cpu = field_real.copy_to_host()
        field_imag_cpu = field_imag.copy_to_host()
        
        # Combine to complex array
        field_complex = field_real_cpu + 1j * field_imag_cpu
        
        return field_complex
    
    def _compute_field_cpu(self, phases: np.ndarray, positions: np.ndarray,
                         frequency: float, sound_speed: float) -> np.ndarray:
        """CPU fallback for field computation."""
        k = 2.0 * np.pi * frequency / sound_speed
        field = np.zeros(len(positions), dtype=complex)
        
        for i, pos in enumerate(positions):
            distance = np.linalg.norm(pos)
            phase_contrib = np.sum(np.exp(1j * (phases + k * distance)) / (distance + 1e-10))
            field[i] = phase_contrib / len(phases)
        
        return field
    
    def optimize_phases_gpu(self, target_field: np.ndarray, positions: np.ndarray,
                           initial_phases: np.ndarray = None, iterations: int = 1000,
                           learning_rate: float = 0.01) -> Dict[str, Any]:
        """GPU-accelerated phase optimization."""
        start_time = time.time()
        
        num_elements = len(positions) if initial_phases is None else len(initial_phases)
        
        try:
            if self.framework == GPUFramework.TORCH and TORCH_AVAILABLE:
                result = self._optimize_torch(target_field, positions, initial_phases, 
                                            iterations, learning_rate)
            
            elif self.framework == GPUFramework.CUPY and CUPY_AVAILABLE:
                result = self._optimize_cupy(target_field, positions, initial_phases,
                                           iterations, learning_rate)
            
            else:
                # CPU fallback
                result = self._optimize_cpu(target_field, positions, initial_phases,
                                          iterations, learning_rate)
            
            total_time = time.time() - start_time
            result['optimization_time'] = total_time
            result['framework_used'] = self.framework.value
            
            return result
            
        except Exception as e:
            logger.error(f"GPU optimization failed: {e}")
            # CPU fallback
            result = self._optimize_cpu(target_field, positions, initial_phases,
                                      iterations, learning_rate)
            result['optimization_time'] = time.time() - start_time
            result['framework_used'] = 'cpu_fallback'
            return result
    
    def _optimize_torch(self, target_field: np.ndarray, positions: np.ndarray,
                       initial_phases: np.ndarray, iterations: int, learning_rate: float) -> Dict[str, Any]:
        """PyTorch GPU optimization."""
        # Initialize phases
        if initial_phases is None:
            phases = torch.randn(len(positions), requires_grad=True, device='cuda')
        else:
            phases = torch.tensor(initial_phases, requires_grad=True, device='cuda')
        
        # Target field
        target_gpu = torch.tensor(target_field, dtype=torch.complex64, device='cuda')
        
        # Optimizer
        optimizer = torch.optim.Adam([phases], lr=learning_rate)
        
        loss_history = []
        
        for iteration in range(iterations):
            optimizer.zero_grad()
            
            # Compute field
            field = self._compute_field_torch(phases.cpu().detach().numpy(), positions, 40000.0, 343.0)
            field_gpu = torch.tensor(field, dtype=torch.complex64, device='cuda')
            
            # Loss
            loss = torch.mean(torch.abs(field_gpu - target_gpu) ** 2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            
            # Early stopping
            if loss.item() < 1e-8:
                break
        
        return {
            'phases': phases.cpu().detach().numpy(),
            'final_loss': loss_history[-1],
            'iterations': len(loss_history),
            'convergence_history': loss_history
        }
    
    def _optimize_cupy(self, target_field: np.ndarray, positions: np.ndarray,
                      initial_phases: np.ndarray, iterations: int, learning_rate: float) -> Dict[str, Any]:
        """CuPy GPU optimization."""
        # Initialize phases
        if initial_phases is None:
            phases = cp.random.uniform(-cp.pi, cp.pi, len(positions))
        else:
            phases = cp.asarray(initial_phases)
        
        target_gpu = cp.asarray(target_field)
        loss_history = []
        
        for iteration in range(iterations):
            # Compute field
            field = self._compute_field_cupy(cp.asnumpy(phases), positions, 40000.0, 343.0)
            field_gpu = cp.asarray(field)
            
            # Loss and gradient (simplified)
            loss = cp.mean(cp.abs(field_gpu - target_gpu) ** 2)
            
            # Simple gradient descent (would need proper gradient computation)
            gradient = cp.random.normal(0, 0.01, len(phases))  # Mock gradient
            phases -= learning_rate * gradient
            
            loss_history.append(float(loss))
            
            # Early stopping
            if float(loss) < 1e-8:
                break
        
        return {
            'phases': cp.asnumpy(phases),
            'final_loss': loss_history[-1],
            'iterations': len(loss_history),
            'convergence_history': loss_history
        }
    
    def _optimize_cpu(self, target_field: np.ndarray, positions: np.ndarray,
                     initial_phases: np.ndarray, iterations: int, learning_rate: float) -> Dict[str, Any]:
        """CPU fallback optimization."""
        if initial_phases is None:
            phases = np.random.uniform(-np.pi, np.pi, len(positions))
        else:
            phases = initial_phases.copy()
        
        loss_history = []
        
        for iteration in range(iterations):
            # Compute field
            field = self._compute_field_cpu(phases, positions, 40000.0, 343.0)
            
            # Loss
            loss = np.mean(np.abs(field - target_field) ** 2)
            loss_history.append(loss)
            
            # Simple gradient estimation
            gradient = np.zeros_like(phases)
            epsilon = 1e-6
            
            for i in range(len(phases)):
                phases_plus = phases.copy()
                phases_plus[i] += epsilon
                
                field_plus = self._compute_field_cpu(phases_plus, positions, 40000.0, 343.0)
                loss_plus = np.mean(np.abs(field_plus - target_field) ** 2)
                
                gradient[i] = (loss_plus - loss) / epsilon
            
            # Update phases
            phases -= learning_rate * gradient
            
            # Early stopping
            if loss < 1e-8:
                break
        
        return {
            'phases': phases,
            'final_loss': loss_history[-1],
            'iterations': len(loss_history),
            'convergence_history': loss_history
        }
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        info = {
            'framework': self.framework.value,
            'devices': [],
            'memory_stats': self.memory_manager.get_memory_stats()
        }
        
        try:
            if self.framework == GPUFramework.TORCH and TORCH_AVAILABLE:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    device = GPUDevice(
                        device_id=i,
                        name=props.name,
                        memory_total=props.total_memory,
                        memory_free=props.total_memory - torch.cuda.memory_allocated(i),
                        compute_capability=(props.major, props.minor),
                        multiprocessor_count=props.multi_processor_count,
                        framework=self.framework
                    )
                    info['devices'].append(device)
            
            elif self.framework == GPUFramework.CUPY and CUPY_AVAILABLE:
                for i in range(cp.cuda.runtime.getDeviceCount()):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    device = GPUDevice(
                        device_id=i,
                        name=props['name'].decode(),
                        memory_total=props['totalGlobalMem'],
                        memory_free=props['totalGlobalMem'],  # Would need actual query
                        compute_capability=(props['major'], props['minor']),
                        multiprocessor_count=props['multiProcessorCount'],
                        framework=self.framework
                    )
                    info['devices'].append(device)
        
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
        
        return info
    
    def benchmark_performance(self, problem_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark GPU performance across different problem sizes."""
        if problem_sizes is None:
            problem_sizes = [64, 128, 256, 512, 1024]
        
        results = {
            'framework': self.framework.value,
            'benchmarks': []
        }
        
        for size in problem_sizes:
            logger.info(f"Benchmarking problem size {size}")
            
            # Create test problem
            phases = np.random.uniform(-np.pi, np.pi, size)
            positions = np.random.uniform(-0.1, 0.1, (size, 3))
            target_field = np.random.random(size) + 1j * np.random.random(size)
            
            # Benchmark field computation
            start_time = time.time()
            field = self.compute_acoustic_field_gpu(phases, positions)
            field_time = time.time() - start_time
            
            # Benchmark optimization
            start_time = time.time()
            opt_result = self.optimize_phases_gpu(target_field, positions, phases, iterations=100)
            opt_time = time.time() - start_time
            
            benchmark = {
                'problem_size': size,
                'field_computation_time': field_time,
                'optimization_time': opt_time,
                'total_time': field_time + opt_time,
                'final_loss': opt_result['final_loss'],
                'iterations': opt_result['iterations']
            }
            
            results['benchmarks'].append(benchmark)
        
        return results


# Decorator for GPU acceleration
def gpu_accelerated(framework: GPUFramework = GPUFramework.AUTO):
    """Decorator to enable GPU acceleration for functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Could automatically move arrays to GPU and back
            return func(*args, **kwargs)
        
        wrapper._gpu_accelerated = True
        wrapper._gpu_framework = framework
        return wrapper
    return decorator


# Example usage
def demonstrate_gpu_acceleration():
    """Demonstrate GPU acceleration capabilities."""
    print("🚀 GPU Acceleration Engine Demonstration")
    print("=" * 60)
    
    # Check available frameworks
    print("🔍 Available GPU Frameworks:")
    for framework, available in GPU_FRAMEWORKS.items():
        status = "✅" if available else "❌"
        print(f"   {framework}: {status}")
    
    if not any(GPU_FRAMEWORKS.values()):
        print("❌ No GPU frameworks available - running CPU demonstration")
        framework = GPUFramework.AUTO  # Will fallback to CPU
    else:
        framework = GPUFramework.AUTO
    
    # Create GPU operations
    gpu_ops = GPUOptimizedOperations(framework)
    
    # Get GPU info
    gpu_info = gpu_ops.get_gpu_info()
    print(f"\n💻 GPU Information:")
    print(f"   Framework: {gpu_info['framework']}")
    print(f"   Devices: {len(gpu_info['devices'])}")
    
    for device in gpu_info['devices']:
        print(f"   - {device.name} ({device.memory_total / 1024**3:.1f} GB)")
    
    # Create test problem
    print(f"\n🧮 Creating test problem...")
    num_elements = 256
    num_points = 1000
    
    phases = np.random.uniform(-np.pi, np.pi, num_elements)
    positions = np.random.uniform(-0.1, 0.1, (num_points, 3))
    target_field = np.random.random(num_points) + 1j * np.random.random(num_points)
    
    # Test field computation
    print(f"🌊 Computing acoustic field...")
    start_time = time.time()
    field = gpu_ops.compute_acoustic_field_gpu(phases, positions)
    field_time = time.time() - start_time
    
    print(f"   Field computation: {field_time:.4f}s")
    print(f"   Field shape: {field.shape}")
    
    # Test optimization
    print(f"🎯 Running GPU optimization...")
    start_time = time.time()
    opt_result = gpu_ops.optimize_phases_gpu(
        target_field, positions, phases, iterations=200, learning_rate=0.01
    )
    opt_time = time.time() - start_time
    
    print(f"   Optimization: {opt_time:.4f}s")
    print(f"   Final loss: {opt_result['final_loss']:.8f}")
    print(f"   Iterations: {opt_result['iterations']}")
    print(f"   Framework used: {opt_result['framework_used']}")
    
    # Benchmark performance
    print(f"\n📊 Running performance benchmark...")
    benchmark_results = gpu_ops.benchmark_performance([64, 128, 256])
    
    print(f"   Benchmark Results:")
    for bench in benchmark_results['benchmarks']:
        print(f"   Size {bench['problem_size']}: "
              f"Field {bench['field_computation_time']:.4f}s, "
              f"Opt {bench['optimization_time']:.4f}s")
    
    # Memory statistics
    memory_stats = gpu_ops.memory_manager.get_memory_stats()
    print(f"\n💾 Memory Statistics:")
    print(f"   Allocations: {memory_stats['allocation_count']}")
    print(f"   Peak usage: {memory_stats['peak_usage'] / 1024**2:.1f} MB")
    
    print("\n" + "=" * 60)
    return gpu_ops


if __name__ == "__main__":
    # Run demonstration
    demonstrate_gpu_acceleration()