"""
Parallel computing and resource management for acoustic holography.
Provides thread pools, process pools, GPU acceleration, and resource optimization.
"""

import threading
import multiprocessing as mp
import concurrent.futures
import queue
import time
import logging
from typing import List, Callable, Any, Optional, Dict, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil
import os
from contextlib import contextmanager

# Handle optional dependencies gracefully
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.multiprocessing as torch_mp
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


logger = logging.getLogger(__name__)


class ComputeDevice(Enum):
    """Available compute devices."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class ProcessingMode(Enum):
    """Processing modes for different workload types."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded" 
    MULTIPROCESS = "multiprocess"
    GPU_ACCELERATED = "gpu"
    DISTRIBUTED = "distributed"


@dataclass
class ComputeResource:
    """Information about available compute resources."""
    device_type: ComputeDevice
    device_id: int
    memory_gb: float
    cores: int
    utilization: float = 0.0
    temperature: Optional[float] = None
    available: bool = True


class ResourceManager:
    """
    Manages compute resources and workload distribution.
    
    Automatically detects available resources and optimizes workload distribution
    across CPUs, GPUs, and other accelerators.
    """
    
    def __init__(self, max_cpu_workers: Optional[int] = None):
        """
        Initialize resource manager.
        
        Args:
            max_cpu_workers: Maximum number of CPU worker threads/processes
        """
        self.max_cpu_workers = max_cpu_workers or min(32, (os.cpu_count() or 1) + 4)
        
        # Detect available resources
        self.cpu_resources = self._detect_cpu_resources()
        self.gpu_resources = self._detect_gpu_resources()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Resource pools
        self.thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        
        # GPU device management
        self.gpu_device_queue = queue.Queue()
        self._initialize_gpu_queue()
        
        # Resource monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        logger.info(f"Initialized ResourceManager with {len(self.cpu_resources)} CPU cores, "
                   f"{len(self.gpu_resources)} GPUs, {self.total_memory_gb:.1f}GB RAM")
    
    def _detect_cpu_resources(self) -> List[ComputeResource]:
        """Detect available CPU resources."""
        cpu_count = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        
        resources = []
        for i in range(physical_cores):
            memory_per_core = self.total_memory_gb / physical_cores
            resources.append(ComputeResource(
                device_type=ComputeDevice.CPU,
                device_id=i,
                memory_gb=memory_per_core,
                cores=cpu_count // physical_cores,  # logical cores per physical
                available=True
            ))
        
        return resources
    
    def _detect_gpu_resources(self) -> List[ComputeResource]:
        """Detect available GPU resources."""
        resources = []
        
        if HAS_TORCH and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                
                resources.append(ComputeResource(
                    device_type=ComputeDevice.CUDA,
                    device_id=i,
                    memory_gb=memory_gb,
                    cores=props.multi_processor_count,
                    available=True
                ))
        
        # Check for Apple Silicon MPS
        if HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            resources.append(ComputeResource(
                device_type=ComputeDevice.MPS,
                device_id=0,
                memory_gb=8.0,  # Estimated unified memory
                cores=8,  # Estimated GPU cores
                available=True
            ))
        
        return resources
    
    def _initialize_gpu_queue(self):
        """Initialize GPU device queue for round-robin allocation."""
        for gpu in self.gpu_resources:
            if gpu.available:
                self.gpu_device_queue.put(gpu)
    
    @contextmanager
    def get_gpu_device(self):
        """Context manager to get and release GPU device."""
        if not self.gpu_resources:
            yield None
            return
        
        try:
            gpu = self.gpu_device_queue.get(timeout=30)
            if HAS_TORCH:
                if gpu.device_type == ComputeDevice.CUDA:
                    torch.cuda.set_device(gpu.device_id)
                    device = f"cuda:{gpu.device_id}"
                elif gpu.device_type == ComputeDevice.MPS:
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = None
            
            yield device
            
        except queue.Empty:
            logger.warning("No GPU devices available, falling back to CPU")
            yield None
        finally:
            # Return device to queue
            if 'gpu' in locals():
                self.gpu_device_queue.put(gpu)
    
    def get_optimal_processing_mode(
        self,
        workload_size: int,
        computation_type: str = "general",
        memory_required_gb: float = 1.0
    ) -> ProcessingMode:
        """
        Determine optimal processing mode for given workload.
        
        Args:
            workload_size: Size of the workload (number of tasks)
            computation_type: Type of computation ("field_calc", "optimization", "fft")
            memory_required_gb: Memory required per task
            
        Returns:
            Recommended processing mode
        """
        # Check if GPU acceleration is beneficial
        if (computation_type in ["field_calc", "optimization", "fft"] 
            and self.gpu_resources and HAS_TORCH):
            
            # Check if workload fits in GPU memory
            max_gpu_memory = max(gpu.memory_gb for gpu in self.gpu_resources)
            if memory_required_gb * workload_size < max_gpu_memory * 0.8:
                return ProcessingMode.GPU_ACCELERATED
        
        # Choose between threading and multiprocessing
        if workload_size == 1:
            return ProcessingMode.SEQUENTIAL
        elif workload_size < 4 or computation_type == "io_bound":
            return ProcessingMode.THREADED
        elif memory_required_gb < 0.5 and workload_size > len(self.cpu_resources):
            return ProcessingMode.MULTIPROCESS
        else:
            return ProcessingMode.THREADED
    
    def get_thread_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """Get or create thread pool."""
        if self.thread_pool is None or self.thread_pool._shutdown:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_cpu_workers,
                thread_name_prefix="acousto_worker"
            )
        return self.thread_pool
    
    def get_process_pool(self) -> concurrent.futures.ProcessPoolExecutor:
        """Get or create process pool."""
        if self.process_pool is None or self.process_pool._shutdown:
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(self.max_cpu_workers, len(self.cpu_resources))
            )
        return self.process_pool
    
    def execute_parallel(
        self,
        func: Callable,
        tasks: List[Any],
        processing_mode: Optional[ProcessingMode] = None,
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """
        Execute tasks in parallel using optimal processing mode.
        
        Args:
            func: Function to execute
            tasks: List of task parameters
            processing_mode: Force specific processing mode
            max_workers: Override maximum workers
            
        Returns:
            List of results in same order as tasks
        """
        if not tasks:
            return []
        
        # Determine processing mode
        if processing_mode is None:
            processing_mode = self.get_optimal_processing_mode(
                workload_size=len(tasks)
            )
        
        logger.info(f"Executing {len(tasks)} tasks using {processing_mode.value} mode")
        
        if processing_mode == ProcessingMode.SEQUENTIAL:
            return [func(task) for task in tasks]
        
        elif processing_mode == ProcessingMode.THREADED:
            executor = self.get_thread_pool()
            futures = [executor.submit(func, task) for task in tasks]
            return [future.result() for future in futures]
        
        elif processing_mode == ProcessingMode.MULTIPROCESS:
            executor = self.get_process_pool()
            futures = [executor.submit(func, task) for task in tasks]
            return [future.result() for future in futures]
        
        elif processing_mode == ProcessingMode.GPU_ACCELERATED:
            return self._execute_gpu_batch(func, tasks)
        
        else:
            # Fallback to threading
            return self.execute_parallel(func, tasks, ProcessingMode.THREADED)
    
    def _execute_gpu_batch(self, func: Callable, tasks: List[Any]) -> List[Any]:
        """Execute tasks on GPU in batches."""
        results = []
        
        with self.get_gpu_device() as device:
            if device is None:
                # Fallback to CPU
                return [func(task) for task in tasks]
            
            # Process in batches to manage GPU memory
            batch_size = self._calculate_gpu_batch_size(len(tasks))
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                batch_results = func(batch, device=device)
                
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)
        
        return results
    
    def _calculate_gpu_batch_size(self, total_tasks: int) -> int:
        """Calculate optimal GPU batch size."""
        if not self.gpu_resources:
            return 1
        
        # Estimate based on GPU memory
        max_gpu_memory = max(gpu.memory_gb for gpu in self.gpu_resources)
        
        # Conservative batch size calculation
        if max_gpu_memory >= 16:
            return min(64, total_tasks)
        elif max_gpu_memory >= 8:
            return min(32, total_tasks)
        else:
            return min(16, total_tasks)
    
    def start_monitoring(self, interval: float = 30.0):
        """Start resource monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        def monitor_loop():
            while self._monitoring_active:
                try:
                    self._update_resource_status()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    time.sleep(interval)
        
        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
    
    def _update_resource_status(self):
        """Update resource utilization information."""
        # Update CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        for i, resource in enumerate(self.cpu_resources):
            if i < len(cpu_percent):
                resource.utilization = cpu_percent[i] / 100.0
        
        # Update GPU utilization
        if HAS_TORCH and torch.cuda.is_available():
            for gpu in self.gpu_resources:
                if gpu.device_type == ComputeDevice.CUDA:
                    try:
                        gpu.utilization = torch.cuda.utilization(gpu.device_id) / 100.0
                    except Exception:
                        pass
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return {
            'cpu_resources': [
                {
                    'device_id': r.device_id,
                    'cores': r.cores,
                    'memory_gb': r.memory_gb,
                    'utilization': r.utilization,
                    'available': r.available
                }
                for r in self.cpu_resources
            ],
            'gpu_resources': [
                {
                    'device_id': r.device_id,
                    'device_type': r.device_type.value,
                    'cores': r.cores,
                    'memory_gb': r.memory_gb,
                    'utilization': r.utilization,
                    'available': r.available
                }
                for r in self.gpu_resources
            ],
            'total_memory_gb': self.total_memory_gb,
            'memory_usage_percent': psutil.virtual_memory().percent
        }
    
    def shutdown(self):
        """Shutdown resource manager and cleanup."""
        self.stop_monitoring()
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class ParallelFieldCalculator:
    """
    Parallel acoustic field calculator with automatic load balancing.
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        """Initialize parallel field calculator."""
        self.resource_manager = resource_manager or ResourceManager()
        self.cache = {}
        
    def calculate_field_parallel(
        self,
        transducer_positions: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
        field_points: np.ndarray,
        frequency: float,
        medium_properties: Dict[str, Any],
        chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate acoustic field in parallel.
        
        Args:
            transducer_positions: Transducer positions (N, 3)
            phases: Phase values (N,)
            amplitudes: Amplitude values (N,)
            field_points: Field evaluation points (M, 3)
            frequency: Frequency in Hz
            medium_properties: Medium properties dict
            chunk_size: Override chunk size for field points
            
        Returns:
            Complex pressure field (M,)
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for field calculations")
        
        num_field_points = field_points.shape[0]
        num_transducers = transducer_positions.shape[0]
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size(
                num_field_points, num_transducers
            )
        
        # Split field points into chunks
        field_chunks = [
            field_points[i:i + chunk_size]
            for i in range(0, num_field_points, chunk_size)
        ]
        
        # Prepare calculation function
        def calculate_chunk(chunk_data):
            chunk_points, device = chunk_data
            return self._calculate_field_chunk(
                transducer_positions, phases, amplitudes,
                chunk_points, frequency, medium_properties, device
            )
        
        # Determine processing mode
        processing_mode = self.resource_manager.get_optimal_processing_mode(
            workload_size=len(field_chunks),
            computation_type="field_calc",
            memory_required_gb=self._estimate_memory_requirement(chunk_size, num_transducers)
        )
        
        # Execute parallel calculation
        if processing_mode == ProcessingMode.GPU_ACCELERATED and HAS_TORCH:
            results = self._calculate_field_gpu_parallel(
                transducer_positions, phases, amplitudes,
                field_chunks, frequency, medium_properties
            )
        else:
            # CPU parallel execution
            chunk_tasks = [(chunk, None) for chunk in field_chunks]
            results = self.resource_manager.execute_parallel(
                calculate_chunk, chunk_tasks, processing_mode
            )
        
        # Combine results
        return np.concatenate(results) if results else np.array([])
    
    def _calculate_field_chunk(
        self,
        transducer_pos: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
        field_points: np.ndarray,
        frequency: float,
        medium_props: Dict[str, Any],
        device: Optional[str] = None
    ) -> np.ndarray:
        """Calculate field for a chunk of points."""
        if device and HAS_TORCH and "cuda" in device:
            return self._calculate_field_chunk_gpu(
                transducer_pos, phases, amplitudes,
                field_points, frequency, medium_props, device
            )
        else:
            return self._calculate_field_chunk_cpu(
                transducer_pos, phases, amplitudes,
                field_points, frequency, medium_props
            )
    
    def _calculate_field_chunk_cpu(
        self,
        transducer_pos: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
        field_points: np.ndarray,
        frequency: float,
        medium_props: Dict[str, Any]
    ) -> np.ndarray:
        """CPU implementation of field calculation."""
        c = medium_props.get('speed_of_sound', 343.0)
        k = 2 * np.pi * frequency / c
        
        # Calculate distances from each transducer to each field point
        distances = np.linalg.norm(
            field_points[:, np.newaxis, :] - transducer_pos[np.newaxis, :, :],
            axis=2
        )
        
        # Green's function
        green = np.exp(1j * k * distances) / (4 * np.pi * distances)
        
        # Apply phases and amplitudes
        source_strength = amplitudes * np.exp(1j * phases)
        
        # Sum contributions
        field = np.sum(green * source_strength[np.newaxis, :], axis=1)
        
        return field
    
    def _calculate_field_chunk_gpu(
        self,
        transducer_pos: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
        field_points: np.ndarray,
        frequency: float,
        medium_props: Dict[str, Any],
        device: str
    ) -> np.ndarray:
        """GPU implementation of field calculation."""
        if not HAS_TORCH:
            return self._calculate_field_chunk_cpu(
                transducer_pos, phases, amplitudes,
                field_points, frequency, medium_props
            )
        
        # Convert to tensors
        t_pos = torch.from_numpy(transducer_pos).float().to(device)
        f_points = torch.from_numpy(field_points).float().to(device)
        t_phases = torch.from_numpy(phases).float().to(device)
        t_amplitudes = torch.from_numpy(amplitudes).float().to(device)
        
        c = medium_props.get('speed_of_sound', 343.0)
        k = 2 * np.pi * frequency / c
        
        # Calculate distances efficiently on GPU
        distances = torch.norm(
            f_points.unsqueeze(1) - t_pos.unsqueeze(0),
            dim=2
        )
        
        # Green's function
        green = torch.exp(1j * k * distances) / (4 * np.pi * distances)
        
        # Apply phases and amplitudes
        source_strength = t_amplitudes * torch.exp(1j * t_phases)
        
        # Sum contributions
        field = torch.sum(green * source_strength.unsqueeze(0), dim=1)
        
        return field.cpu().numpy()
    
    def _calculate_field_gpu_parallel(
        self,
        transducer_pos: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
        field_chunks: List[np.ndarray],
        frequency: float,
        medium_props: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Calculate field using multiple GPUs in parallel."""
        results = []
        
        with self.resource_manager.get_gpu_device() as device:
            for chunk in field_chunks:
                result = self._calculate_field_chunk_gpu(
                    transducer_pos, phases, amplitudes,
                    chunk, frequency, medium_props, device
                )
                results.append(result)
        
        return results
    
    def _calculate_optimal_chunk_size(
        self,
        num_field_points: int,
        num_transducers: int
    ) -> int:
        """Calculate optimal chunk size for field calculation."""
        # Estimate memory usage per point
        bytes_per_point = num_transducers * 8  # Complex64
        
        # Target chunk memory usage (100MB)
        target_memory = 100 * 1024 * 1024
        optimal_chunk = target_memory // bytes_per_point
        
        # Constraints
        min_chunk = 100
        max_chunk = min(10000, num_field_points)
        
        return max(min_chunk, min(optimal_chunk, max_chunk))
    
    def _estimate_memory_requirement(
        self,
        chunk_size: int,
        num_transducers: int
    ) -> float:
        """Estimate memory requirement in GB."""
        bytes_needed = chunk_size * num_transducers * 8  # Complex64
        return bytes_needed / (1024**3)


# Global resource manager
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get or create global resource manager."""
    global _global_resource_manager
    
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
        _global_resource_manager.start_monitoring()
    
    return _global_resource_manager


def parallel_compute(processing_mode: Optional[ProcessingMode] = None):
    """Decorator for automatic parallel execution."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Check if function should be parallelized
            if 'parallel' in kwargs and not kwargs.pop('parallel'):
                return func(*args, **kwargs)
            
            # Get resource manager
            rm = get_resource_manager()
            
            # Determine if parallelization is beneficial
            # This is a simplified heuristic
            input_size = 1
            for arg in args:
                if hasattr(arg, '__len__'):
                    input_size = max(input_size, len(arg))
            
            if input_size < 10:  # Small inputs don't benefit from parallelization
                return func(*args, **kwargs)
            
            # For now, just use optimal mode determination
            mode = processing_mode or rm.get_optimal_processing_mode(input_size)
            
            if mode == ProcessingMode.SEQUENTIAL:
                return func(*args, **kwargs)
            else:
                # This would need more sophisticated task decomposition
                # For now, fall back to original function
                return func(*args, **kwargs)
        
        return wrapper
    return decorator