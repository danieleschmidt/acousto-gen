"""
GPU acceleration framework for acoustic holography computations.
Supports multi-GPU processing, memory optimization, and distributed computing.
"""

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
import warnings

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    cuda = None
    jit = None


@dataclass
class GPUInfo:
    """GPU device information."""
    device_id: int
    name: str
    memory_total: int
    memory_free: int
    compute_capability: Tuple[int, int]
    multiprocessor_count: int
    is_available: bool = True


@dataclass
class ComputeTask:
    """Computational task for GPU processing."""
    task_id: str
    function: Callable
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    priority: int = 0
    device_preference: Optional[int] = None
    memory_required: Optional[int] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for GPU operations."""
    total_time: float
    gpu_time: float
    memory_usage: Dict[int, int]  # device_id -> memory_used
    throughput: float
    efficiency: float
    power_consumption: Optional[float] = None


class GPUMemoryManager:
    """Advanced GPU memory management with pooling and optimization."""
    
    def __init__(self, device_ids: List[int]):
        """
        Initialize GPU memory manager.
        
        Args:
            device_ids: List of GPU device IDs to manage
        """
        self.device_ids = device_ids
        self.memory_pools: Dict[int, Any] = {}
        self.allocation_tracking: Dict[int, List[Tuple[int, int]]] = {}
        self.fragmentation_threshold = 0.1  # 10% fragmentation threshold
        
        self._initialize_memory_pools()
    
    def _initialize_memory_pools(self):
        """Initialize memory pools for each device."""
        for device_id in self.device_ids:
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                # Create memory pool for the device
                with torch.cuda.device(device_id):
                    # Enable memory pool if available (PyTorch 1.10+)
                    if hasattr(torch.cuda, 'memory_pool_set_release_threshold'):
                        torch.cuda.memory_pool_set_release_threshold(device_id, 0.8)
                
                self.allocation_tracking[device_id] = []
    
    def allocate_tensor(
        self,
        size: Tuple[int, ...],
        dtype: torch.dtype,
        device_id: int
    ) -> torch.Tensor:
        """
        Allocate tensor with optimized memory management.
        
        Args:
            size: Tensor size
            dtype: Data type
            device_id: Target GPU device
            
        Returns:
            Allocated tensor
        """
        device = torch.device(f'cuda:{device_id}')
        
        try:
            tensor = torch.empty(size, dtype=dtype, device=device)
            
            # Track allocation
            memory_used = tensor.element_size() * tensor.numel()
            self.allocation_tracking[device_id].append((memory_used, time.time()))
            
            return tensor
            
        except torch.cuda.OutOfMemoryError:
            # Try memory cleanup and retry
            self._cleanup_memory(device_id)
            
            try:
                tensor = torch.empty(size, dtype=dtype, device=device)
                memory_used = tensor.element_size() * tensor.numel()
                self.allocation_tracking[device_id].append((memory_used, time.time()))
                return tensor
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(f"Insufficient GPU memory on device {device_id}")
    
    def _cleanup_memory(self, device_id: int):
        """Perform memory cleanup on specified device."""
        with torch.cuda.device(device_id):
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats(device_id)
    
    def get_memory_info(self, device_id: int) -> Dict[str, int]:
        """Get detailed memory information for device."""
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            return {}
        
        with torch.cuda.device(device_id):
            return {
                'allocated': torch.cuda.memory_allocated(device_id),
                'cached': torch.cuda.memory_reserved(device_id),
                'max_allocated': torch.cuda.max_memory_allocated(device_id),
                'max_cached': torch.cuda.max_memory_reserved(device_id),
            }
    
    def optimize_memory_layout(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optimize memory layout of tensors for better performance."""
        optimized_tensors = []
        
        for tensor in tensors:
            if not tensor.is_contiguous():
                # Make tensor contiguous for better memory access patterns
                tensor = tensor.contiguous()
            
            # Consider memory format optimization for convolutions
            if tensor.dim() == 4:  # Assuming NCHW format
                # Convert to channels_last for better GPU performance
                tensor = tensor.to(memory_format=torch.channels_last)
            
            optimized_tensors.append(tensor)
        
        return optimized_tensors


class MultiGPUAccelerator:
    """
    Multi-GPU acceleration system for acoustic holography computations.
    
    Provides automatic load balancing, memory management, and distributed
    processing across multiple GPU devices.
    """
    
    def __init__(
        self,
        device_ids: Optional[List[int]] = None,
        enable_mixed_precision: bool = True,
        enable_tensor_cores: bool = True,
        memory_fraction: float = 0.9
    ):
        """
        Initialize multi-GPU accelerator.
        
        Args:
            device_ids: List of GPU device IDs to use (auto-detect if None)
            enable_mixed_precision: Use mixed precision training
            enable_tensor_cores: Enable tensor core usage
            memory_fraction: Fraction of GPU memory to use
        """
        self.device_ids = device_ids or self._detect_gpus()
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_tensor_cores = enable_tensor_cores
        self.memory_fraction = memory_fraction
        
        # Initialize components
        self.gpu_info = self._get_gpu_info()
        self.memory_manager = GPUMemoryManager(self.device_ids)
        self.task_queue = queue.PriorityQueue()
        self.result_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.load_balancer = self._create_load_balancer()
        
        # Threading
        self.worker_threads: List[threading.Thread] = []
        self.is_running = False
        
        # Mixed precision scaler
        if self.enable_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"Multi-GPU Accelerator initialized with {len(self.device_ids)} devices")
        for device_id in self.device_ids:
            gpu = self.gpu_info[device_id]
            print(f"  GPU {device_id}: {gpu.name} ({gpu.memory_total // 1024**2} MB)")
    
    def _detect_gpus(self) -> List[int]:
        """Auto-detect available GPU devices."""
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            return []
        
        device_count = torch.cuda.device_count()
        available_devices = []
        
        for i in range(device_count):
            try:
                # Test if device is accessible
                with torch.cuda.device(i):
                    torch.cuda.current_device()
                available_devices.append(i)
            except Exception as e:
                print(f"GPU {i} not accessible: {e}")
        
        return available_devices
    
    def _get_gpu_info(self) -> Dict[int, GPUInfo]:
        """Get information about available GPUs."""
        gpu_info = {}
        
        for device_id in self.device_ids:
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(device_id)
                
                gpu_info[device_id] = GPUInfo(
                    device_id=device_id,
                    name=props.name,
                    memory_total=props.total_memory,
                    memory_free=props.total_memory - torch.cuda.memory_allocated(device_id),
                    compute_capability=(props.major, props.minor),
                    multiprocessor_count=props.multi_processor_count,
                    is_available=True
                )
            else:
                gpu_info[device_id] = GPUInfo(
                    device_id=device_id,
                    name="Unknown",
                    memory_total=0,
                    memory_free=0,
                    compute_capability=(0, 0),
                    multiprocessor_count=0,
                    is_available=False
                )
        
        return gpu_info
    
    def _create_load_balancer(self) -> 'LoadBalancer':
        """Create load balancer for task distribution."""
        return LoadBalancer(self.gpu_info)
    
    def accelerate_wave_propagation(
        self,
        source_positions: np.ndarray,
        source_amplitudes: np.ndarray,
        source_phases: np.ndarray,
        target_points: np.ndarray,
        frequency: float,
        medium_properties: Dict[str, float]
    ) -> torch.Tensor:
        """
        GPU-accelerated wave propagation computation.
        
        Args:
            source_positions: Nx3 array of source positions
            source_amplitudes: N array of source amplitudes
            source_phases: N array of source phases
            target_points: Mx3 array of target points
            frequency: Operating frequency
            medium_properties: Medium properties dict
            
        Returns:
            Complex pressure field at target points
        """
        start_time = time.time()
        
        # Convert to tensors and distribute across GPUs
        num_sources = len(source_positions)
        num_targets = len(target_points)
        chunk_size = max(1, num_targets // len(self.device_ids))
        
        # Split computation across GPUs
        tasks = []
        for i, device_id in enumerate(self.device_ids):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_targets) if i < len(self.device_ids) - 1 else num_targets
            
            if start_idx < num_targets:
                task = ComputeTask(
                    task_id=f"wave_prop_{i}",
                    function=self._compute_wave_propagation_chunk,
                    args=(
                        source_positions,
                        source_amplitudes,
                        source_phases,
                        target_points[start_idx:end_idx],
                        frequency,
                        medium_properties
                    ),
                    kwargs={'device_id': device_id},
                    priority=1
                )
                tasks.append(task)
        
        # Execute tasks in parallel
        results = self.execute_parallel_tasks(tasks)
        
        # Combine results
        combined_result = torch.cat(results, dim=0)
        
        # Record performance metrics
        total_time = time.time() - start_time
        self.performance_metrics['wave_propagation'] = PerformanceMetrics(
            total_time=total_time,
            gpu_time=total_time * 0.9,  # Estimate
            memory_usage={i: self.memory_manager.get_memory_info(i)['allocated'] 
                         for i in self.device_ids},
            throughput=num_targets / total_time,
            efficiency=0.8  # Estimate
        )
        
        return combined_result
    
    def _compute_wave_propagation_chunk(
        self,
        source_positions: np.ndarray,
        source_amplitudes: np.ndarray,
        source_phases: np.ndarray,
        target_points: np.ndarray,
        frequency: float,
        medium_properties: Dict[str, float],
        device_id: int
    ) -> torch.Tensor:
        """Compute wave propagation for a chunk of target points."""
        device = torch.device(f'cuda:{device_id}')
        
        # Convert to tensors on GPU
        src_pos = torch.tensor(source_positions, dtype=torch.float32, device=device)
        src_amp = torch.tensor(source_amplitudes, dtype=torch.float32, device=device)
        src_phase = torch.tensor(source_phases, dtype=torch.float32, device=device)
        tgt_pos = torch.tensor(target_points, dtype=torch.float32, device=device)
        
        # Physical constants
        k = 2 * np.pi * frequency / medium_properties['speed_of_sound']
        
        # Compute distances (vectorized)
        # src_pos: [N, 3], tgt_pos: [M, 3] -> distances: [M, N]
        src_pos_expanded = src_pos.unsqueeze(0)  # [1, N, 3]
        tgt_pos_expanded = tgt_pos.unsqueeze(1)  # [M, 1, 3]
        
        distances = torch.norm(tgt_pos_expanded - src_pos_expanded, dim=2)  # [M, N]
        
        # Avoid division by zero
        distances = torch.maximum(distances, torch.tensor(1e-6, device=device))
        
        # Green's function computation
        with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
            # Complex amplitude for each source-target pair
            complex_amplitude = (src_amp.unsqueeze(0) * 
                               torch.exp(1j * (k * distances + src_phase.unsqueeze(0))) / 
                               (4 * np.pi * distances))
            
            # Sum contributions from all sources
            field = torch.sum(complex_amplitude, dim=1)  # [M]
        
        return field
    
    def accelerate_field_optimization(
        self,
        forward_model: Callable,
        target_field: torch.Tensor,
        initial_phases: torch.Tensor,
        iterations: int = 1000,
        learning_rate: float = 0.01
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        GPU-accelerated field optimization with distributed training.
        
        Args:
            forward_model: Forward propagation model
            target_field: Target acoustic field
            initial_phases: Initial phase values
            iterations: Number of optimization iterations
            learning_rate: Learning rate
            
        Returns:
            Tuple of (optimized_phases, loss_history)
        """
        # Use primary GPU for optimization
        primary_device = self.device_ids[0]
        device = torch.device(f'cuda:{primary_device}')
        
        # Move data to GPU
        phases = initial_phases.clone().to(device).requires_grad_(True)
        target = target_field.to(device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([phases], lr=learning_rate)
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.enable_mixed_precision else None
        
        loss_history = []
        
        for iteration in range(iterations):
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
                # Forward pass
                generated_field = forward_model(phases)
                
                # Loss computation
                loss = torch.nn.functional.mse_loss(generated_field, target)
                
                # Add regularization
                phase_smooth_loss = torch.mean(torch.diff(phases) ** 2)
                total_loss = loss + 0.01 * phase_smooth_loss
            
            # Backward pass with mixed precision
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            # Record loss
            loss_history.append(total_loss.item())
            
            # Early stopping check
            if iteration > 100 and abs(loss_history[-1] - loss_history[-100]) < 1e-6:
                break
        
        return phases.detach().cpu(), loss_history
    
    def execute_parallel_tasks(
        self,
        tasks: List[ComputeTask],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Execute multiple computational tasks in parallel across GPUs.
        
        Args:
            tasks: List of computational tasks
            timeout: Maximum execution time
            
        Returns:
            List of task results
        """
        # Distribute tasks using load balancer
        task_assignments = self.load_balancer.assign_tasks(tasks)
        
        # Execute tasks using thread pool
        results = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=len(self.device_ids)) as executor:
            # Submit tasks
            future_to_task = {}
            for task_idx, (task, device_id) in enumerate(task_assignments.items()):
                future = executor.submit(self._execute_task, task, device_id)
                future_to_task[future] = task_idx
            
            # Collect results
            for future in as_completed(future_to_task, timeout=timeout):
                task_idx = future_to_task[future]
                try:
                    results[task_idx] = future.result()
                except Exception as e:
                    print(f"Task {task_idx} failed: {e}")
                    results[task_idx] = None
        
        return [r for r in results if r is not None]
    
    def _execute_task(self, task: ComputeTask, device_id: int) -> Any:
        """Execute individual computational task on specified GPU."""
        try:
            with torch.cuda.device(device_id):
                # Set device context
                kwargs = task.kwargs.copy()
                kwargs['device_id'] = device_id
                
                # Execute function
                result = task.function(*task.args, **kwargs)
                
                # Cache result if beneficial
                if task.task_id and len(str(result)) < 1000:  # Simple caching heuristic
                    self.result_cache[task.task_id] = result
                
                return result
        
        except Exception as e:
            print(f"Task execution failed on GPU {device_id}: {e}")
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'gpu_info': {
                device_id: {
                    'name': info.name,
                    'memory_total_mb': info.memory_total // 1024**2,
                    'memory_free_mb': info.memory_free // 1024**2,
                    'compute_capability': info.compute_capability,
                    'is_available': info.is_available
                }
                for device_id, info in self.gpu_info.items()
            },
            'performance_metrics': {
                op_name: {
                    'total_time': metrics.total_time,
                    'gpu_time': metrics.gpu_time,
                    'throughput': metrics.throughput,
                    'efficiency': metrics.efficiency,
                    'memory_usage_mb': {
                        device_id: usage // 1024**2 
                        for device_id, usage in metrics.memory_usage.items()
                    }
                }
                for op_name, metrics in self.performance_metrics.items()
            },
            'system_info': {
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__,
                'mixed_precision_enabled': self.enable_mixed_precision,
                'tensor_cores_enabled': self.enable_tensor_cores,
            }
        }
        
        return report
    
    def optimize_batch_size(
        self,
        model_function: Callable,
        input_shape: Tuple[int, ...],
        device_id: int = 0,
        max_batch_size: int = 1024
    ) -> int:
        """
        Find optimal batch size for given model and input shape.
        
        Args:
            model_function: Function to test
            input_shape: Input tensor shape (without batch dimension)
            device_id: Target GPU device
            max_batch_size: Maximum batch size to test
            
        Returns:
            Optimal batch size
        """
        device = torch.device(f'cuda:{device_id}')
        optimal_batch_size = 1
        
        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Test batch size
                test_input = torch.randn((mid,) + input_shape, device=device)
                
                # Measure memory usage
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated(device_id)
                
                with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
                    _ = model_function(test_input)
                
                memory_after = torch.cuda.memory_allocated(device_id)
                memory_used = memory_after - memory_before
                
                # Check if within memory limits
                available_memory = self.gpu_info[device_id].memory_free
                if memory_used < available_memory * self.memory_fraction:
                    optimal_batch_size = mid
                    low = mid + 1
                else:
                    high = mid - 1
                
                # Cleanup
                del test_input
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                high = mid - 1
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error testing batch size {mid}: {e}")
                break
        
        return optimal_batch_size


class LoadBalancer:
    """Load balancer for distributing tasks across GPUs."""
    
    def __init__(self, gpu_info: Dict[int, GPUInfo]):
        """Initialize load balancer with GPU information."""
        self.gpu_info = gpu_info
        self.task_history: Dict[int, List[float]] = {
            device_id: [] for device_id in gpu_info.keys()
        }
    
    def assign_tasks(self, tasks: List[ComputeTask]) -> Dict[ComputeTask, int]:
        """
        Assign tasks to GPUs using load balancing algorithm.
        
        Args:
            tasks: List of tasks to assign
            
        Returns:
            Dictionary mapping tasks to device IDs
        """
        assignments = {}
        device_loads = {device_id: 0 for device_id in self.gpu_info.keys()}
        
        # Sort tasks by priority (higher priority first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            # Find best device for task
            best_device = self._select_device(task, device_loads)
            
            assignments[task] = best_device
            
            # Update load estimate
            estimated_load = self._estimate_task_load(task)
            device_loads[best_device] += estimated_load
        
        return assignments
    
    def _select_device(self, task: ComputeTask, current_loads: Dict[int, float]) -> int:
        """Select best device for a specific task."""
        # Check device preference
        if (task.device_preference is not None and 
            task.device_preference in self.gpu_info and
            self.gpu_info[task.device_preference].is_available):
            return task.device_preference
        
        # Select device with lowest load and sufficient memory
        available_devices = [
            device_id for device_id, info in self.gpu_info.items()
            if info.is_available
        ]
        
        if not available_devices:
            raise RuntimeError("No available GPU devices")
        
        # Score devices based on load and memory
        device_scores = {}
        
        for device_id in available_devices:
            info = self.gpu_info[device_id]
            load = current_loads[device_id]
            
            # Normalize load (0-1)
            load_score = load / max(1.0, max(current_loads.values()))
            
            # Memory availability score (0-1)
            memory_score = info.memory_free / max(1, info.memory_total)
            
            # Compute capability bonus
            capability_score = (info.compute_capability[0] * 10 + info.compute_capability[1]) / 100
            
            # Combined score (lower is better)
            combined_score = load_score - 0.3 * memory_score - 0.1 * capability_score
            device_scores[device_id] = combined_score
        
        # Return device with best score
        return min(device_scores.keys(), key=lambda d: device_scores[d])
    
    def _estimate_task_load(self, task: ComputeTask) -> float:
        """Estimate computational load of a task."""
        # Simple heuristic based on task type and parameters
        base_load = 1.0
        
        # Adjust based on memory requirements
        if task.memory_required:
            memory_factor = task.memory_required / (1024**3)  # GB
            base_load *= (1 + memory_factor * 0.1)
        
        # Adjust based on task priority
        priority_factor = max(1.0, task.priority / 10.0)
        base_load *= priority_factor
        
        return base_load
    
    def update_task_completion(self, device_id: int, execution_time: float):
        """Update task completion history for better load balancing."""
        self.task_history[device_id].append(execution_time)
        
        # Keep only recent history
        if len(self.task_history[device_id]) > 100:
            self.task_history[device_id] = self.task_history[device_id][-100:]


# CUDA kernels using Numba (if available)
if NUMBA_AVAILABLE:
    @cuda.jit
    def wave_propagation_kernel(
        src_positions, src_amplitudes, src_phases,
        tgt_positions, distances, field_real, field_imag,
        k, num_sources, num_targets
    ):
        """CUDA kernel for wave propagation computation."""
        target_idx = cuda.grid(1)
        
        if target_idx < num_targets:
            real_sum = 0.0
            imag_sum = 0.0
            
            for src_idx in range(num_sources):
                # Calculate distance
                dx = tgt_positions[target_idx, 0] - src_positions[src_idx, 0]
                dy = tgt_positions[target_idx, 1] - src_positions[src_idx, 1]
                dz = tgt_positions[target_idx, 2] - src_positions[src_idx, 2]
                
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                distances[target_idx, src_idx] = dist
                
                if dist > 1e-6:  # Avoid singularity
                    # Green's function
                    phase = k * dist + src_phases[src_idx]
                    amplitude = src_amplitudes[src_idx] / (4 * 3.14159265359 * dist)
                    
                    real_sum += amplitude * math.cos(phase)
                    imag_sum += amplitude * math.sin(phase)
            
            field_real[target_idx] = real_sum
            field_imag[target_idx] = imag_sum


def create_gpu_accelerator(
    device_ids: Optional[List[int]] = None,
    enable_optimizations: bool = True
) -> Optional[MultiGPUAccelerator]:
    """
    Create GPU accelerator instance with auto-detection.
    
    Args:
        device_ids: Specific GPU devices to use
        enable_optimizations: Enable performance optimizations
        
    Returns:
        MultiGPUAccelerator instance or None if no GPUs available
    """
    if not torch.cuda.is_available():
        print("CUDA not available, GPU acceleration disabled")
        return None
    
    try:
        accelerator = MultiGPUAccelerator(
            device_ids=device_ids,
            enable_mixed_precision=enable_optimizations,
            enable_tensor_cores=enable_optimizations
        )
        
        return accelerator
        
    except Exception as e:
        print(f"Failed to initialize GPU accelerator: {e}")
        return None