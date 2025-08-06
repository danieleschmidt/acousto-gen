"""
Concurrent and parallel processing for acoustic holography computations.
Provides GPU acceleration, multi-threading, and distributed computing support.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass
from threading import Lock, Event
import queue
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ComputeTask:
    """Compute task definition."""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    estimated_time: float = 1.0
    gpu_required: bool = False
    memory_required_mb: float = 100.0


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    result: Any
    execution_time: float
    success: bool
    error: Optional[Exception] = None


class ResourceManager:
    """Manages computational resources and load balancing."""
    
    def __init__(self):
        """Initialize resource manager."""
        self._lock = Lock()
        self._cpu_cores = self._detect_cpu_cores()
        self._gpu_available = self._detect_gpu()
        self._memory_gb = self._detect_memory()
        
        # Resource tracking
        self._cpu_usage = 0.0
        self._gpu_usage = 0.0
        self._memory_usage_mb = 0.0
        
        # Task queues by priority
        self._high_priority_queue = queue.PriorityQueue()
        self._normal_priority_queue = queue.PriorityQueue()
        self._low_priority_queue = queue.PriorityQueue()
        
        logger.info(f"Detected {self._cpu_cores} CPU cores, "
                   f"GPU available: {self._gpu_available}, "
                   f"Memory: {self._memory_gb:.1f} GB")
    
    def _detect_cpu_cores(self) -> int:
        """Detect number of CPU cores."""
        import multiprocessing
        return multiprocessing.cpu_count()
    
    def _detect_gpu(self) -> bool:
        """Detect GPU availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _detect_memory(self) -> float:
        """Detect available memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 8.0  # Fallback estimate
    
    def get_optimal_thread_count(self, task_type: str = "cpu") -> int:
        """Get optimal number of threads for task type."""
        if task_type == "gpu" and self._gpu_available:
            # GPU tasks typically don't need many threads
            return min(4, self._cpu_cores // 2)
        elif task_type == "io":
            # I/O bound tasks can use more threads
            return min(32, self._cpu_cores * 4)
        else:
            # CPU bound tasks
            return max(1, self._cpu_cores - 1)
    
    def can_execute_task(self, task: ComputeTask) -> bool:
        """Check if task can be executed given current resources."""
        with self._lock:
            if task.gpu_required and not self._gpu_available:
                return False
            
            memory_available = (self._memory_gb * 1024) - self._memory_usage_mb
            if task.memory_required_mb > memory_available:
                return False
            
            return True
    
    def allocate_resources(self, task: ComputeTask) -> bool:
        """Allocate resources for task execution."""
        with self._lock:
            if not self.can_execute_task(task):
                return False
            
            self._memory_usage_mb += task.memory_required_mb
            if task.gpu_required:
                self._gpu_usage = min(1.0, self._gpu_usage + 0.1)
            
            return True
    
    def release_resources(self, task: ComputeTask):
        """Release resources after task completion."""
        with self._lock:
            self._memory_usage_mb = max(0, self._memory_usage_mb - task.memory_required_mb)
            if task.gpu_required:
                self._gpu_usage = max(0.0, self._gpu_usage - 0.1)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource utilization."""
        with self._lock:
            return {
                'cpu_cores': self._cpu_cores,
                'gpu_available': self._gpu_available,
                'total_memory_gb': self._memory_gb,
                'cpu_usage': self._cpu_usage,
                'gpu_usage': self._gpu_usage,
                'memory_usage_mb': self._memory_usage_mb,
                'memory_utilization': self._memory_usage_mb / (self._memory_gb * 1024)
            }


class ConcurrentExecutor:
    """Concurrent task executor with intelligent scheduling."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        enable_gpu: bool = True,
        enable_multiprocessing: bool = True
    ):
        """
        Initialize concurrent executor.
        
        Args:
            max_workers: Maximum number of worker threads
            enable_gpu: Whether to enable GPU acceleration
            enable_multiprocessing: Whether to enable multiprocessing
        """
        self.resource_manager = ResourceManager()
        self.max_workers = max_workers or self.resource_manager.get_optimal_thread_count()
        self.enable_gpu = enable_gpu and self.resource_manager._gpu_available
        self.enable_multiprocessing = enable_multiprocessing
        
        # Executors
        self._thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._process_executor = ProcessPoolExecutor(max_workers=min(4, self.max_workers)) if enable_multiprocessing else None
        
        # Task tracking
        self._running_tasks: Dict[str, ComputeTask] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._task_lock = Lock()
        
        # Performance stats
        self._total_tasks = 0
        self._completed_count = 0
        self._failed_count = 0
        self._total_execution_time = 0.0
        
        logger.info(f"Concurrent executor initialized with {self.max_workers} workers, "
                   f"GPU: {self.enable_gpu}, Multiprocessing: {self.enable_multiprocessing}")
    
    def submit_task(
        self,
        task_id: str,
        function: Callable,
        *args,
        priority: int = 0,
        gpu_required: bool = False,
        use_process: bool = False,
        **kwargs
    ) -> str:
        """
        Submit task for execution.
        
        Args:
            task_id: Unique task identifier
            function: Function to execute
            *args: Function arguments
            priority: Task priority (higher = more important)
            gpu_required: Whether task requires GPU
            use_process: Whether to use separate process
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task = ComputeTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            gpu_required=gpu_required and self.enable_gpu
        )
        
        with self._task_lock:
            if task_id in self._running_tasks:
                raise ValueError(f"Task {task_id} is already running")
            
            self._running_tasks[task_id] = task
            self._total_tasks += 1
        
        # Choose executor based on requirements
        if use_process and self._process_executor:
            future = self._process_executor.submit(self._execute_task, task)
        else:
            future = self._thread_executor.submit(self._execute_task, task)
        
        # Add callback for completion
        future.add_done_callback(lambda f: self._task_completed(task_id, f))
        
        logger.debug(f"Submitted task {task_id} with priority {priority}")
        return task_id
    
    def _execute_task(self, task: ComputeTask) -> TaskResult:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            # Allocate resources
            if not self.resource_manager.allocate_resources(task):
                raise RuntimeError("Insufficient resources to execute task")
            
            # Execute function
            if task.gpu_required:
                result = self._execute_gpu_task(task)
            else:
                result = task.function(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                result=result,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                result=None,
                execution_time=execution_time,
                success=False,
                error=e
            )
            
        finally:
            self.resource_manager.release_resources(task)
    
    def _execute_gpu_task(self, task: ComputeTask) -> Any:
        """Execute task on GPU."""
        if not self.enable_gpu:
            # Fallback to CPU
            return task.function(*task.args, **task.kwargs)
        
        try:
            import torch
            
            # Move data to GPU if possible
            gpu_args = []
            for arg in task.args:
                if isinstance(arg, np.ndarray):
                    gpu_args.append(torch.from_numpy(arg).cuda())
                else:
                    gpu_args.append(arg)
            
            gpu_kwargs = {}
            for k, v in task.kwargs.items():
                if isinstance(v, np.ndarray):
                    gpu_kwargs[k] = torch.from_numpy(v).cuda()
                else:
                    gpu_kwargs[k] = v
            
            # Execute on GPU
            with torch.cuda.device(0):
                result = task.function(*gpu_args, **gpu_kwargs)
            
            # Convert result back to numpy if needed
            if hasattr(result, 'cpu') and hasattr(result, 'numpy'):
                return result.cpu().numpy()
            else:
                return result
                
        except Exception as e:
            logger.warning(f"GPU execution failed for task {task.task_id}, falling back to CPU: {e}")
            return task.function(*task.args, **task.kwargs)
    
    def _task_completed(self, task_id: str, future):
        """Handle task completion."""
        try:
            result = future.result()
            
            with self._task_lock:
                if task_id in self._running_tasks:
                    del self._running_tasks[task_id]
                
                self._completed_tasks[task_id] = result
                
                if result.success:
                    self._completed_count += 1
                else:
                    self._failed_count += 1
                
                self._total_execution_time += result.execution_time
            
            logger.debug(f"Task {task_id} completed in {result.execution_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error handling completion of task {task_id}: {e}")
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        Get task result.
        
        Args:
            task_id: Task identifier
            timeout: Timeout in seconds
            
        Returns:
            Task result or None if not completed
        """
        start_time = time.time()
        
        while True:
            with self._task_lock:
                if task_id in self._completed_tasks:
                    return self._completed_tasks[task_id]
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.1)
    
    def wait_for_all(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for all running tasks to complete."""
        start_time = time.time()
        
        while True:
            with self._task_lock:
                if not self._running_tasks:
                    return self._completed_tasks.copy()
            
            if timeout and (time.time() - start_time) > timeout:
                break
            
            time.sleep(0.1)
        
        return self._completed_tasks.copy()
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel running task."""
        with self._task_lock:
            if task_id in self._running_tasks:
                # Note: Actual cancellation depends on the executor
                logger.info(f"Attempting to cancel task {task_id}")
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        with self._task_lock:
            avg_execution_time = (self._total_execution_time / max(self._completed_count, 1))
            
            stats = {
                'total_tasks': self._total_tasks,
                'running_tasks': len(self._running_tasks),
                'completed_tasks': self._completed_count,
                'failed_tasks': self._failed_count,
                'success_rate': self._completed_count / max(self._total_tasks, 1),
                'avg_execution_time': avg_execution_time,
                'total_execution_time': self._total_execution_time,
                'max_workers': self.max_workers,
                'gpu_enabled': self.enable_gpu,
                'multiprocessing_enabled': self.enable_multiprocessing
            }
            
            # Add resource stats
            stats.update(self.resource_manager.get_resource_stats())
            
            return stats
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor."""
        logger.info("Shutting down concurrent executor")
        
        if self._thread_executor:
            self._thread_executor.shutdown(wait=wait)
        
        if self._process_executor:
            self._process_executor.shutdown(wait=wait)


# Global executor instance
_executor: Optional[ConcurrentExecutor] = None


def get_executor() -> ConcurrentExecutor:
    """Get global concurrent executor."""
    global _executor
    if _executor is None:
        _executor = ConcurrentExecutor()
    return _executor


def parallel_map(
    function: Callable,
    items: List[Any],
    max_workers: Optional[int] = None,
    use_gpu: bool = False,
    chunk_size: Optional[int] = None
) -> List[Any]:
    """
    Apply function to items in parallel.
    
    Args:
        function: Function to apply
        items: List of items to process
        max_workers: Maximum number of workers
        use_gpu: Whether to use GPU acceleration
        chunk_size: Size of work chunks
        
    Returns:
        List of results in original order
    """
    if not items:
        return []
    
    executor = get_executor()
    
    # Submit tasks
    task_ids = []
    for i, item in enumerate(items):
        task_id = f"parallel_map_{i}"
        executor.submit_task(
            task_id,
            function,
            item,
            gpu_required=use_gpu
        )
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        result = executor.get_result(task_id)
        if result and result.success:
            results.append(result.result)
        else:
            error = result.error if result else Exception("Task failed")
            raise error
    
    return results


def parallel_field_computation(
    field_function: Callable,
    parameter_sets: List[Dict[str, Any]],
    use_gpu: bool = True
) -> List[np.ndarray]:
    """
    Compute multiple acoustic fields in parallel.
    
    Args:
        field_function: Field computation function
        parameter_sets: List of parameter dictionaries
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        List of computed fields
    """
    def compute_field(params):
        return field_function(**params)
    
    return parallel_map(
        compute_field,
        parameter_sets,
        use_gpu=use_gpu
    )


def parallel_optimization(
    optimization_function: Callable,
    targets: List[np.ndarray],
    **optimization_kwargs
) -> List[Dict[str, Any]]:
    """
    Run multiple optimizations in parallel.
    
    Args:
        optimization_function: Optimization function
        targets: List of target fields
        **optimization_kwargs: Common optimization parameters
        
    Returns:
        List of optimization results
    """
    def optimize_target(target):
        return optimization_function(target, **optimization_kwargs)
    
    return parallel_map(
        optimize_target,
        targets,
        use_gpu=True
    )


async def async_field_computation(
    field_function: Callable,
    parameters: Dict[str, Any]
) -> np.ndarray:
    """
    Asynchronous field computation.
    
    Args:
        field_function: Field computation function
        parameters: Function parameters
        
    Returns:
        Computed field
    """
    loop = asyncio.get_event_loop()
    
    # Run in thread pool
    result = await loop.run_in_executor(
        None,  # Use default executor
        field_function,
        **parameters
    )
    
    return result


def cleanup_executor():
    """Clean up global executor."""
    global _executor
    if _executor:
        _executor.shutdown(wait=True)
        _executor = None