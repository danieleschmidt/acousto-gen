"""
Distributed Computing Engine
Generation 3: MAKE IT SCALE - High-performance distributed optimization for acoustic holography.
"""

import numpy as np
import time
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import queue
import socket
import struct
import pickle
import logging
from functools import wraps
import psutil
import uuid

# Configure distributed computing logger
logger = logging.getLogger('acousto_gen.distributed')
logger.setLevel(logging.INFO)


class NodeType(Enum):
    """Types of nodes in the distributed system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    HYBRID = "hybrid"


class TaskType(Enum):
    """Types of distributed tasks."""
    OPTIMIZATION = "optimization"
    FIELD_COMPUTATION = "field_computation"
    GRADIENT_ESTIMATION = "gradient_estimation"
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    node_type: NodeType
    host: str
    port: int
    cpu_cores: int
    memory_gb: float
    gpu_available: bool = False
    status: str = "idle"
    last_heartbeat: float = field(default_factory=time.time)
    current_tasks: List[str] = field(default_factory=list)
    performance_score: float = 1.0
    load_factor: float = 0.0
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())


@dataclass
class DistributedTask:
    """Represents a task to be executed in the distributed system."""
    task_id: str
    task_type: TaskType
    function_name: str
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    priority: int = 1
    estimated_duration: float = 60.0
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class DistributedResult:
    """Result from distributed computation."""
    task_id: str
    success: bool
    result: Any
    execution_time: float
    node_id: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Intelligent task scheduler for distributed optimization."""
    
    def __init__(self):
        self.pending_tasks = queue.PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_dependencies = {}
        self.lock = threading.RLock()
        
    def schedule_task(self, task: DistributedTask, nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Schedule a task to the best available node."""
        with self.lock:
            # Check dependencies
            if not self._dependencies_satisfied(task):
                self.pending_tasks.put((task.priority, task.created_at, task))
                return None
            
            # Find best node for task
            available_nodes = [
                node for node in nodes 
                if node.status in ["idle", "busy"] and len(node.current_tasks) < node.cpu_cores
            ]
            
            if not available_nodes:
                self.pending_tasks.put((task.priority, task.created_at, task))
                return None
            
            # Score nodes based on suitability
            best_node = self._select_best_node(task, available_nodes)
            
            # Assign task
            task.assigned_node = best_node.node_id
            task.status = TaskStatus.ASSIGNED
            best_node.current_tasks.append(task.task_id)
            self.running_tasks[task.task_id] = task
            
            logger.info(f"Scheduled task {task.task_id} to node {best_node.node_id}")
            return best_node
    
    def _dependencies_satisfied(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _select_best_node(self, task: DistributedTask, nodes: List[ComputeNode]) -> ComputeNode:
        """Select the best node for a task using intelligent scoring."""
        scores = []
        
        for node in nodes:
            score = 0.0
            
            # Performance score (higher is better)
            score += node.performance_score * 0.4
            
            # Load factor (lower is better)
            score += (1.0 - node.load_factor) * 0.3
            
            # Task type affinity
            if task.task_type == TaskType.FIELD_COMPUTATION and node.gpu_available:
                score += 0.2  # GPU helps with field computation
            
            # Resource availability
            cpu_utilization = len(node.current_tasks) / max(node.cpu_cores, 1)
            score += (1.0 - cpu_utilization) * 0.1
            
            scores.append((score, node))
        
        # Return node with highest score
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]
    
    def complete_task(self, task_id: str, result: Any, execution_time: float):
        """Mark task as completed."""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result
                
                self.completed_tasks[task_id] = task
                
                # Update node status
                if task.assigned_node:
                    for node in []:  # Would access nodes from coordinator
                        if node.node_id == task.assigned_node:
                            if task_id in node.current_tasks:
                                node.current_tasks.remove(task_id)
                            break
                
                logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
    
    def fail_task(self, task_id: str, error: str):
        """Mark task as failed."""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
                task.status = TaskStatus.FAILED
                task.error = error
                
                self.failed_tasks[task_id] = task
                logger.error(f"Task {task_id} failed: {error}")
    
    def get_pending_tasks(self) -> List[DistributedTask]:
        """Get all pending tasks."""
        with self.lock:
            tasks = []
            temp_queue = queue.PriorityQueue()
            
            while not self.pending_tasks.empty():
                item = self.pending_tasks.get()
                tasks.append(item[2])  # Get task from priority tuple
                temp_queue.put(item)
            
            # Restore queue
            self.pending_tasks = temp_queue
            return tasks


class WorkerNode:
    """Worker node for distributed computation."""
    
    def __init__(self, host: str = "localhost", port: int = 0):
        # Node information
        self.node_info = ComputeNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.WORKER,
            host=host,
            port=port,
            cpu_cores=multiprocessing.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            gpu_available=self._detect_gpu()
        )
        
        # Task execution
        self.executor = ProcessPoolExecutor(max_workers=self.node_info.cpu_cores)
        self.current_tasks = {}
        self.task_queue = asyncio.Queue()
        
        # Communication
        self.coordinator_host = None
        self.coordinator_port = None
        self.running = False
        
        # Performance tracking
        self.execution_times = {}
        self.success_count = 0
        self.failure_count = 0
        
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import cupy
                return True
            except ImportError:
                return False
    
    async def start(self, coordinator_host: str, coordinator_port: int):
        """Start worker node."""
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.running = True
        
        logger.info(f"Starting worker node {self.node_info.node_id}")
        
        # Register with coordinator
        await self._register_with_coordinator()
        
        # Start task processing loop
        asyncio.create_task(self._process_tasks())
        
        # Start heartbeat
        asyncio.create_task(self._send_heartbeats())
    
    async def stop(self):
        """Stop worker node."""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info(f"Stopped worker node {self.node_info.node_id}")
    
    async def _register_with_coordinator(self):
        """Register this worker with the coordinator."""
        # In a real implementation, this would send registration message
        logger.info(f"Registered with coordinator at {self.coordinator_host}:{self.coordinator_port}")
    
    async def _send_heartbeats(self):
        """Send periodic heartbeats to coordinator."""
        while self.running:
            try:
                # Update node status
                self.node_info.last_heartbeat = time.time()
                self.node_info.load_factor = len(self.current_tasks) / self.node_info.cpu_cores
                
                # Send heartbeat (would use actual network communication)
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(60)
    
    async def _process_tasks(self):
        """Process tasks from the task queue."""
        while self.running:
            try:
                # Wait for task
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Execute task
                await self._execute_task(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Task processing error: {e}")
    
    async def _execute_task(self, task: DistributedTask):
        """Execute a distributed task."""
        start_time = time.time()
        
        try:
            logger.info(f"Executing task {task.task_id} ({task.task_type.value})")
            
            # Get task function
            function = self._get_task_function(task.function_name)
            
            # Execute in process pool
            future = self.executor.submit(function, **task.parameters)
            result = await asyncio.wrap_future(future)
            
            execution_time = time.time() - start_time
            
            # Create result
            dist_result = DistributedResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                node_id=self.node_info.node_id
            )
            
            # Update performance metrics
            self.success_count += 1
            self.execution_times[task.task_type.value] = execution_time
            
            # Update performance score
            self._update_performance_score(execution_time, task.estimated_duration)
            
            # Send result back to coordinator
            await self._send_result(dist_result)
            
            logger.info(f"Task {task.task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create error result
            dist_result = DistributedResult(
                task_id=task.task_id,
                success=False,
                result=None,
                execution_time=execution_time,
                node_id=self.node_info.node_id,
                error_message=str(e)
            )
            
            self.failure_count += 1
            
            # Send error result
            await self._send_result(dist_result)
            
            logger.error(f"Task {task.task_id} failed: {e}")
    
    def _get_task_function(self, function_name: str) -> Callable:
        """Get the function to execute for a task."""
        # Registry of available functions
        functions = {
            'optimize_phases': self._optimize_phases_task,
            'compute_field': self._compute_field_task,
            'estimate_gradient': self._estimate_gradient_task,
            'validate_result': self._validate_result_task,
            'preprocess_data': self._preprocess_data_task
        }
        
        if function_name not in functions:
            raise ValueError(f"Unknown function: {function_name}")
        
        return functions[function_name]
    
    def _optimize_phases_task(self, phases: np.ndarray, target_field: np.ndarray, 
                             iterations: int = 100, **kwargs) -> Dict[str, Any]:
        """Distributed optimization task."""
        # Simple gradient descent optimization
        learning_rate = kwargs.get('learning_rate', 0.01)
        current_phases = phases.copy()
        
        for i in range(iterations):
            # Mock field computation
            field = np.random.random(target_field.shape) * np.mean(np.abs(current_phases))
            
            # Mock gradient computation
            gradient = np.random.normal(0, 0.1, len(current_phases))
            
            # Update phases
            current_phases -= learning_rate * gradient
            
            # Early stopping
            loss = np.mean(np.abs(field - target_field)**2)
            if loss < 1e-6:
                break
        
        return {
            'phases': current_phases,
            'final_loss': loss,
            'iterations': i + 1
        }
    
    def _compute_field_task(self, phases: np.ndarray, positions: np.ndarray, 
                           frequency: float = 40000.0, **kwargs) -> np.ndarray:
        """Distributed field computation task."""
        # Mock acoustic field computation
        num_points = len(positions)
        field = np.zeros(num_points, dtype=complex)
        
        for i, pos in enumerate(positions):
            # Simplified acoustic propagation
            distances = np.sqrt(np.sum((pos - np.array([0, 0, 0]))**2))
            phase_contrib = np.sum(np.exp(1j * (phases + 2 * np.pi * frequency * distances / 343.0)))
            field[i] = phase_contrib / len(phases)
        
        return field
    
    def _estimate_gradient_task(self, phases: np.ndarray, target_field: np.ndarray,
                               epsilon: float = 1e-6, **kwargs) -> np.ndarray:
        """Distributed gradient estimation task."""
        gradient = np.zeros_like(phases)
        
        # Finite difference gradient estimation
        for i in range(len(phases)):
            phases_plus = phases.copy()
            phases_plus[i] += epsilon
            
            # Mock field computation
            field_plus = np.random.random(target_field.shape) * np.mean(np.abs(phases_plus))
            field_base = np.random.random(target_field.shape) * np.mean(np.abs(phases))
            
            # Loss gradient
            loss_plus = np.mean(np.abs(field_plus - target_field)**2)
            loss_base = np.mean(np.abs(field_base - target_field)**2)
            
            gradient[i] = (loss_plus - loss_base) / epsilon
        
        return gradient
    
    def _validate_result_task(self, phases: np.ndarray, target_field: np.ndarray,
                             **kwargs) -> Dict[str, Any]:
        """Distributed validation task."""
        # Mock validation
        field = np.random.random(target_field.shape) * np.mean(np.abs(phases))
        
        loss = np.mean(np.abs(field - target_field)**2)
        correlation = np.corrcoef(field.flatten(), target_field.flatten())[0, 1]
        
        return {
            'loss': loss,
            'correlation': correlation,
            'valid': loss < 0.1 and correlation > 0.8
        }
    
    def _preprocess_data_task(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Distributed preprocessing task."""
        # Mock preprocessing
        processed = data.copy()
        
        # Normalization
        processed = (processed - np.mean(processed)) / np.std(processed)
        
        # Filtering
        if len(processed.shape) > 1:
            processed = np.abs(processed)  # Magnitude only
        
        return processed
    
    def _update_performance_score(self, actual_time: float, estimated_time: float):
        """Update node performance score based on execution efficiency."""
        if estimated_time > 0:
            efficiency = estimated_time / actual_time
            # Update running average
            alpha = 0.1  # Learning rate
            self.node_info.performance_score = (
                (1 - alpha) * self.node_info.performance_score + 
                alpha * min(2.0, max(0.1, efficiency))
            )
    
    async def _send_result(self, result: DistributedResult):
        """Send task result back to coordinator."""
        # In real implementation, would send over network
        logger.info(f"Sending result for task {result.task_id}")
    
    def add_task(self, task: DistributedTask):
        """Add task to execution queue."""
        asyncio.create_task(self.task_queue.put(task))


class CoordinatorNode:
    """Coordinator node for distributed optimization."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        
        # Node management
        self.worker_nodes = {}
        self.scheduler = TaskScheduler()
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        
        # Performance monitoring
        self.system_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'throughput': 0.0,
            'node_utilization': 0.0
        }
        
        # Communication
        self.running = False
        
    async def start(self):
        """Start coordinator node."""
        self.running = True
        logger.info(f"Starting coordinator node at {self.host}:{self.port}")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_nodes())
        asyncio.create_task(self._monitor_performance())
        asyncio.create_task(self._schedule_pending_tasks())
    
    async def stop(self):
        """Stop coordinator node."""
        self.running = False
        logger.info("Stopping coordinator node")
    
    def register_worker(self, worker_node: ComputeNode):
        """Register a worker node."""
        self.worker_nodes[worker_node.node_id] = worker_node
        logger.info(f"Registered worker {worker_node.node_id}")
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed execution."""
        self.active_tasks[task.task_id] = task
        self.system_metrics['total_tasks'] += 1
        
        # Try to schedule immediately
        available_nodes = list(self.worker_nodes.values())
        scheduled_node = self.scheduler.schedule_task(task, available_nodes)
        
        if scheduled_node:
            logger.info(f"Task {task.task_id} scheduled to {scheduled_node.node_id}")
        else:
            logger.info(f"Task {task.task_id} queued for later scheduling")
        
        return task.task_id
    
    async def submit_optimization(self, phases: np.ndarray, target_field: np.ndarray,
                                 iterations: int = 1000, parallel_workers: int = 4) -> Dict[str, Any]:
        """Submit a distributed optimization job."""
        start_time = time.time()
        
        # Split work across workers
        iterations_per_worker = iterations // parallel_workers
        
        # Create distributed tasks
        task_ids = []
        for i in range(parallel_workers):
            task = DistributedTask(
                task_id=f"opt_{uuid.uuid4()}",
                task_type=TaskType.OPTIMIZATION,
                function_name='optimize_phases',
                parameters={
                    'phases': phases,
                    'target_field': target_field,
                    'iterations': iterations_per_worker,
                    'learning_rate': 0.01
                },
                priority=1,
                estimated_duration=iterations_per_worker * 0.01
            )
            
            task_id = self.submit_task(task)
            task_ids.append(task_id)
        
        # Wait for completion
        results = await self._wait_for_tasks(task_ids)
        
        # Combine results (select best)
        best_result = min(results, key=lambda r: r.get('final_loss', float('inf')))
        
        total_time = time.time() - start_time
        
        return {
            'phases': best_result['phases'],
            'final_loss': best_result['final_loss'],
            'total_time': total_time,
            'distributed_workers': parallel_workers,
            'individual_results': results
        }
    
    async def _wait_for_tasks(self, task_ids: List[str], timeout: float = 300.0) -> List[Any]:
        """Wait for a set of tasks to complete."""
        start_time = time.time()
        results = []
        
        while len(results) < len(task_ids) and (time.time() - start_time) < timeout:
            for task_id in task_ids:
                if task_id in self.completed_tasks:
                    task = self.completed_tasks[task_id]
                    if task.result not in results:
                        results.append(task.result)
            
            await asyncio.sleep(0.1)
        
        return results
    
    async def _monitor_nodes(self):
        """Monitor worker node health."""
        while self.running:
            try:
                current_time = time.time()
                
                # Check node heartbeats
                dead_nodes = []
                for node_id, node in self.worker_nodes.items():
                    if current_time - node.last_heartbeat > 120:  # 2 minutes timeout
                        dead_nodes.append(node_id)
                        logger.warning(f"Node {node_id} appears to be dead")
                
                # Remove dead nodes
                for node_id in dead_nodes:
                    del self.worker_nodes[node_id]
                    
                    # Reschedule their tasks
                    for task in self.active_tasks.values():
                        if task.assigned_node == node_id and task.status == TaskStatus.RUNNING:
                            task.status = TaskStatus.PENDING
                            task.assigned_node = None
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Node monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _monitor_performance(self):
        """Monitor system performance metrics."""
        while self.running:
            try:
                # Calculate metrics
                if self.system_metrics['total_tasks'] > 0:
                    success_rate = (
                        self.system_metrics['completed_tasks'] / 
                        self.system_metrics['total_tasks']
                    )
                else:
                    success_rate = 0.0
                
                # Node utilization
                if self.worker_nodes:
                    total_capacity = sum(node.cpu_cores for node in self.worker_nodes.values())
                    total_load = sum(len(node.current_tasks) for node in self.worker_nodes.values())
                    self.system_metrics['node_utilization'] = total_load / max(total_capacity, 1)
                
                # Throughput (tasks per minute)
                self.system_metrics['throughput'] = self.system_metrics['completed_tasks'] / max(1, time.time() / 60)
                
                logger.info(f"Performance: {success_rate:.2%} success, "
                           f"{self.system_metrics['node_utilization']:.1%} utilization, "
                           f"{self.system_metrics['throughput']:.1f} tasks/min")
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _schedule_pending_tasks(self):
        """Continuously schedule pending tasks."""
        while self.running:
            try:
                pending_tasks = self.scheduler.get_pending_tasks()
                available_nodes = [
                    node for node in self.worker_nodes.values()
                    if len(node.current_tasks) < node.cpu_cores
                ]
                
                for task in pending_tasks[:len(available_nodes)]:
                    scheduled_node = self.scheduler.schedule_task(task, available_nodes)
                    if scheduled_node:
                        # Send task to worker (in real implementation)
                        logger.info(f"Scheduled pending task {task.task_id}")
                
                await asyncio.sleep(10)  # Schedule every 10 seconds
                
            except Exception as e:
                logger.error(f"Task scheduling error: {e}")
                await asyncio.sleep(30)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'coordinator': {
                'host': self.host,
                'port': self.port,
                'running': self.running
            },
            'workers': {
                'total_nodes': len(self.worker_nodes),
                'active_nodes': len([n for n in self.worker_nodes.values() if n.status != 'dead']),
                'total_cpu_cores': sum(n.cpu_cores for n in self.worker_nodes.values()),
                'total_memory_gb': sum(n.memory_gb for n in self.worker_nodes.values()),
                'gpu_nodes': len([n for n in self.worker_nodes.values() if n.gpu_available])
            },
            'tasks': {
                'active_tasks': len(self.active_tasks),
                'pending_tasks': self.scheduler.pending_tasks.qsize(),
                'completed_tasks': len(self.completed_tasks)
            },
            'performance': self.system_metrics
        }


class DistributedOptimizer:
    """High-level interface for distributed acoustic optimization."""
    
    def __init__(self, coordinator_host: str = "localhost", coordinator_port: int = 8000):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.coordinator = None
        self.workers = []
        
    async def start_cluster(self, num_workers: int = None):
        """Start distributed computing cluster."""
        if num_workers is None:
            num_workers = min(8, multiprocessing.cpu_count())
        
        # Start coordinator
        self.coordinator = CoordinatorNode(self.coordinator_host, self.coordinator_port)
        await self.coordinator.start()
        
        # Start workers
        for i in range(num_workers):
            worker = WorkerNode()
            await worker.start(self.coordinator_host, self.coordinator_port)
            
            # Register with coordinator
            self.coordinator.register_worker(worker.node_info)
            self.workers.append(worker)
        
        logger.info(f"Started distributed cluster with {num_workers} workers")
    
    async def stop_cluster(self):
        """Stop distributed computing cluster."""
        # Stop workers
        for worker in self.workers:
            await worker.stop()
        
        # Stop coordinator
        if self.coordinator:
            await self.coordinator.stop()
        
        logger.info("Stopped distributed cluster")
    
    async def optimize_distributed(self, phases: np.ndarray, target_field: np.ndarray,
                                  iterations: int = 1000, parallel_workers: int = None) -> Dict[str, Any]:
        """Run distributed optimization."""
        if not self.coordinator:
            raise RuntimeError("Cluster not started")
        
        if parallel_workers is None:
            parallel_workers = min(len(self.workers), 4)
        
        logger.info(f"Starting distributed optimization with {parallel_workers} workers")
        
        result = await self.coordinator.submit_optimization(
            phases, target_field, iterations, parallel_workers
        )
        
        logger.info(f"Distributed optimization completed in {result['total_time']:.2f}s")
        return result
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status information."""
        if not self.coordinator:
            return {'status': 'cluster_not_started'}
        
        return self.coordinator.get_system_status()


# Decorators for distributed execution
def distributed_task(task_type: TaskType, estimated_duration: float = 60.0):
    """Decorator to mark functions for distributed execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Could automatically submit to distributed system
            return func(*args, **kwargs)
        
        wrapper._distributed_task = True
        wrapper._task_type = task_type
        wrapper._estimated_duration = estimated_duration
        return wrapper
    return decorator


# Example usage
async def demonstrate_distributed_computing():
    """Demonstrate distributed computing capabilities."""
    print("🌐 Distributed Computing Engine Demonstration")
    print("=" * 60)
    
    # Create distributed optimizer
    optimizer = DistributedOptimizer()
    
    try:
        # Start cluster
        print("🚀 Starting distributed cluster...")
        await optimizer.start_cluster(num_workers=4)
        
        # Create test problem
        phases = np.random.uniform(-np.pi, np.pi, 256)
        target_field = np.random.random((32, 32, 32))
        
        print("🎯 Running distributed optimization...")
        
        # Run distributed optimization
        result = await optimizer.optimize_distributed(
            phases=phases,
            target_field=target_field,
            iterations=1000,
            parallel_workers=4
        )
        
        print(f"✅ Optimization completed!")
        print(f"   Final loss: {result['final_loss']:.8f}")
        print(f"   Total time: {result['total_time']:.2f}s")
        print(f"   Workers used: {result['distributed_workers']}")
        
        # Get cluster status
        status = optimizer.get_cluster_status()
        print(f"\n📊 Cluster Status:")
        print(f"   Total nodes: {status['workers']['total_nodes']}")
        print(f"   Total CPU cores: {status['workers']['total_cpu_cores']}")
        print(f"   Completed tasks: {status['tasks']['completed_tasks']}")
        
    finally:
        # Stop cluster
        print("\n🛑 Stopping distributed cluster...")
        await optimizer.stop_cluster()
    
    print("\n" + "=" * 60)
    return optimizer


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_distributed_computing())