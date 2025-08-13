"""
Distributed Optimization Engine
Implements scalable, high-performance optimization across multiple workers and nodes.
"""

import asyncio
import logging
import time
import uuid
import json
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from queue import Queue
import multiprocessing as mp
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Distributed optimization strategies."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PARAMETER_SERVER = "parameter_server"
    FEDERATED = "federated"
    ASYNCHRONOUS = "asynchronous"
    SYNCHRONOUS = "synchronous"


class WorkerStatus(Enum):
    """Worker node status."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class OptimizationTask:
    """Individual optimization task."""
    task_id: str
    target_field: np.ndarray
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: float = 300.0
    retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_worker: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class WorkerNode:
    """Distributed worker node."""
    worker_id: str
    host: str
    port: int
    capabilities: Dict[str, Any]
    status: WorkerStatus = WorkerStatus.IDLE
    current_task: Optional[str] = None
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_task_time: float = 0.0
    last_heartbeat: Optional[datetime] = None
    resources: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Optimization result with performance metrics."""
    task_id: str
    success: bool
    phases: Optional[np.ndarray] = None
    final_loss: float = float('inf')
    iterations: int = 0
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    convergence_history: List[float] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedTaskScheduler:
    """Intelligent task scheduler for distributed optimization."""
    
    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.priority_queue = asyncio.PriorityQueue()
        self.completed_tasks: Dict[str, OptimizationResult] = {}
        self.active_tasks: Dict[str, OptimizationTask] = {}
        self.worker_assignments: Dict[str, str] = {}  # task_id -> worker_id
        
        # Scheduling policies
        self.load_balancing_strategy = "round_robin"  # round_robin, least_loaded, capability_based
        self.task_timeout = 300.0
        self.max_retries = 3
        
    async def submit_task(self, task: OptimizationTask) -> str:
        """Submit a task for distributed execution."""
        
        self.active_tasks[task.task_id] = task
        
        # Add to appropriate queue based on priority
        if task.priority > 5:
            await self.priority_queue.put((task.priority, task))
        else:
            await self.task_queue.put(task)
        
        logger.info(f"Task {task.task_id} submitted with priority {task.priority}")
        return task.task_id
    
    async def get_next_task(self, worker_capabilities: Dict[str, Any]) -> Optional[OptimizationTask]:
        """Get the next task for a worker based on capabilities."""
        
        # Try priority queue first
        try:
            if not self.priority_queue.empty():
                priority, task = await asyncio.wait_for(self.priority_queue.get(), timeout=0.1)
                if self._task_compatible_with_worker(task, worker_capabilities):
                    return task
                else:
                    # Put back if not compatible
                    await self.priority_queue.put((priority, task))
        except asyncio.TimeoutError:
            pass
        
        # Try regular queue
        try:
            task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
            if self._task_compatible_with_worker(task, worker_capabilities):
                return task
            else:
                # Put back if not compatible
                await self.task_queue.put(task)
        except asyncio.TimeoutError:
            pass
        
        return None
    
    def _task_compatible_with_worker(self, task: OptimizationTask, worker_capabilities: Dict[str, Any]) -> bool:
        """Check if a task is compatible with worker capabilities."""
        
        # Check if worker has required capabilities
        required_memory = task.parameters.get("required_memory", 1024)  # MB
        required_compute = task.parameters.get("required_compute", "cpu")
        
        worker_memory = worker_capabilities.get("memory", 0)
        worker_compute = worker_capabilities.get("compute_types", ["cpu"])
        
        if worker_memory < required_memory:
            return False
        
        if required_compute not in worker_compute:
            return False
        
        return True
    
    async def mark_task_completed(self, task_id: str, result: OptimizationResult):
        """Mark a task as completed."""
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.completed_at = datetime.now(timezone.utc)
            task.result = result
            
            self.completed_tasks[task_id] = result
            del self.active_tasks[task_id]
            
            if task_id in self.worker_assignments:
                del self.worker_assignments[task_id]
            
            logger.info(f"Task {task_id} completed successfully")
    
    async def mark_task_failed(self, task_id: str, error: str):
        """Mark a task as failed and potentially retry."""
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.error = error
            task.retries -= 1
            
            if task.retries > 0:
                logger.warning(f"Task {task_id} failed, retrying ({task.retries} attempts left)")
                # Reset task for retry
                task.assigned_worker = None
                task.started_at = None
                await self.task_queue.put(task)
            else:
                logger.error(f"Task {task_id} failed permanently: {error}")
                del self.active_tasks[task_id]
                
                if task_id in self.worker_assignments:
                    del self.worker_assignments[task_id]


class DistributedOptimizationEngine:
    """Main distributed optimization engine."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.scheduler = DistributedTaskScheduler()
        self.optimization_strategy = OptimizationStrategy.DATA_PARALLEL
        
        # Performance tracking
        self.throughput_history: List[Tuple[datetime, float]] = []
        self.latency_history: List[Tuple[datetime, float]] = []
        
        # Resource management
        self.resource_monitor = ResourceMonitor()
        self.auto_scaler = AutoScaler(self)
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.heartbeat_interval = 30.0  # seconds
        
        # Process pools for local computation
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        self.thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count() * 2)
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_latency": 0.0,
            "current_throughput": 0.0,
            "total_workers": 0,
            "active_workers": 0
        }
    
    async def register_worker(self, worker_node: WorkerNode) -> bool:
        """Register a new worker node."""
        
        try:
            # Validate worker capabilities
            if not self._validate_worker_capabilities(worker_node):
                logger.error(f"Worker {worker_node.worker_id} has invalid capabilities")
                return False
            
            self.worker_nodes[worker_node.worker_id] = worker_node
            worker_node.last_heartbeat = datetime.now(timezone.utc)
            
            logger.info(f"Worker {worker_node.worker_id} registered successfully")
            
            # Update statistics
            self.stats["total_workers"] = len(self.worker_nodes)
            self._update_active_workers()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register worker {worker_node.worker_id}: {e}")
            return False
    
    def _validate_worker_capabilities(self, worker_node: WorkerNode) -> bool:
        """Validate worker node capabilities."""
        
        required_capabilities = ["memory", "cpu_cores", "compute_types"]
        
        for capability in required_capabilities:
            if capability not in worker_node.capabilities:
                return False
        
        # Validate resource constraints
        memory = worker_node.capabilities.get("memory", 0)
        cpu_cores = worker_node.capabilities.get("cpu_cores", 0)
        
        if memory < 512 or cpu_cores < 1:  # Minimum requirements
            return False
        
        return True
    
    async def submit_optimization_task(
        self,
        target_field: np.ndarray,
        optimization_parameters: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """Submit an optimization task for distributed execution."""
        
        task_id = str(uuid.uuid4())
        
        task = OptimizationTask(
            task_id=task_id,
            target_field=target_field,
            parameters=optimization_parameters,
            priority=priority
        )
        
        await self.scheduler.submit_task(task)
        self.stats["total_tasks"] += 1
        
        logger.info(f"Optimization task {task_id} submitted")
        return task_id
    
    async def get_optimization_result(self, task_id: str, timeout: float = None) -> OptimizationResult:
        """Get the result of an optimization task."""
        
        start_time = time.time()
        timeout = timeout or 300.0
        
        while time.time() - start_time < timeout:
            if task_id in self.scheduler.completed_tasks:
                return self.scheduler.completed_tasks[task_id]
            
            await asyncio.sleep(1.0)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    async def start_distributed_processing(self):
        """Start the distributed processing system."""
        
        logger.info("Starting distributed optimization engine")
        
        # Start worker management tasks
        asyncio.create_task(self._worker_heartbeat_monitor())
        asyncio.create_task(self._task_distribution_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._auto_scaling_loop())
        
        # Start resource monitoring
        await self.resource_monitor.start()
        
        logger.info("Distributed processing started successfully")
    
    async def stop_distributed_processing(self):
        """Stop the distributed processing system."""
        
        logger.info("Stopping distributed optimization engine")
        
        # Shutdown process pools
        self.process_pool.shutdown(wait=True)
        self.thread_pool.shutdown(wait=True)
        
        # Stop resource monitoring
        await self.resource_monitor.stop()
        
        logger.info("Distributed processing stopped")
    
    async def _worker_heartbeat_monitor(self):
        """Monitor worker heartbeats and manage worker status."""
        
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                for worker_id, worker in self.worker_nodes.items():
                    if worker.last_heartbeat:
                        time_since_heartbeat = (current_time - worker.last_heartbeat).total_seconds()
                        
                        if time_since_heartbeat > self.heartbeat_interval * 3:  # 3x heartbeat interval
                            if worker.status != WorkerStatus.DISCONNECTED:
                                logger.warning(f"Worker {worker_id} appears disconnected")
                                worker.status = WorkerStatus.DISCONNECTED
                                
                                # Reassign tasks if worker was busy
                                if worker.current_task:
                                    await self._reassign_task(worker.current_task)
                
                self._update_active_workers()
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(60)
    
    async def _task_distribution_loop(self):
        """Main loop for distributing tasks to workers."""
        
        while True:
            try:
                # Find available workers
                available_workers = [
                    worker for worker in self.worker_nodes.values()
                    if worker.status == WorkerStatus.IDLE
                ]
                
                if not available_workers:
                    await asyncio.sleep(1.0)
                    continue
                
                # Get tasks and assign to workers
                for worker in available_workers:
                    task = await self.scheduler.get_next_task(worker.capabilities)
                    
                    if task:
                        await self._assign_task_to_worker(task, worker)
                    else:
                        break  # No more tasks available
                
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
                
            except Exception as e:
                logger.error(f"Error in task distribution loop: {e}")
                await asyncio.sleep(5)
    
    async def _assign_task_to_worker(self, task: OptimizationTask, worker: WorkerNode):
        """Assign a task to a specific worker."""
        
        try:
            worker.status = WorkerStatus.BUSY
            worker.current_task = task.task_id
            task.assigned_worker = worker.worker_id
            task.started_at = datetime.now(timezone.utc)
            
            self.scheduler.worker_assignments[task.task_id] = worker.worker_id
            
            # Execute task (in real implementation, this would send to remote worker)
            asyncio.create_task(self._execute_task_on_worker(task, worker))
            
            logger.info(f"Assigned task {task.task_id} to worker {worker.worker_id}")
            
        except Exception as e:
            logger.error(f"Failed to assign task {task.task_id} to worker {worker.worker_id}: {e}")
            worker.status = WorkerStatus.ERROR
    
    async def _execute_task_on_worker(self, task: OptimizationTask, worker: WorkerNode):
        """Execute a task on a worker (local simulation)."""
        
        start_time = time.time()
        
        try:
            # Simulate optimization execution
            if self.optimization_strategy == OptimizationStrategy.DATA_PARALLEL:
                result = await self._execute_data_parallel_optimization(task, worker)
            elif self.optimization_strategy == OptimizationStrategy.MODEL_PARALLEL:
                result = await self._execute_model_parallel_optimization(task, worker)
            else:
                result = await self._execute_standard_optimization(task, worker)
            
            execution_time = time.time() - start_time
            
            # Update worker statistics
            worker.total_tasks += 1
            worker.successful_tasks += 1
            worker.average_task_time = (
                (worker.average_task_time * (worker.total_tasks - 1) + execution_time) /
                worker.total_tasks
            )
            
            # Mark task as completed
            optimization_result = OptimizationResult(
                task_id=task.task_id,
                success=True,
                phases=result.get("phases"),
                final_loss=result.get("final_loss", 0.0),
                iterations=result.get("iterations", 0),
                execution_time=execution_time,
                worker_id=worker.worker_id,
                convergence_history=result.get("convergence_history", []),
                resource_usage=result.get("resource_usage", {}),
                metadata=result.get("metadata", {})
            )
            
            await self.scheduler.mark_task_completed(task.task_id, optimization_result)
            self.stats["completed_tasks"] += 1
            
            # Update performance metrics
            self.latency_history.append((datetime.now(timezone.utc), execution_time))
            self._update_throughput()
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed on worker {worker.worker_id}: {e}")
            
            worker.failed_tasks += 1
            worker.status = WorkerStatus.ERROR
            
            await self.scheduler.mark_task_failed(task.task_id, str(e))
            self.stats["failed_tasks"] += 1
            
        finally:
            # Reset worker status
            worker.status = WorkerStatus.IDLE
            worker.current_task = None
    
    async def _execute_data_parallel_optimization(
        self,
        task: OptimizationTask,
        worker: WorkerNode
    ) -> Dict[str, Any]:
        """Execute data-parallel optimization strategy."""
        
        # Split target field into chunks
        target_field = task.target_field
        chunk_size = max(1, target_field.size // worker.capabilities.get("cpu_cores", 1))
        
        # Simulate parallel processing
        await asyncio.sleep(0.1 * chunk_size / 1000)  # Simulate computation time
        
        # Generate mock results
        num_transducers = task.parameters.get("num_transducers", 256)
        phases = np.random.uniform(0, 2*np.pi, num_transducers)
        final_loss = np.random.exponential(0.001)
        iterations = np.random.randint(50, 500)
        
        return {
            "phases": phases,
            "final_loss": final_loss,
            "iterations": iterations,
            "convergence_history": [final_loss * (1 + i * 0.1) for i in range(iterations//10)],
            "resource_usage": {
                "cpu_utilization": np.random.uniform(70, 95),
                "memory_usage": np.random.uniform(200, 800),
                "execution_time": time.time()
            },
            "metadata": {
                "strategy": "data_parallel",
                "chunk_size": chunk_size,
                "worker_cores": worker.capabilities.get("cpu_cores", 1)
            }
        }
    
    async def _execute_model_parallel_optimization(
        self,
        task: OptimizationTask,
        worker: WorkerNode
    ) -> Dict[str, Any]:
        """Execute model-parallel optimization strategy."""
        
        # Simulate model parallelism with longer execution time
        await asyncio.sleep(0.2)
        
        num_transducers = task.parameters.get("num_transducers", 256)
        phases = np.random.uniform(0, 2*np.pi, num_transducers)
        final_loss = np.random.exponential(0.001) * 0.8  # Better convergence
        iterations = np.random.randint(30, 300)
        
        return {
            "phases": phases,
            "final_loss": final_loss,
            "iterations": iterations,
            "convergence_history": [final_loss * (1 + i * 0.05) for i in range(iterations//5)],
            "resource_usage": {
                "cpu_utilization": np.random.uniform(60, 85),
                "memory_usage": np.random.uniform(400, 1200),
                "gpu_utilization": np.random.uniform(80, 95)
            },
            "metadata": {
                "strategy": "model_parallel",
                "model_shards": worker.capabilities.get("cpu_cores", 1)
            }
        }
    
    async def _execute_standard_optimization(
        self,
        task: OptimizationTask,
        worker: WorkerNode
    ) -> Dict[str, Any]:
        """Execute standard single-worker optimization."""
        
        # Simulate standard optimization
        await asyncio.sleep(0.15)
        
        num_transducers = task.parameters.get("num_transducers", 256)
        phases = np.random.uniform(0, 2*np.pi, num_transducers)
        final_loss = np.random.exponential(0.001)
        iterations = np.random.randint(100, 800)
        
        return {
            "phases": phases,
            "final_loss": final_loss,
            "iterations": iterations,
            "convergence_history": [final_loss * (1 + i * 0.08) for i in range(iterations//8)],
            "resource_usage": {
                "cpu_utilization": np.random.uniform(50, 80),
                "memory_usage": np.random.uniform(150, 600)
            },
            "metadata": {
                "strategy": "standard",
                "single_worker": True
            }
        }
    
    async def _reassign_task(self, task_id: str):
        """Reassign a task when worker becomes unavailable."""
        
        if task_id in self.scheduler.active_tasks:
            task = self.scheduler.active_tasks[task_id]
            task.assigned_worker = None
            task.started_at = None
            
            # Put task back in queue
            await self.scheduler.task_queue.put(task)
            
            logger.info(f"Task {task_id} reassigned due to worker unavailability")
    
    async def _performance_monitoring_loop(self):
        """Monitor and update performance metrics."""
        
        while True:
            try:
                # Update throughput calculation
                self._update_throughput()
                
                # Update latency statistics
                if self.latency_history:
                    recent_latencies = [
                        latency for timestamp, latency in self.latency_history
                        if (datetime.now(timezone.utc) - timestamp).total_seconds() < 300  # Last 5 minutes
                    ]
                    
                    if recent_latencies:
                        self.stats["average_latency"] = sum(recent_latencies) / len(recent_latencies)
                
                # Log performance metrics
                if self.stats["completed_tasks"] % 10 == 0 and self.stats["completed_tasks"] > 0:
                    logger.info(f"Performance: {self.stats['current_throughput']:.2f} tasks/min, "
                              f"avg latency: {self.stats['average_latency']:.2f}s")
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    def _update_throughput(self):
        """Update throughput calculation."""
        
        current_time = datetime.now(timezone.utc)
        
        # Count tasks completed in last minute
        recent_completions = sum(
            1 for timestamp, _ in self.throughput_history
            if (current_time - timestamp).total_seconds() < 60
        )
        
        self.stats["current_throughput"] = recent_completions
        
        # Add current timestamp to history
        self.throughput_history.append((current_time, 1))
        
        # Clean old entries
        self.throughput_history = [
            (timestamp, count) for timestamp, count in self.throughput_history
            if (current_time - timestamp).total_seconds() < 300  # Keep 5 minutes
        ]
    
    def _update_active_workers(self):
        """Update count of active workers."""
        
        active_count = sum(
            1 for worker in self.worker_nodes.values()
            if worker.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]
        )
        
        self.stats["active_workers"] = active_count
    
    async def _auto_scaling_loop(self):
        """Auto-scaling loop for dynamic resource management."""
        
        while True:
            try:
                await self.auto_scaler.evaluate_scaling_decision()
                await asyncio.sleep(60)  # Evaluate every minute
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(120)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        worker_status = {}
        for worker_id, worker in self.worker_nodes.items():
            worker_status[worker_id] = {
                "status": worker.status.value,
                "current_task": worker.current_task,
                "total_tasks": worker.total_tasks,
                "success_rate": worker.successful_tasks / worker.total_tasks if worker.total_tasks > 0 else 0,
                "average_task_time": worker.average_task_time,
                "capabilities": worker.capabilities,
                "last_heartbeat": worker.last_heartbeat.isoformat() if worker.last_heartbeat else None
            }
        
        queue_status = {
            "active_tasks": len(self.scheduler.active_tasks),
            "completed_tasks": len(self.scheduler.completed_tasks),
            "queued_tasks": self.scheduler.task_queue.qsize(),
            "priority_tasks": self.scheduler.priority_queue.qsize()
        }
        
        return {
            "statistics": self.stats,
            "workers": worker_status,
            "queue_status": queue_status,
            "optimization_strategy": self.optimization_strategy.value,
            "resource_utilization": self.resource_monitor.get_current_utilization(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class ResourceMonitor:
    """Monitor system resources for optimization and scaling decisions."""
    
    def __init__(self):
        self.cpu_history: List[Tuple[datetime, float]] = []
        self.memory_history: List[Tuple[datetime, float]] = []
        self.monitoring_active = False
        
    async def start(self):
        """Start resource monitoring."""
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitoring started")
    
    async def stop(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        logger.info("Resource monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.monitoring_active:
            try:
                import psutil
                
                current_time = datetime.now(timezone.utc)
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                self.cpu_history.append((current_time, cpu_percent))
                self.memory_history.append((current_time, memory_percent))
                
                # Keep only last hour of data
                cutoff_time = current_time - timedelta(hours=1)
                self.cpu_history = [(t, v) for t, v in self.cpu_history if t > cutoff_time]
                self.memory_history = [(t, v) for t, v in self.memory_history if t > cutoff_time]
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(30)
    
    def get_current_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        
        if not self.cpu_history or not self.memory_history:
            return {"cpu": 0.0, "memory": 0.0}
        
        latest_cpu = self.cpu_history[-1][1]
        latest_memory = self.memory_history[-1][1]
        
        return {
            "cpu": latest_cpu,
            "memory": latest_memory,
            "cpu_trend": self._calculate_trend(self.cpu_history),
            "memory_trend": self._calculate_trend(self.memory_history)
        }
    
    def _calculate_trend(self, history: List[Tuple[datetime, float]]) -> float:
        """Calculate trend over the last 10 minutes."""
        
        if len(history) < 2:
            return 0.0
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        recent_data = [(t, v) for t, v in history if t > cutoff_time]
        
        if len(recent_data) < 2:
            return 0.0
        
        # Simple linear trend
        values = [v for _, v in recent_data]
        return (values[-1] - values[0]) / len(values)


class AutoScaler:
    """Automatic scaling system for worker nodes."""
    
    def __init__(self, optimization_engine: DistributedOptimizationEngine):
        self.engine = optimization_engine
        self.scaling_policies = {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "queue_length_threshold": 10,
            "scale_up_cooldown": 300,  # seconds
            "scale_down_cooldown": 600,  # seconds
            "min_workers": 1,
            "max_workers": 10
        }
        self.last_scale_action = None
        self.last_scale_time = None
    
    async def evaluate_scaling_decision(self):
        """Evaluate whether to scale up or down."""
        
        try:
            current_time = datetime.now(timezone.utc)
            
            # Check cooldown period
            if (self.last_scale_time and 
                (current_time - self.last_scale_time).total_seconds() < self.scaling_policies["scale_up_cooldown"]):
                return
            
            # Get current metrics
            resource_util = self.engine.resource_monitor.get_current_utilization()
            queue_length = self.engine.scheduler.task_queue.qsize() + self.engine.scheduler.priority_queue.qsize()
            active_workers = self.engine.stats["active_workers"]
            
            # Scaling decision logic
            should_scale_up = (
                resource_util["cpu"] > self.scaling_policies["cpu_threshold"] or
                resource_util["memory"] > self.scaling_policies["memory_threshold"] or
                queue_length > self.scaling_policies["queue_length_threshold"]
            ) and active_workers < self.scaling_policies["max_workers"]
            
            should_scale_down = (
                resource_util["cpu"] < 30 and
                resource_util["memory"] < 40 and
                queue_length == 0 and
                active_workers > self.scaling_policies["min_workers"]
            )
            
            if should_scale_up:
                await self._scale_up()
            elif should_scale_down and (
                not self.last_scale_time or
                (current_time - self.last_scale_time).total_seconds() > self.scaling_policies["scale_down_cooldown"]
            ):
                await self._scale_down()
                
        except Exception as e:
            logger.error(f"Error in auto-scaling evaluation: {e}")
    
    async def _scale_up(self):
        """Scale up by adding a new worker."""
        
        try:
            # Create a new simulated worker
            new_worker_id = f"worker_{len(self.engine.worker_nodes) + 1}"
            
            new_worker = WorkerNode(
                worker_id=new_worker_id,
                host="localhost",
                port=8000 + len(self.engine.worker_nodes),
                capabilities={
                    "memory": 2048,
                    "cpu_cores": 4,
                    "compute_types": ["cpu", "gpu"]
                }
            )
            
            success = await self.engine.register_worker(new_worker)
            
            if success:
                self.last_scale_action = "scale_up"
                self.last_scale_time = datetime.now(timezone.utc)
                logger.info(f"Scaled up: Added worker {new_worker_id}")
            
        except Exception as e:
            logger.error(f"Error scaling up: {e}")
    
    async def _scale_down(self):
        """Scale down by removing a worker."""
        
        try:
            # Find an idle worker to remove
            idle_workers = [
                worker for worker in self.engine.worker_nodes.values()
                if worker.status == WorkerStatus.IDLE
            ]
            
            if idle_workers:
                worker_to_remove = idle_workers[0]
                worker_to_remove.status = WorkerStatus.DISCONNECTED
                
                self.last_scale_action = "scale_down"
                self.last_scale_time = datetime.now(timezone.utc)
                logger.info(f"Scaled down: Removed worker {worker_to_remove.worker_id}")
            
        except Exception as e:
            logger.error(f"Error scaling down: {e}")


# Example usage and demonstration
async def demonstrate_distributed_optimization():
    """Demonstrate distributed optimization capabilities."""
    
    print("🚀 Distributed Optimization Engine Demonstration")
    print("=" * 60)
    
    # Initialize distributed engine
    engine = DistributedOptimizationEngine()
    
    # Register some worker nodes
    workers = [
        WorkerNode(
            worker_id=f"worker_{i}",
            host="localhost",
            port=8000 + i,
            capabilities={
                "memory": 1024 + i * 512,
                "cpu_cores": 2 + i,
                "compute_types": ["cpu"] + (["gpu"] if i % 2 == 0 else [])
            }
        )
        for i in range(3)
    ]
    
    for worker in workers:
        await engine.register_worker(worker)
    
    print(f"Registered {len(workers)} worker nodes")
    
    # Start distributed processing
    await engine.start_distributed_processing()
    
    # Submit some optimization tasks
    print("\n📤 Submitting optimization tasks...")
    
    task_ids = []
    for i in range(5):
        target_field = np.random.random((32, 32, 32))
        optimization_params = {
            "num_transducers": 256,
            "iterations": 1000,
            "learning_rate": 0.01,
            "required_memory": 512 + i * 128,
            "required_compute": "cpu"
        }
        
        task_id = await engine.submit_optimization_task(
            target_field=target_field,
            optimization_parameters=optimization_params,
            priority=i % 3 + 1
        )
        
        task_ids.append(task_id)
        print(f"  Task {i+1}: {task_id}")
    
    # Wait for tasks to complete
    print("\n⏳ Waiting for tasks to complete...")
    results = []
    
    for task_id in task_ids:
        try:
            result = await engine.get_optimization_result(task_id, timeout=30.0)
            results.append(result)
            print(f"  ✅ Task {task_id[:8]}... completed in {result.execution_time:.2f}s")
        except TimeoutError:
            print(f"  ⏰ Task {task_id[:8]}... timed out")
    
    # Display system status
    print("\n📊 System Status:")
    status = engine.get_system_status()
    
    print(f"  Total tasks: {status['statistics']['total_tasks']}")
    print(f"  Completed: {status['statistics']['completed_tasks']}")
    print(f"  Failed: {status['statistics']['failed_tasks']}")
    print(f"  Active workers: {status['statistics']['active_workers']}")
    print(f"  Current throughput: {status['statistics']['current_throughput']:.2f} tasks/min")
    
    print("\n👥 Worker Status:")
    for worker_id, worker_info in status['workers'].items():
        print(f"  {worker_id}: {worker_info['status']} "
              f"(success rate: {worker_info['success_rate']:.1%})")
    
    # Stop distributed processing
    await engine.stop_distributed_processing()
    
    print("\n" + "=" * 60)
    return engine, results


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_distributed_optimization())