"""
Distributed cluster management for large-scale acoustic holography computations.
Supports multi-node processing, job scheduling, and fault tolerance.
"""

import asyncio
import json
import time
import threading
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import socket
import pickle
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import psutil
import zmq
import redis
from pathlib import Path

try:
    import dask
    from dask.distributed import Client, as_completed as dask_completed
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dask = None
    Client = None
    delayed = None

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None


class NodeStatus(Enum):
    """Node status types."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class JobStatus(Enum):
    """Job status types."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    cpu_count: int
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: float
    status: NodeStatus = NodeStatus.IDLE
    current_jobs: List[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputeJob:
    """Distributed computation job."""
    job_id: str
    job_type: str
    function: str  # Serialized function
    args: bytes  # Pickled arguments
    kwargs: bytes  # Pickled keyword arguments
    priority: int = 0
    estimated_runtime: float = 0
    memory_required: float = 0
    gpu_required: bool = False
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout: float = 3600  # 1 hour default
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: JobStatus = JobStatus.PENDING
    assigned_node: Optional[str] = None
    result: Optional[bytes] = None  # Pickled result
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class ClusterStats:
    """Cluster performance statistics."""
    total_nodes: int
    active_nodes: int
    total_jobs_completed: int
    total_jobs_failed: int
    average_job_time: float
    cluster_efficiency: float
    resource_utilization: Dict[str, float]
    fault_tolerance_events: int


class DistributedScheduler:
    """
    Advanced job scheduler for distributed acoustic holography computations.
    
    Implements priority-based scheduling, load balancing, and fault tolerance
    across multiple compute nodes.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        scheduler_port: int = 9999,
        enable_fault_tolerance: bool = True
    ):
        """
        Initialize distributed scheduler.
        
        Args:
            redis_host: Redis server hostname for coordination
            redis_port: Redis server port
            scheduler_port: Port for scheduler communication
            enable_fault_tolerance: Enable automatic fault recovery
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.scheduler_port = scheduler_port
        self.enable_fault_tolerance = enable_fault_tolerance
        
        # Connection pools
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.zmq_context = zmq.Context()
        
        # Scheduler state
        self.nodes: Dict[str, NodeInfo] = {}
        self.jobs: Dict[str, ComputeJob] = {}
        self.job_queue: List[str] = []  # Job IDs in priority order
        
        # Statistics
        self.stats = ClusterStats(
            total_nodes=0,
            active_nodes=0,
            total_jobs_completed=0,
            total_jobs_failed=0,
            average_job_time=0,
            cluster_efficiency=0,
            resource_utilization={},
            fault_tolerance_events=0
        )
        
        # Threading
        self.scheduler_thread = None
        self.heartbeat_thread = None
        self.is_running = False
        
        # Fault tolerance
        self.failed_nodes: Set[str] = set()
        self.job_checkpoints: Dict[str, Dict] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        print("Distributed scheduler initialized")
    
    def start_scheduler(self) -> None:
        """Start the distributed scheduler."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # Start heartbeat monitor
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.heartbeat_thread.start()
        
        self.logger.info("Distributed scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the distributed scheduler."""
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        
        self.zmq_context.term()
        self.logger.info("Distributed scheduler stopped")
    
    def register_node(
        self,
        hostname: str,
        ip_address: str,
        port: int,
        capabilities: Set[str] = None
    ) -> str:
        """
        Register a compute node with the cluster.
        
        Args:
            hostname: Node hostname
            ip_address: Node IP address
            port: Node communication port
            capabilities: Node capabilities (e.g., 'gpu', 'high_memory')
            
        Returns:
            Node ID
        """
        node_id = f"{hostname}_{ip_address}_{port}"
        
        # Get system information
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU detection (simplified)
        gpu_count = 0
        gpu_memory_gb = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    props = torch.cuda.get_device_properties(0)
                    gpu_memory_gb = props.total_memory / (1024**3)
        except ImportError:
            pass
        
        node_info = NodeInfo(
            node_id=node_id,
            hostname=hostname,
            ip_address=ip_address,
            port=port,
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            capabilities=capabilities or set()
        )
        
        self.nodes[node_id] = node_info
        self.stats.total_nodes += 1
        self.stats.active_nodes += 1
        
        # Store in Redis for persistence
        self.redis_client.hset(
            "cluster:nodes",
            node_id,
            json.dumps(asdict(node_info), default=str)
        )
        
        self.logger.info(f"Node registered: {node_id}")
        return node_id
    
    def submit_job(
        self,
        function: Callable,
        *args,
        job_type: str = "acoustic_computation",
        priority: int = 0,
        estimated_runtime: float = 0,
        memory_required: float = 0,
        gpu_required: bool = False,
        dependencies: List[str] = None,
        **kwargs
    ) -> str:
        """
        Submit a computation job to the cluster.
        
        Args:
            function: Function to execute
            *args: Function arguments
            job_type: Type of job for scheduling
            priority: Job priority (higher = more important)
            estimated_runtime: Expected runtime in seconds
            memory_required: Memory requirement in GB
            gpu_required: Whether job needs GPU
            dependencies: List of job IDs this job depends on
            **kwargs: Function keyword arguments
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        # Serialize function and arguments
        function_bytes = pickle.dumps(function)
        args_bytes = pickle.dumps(args)
        kwargs_bytes = pickle.dumps(kwargs)
        
        job = ComputeJob(
            job_id=job_id,
            job_type=job_type,
            function=function_bytes.hex(),  # Store as hex string
            args=args_bytes,
            kwargs=kwargs_bytes,
            priority=priority,
            estimated_runtime=estimated_runtime,
            memory_required=memory_required,
            gpu_required=gpu_required,
            dependencies=dependencies or []
        )
        
        self.jobs[job_id] = job
        self._enqueue_job(job_id)
        
        # Store in Redis
        self.redis_client.hset(
            "cluster:jobs",
            job_id,
            json.dumps(asdict(job), default=str)
        )
        
        self.logger.info(f"Job submitted: {job_id} (type: {job_type}, priority: {priority})")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get status of a specific job."""
        if job_id in self.jobs:
            return self.jobs[job_id].status
        
        # Check Redis
        job_data = self.redis_client.hget("cluster:jobs", job_id)
        if job_data:
            job_dict = json.loads(job_data)
            return JobStatus(job_dict['status'])
        
        return None
    
    def get_job_result(self, job_id: str) -> Optional[Any]:
        """Get result of a completed job."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        if job.status == JobStatus.COMPLETED and job.result:
            return pickle.loads(job.result)
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            if job_id in self.job_queue:
                self.job_queue.remove(job_id)
            return True
        
        elif job.status == JobStatus.RUNNING and job.assigned_node:
            # Send cancellation signal to node
            self._send_node_command(job.assigned_node, "cancel_job", {"job_id": job_id})
            job.status = JobStatus.CANCELLED
            return True
        
        return False
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self.is_running:
            try:
                self._schedule_jobs()
                self._update_statistics()
                time.sleep(1)  # Schedule every second
                
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(5)
    
    def _schedule_jobs(self) -> None:
        """Schedule pending jobs to available nodes."""
        if not self.job_queue:
            return
        
        # Get available nodes
        available_nodes = [
            node for node in self.nodes.values()
            if node.status == NodeStatus.IDLE and node.node_id not in self.failed_nodes
        ]
        
        if not available_nodes:
            return
        
        # Sort jobs by priority
        self.job_queue.sort(key=lambda jid: self.jobs[jid].priority, reverse=True)
        
        # Schedule jobs
        scheduled_jobs = []
        
        for job_id in self.job_queue:
            job = self.jobs[job_id]
            
            # Check dependencies
            if not self._dependencies_satisfied(job):
                continue
            
            # Find suitable node
            suitable_node = self._find_suitable_node(job, available_nodes)
            
            if suitable_node:
                # Assign job to node
                self._assign_job_to_node(job, suitable_node)
                scheduled_jobs.append(job_id)
                available_nodes.remove(suitable_node)
                
                if not available_nodes:
                    break
        
        # Remove scheduled jobs from queue
        for job_id in scheduled_jobs:
            self.job_queue.remove(job_id)
    
    def _dependencies_satisfied(self, job: ComputeJob) -> bool:
        """Check if job dependencies are satisfied."""
        for dep_id in job.dependencies:
            if dep_id in self.jobs:
                if self.jobs[dep_id].status != JobStatus.COMPLETED:
                    return False
        return True
    
    def _find_suitable_node(
        self,
        job: ComputeJob,
        available_nodes: List[NodeInfo]
    ) -> Optional[NodeInfo]:
        """Find the most suitable node for a job."""
        suitable_nodes = []
        
        for node in available_nodes:
            # Check GPU requirement
            if job.gpu_required and node.gpu_count == 0:
                continue
            
            # Check memory requirement
            if job.memory_required > node.memory_gb:
                continue
            
            # Check capabilities
            job_capabilities = self._get_job_capabilities(job)
            if job_capabilities and not job_capabilities.issubset(node.capabilities):
                continue
            
            suitable_nodes.append(node)
        
        if not suitable_nodes:
            return None
        
        # Rank nodes by suitability
        def node_score(node: NodeInfo) -> float:
            # Prefer nodes with more resources
            score = node.cpu_count + node.memory_gb / 10
            
            if job.gpu_required and node.gpu_count > 0:
                score += node.gpu_memory_gb
            
            # Prefer nodes with fewer active jobs
            score -= len(node.current_jobs) * 10
            
            return score
        
        return max(suitable_nodes, key=node_score)
    
    def _get_job_capabilities(self, job: ComputeJob) -> Set[str]:
        """Get required capabilities for a job."""
        capabilities = set()
        
        if job.gpu_required:
            capabilities.add("gpu")
        
        if job.memory_required > 32:  # > 32GB
            capabilities.add("high_memory")
        
        # Add job-type specific capabilities
        if job.job_type == "neural_training":
            capabilities.add("ml")
        elif job.job_type == "wave_propagation":
            capabilities.add("scientific_computing")
        
        return capabilities
    
    def _assign_job_to_node(self, job: ComputeJob, node: NodeInfo) -> None:
        """Assign job to a specific node."""
        job.assigned_node = node.node_id
        job.started_at = time.time()
        job.status = JobStatus.RUNNING
        
        node.current_jobs.append(job.job_id)
        node.status = NodeStatus.BUSY
        
        # Send job to node
        job_data = {
            "job_id": job.job_id,
            "function": job.function,
            "args": job.args.hex() if isinstance(job.args, bytes) else job.args,
            "kwargs": job.kwargs.hex() if isinstance(job.kwargs, bytes) else job.kwargs,
            "timeout": job.timeout
        }
        
        self._send_node_command(node.node_id, "execute_job", job_data)
        
        self.logger.info(f"Job {job.job_id} assigned to node {node.node_id}")
    
    def _send_node_command(self, node_id: str, command: str, data: Dict[str, Any]) -> None:
        """Send command to a specific node."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        try:
            # Use ZeroMQ for node communication
            socket = self.zmq_context.socket(zmq.REQ)
            socket.connect(f"tcp://{node.ip_address}:{node.port}")
            
            message = {
                "command": command,
                "data": data
            }
            
            socket.send_json(message)
            
            # Wait for response (with timeout)
            if socket.poll(5000):  # 5 second timeout
                response = socket.recv_json()
                self.logger.debug(f"Node {node_id} response: {response}")
            else:
                self.logger.warning(f"No response from node {node_id}")
                self._handle_node_timeout(node_id)
            
            socket.close()
            
        except Exception as e:
            self.logger.error(f"Failed to send command to node {node_id}: {e}")
            self._handle_node_failure(node_id)
    
    def _handle_job_completion(self, job_id: str, result: bytes, success: bool) -> None:
        """Handle job completion notification from node."""
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        job.completed_at = time.time()
        
        if success:
            job.status = JobStatus.COMPLETED
            job.result = result
            self.stats.total_jobs_completed += 1
            
            # Calculate and update average job time
            runtime = job.completed_at - job.started_at
            self.stats.average_job_time = (
                (self.stats.average_job_time * (self.stats.total_jobs_completed - 1) + runtime) /
                self.stats.total_jobs_completed
            )
            
        else:
            job.status = JobStatus.FAILED
            job.error = result.decode() if result else "Unknown error"
            self.stats.total_jobs_failed += 1
            
            # Retry if possible
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.PENDING
                job.assigned_node = None
                self._enqueue_job(job_id)
                self.logger.info(f"Retrying job {job_id} (attempt {job.retry_count})")
        
        # Free up the node
        if job.assigned_node:
            node = self.nodes[job.assigned_node]
            if job_id in node.current_jobs:
                node.current_jobs.remove(job_id)
            
            if not node.current_jobs:
                node.status = NodeStatus.IDLE
        
        self.logger.info(f"Job {job_id} {'completed' if success else 'failed'}")
    
    def _enqueue_job(self, job_id: str) -> None:
        """Add job to the scheduling queue."""
        if job_id not in self.job_queue:
            self.job_queue.append(job_id)
    
    def _heartbeat_monitor(self) -> None:
        """Monitor node heartbeats and handle failures."""
        while self.is_running:
            try:
                current_time = time.time()
                timeout_threshold = 30  # 30 seconds
                
                for node_id, node in list(self.nodes.items()):
                    if current_time - node.last_heartbeat > timeout_threshold:
                        self.logger.warning(f"Node {node_id} heartbeat timeout")
                        self._handle_node_failure(node_id)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                time.sleep(10)
    
    def _handle_node_failure(self, node_id: str) -> None:
        """Handle node failure and reassign jobs."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.status = NodeStatus.OFFLINE
        self.failed_nodes.add(node_id)
        
        if node_id in self.nodes:
            self.stats.active_nodes -= 1
        
        # Reassign running jobs
        for job_id in list(node.current_jobs):
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = JobStatus.PENDING
                job.assigned_node = None
                self._enqueue_job(job_id)
                
                self.logger.info(f"Reassigning job {job_id} due to node failure")
        
        node.current_jobs.clear()
        self.stats.fault_tolerance_events += 1
        
        self.logger.error(f"Node {node_id} marked as failed")
    
    def _handle_node_timeout(self, node_id: str) -> None:
        """Handle node communication timeout."""
        self.logger.warning(f"Node {node_id} communication timeout")
        # For now, just log. Could implement more sophisticated handling
    
    def _update_statistics(self) -> None:
        """Update cluster statistics."""
        active_nodes = sum(1 for node in self.nodes.values() 
                          if node.status in [NodeStatus.IDLE, NodeStatus.BUSY])
        
        self.stats.active_nodes = active_nodes
        
        # Calculate resource utilization
        total_cpu = sum(node.cpu_count for node in self.nodes.values() if node.status != NodeStatus.OFFLINE)
        used_cpu = sum(len(node.current_jobs) for node in self.nodes.values() if node.status == NodeStatus.BUSY)
        
        if total_cpu > 0:
            self.stats.resource_utilization['cpu'] = used_cpu / total_cpu
        
        # Calculate cluster efficiency
        total_jobs = self.stats.total_jobs_completed + self.stats.total_jobs_failed
        if total_jobs > 0:
            self.stats.cluster_efficiency = self.stats.total_jobs_completed / total_jobs
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        return {
            "nodes": {
                node_id: {
                    "hostname": node.hostname,
                    "status": node.status.value,
                    "cpu_count": node.cpu_count,
                    "memory_gb": node.memory_gb,
                    "gpu_count": node.gpu_count,
                    "current_jobs": len(node.current_jobs),
                    "last_heartbeat": node.last_heartbeat
                }
                for node_id, node in self.nodes.items()
            },
            "jobs": {
                "pending": len([j for j in self.jobs.values() if j.status == JobStatus.PENDING]),
                "running": len([j for j in self.jobs.values() if j.status == JobStatus.RUNNING]),
                "completed": len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED]),
                "failed": len([j for j in self.jobs.values() if j.status == JobStatus.FAILED])
            },
            "statistics": asdict(self.stats),
            "queue_length": len(self.job_queue)
        }


class AcousticClusterClient:
    """
    Client interface for submitting acoustic holography jobs to the cluster.
    
    Provides high-level functions for distributed acoustic computations.
    """
    
    def __init__(
        self,
        scheduler_host: str = "localhost",
        scheduler_port: int = 9999,
        redis_host: str = "localhost",
        redis_port: int = 6379
    ):
        """
        Initialize cluster client.
        
        Args:
            scheduler_host: Scheduler hostname
            scheduler_port: Scheduler port
            redis_host: Redis hostname
            redis_port: Redis port
        """
        self.scheduler_host = scheduler_host
        self.scheduler_port = scheduler_port
        
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.zmq_context = zmq.Context()
        
        # High-level job tracking
        self.submitted_jobs: Dict[str, Dict[str, Any]] = {}
    
    def submit_wave_propagation(
        self,
        source_positions: np.ndarray,
        source_amplitudes: np.ndarray,
        source_phases: np.ndarray,
        target_points: np.ndarray,
        frequency: float,
        medium_properties: Dict[str, float],
        priority: int = 0,
        chunk_size: int = 1000
    ) -> List[str]:
        """
        Submit distributed wave propagation computation.
        
        Args:
            source_positions: Source positions
            source_amplitudes: Source amplitudes
            source_phases: Source phases
            target_points: Target points
            frequency: Operating frequency
            medium_properties: Medium properties
            priority: Job priority
            chunk_size: Points per computation chunk
            
        Returns:
            List of job IDs
        """
        from ..physics.propagation.wave_propagator import WavePropagator
        
        # Split computation into chunks
        num_chunks = (len(target_points) + chunk_size - 1) // chunk_size
        job_ids = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(target_points))
            
            chunk_points = target_points[start_idx:end_idx]
            
            # Create computation function
            def compute_chunk(src_pos, src_amp, src_phase, tgt_points, freq, medium):
                propagator = WavePropagator(frequency=freq, medium=medium)
                return propagator.compute_field_from_sources(
                    src_pos, src_amp, src_phase, tgt_points
                )
            
            # Submit job
            job_id = self.submit_job(
                compute_chunk,
                source_positions,
                source_amplitudes,
                source_phases,
                chunk_points,
                frequency,
                medium_properties,
                job_type="wave_propagation",
                priority=priority,
                estimated_runtime=len(chunk_points) * 0.001  # Rough estimate
            )
            
            job_ids.append(job_id)
            
            self.submitted_jobs[job_id] = {
                'type': 'wave_propagation_chunk',
                'chunk_index': i,
                'total_chunks': num_chunks,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
        
        return job_ids
    
    def submit_field_optimization(
        self,
        target_field: 'AcousticField',
        transducer_array: 'TransducerArray',
        optimization_params: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """
        Submit field optimization job.
        
        Args:
            target_field: Target acoustic field
            transducer_array: Transducer array
            optimization_params: Optimization parameters
            priority: Job priority
            
        Returns:
            Job ID
        """
        from ..optimization.hologram_optimizer import GradientOptimizer
        
        def optimize_field(target, array, params):
            optimizer = GradientOptimizer(
                num_elements=len(array.elements),
                **params
            )
            
            # Create simple forward model
            def forward_model(phases):
                # Simplified forward model for demonstration
                return torch.randn_like(target.data, dtype=torch.complex64)
            
            import torch
            result = optimizer.optimize(
                forward_model=forward_model,
                target_field=torch.tensor(target.data, dtype=torch.complex64),
                iterations=params.get('iterations', 1000)
            )
            
            return result.phases
        
        job_id = self.submit_job(
            optimize_field,
            target_field,
            transducer_array,
            optimization_params,
            job_type="field_optimization",
            priority=priority,
            estimated_runtime=optimization_params.get('iterations', 1000) * 0.01,
            gpu_required=True
        )
        
        self.submitted_jobs[job_id] = {
            'type': 'field_optimization',
            'target_shape': target_field.shape,
            'num_transducers': len(transducer_array.elements)
        }
        
        return job_id
    
    def submit_job(
        self,
        function: Callable,
        *args,
        job_type: str = "generic",
        priority: int = 0,
        estimated_runtime: float = 0,
        memory_required: float = 0,
        gpu_required: bool = False,
        **kwargs
    ) -> str:
        """Submit generic computation job."""
        # Communicate with scheduler
        socket = self.zmq_context.socket(zmq.REQ)
        socket.connect(f"tcp://{self.scheduler_host}:{self.scheduler_port}")
        
        # Serialize job data
        job_data = {
            "action": "submit_job",
            "function": pickle.dumps(function).hex(),
            "args": pickle.dumps(args).hex(),
            "kwargs": pickle.dumps(kwargs).hex(),
            "job_type": job_type,
            "priority": priority,
            "estimated_runtime": estimated_runtime,
            "memory_required": memory_required,
            "gpu_required": gpu_required
        }
        
        socket.send_json(job_data)
        
        # Get job ID response
        if socket.poll(5000):  # 5 second timeout
            response = socket.recv_json()
            job_id = response.get("job_id")
        else:
            raise RuntimeError("Failed to submit job: scheduler timeout")
        
        socket.close()
        
        return job_id
    
    def get_results(self, job_ids: List[str], timeout: float = None) -> List[Any]:
        """
        Wait for and collect results from multiple jobs.
        
        Args:
            job_ids: List of job IDs to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            List of job results
        """
        results = [None] * len(job_ids)
        completed_jobs = set()
        start_time = time.time()
        
        while len(completed_jobs) < len(job_ids):
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break
            
            for i, job_id in enumerate(job_ids):
                if i in completed_jobs:
                    continue
                
                # Check job status
                status = self.get_job_status(job_id)
                
                if status == JobStatus.COMPLETED:
                    result = self.get_job_result(job_id)
                    results[i] = result
                    completed_jobs.add(i)
                elif status == JobStatus.FAILED:
                    completed_jobs.add(i)  # Mark as completed (with None result)
            
            time.sleep(0.5)  # Poll every 500ms
        
        return results
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get status of a job."""
        job_data = self.redis_client.hget("cluster:jobs", job_id)
        if job_data:
            job_dict = json.loads(job_data)
            return JobStatus(job_dict['status'])
        return None
    
    def get_job_result(self, job_id: str) -> Any:
        """Get result of a completed job."""
        job_data = self.redis_client.hget("cluster:jobs", job_id)
        if job_data:
            job_dict = json.loads(job_data)
            if job_dict['status'] == 'completed' and job_dict.get('result'):
                return pickle.loads(bytes.fromhex(job_dict['result']))
        return None


def create_cluster_manager(
    enable_dask: bool = True,
    enable_ray: bool = True,
    redis_config: Optional[Dict[str, Any]] = None
) -> Optional[DistributedScheduler]:
    """
    Create distributed cluster manager with available backends.
    
    Args:
        enable_dask: Enable Dask backend
        enable_ray: Enable Ray backend
        redis_config: Redis configuration
        
    Returns:
        DistributedScheduler instance or None if setup failed
    """
    redis_config = redis_config or {"host": "localhost", "port": 6379}
    
    try:
        scheduler = DistributedScheduler(
            redis_host=redis_config["host"],
            redis_port=redis_config["port"]
        )
        
        scheduler.start_scheduler()
        return scheduler
        
    except Exception as e:
        print(f"Failed to create cluster manager: {e}")
        return None