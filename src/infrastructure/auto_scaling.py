"""
Auto-scaling and load balancing for Acousto-Gen applications.
Provides dynamic resource scaling based on workload and performance metrics.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque
import statistics
import psutil

# Handle optional dependencies
try:
    from optimization.parallel_computing import ResourceManager, get_resource_manager
    HAS_RESOURCE_MANAGER = True
except ImportError:
    HAS_RESOURCE_MANAGER = False

try:
    from monitoring.metrics import metrics_collector
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Auto-scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_AWARE = "resource_aware"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    request_rate: float
    response_time_95p: float
    queue_depth: int
    error_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerNode:
    """Represents a worker node in the cluster."""
    node_id: str
    endpoint: str
    capacity: int
    current_load: int = 0
    last_response_time: float = 0.0
    health_status: str = "healthy"
    last_health_check: float = field(default_factory=time.time)
    
    @property
    def utilization(self) -> float:
        """Calculate current utilization percentage."""
        return (self.current_load / max(self.capacity, 1)) * 100
    
    @property
    def available_capacity(self) -> int:
        """Calculate available capacity."""
        return max(0, self.capacity - self.current_load)


class AutoScaler:
    """
    Intelligent auto-scaling system for compute resources.
    
    Monitors system metrics and automatically adjusts resource allocation
    to maintain optimal performance while minimizing costs.
    """
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        target_cpu_utilization: float = 70.0,
        target_memory_utilization: float = 80.0,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 30.0,
        cooldown_period: float = 300.0,  # 5 minutes
        evaluation_period: float = 60.0,  # 1 minute
        emergency_threshold: float = 95.0
    ):
        """
        Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            target_cpu_utilization: Target CPU utilization percentage
            target_memory_utilization: Target memory utilization percentage
            scale_up_threshold: Threshold to trigger scale up
            scale_down_threshold: Threshold to trigger scale down
            cooldown_period: Minimum time between scaling actions
            evaluation_period: How often to evaluate scaling decisions
            emergency_threshold: Emergency scaling threshold
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_utilization = target_cpu_utilization
        self.target_memory_utilization = target_memory_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.evaluation_period = evaluation_period
        self.emergency_threshold = emergency_threshold
        
        # State tracking
        self.current_workers = min_workers
        self.last_scaling_action = 0.0
        self.metrics_history = deque(maxlen=100)
        
        # Callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        # Monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Resource manager integration
        self.resource_manager = None
        if HAS_RESOURCE_MANAGER:
            self.resource_manager = get_resource_manager()
        
        logger.info(f"Initialized AutoScaler: {min_workers}-{max_workers} workers, "
                   f"target CPU: {target_cpu_utilization}%")
    
    def set_scale_callbacks(
        self,
        scale_up_callback: Callable[[int], bool],
        scale_down_callback: Callable[[int], bool]
    ):
        """Set callbacks for scaling actions."""
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
    
    def start_monitoring(self):
        """Start automatic scaling monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        def monitoring_loop():
            while self._monitoring_active:
                try:
                    self._evaluate_scaling()
                    time.sleep(self.evaluation_period)
                except Exception as e:
                    logger.error(f"Auto-scaling monitoring error: {e}")
                    time.sleep(self.evaluation_period)
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Started auto-scaling monitoring")
    
    def stop_monitoring(self):
        """Stop automatic scaling monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
    
    def _evaluate_scaling(self):
        """Evaluate whether scaling action is needed."""
        current_metrics = self._collect_metrics()
        self.metrics_history.append(current_metrics)
        
        # Need at least 3 data points for trend analysis
        if len(self.metrics_history) < 3:
            return
        
        # Calculate scaling decision
        scaling_action = self._determine_scaling_action(current_metrics)
        
        if scaling_action != ScalingAction.MAINTAIN:
            self._execute_scaling_action(scaling_action, current_metrics)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        cpu_util = psutil.cpu_percent(interval=1)
        memory_util = psutil.virtual_memory().percent
        
        gpu_util = 0.0
        if self.resource_manager and self.resource_manager.gpu_resources:
            gpu_utils = [gpu.utilization * 100 for gpu in self.resource_manager.gpu_resources]
            gpu_util = statistics.mean(gpu_utils) if gpu_utils else 0.0
        
        # Get metrics from monitoring system if available
        request_rate = 0.0
        response_time_95p = 0.0
        error_rate = 0.0
        queue_depth = 0
        
        if HAS_METRICS and metrics_collector:
            try:
                system_metrics = metrics_collector.get_system_metrics()
                # Extract relevant metrics if available
                request_rate = system_metrics.get('request_rate', 0.0)
                response_time_95p = system_metrics.get('response_time_95p', 0.0)
                error_rate = system_metrics.get('error_rate', 0.0)
            except Exception as e:
                logger.debug(f"Could not collect metrics: {e}")
        
        return ScalingMetrics(
            cpu_utilization=cpu_util,
            memory_utilization=memory_util,
            gpu_utilization=gpu_util,
            request_rate=request_rate,
            response_time_95p=response_time_95p,
            queue_depth=queue_depth,
            error_rate=error_rate
        )
    
    def _determine_scaling_action(self, metrics: ScalingMetrics) -> ScalingAction:
        """Determine what scaling action to take."""
        # Check cooldown period
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return ScalingAction.MAINTAIN
        
        # Emergency scaling
        if (metrics.cpu_utilization > self.emergency_threshold or 
            metrics.memory_utilization > self.emergency_threshold or
            metrics.error_rate > 0.1):  # 10% error rate
            return ScalingAction.EMERGENCY_SCALE
        
        # Calculate trend over recent history
        recent_metrics = list(self.metrics_history)[-5:]  # Last 5 evaluations
        
        if len(recent_metrics) >= 3:
            cpu_trend = self._calculate_trend([m.cpu_utilization for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_utilization for m in recent_metrics])
            
            # Scale up conditions
            if (metrics.cpu_utilization > self.scale_up_threshold or
                metrics.memory_utilization > self.scale_up_threshold or
                (cpu_trend > 5 and metrics.cpu_utilization > self.target_cpu_utilization) or
                (memory_trend > 5 and metrics.memory_utilization > self.target_memory_utilization)):
                
                if self.current_workers < self.max_workers:
                    return ScalingAction.SCALE_UP
            
            # Scale down conditions
            elif (metrics.cpu_utilization < self.scale_down_threshold and
                  metrics.memory_utilization < self.scale_down_threshold and
                  cpu_trend < -2 and memory_trend < -2):
                
                if self.current_workers > self.min_workers:
                    return ScalingAction.SCALE_DOWN
        
        return ScalingAction.MAINTAIN
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of recent values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression slope
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _execute_scaling_action(self, action: ScalingAction, metrics: ScalingMetrics):
        """Execute the determined scaling action."""
        if action == ScalingAction.SCALE_UP:
            new_workers = min(self.max_workers, self.current_workers + 1)
            if self._scale_up(new_workers):
                self.current_workers = new_workers
                self.last_scaling_action = time.time()
                logger.info(f"Scaled up to {new_workers} workers (CPU: {metrics.cpu_utilization:.1f}%)")
        
        elif action == ScalingAction.SCALE_DOWN:
            new_workers = max(self.min_workers, self.current_workers - 1)
            if self._scale_down(new_workers):
                self.current_workers = new_workers
                self.last_scaling_action = time.time()
                logger.info(f"Scaled down to {new_workers} workers (CPU: {metrics.cpu_utilization:.1f}%)")
        
        elif action == ScalingAction.EMERGENCY_SCALE:
            # Emergency scale up by 50% or add 2 workers, whichever is more
            additional_workers = max(2, int(self.current_workers * 0.5))
            new_workers = min(self.max_workers, self.current_workers + additional_workers)
            
            if self._scale_up(new_workers):
                self.current_workers = new_workers
                self.last_scaling_action = time.time()
                logger.warning(f"Emergency scaled up to {new_workers} workers!")
    
    def _scale_up(self, target_workers: int) -> bool:
        """Execute scale up action."""
        if self.scale_up_callback:
            try:
                return self.scale_up_callback(target_workers)
            except Exception as e:
                logger.error(f"Scale up callback failed: {e}")
                return False
        else:
            # Default implementation - just log
            logger.info(f"Would scale up to {target_workers} workers")
            return True
    
    def _scale_down(self, target_workers: int) -> bool:
        """Execute scale down action."""
        if self.scale_down_callback:
            try:
                return self.scale_down_callback(target_workers)
            except Exception as e:
                logger.error(f"Scale down callback failed: {e}")
                return False
        else:
            # Default implementation - just log
            logger.info(f"Would scale down to {target_workers} workers")
            return True
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'last_scaling_action': self.last_scaling_action,
            'monitoring_active': self._monitoring_active,
            'latest_metrics': {
                'cpu_utilization': latest_metrics.cpu_utilization if latest_metrics else 0,
                'memory_utilization': latest_metrics.memory_utilization if latest_metrics else 0,
                'gpu_utilization': latest_metrics.gpu_utilization if latest_metrics else 0,
                'timestamp': latest_metrics.timestamp if latest_metrics else 0
            } if latest_metrics else None,
            'metrics_history_size': len(self.metrics_history)
        }


class LoadBalancer:
    """
    Intelligent load balancer for distributing workload across worker nodes.
    
    Supports multiple balancing strategies and automatic node health monitoring.
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE,
        health_check_interval: float = 30.0,
        unhealthy_threshold: int = 3
    ):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy to use
            health_check_interval: Interval between health checks
            unhealthy_threshold: Failures before marking node unhealthy
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.unhealthy_threshold = unhealthy_threshold
        
        # Node management
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.round_robin_index = 0
        
        # Health checking
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_checks = False
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        logger.info(f"Initialized LoadBalancer with {strategy.value} strategy")
    
    def add_worker_node(
        self,
        node_id: str,
        endpoint: str,
        capacity: int
    ):
        """Add a worker node to the load balancer."""
        node = WorkerNode(
            node_id=node_id,
            endpoint=endpoint,
            capacity=capacity
        )
        
        self.worker_nodes[node_id] = node
        logger.info(f"Added worker node: {node_id} ({endpoint}) with capacity {capacity}")
    
    def remove_worker_node(self, node_id: str):
        """Remove a worker node from the load balancer."""
        if node_id in self.worker_nodes:
            del self.worker_nodes[node_id]
            logger.info(f"Removed worker node: {node_id}")
    
    def select_worker_node(self) -> Optional[WorkerNode]:
        """Select the best worker node based on the current strategy."""
        healthy_nodes = [
            node for node in self.worker_nodes.values()
            if node.health_status == "healthy" and node.available_capacity > 0
        ]
        
        if not healthy_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(healthy_nodes)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_nodes, key=lambda n: n.current_load)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            return self._select_weighted_response_time(healthy_nodes)
        
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._select_resource_aware(healthy_nodes)
        
        else:
            return healthy_nodes[0]
    
    def _select_round_robin(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node using round-robin strategy."""
        if not nodes:
            return None
        
        node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index = (self.round_robin_index + 1) % len(nodes)
        return node
    
    def _select_weighted_response_time(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node based on weighted response time."""
        if not nodes:
            return None
        
        # Calculate weights (inverse of response time)
        weights = []
        for node in nodes:
            # Use inverse response time, with minimum to avoid division by zero
            response_time = max(node.last_response_time, 0.001)
            weight = 1.0 / response_time
            weights.append(weight)
        
        # Select based on weights
        total_weight = sum(weights)
        if total_weight == 0:
            return nodes[0]
        
        import random
        r = random.uniform(0, total_weight)
        
        current_weight = 0
        for node, weight in zip(nodes, weights):
            current_weight += weight
            if r <= current_weight:
                return node
        
        return nodes[-1]
    
    def _select_resource_aware(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node based on available resources."""
        if not nodes:
            return None
        
        # Score nodes based on available capacity and utilization
        best_node = None
        best_score = -1
        
        for node in nodes:
            # Calculate score: higher available capacity and lower utilization is better
            utilization_score = 1.0 - (node.utilization / 100.0)
            capacity_score = node.available_capacity / max(node.capacity, 1)
            
            # Combined score with emphasis on available capacity
            score = (capacity_score * 0.7) + (utilization_score * 0.3)
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def assign_work(self, node_id: str, work_units: int = 1) -> bool:
        """Assign work to a specific node."""
        if node_id not in self.worker_nodes:
            return False
        
        node = self.worker_nodes[node_id]
        if node.available_capacity >= work_units:
            node.current_load += work_units
            return True
        
        return False
    
    def complete_work(self, node_id: str, work_units: int = 1, response_time: float = 0.0):
        """Mark work as completed on a node."""
        if node_id not in self.worker_nodes:
            return
        
        node = self.worker_nodes[node_id]
        node.current_load = max(0, node.current_load - work_units)
        
        if response_time > 0:
            # Update exponential moving average of response time
            alpha = 0.1
            if node.last_response_time == 0:
                node.last_response_time = response_time
            else:
                node.last_response_time = (alpha * response_time + 
                                         (1 - alpha) * node.last_response_time)
        
        self.successful_requests += 1
        self.total_requests += 1
    
    def mark_work_failed(self, node_id: str, work_units: int = 1):
        """Mark work as failed on a node."""
        if node_id not in self.worker_nodes:
            return
        
        node = self.worker_nodes[node_id]
        node.current_load = max(0, node.current_load - work_units)
        
        self.failed_requests += 1
        self.total_requests += 1
    
    def start_health_monitoring(self):
        """Start health monitoring for worker nodes."""
        if self._health_check_thread is not None:
            return
        
        self._stop_health_checks = False
        
        def health_check_loop():
            while not self._stop_health_checks:
                try:
                    self._perform_health_checks()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    time.sleep(self.health_check_interval)
        
        self._health_check_thread = threading.Thread(target=health_check_loop, daemon=True)
        self._health_check_thread.start()
        
        logger.info("Started load balancer health monitoring")
    
    def stop_health_monitoring(self):
        """Stop health monitoring."""
        self._stop_health_checks = True
        if self._health_check_thread:
            self._health_check_thread.join(timeout=2.0)
            self._health_check_thread = None
    
    def _perform_health_checks(self):
        """Perform health checks on all nodes."""
        for node_id, node in self.worker_nodes.items():
            try:
                # Simple health check - in production, this would be an HTTP request
                # For now, just mark as healthy if last check was recent
                current_time = time.time()
                
                # Simulate occasional health check failures for testing
                import random
                if random.random() < 0.05:  # 5% failure rate
                    raise Exception("Simulated health check failure")
                
                node.health_status = "healthy"
                node.last_health_check = current_time
                
            except Exception as e:
                logger.warning(f"Health check failed for node {node_id}: {e}")
                node.health_status = "unhealthy"
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        healthy_nodes = sum(1 for node in self.worker_nodes.values() 
                          if node.health_status == "healthy")
        
        total_capacity = sum(node.capacity for node in self.worker_nodes.values())
        total_load = sum(node.current_load for node in self.worker_nodes.values())
        
        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100
        
        return {
            'strategy': self.strategy.value,
            'total_nodes': len(self.worker_nodes),
            'healthy_nodes': healthy_nodes,
            'total_capacity': total_capacity,
            'total_load': total_load,
            'utilization_percent': (total_load / max(total_capacity, 1)) * 100,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate_percent': success_rate,
            'nodes': [
                {
                    'node_id': node.node_id,
                    'endpoint': node.endpoint,
                    'capacity': node.capacity,
                    'current_load': node.current_load,
                    'utilization_percent': node.utilization,
                    'health_status': node.health_status,
                    'last_response_time': node.last_response_time
                }
                for node in self.worker_nodes.values()
            ]
        }


# Global instances
_global_auto_scaler: Optional[AutoScaler] = None
_global_load_balancer: Optional[LoadBalancer] = None


def get_auto_scaler() -> AutoScaler:
    """Get or create global auto-scaler."""
    global _global_auto_scaler
    
    if _global_auto_scaler is None:
        _global_auto_scaler = AutoScaler()
    
    return _global_auto_scaler


def get_load_balancer() -> LoadBalancer:
    """Get or create global load balancer."""
    global _global_load_balancer
    
    if _global_load_balancer is None:
        _global_load_balancer = LoadBalancer()
    
    return _global_load_balancer