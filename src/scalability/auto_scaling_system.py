"""
Auto-Scaling System for Acousto-Gen Generation 3.
Implements intelligent resource scaling and load adaptation.
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import psutil
import json
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    WORKERS = "workers"
    CACHE = "cache"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    queue_depth: int
    average_response_time: float
    error_rate: float
    throughput: float


@dataclass
class ScalingRule:
    """Defines scaling behavior based on metrics."""
    name: str
    resource_type: ResourceType
    metric_name: str
    threshold_up: float
    threshold_down: float
    min_instances: int
    max_instances: int
    cooldown_period: float
    scale_factor: float = 1.5
    enabled: bool = True


class MetricsCollector:
    """Collects system metrics for scaling decisions."""
    
    def __init__(self, collection_interval: float = 5.0):
        """Initialize metrics collector."""
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=100)  # Keep last 100 measurements
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start metrics collection."""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        logger.info("Metrics collection started")
    
    def stop(self):
        """Stop metrics collection."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Metrics collection stopped")
    
    def _collect_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                metrics = self._collect_current_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics if available
        gpu_usage = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization() or 0.0
        except:
            pass
        
        # Application metrics (would be injected in real system)
        queue_depth = 0  # To be updated by application
        avg_response_time = 0.0  # To be updated by application
        error_rate = 0.0  # To be updated by application
        throughput = 0.0  # To be updated by application
        
        return ScalingMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            queue_depth=queue_depth,
            average_response_time=avg_response_time,
            error_rate=error_rate,
            throughput=throughput
        )
    
    def get_recent_metrics(self, time_window: float = 60.0) -> List[ScalingMetrics]:
        """Get metrics within time window."""
        cutoff_time = time.time() - time_window
        
        with self._lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_average_metrics(self, time_window: float = 60.0) -> Optional[ScalingMetrics]:
        """Get average metrics over time window."""
        recent_metrics = self.get_recent_metrics(time_window)
        
        if not recent_metrics:
            return None
        
        return ScalingMetrics(
            timestamp=time.time(),
            cpu_usage=np.mean([m.cpu_usage for m in recent_metrics]),
            memory_usage=np.mean([m.memory_usage for m in recent_metrics]),
            gpu_usage=np.mean([m.gpu_usage for m in recent_metrics if m.gpu_usage is not None]),
            queue_depth=int(np.mean([m.queue_depth for m in recent_metrics])),
            average_response_time=np.mean([m.average_response_time for m in recent_metrics]),
            error_rate=np.mean([m.error_rate for m in recent_metrics]),
            throughput=np.mean([m.throughput for m in recent_metrics])
        )
    
    def update_application_metrics(self, queue_depth: int = None, 
                                  avg_response_time: float = None,
                                  error_rate: float = None,
                                  throughput: float = None):
        """Update application-specific metrics."""
        # This would be called by the application to update metrics
        pass


class ResourceScaler:
    """Handles scaling of specific resource types."""
    
    def __init__(self, resource_type: ResourceType):
        """Initialize resource scaler."""
        self.resource_type = resource_type
        self.current_instances = 1
        self.target_instances = 1
        self.scaling_operations = []
        self._lock = threading.Lock()
    
    def scale_to(self, target_instances: int, reason: str = ""):
        """Scale to target number of instances."""
        with self._lock:
            if target_instances == self.current_instances:
                return False  # No scaling needed
            
            scaling_operation = {
                'timestamp': time.time(),
                'from_instances': self.current_instances,
                'to_instances': target_instances,
                'reason': reason,
                'resource_type': self.resource_type.value
            }
            
            self.scaling_operations.append(scaling_operation)
            
            # Simulate scaling operation
            if self.resource_type == ResourceType.WORKERS:
                success = self._scale_workers(target_instances)
            elif self.resource_type == ResourceType.CACHE:
                success = self._scale_cache(target_instances)
            elif self.resource_type == ResourceType.CPU:
                success = self._scale_cpu(target_instances)
            else:
                success = True  # Default success
            
            if success:
                self.current_instances = target_instances
                logger.info(f"Scaled {self.resource_type.value} from {scaling_operation['from_instances']} to {target_instances}: {reason}")
            else:
                logger.error(f"Failed to scale {self.resource_type.value} to {target_instances}")
            
            return success
    
    def _scale_workers(self, target_instances: int) -> bool:
        """Scale worker processes."""
        # In a real implementation, this would start/stop worker processes
        # For now, just simulate the scaling
        time.sleep(0.1)  # Simulate scaling time
        return True
    
    def _scale_cache(self, target_size: int) -> bool:
        """Scale cache size."""
        # In a real implementation, this would resize cache
        time.sleep(0.05)
        return True
    
    def _scale_cpu(self, target_allocation: int) -> bool:
        """Scale CPU allocation (for containerized environments)."""
        # This would adjust CPU limits in containers
        return True
    
    def get_scaling_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent scaling operations."""
        with self._lock:
            return self.scaling_operations[-limit:]


class AutoScalingEngine:
    """Main auto-scaling engine that coordinates all scaling decisions."""
    
    def __init__(self):
        """Initialize auto-scaling engine."""
        self.metrics_collector = MetricsCollector()
        self.scalers = {}  # ResourceType -> ResourceScaler
        self.scaling_rules = []
        self.last_scaling_time = defaultdict(float)
        self.running = False
        self._thread = None
        
        # Initialize scalers for each resource type
        for resource_type in ResourceType:
            self.scalers[resource_type] = ResourceScaler(resource_type)
        
        # Set up default scaling rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Set up default scaling rules."""
        # CPU-based worker scaling
        self.scaling_rules.append(ScalingRule(
            name="CPU High Load Worker Scaling",
            resource_type=ResourceType.WORKERS,
            metric_name="cpu_usage",
            threshold_up=75.0,
            threshold_down=25.0,
            min_instances=1,
            max_instances=16,
            cooldown_period=60.0,  # 1 minute cooldown
            scale_factor=1.5
        ))
        
        # Memory-based cache scaling
        self.scaling_rules.append(ScalingRule(
            name="Memory Pressure Cache Scaling",
            resource_type=ResourceType.CACHE,
            metric_name="memory_usage",
            threshold_up=85.0,
            threshold_down=40.0,
            min_instances=100,  # MB
            max_instances=2000,  # MB
            cooldown_period=30.0,
            scale_factor=1.3
        ))
        
        # Queue depth based scaling
        self.scaling_rules.append(ScalingRule(
            name="Queue Depth Worker Scaling",
            resource_type=ResourceType.WORKERS,
            metric_name="queue_depth",
            threshold_up=10.0,
            threshold_down=2.0,
            min_instances=1,
            max_instances=32,
            cooldown_period=45.0,
            scale_factor=2.0
        ))
        
        # Response time based scaling
        self.scaling_rules.append(ScalingRule(
            name="Response Time Worker Scaling", 
            resource_type=ResourceType.WORKERS,
            metric_name="average_response_time",
            threshold_up=5.0,  # 5 seconds
            threshold_down=1.0,  # 1 second
            min_instances=1,
            max_instances=16,
            cooldown_period=90.0,
            scale_factor=1.8
        ))
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule: {rule.name}")
    
    def start(self):
        """Start auto-scaling engine."""
        if self.running:
            return
        
        self.running = True
        
        # Start metrics collection
        self.metrics_collector.start()
        
        # Start scaling decision loop
        self._thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._thread.start()
        
        logger.info("Auto-scaling engine started")
    
    def stop(self):
        """Stop auto-scaling engine."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop metrics collection
        self.metrics_collector.stop()
        
        # Stop scaling thread
        if self._thread:
            self._thread.join(timeout=5.0)
        
        logger.info("Auto-scaling engine stopped")
    
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.running:
            try:
                self._evaluate_scaling_rules()
                time.sleep(10.0)  # Evaluate every 10 seconds
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                time.sleep(10.0)
    
    def _evaluate_scaling_rules(self):
        """Evaluate all scaling rules and make scaling decisions."""
        # Get current metrics
        current_metrics = self.metrics_collector.get_average_metrics(60.0)
        if not current_metrics:
            return
        
        current_time = time.time()
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            last_scaling = self.last_scaling_time[rule.name]
            if current_time - last_scaling < rule.cooldown_period:
                continue
            
            # Get metric value
            metric_value = getattr(current_metrics, rule.metric_name, 0.0)
            
            # Determine scaling direction
            scaling_direction = self._determine_scaling_direction(metric_value, rule)
            
            if scaling_direction != ScalingDirection.STABLE:
                self._execute_scaling_decision(rule, scaling_direction, metric_value, current_metrics)
                self.last_scaling_time[rule.name] = current_time
    
    def _determine_scaling_direction(self, metric_value: float, rule: ScalingRule) -> ScalingDirection:
        """Determine if scaling up, down, or stable."""
        current_instances = self.scalers[rule.resource_type].current_instances
        
        # Check for scale up conditions
        if metric_value > rule.threshold_up and current_instances < rule.max_instances:
            return ScalingDirection.UP
        
        # Check for scale down conditions
        if metric_value < rule.threshold_down and current_instances > rule.min_instances:
            return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    def _execute_scaling_decision(self, rule: ScalingRule, direction: ScalingDirection, 
                                 metric_value: float, metrics: ScalingMetrics):
        """Execute scaling decision."""
        scaler = self.scalers[rule.resource_type]
        current_instances = scaler.current_instances
        
        if direction == ScalingDirection.UP:
            target_instances = min(
                int(current_instances * rule.scale_factor),
                rule.max_instances
            )
            reason = f"{rule.metric_name}={metric_value:.2f} > {rule.threshold_up} (scale up)"
            
        elif direction == ScalingDirection.DOWN:
            target_instances = max(
                int(current_instances / rule.scale_factor),
                rule.min_instances
            )
            reason = f"{rule.metric_name}={metric_value:.2f} < {rule.threshold_down} (scale down)"
        
        else:
            return
        
        # Execute scaling
        success = scaler.scale_to(target_instances, reason)
        
        if success:
            logger.info(f"Auto-scaling: {rule.name} - {reason}")
            
            # Record scaling event
            self._record_scaling_event(rule, direction, current_instances, target_instances, metrics)
        else:
            logger.warning(f"Auto-scaling failed: {rule.name} - {reason}")
    
    def _record_scaling_event(self, rule: ScalingRule, direction: ScalingDirection,
                             from_instances: int, to_instances: int, metrics: ScalingMetrics):
        """Record scaling event for analysis."""
        event = {
            'timestamp': time.time(),
            'rule_name': rule.name,
            'resource_type': rule.resource_type.value,
            'direction': direction.value,
            'from_instances': from_instances,
            'to_instances': to_instances,
            'metrics_snapshot': {
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'queue_depth': metrics.queue_depth,
                'response_time': metrics.average_response_time
            }
        }
        
        # In a real system, this would be logged to a database or monitoring system
        logger.debug(f"Scaling event recorded: {json.dumps(event, indent=2)}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        current_metrics = self.metrics_collector.get_average_metrics(60.0)
        
        status = {
            'engine_running': self.running,
            'current_metrics': current_metrics.__dict__ if current_metrics else None,
            'resource_instances': {
                resource_type.value: scaler.current_instances 
                for resource_type, scaler in self.scalers.items()
            },
            'active_rules': len([r for r in self.scaling_rules if r.enabled]),
            'total_rules': len(self.scaling_rules),
            'recent_scaling_operations': self._get_recent_scaling_operations(10)
        }
        
        return status
    
    def _get_recent_scaling_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent scaling operations across all resources."""
        all_operations = []
        
        for scaler in self.scalers.values():
            operations = scaler.get_scaling_history(limit)
            all_operations.extend(operations)
        
        # Sort by timestamp and return most recent
        all_operations.sort(key=lambda x: x['timestamp'], reverse=True)
        return all_operations[:limit]
    
    def force_scale(self, resource_type: ResourceType, target_instances: int, reason: str = "Manual scaling"):
        """Manually force scaling of a resource."""
        if resource_type not in self.scalers:
            raise ValueError(f"Unknown resource type: {resource_type}")
        
        scaler = self.scalers[resource_type]
        success = scaler.scale_to(target_instances, reason)
        
        if success:
            logger.info(f"Manual scaling successful: {resource_type.value} to {target_instances}")
        else:
            logger.error(f"Manual scaling failed: {resource_type.value} to {target_instances}")
        
        return success
    
    def update_rule(self, rule_name: str, **kwargs):
        """Update an existing scaling rule."""
        for i, rule in enumerate(self.scaling_rules):
            if rule.name == rule_name:
                # Update rule attributes
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                        logger.info(f"Updated rule {rule_name}: {key} = {value}")
                break
        else:
            logger.warning(f"Rule not found: {rule_name}")


# Global auto-scaling engine instance
auto_scaler = AutoScalingEngine()


def with_auto_scaling(resource_type: ResourceType = ResourceType.WORKERS):
    """Decorator to enable auto-scaling for a function."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Record start time for response time metrics
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Update success metrics
                response_time = time.time() - start_time
                auto_scaler.metrics_collector.update_application_metrics(
                    avg_response_time=response_time,
                    error_rate=0.0
                )
                
                return result
                
            except Exception as e:
                # Update error metrics
                response_time = time.time() - start_time
                auto_scaler.metrics_collector.update_application_metrics(
                    avg_response_time=response_time,
                    error_rate=1.0
                )
                raise
        
        return wrapper
    return decorator