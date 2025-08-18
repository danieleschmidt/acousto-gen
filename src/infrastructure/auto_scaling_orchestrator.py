"""
Auto-Scaling Orchestrator
Generation 3: MAKE IT SCALE - Intelligent auto-scaling for acoustic optimization workloads.
"""

import numpy as np
import time
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import psutil
import subprocess
import uuid
from datetime import datetime, timedelta

# Configure auto-scaling logger
logger = logging.getLogger('acousto_gen.autoscaling')
logger.setLevel(logging.INFO)


class ScalingMetric(Enum):
    """Metrics used for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Add instances
    SCALE_IN = "scale_in"    # Remove instances
    NO_ACTION = "no_action"


class InstanceType(Enum):
    """Types of compute instances."""
    CPU_OPTIMIZED = "cpu_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    GPU_ENABLED = "gpu_enabled"
    BALANCED = "balanced"
    COMPUTE_INTENSIVE = "compute_intensive"


@dataclass
class ScalingRule:
    """Defines a scaling rule."""
    rule_id: str
    metric: ScalingMetric
    threshold_up: float
    threshold_down: float
    action_up: ScalingAction
    action_down: ScalingAction
    cooldown_period: float  # seconds
    evaluation_window: float  # seconds
    enabled: bool = True
    last_triggered: float = 0.0
    priority: int = 1  # Higher number = higher priority
    
    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = str(uuid.uuid4())


@dataclass
class ComputeInstance:
    """Represents a compute instance."""
    instance_id: str
    instance_type: InstanceType
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    status: str  # pending, running, stopping, stopped
    created_at: float
    last_heartbeat: float
    current_load: float = 0.0
    optimization_tasks: int = 0
    performance_score: float = 1.0
    cost_per_hour: float = 0.0
    region: str = "local"
    
    def __post_init__(self):
        if not self.instance_id:
            self.instance_id = str(uuid.uuid4())


@dataclass
class ScalingEvent:
    """Records a scaling event."""
    timestamp: float
    action: ScalingAction
    metric: ScalingMetric
    metric_value: float
    threshold: float
    instances_before: int
    instances_after: int
    reason: str
    duration: float = 0.0
    success: bool = True


@dataclass
class ResourcePrediction:
    """Predicted resource requirements."""
    timestamp: float
    predicted_cpu_usage: float
    predicted_memory_usage: float
    predicted_queue_length: float
    predicted_throughput: float
    confidence: float
    time_horizon: float  # seconds into future


class MetricsCollector:
    """Collects metrics for scaling decisions."""
    
    def __init__(self, window_size: int = 1000):
        self.metrics = {metric: deque(maxlen=window_size) for metric in ScalingMetric}
        self.lock = threading.RLock()
        
    def record_metric(self, metric: ScalingMetric, value: float, timestamp: float = None):
        """Record a metric value."""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            self.metrics[metric].append((timestamp, value))
    
    def get_average(self, metric: ScalingMetric, window_seconds: float = 300.0) -> Optional[float]:
        """Get average metric value over time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.lock:
            values = [
                value for timestamp, value in self.metrics[metric]
                if timestamp >= cutoff_time
            ]
            
            return np.mean(values) if values else None
    
    def get_percentile(self, metric: ScalingMetric, percentile: float = 95.0,
                      window_seconds: float = 300.0) -> Optional[float]:
        """Get percentile of metric values."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.lock:
            values = [
                value for timestamp, value in self.metrics[metric]
                if timestamp >= cutoff_time
            ]
            
            return np.percentile(values, percentile) if values else None
    
    def get_trend(self, metric: ScalingMetric, window_seconds: float = 300.0) -> Optional[float]:
        """Get trend (slope) of metric over time."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.lock:
            data_points = [
                (timestamp, value) for timestamp, value in self.metrics[metric]
                if timestamp >= cutoff_time
            ]
            
            if len(data_points) < 2:
                return None
            
            timestamps, values = zip(*data_points)
            
            # Simple linear regression
            n = len(timestamps)
            sum_x = sum(timestamps)
            sum_y = sum(values)
            sum_xy = sum(t * v for t, v in zip(timestamps, values))
            sum_x2 = sum(t * t for t in timestamps)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope


class PredictiveModel:
    """Predictive model for resource requirements."""
    
    def __init__(self):
        self.training_data = {
            'features': [],  # [cpu, memory, queue_length, time_of_day, day_of_week]
            'targets': []    # [future_cpu, future_memory, future_queue]
        }
        self.model_weights = None
        self.is_trained = False
        
    def add_training_sample(self, current_metrics: Dict[str, float], 
                           future_metrics: Dict[str, float]):
        """Add training sample to the model."""
        # Extract features
        current_time = time.time()
        dt = datetime.fromtimestamp(current_time)
        
        features = [
            current_metrics.get('cpu_utilization', 0.0),
            current_metrics.get('memory_utilization', 0.0),
            current_metrics.get('queue_length', 0.0),
            dt.hour / 24.0,  # Normalized time of day
            dt.weekday() / 7.0  # Normalized day of week
        ]
        
        targets = [
            future_metrics.get('cpu_utilization', 0.0),
            future_metrics.get('memory_utilization', 0.0),
            future_metrics.get('queue_length', 0.0)
        ]
        
        self.training_data['features'].append(features)
        self.training_data['targets'].append(targets)
        
        # Limit training data size
        max_samples = 10000
        if len(self.training_data['features']) > max_samples:
            self.training_data['features'] = self.training_data['features'][-max_samples:]
            self.training_data['targets'] = self.training_data['targets'][-max_samples:]
    
    def train_model(self):
        """Train the predictive model."""
        if len(self.training_data['features']) < 100:
            logger.warning("Insufficient training data for predictive model")
            return
        
        try:
            # Simple linear regression model
            X = np.array(self.training_data['features'])
            y = np.array(self.training_data['targets'])
            
            # Add bias term
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            # Least squares solution
            self.model_weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
            self.is_trained = True
            
            logger.info(f"Trained predictive model with {len(self.training_data['features'])} samples")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def predict(self, current_metrics: Dict[str, float], 
               time_horizon: float = 300.0) -> ResourcePrediction:
        """Predict future resource requirements."""
        if not self.is_trained:
            # Return current values as prediction
            return ResourcePrediction(
                timestamp=time.time(),
                predicted_cpu_usage=current_metrics.get('cpu_utilization', 50.0),
                predicted_memory_usage=current_metrics.get('memory_utilization', 50.0),
                predicted_queue_length=current_metrics.get('queue_length', 0.0),
                predicted_throughput=current_metrics.get('throughput', 1.0),
                confidence=0.5,
                time_horizon=time_horizon
            )
        
        try:
            # Prepare features
            current_time = time.time() + time_horizon
            dt = datetime.fromtimestamp(current_time)
            
            features = [
                current_metrics.get('cpu_utilization', 0.0),
                current_metrics.get('memory_utilization', 0.0),
                current_metrics.get('queue_length', 0.0),
                dt.hour / 24.0,
                dt.weekday() / 7.0
            ]
            
            # Add bias term
            features_with_bias = np.array([1.0] + features)
            
            # Predict
            predictions = features_with_bias @ self.model_weights
            
            # Calculate confidence (simplified)
            confidence = min(1.0, max(0.1, 1.0 - abs(predictions[0] - current_metrics.get('cpu_utilization', 0.0)) / 100.0))
            
            return ResourcePrediction(
                timestamp=time.time(),
                predicted_cpu_usage=max(0.0, min(100.0, predictions[0])),
                predicted_memory_usage=max(0.0, min(100.0, predictions[1])),
                predicted_queue_length=max(0.0, predictions[2]),
                predicted_throughput=current_metrics.get('throughput', 1.0),
                confidence=confidence,
                time_horizon=time_horizon
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback prediction
            return ResourcePrediction(
                timestamp=time.time(),
                predicted_cpu_usage=current_metrics.get('cpu_utilization', 50.0),
                predicted_memory_usage=current_metrics.get('memory_utilization', 50.0),
                predicted_queue_length=current_metrics.get('queue_length', 0.0),
                predicted_throughput=current_metrics.get('throughput', 1.0),
                confidence=0.3,
                time_horizon=time_horizon
            )


class InstanceManager:
    """Manages compute instances."""
    
    def __init__(self):
        self.instances = {}
        self.instance_templates = {
            InstanceType.CPU_OPTIMIZED: {
                'cpu_cores': 8,
                'memory_gb': 16.0,
                'gpu_count': 0,
                'cost_per_hour': 0.40
            },
            InstanceType.MEMORY_OPTIMIZED: {
                'cpu_cores': 4,
                'memory_gb': 32.0,
                'gpu_count': 0,
                'cost_per_hour': 0.50
            },
            InstanceType.GPU_ENABLED: {
                'cpu_cores': 8,
                'memory_gb': 32.0,
                'gpu_count': 1,
                'cost_per_hour': 1.20
            },
            InstanceType.BALANCED: {
                'cpu_cores': 4,
                'memory_gb': 16.0,
                'gpu_count': 0,
                'cost_per_hour': 0.30
            }
        }
        self.lock = threading.RLock()
    
    async def launch_instance(self, instance_type: InstanceType, 
                            region: str = "local") -> ComputeInstance:
        """Launch a new compute instance."""
        template = self.instance_templates[instance_type]
        
        instance = ComputeInstance(
            instance_id=f"{instance_type.value}_{uuid.uuid4()}",
            instance_type=instance_type,
            cpu_cores=template['cpu_cores'],
            memory_gb=template['memory_gb'],
            gpu_count=template['gpu_count'],
            status="pending",
            created_at=time.time(),
            last_heartbeat=time.time(),
            cost_per_hour=template['cost_per_hour'],
            region=region
        )
        
        with self.lock:
            self.instances[instance.instance_id] = instance
        
        # Simulate instance launch
        await asyncio.sleep(2)  # Simulated launch time
        
        instance.status = "running"
        instance.last_heartbeat = time.time()
        
        logger.info(f"Launched instance {instance.instance_id} ({instance_type.value})")
        return instance
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a compute instance."""
        with self.lock:
            if instance_id not in self.instances:
                return False
            
            instance = self.instances[instance_id]
            instance.status = "stopping"
        
        # Simulate termination
        await asyncio.sleep(1)
        
        with self.lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
        
        logger.info(f"Terminated instance {instance_id}")
        return True
    
    def get_active_instances(self) -> List[ComputeInstance]:
        """Get list of active instances."""
        with self.lock:
            return [
                instance for instance in self.instances.values()
                if instance.status == "running"
            ]
    
    def get_total_capacity(self) -> Dict[str, float]:
        """Get total compute capacity."""
        active_instances = self.get_active_instances()
        
        return {
            'total_cpu_cores': sum(i.cpu_cores for i in active_instances),
            'total_memory_gb': sum(i.memory_gb for i in active_instances),
            'total_gpu_count': sum(i.gpu_count for i in active_instances),
            'instance_count': len(active_instances),
            'total_cost_per_hour': sum(i.cost_per_hour for i in active_instances)
        }
    
    def update_instance_load(self, instance_id: str, cpu_load: float, 
                            memory_load: float, task_count: int):
        """Update instance load metrics."""
        with self.lock:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                instance.current_load = (cpu_load + memory_load) / 2.0
                instance.optimization_tasks = task_count
                instance.last_heartbeat = time.time()


class AutoScalingOrchestrator:
    """Main auto-scaling orchestrator."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Components
        self.metrics_collector = MetricsCollector()
        self.predictive_model = PredictiveModel()
        self.instance_manager = InstanceManager()
        
        # Scaling rules
        self.scaling_rules = {}
        self.scaling_events = deque(maxlen=1000)
        
        # State
        self.running = False
        self.min_instances = self.config.get('min_instances', 1)
        self.max_instances = self.config.get('max_instances', 10)
        self.target_utilization = self.config.get('target_utilization', 70.0)
        
        # Performance tracking
        self.cost_optimization_enabled = self.config.get('cost_optimization', True)
        self.performance_target = self.config.get('performance_target', 'balanced')  # cost, performance, balanced
        
        # Initialize default scaling rules
        self._initialize_default_rules()
        
        # Background tasks
        self.monitoring_task = None
        self.scaling_task = None
        self.prediction_task = None
    
    def _initialize_default_rules(self):
        """Initialize default scaling rules."""
        # CPU-based scaling
        cpu_rule = ScalingRule(
            rule_id="cpu_scaling",
            metric=ScalingMetric.CPU_UTILIZATION,
            threshold_up=80.0,
            threshold_down=20.0,
            action_up=ScalingAction.SCALE_OUT,
            action_down=ScalingAction.SCALE_IN,
            cooldown_period=300.0,  # 5 minutes
            evaluation_window=180.0,  # 3 minutes
            priority=1
        )
        
        # Memory-based scaling
        memory_rule = ScalingRule(
            rule_id="memory_scaling",
            metric=ScalingMetric.MEMORY_UTILIZATION,
            threshold_up=85.0,
            threshold_down=25.0,
            action_up=ScalingAction.SCALE_OUT,
            action_down=ScalingAction.SCALE_IN,
            cooldown_period=300.0,
            evaluation_window=180.0,
            priority=2
        )
        
        # Queue-based scaling
        queue_rule = ScalingRule(
            rule_id="queue_scaling",
            metric=ScalingMetric.QUEUE_LENGTH,
            threshold_up=10.0,
            threshold_down=2.0,
            action_up=ScalingAction.SCALE_OUT,
            action_down=ScalingAction.SCALE_IN,
            cooldown_period=180.0,  # 3 minutes
            evaluation_window=120.0,  # 2 minutes
            priority=3
        )
        
        self.scaling_rules.update({
            cpu_rule.rule_id: cpu_rule,
            memory_rule.rule_id: memory_rule,
            queue_rule.rule_id: queue_rule
        })
    
    async def start(self):
        """Start the auto-scaling orchestrator."""
        self.running = True
        
        # Launch minimum instances
        await self._ensure_minimum_instances()
        
        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        self.prediction_task = asyncio.create_task(self._prediction_loop())
        
        logger.info("Auto-scaling orchestrator started")
    
    async def stop(self):
        """Stop the auto-scaling orchestrator."""
        self.running = False
        
        # Cancel background tasks
        tasks = [self.monitoring_task, self.scaling_task, self.prediction_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Auto-scaling orchestrator stopped")
    
    async def _ensure_minimum_instances(self):
        """Ensure minimum number of instances are running."""
        active_instances = self.instance_manager.get_active_instances()
        
        while len(active_instances) < self.min_instances:
            instance_type = self._select_optimal_instance_type()
            await self.instance_manager.launch_instance(instance_type)
            active_instances = self.instance_manager.get_active_instances()
    
    def _select_optimal_instance_type(self, workload_characteristics: Dict[str, float] = None) -> InstanceType:
        """Select optimal instance type based on workload."""
        if workload_characteristics is None:
            workload_characteristics = {}
        
        cpu_intensive = workload_characteristics.get('cpu_intensive', False)
        memory_intensive = workload_characteristics.get('memory_intensive', False)
        gpu_required = workload_characteristics.get('gpu_required', False)
        
        if gpu_required:
            return InstanceType.GPU_ENABLED
        elif cpu_intensive:
            return InstanceType.CPU_OPTIMIZED
        elif memory_intensive:
            return InstanceType.MEMORY_OPTIMIZED
        else:
            return InstanceType.BALANCED
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Update instance metrics
                await self._update_instance_metrics()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        current_time = time.time()
        
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.record_metric(ScalingMetric.CPU_UTILIZATION, cpu_percent, current_time)
        
        # Memory utilization
        memory = psutil.virtual_memory()
        self.metrics_collector.record_metric(ScalingMetric.MEMORY_UTILIZATION, memory.percent, current_time)
        
        # Mock additional metrics
        queue_length = np.random.poisson(5)  # Mock queue length
        response_time = np.random.exponential(0.5)  # Mock response time
        throughput = np.random.gamma(2, 2)  # Mock throughput
        error_rate = np.random.beta(1, 20) * 100  # Mock error rate
        
        self.metrics_collector.record_metric(ScalingMetric.QUEUE_LENGTH, queue_length, current_time)
        self.metrics_collector.record_metric(ScalingMetric.RESPONSE_TIME, response_time, current_time)
        self.metrics_collector.record_metric(ScalingMetric.THROUGHPUT, throughput, current_time)
        self.metrics_collector.record_metric(ScalingMetric.ERROR_RATE, error_rate, current_time)
    
    async def _update_instance_metrics(self):
        """Update metrics for individual instances."""
        active_instances = self.instance_manager.get_active_instances()
        
        for instance in active_instances:
            # Mock instance-specific metrics
            cpu_load = np.random.uniform(10, 90)
            memory_load = np.random.uniform(10, 80)
            task_count = np.random.randint(0, 10)
            
            self.instance_manager.update_instance_load(
                instance.instance_id, cpu_load, memory_load, task_count
            )
    
    async def _scaling_loop(self):
        """Background scaling decision loop."""
        while self.running:
            try:
                # Evaluate scaling rules
                scaling_decisions = await self._evaluate_scaling_rules()
                
                # Execute scaling actions
                for decision in scaling_decisions:
                    await self._execute_scaling_action(decision)
                
                await asyncio.sleep(60)  # Evaluate every minute
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(120)
    
    async def _evaluate_scaling_rules(self) -> List[Dict[str, Any]]:
        """Evaluate all scaling rules and return decisions."""
        current_time = time.time()
        decisions = []
        
        # Sort rules by priority
        sorted_rules = sorted(self.scaling_rules.values(), key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if current_time - rule.last_triggered < rule.cooldown_period:
                continue
            
            # Get metric value
            metric_value = self.metrics_collector.get_average(rule.metric, rule.evaluation_window)
            if metric_value is None:
                continue
            
            # Evaluate thresholds
            action = ScalingAction.NO_ACTION
            
            if metric_value > rule.threshold_up:
                action = rule.action_up
                threshold = rule.threshold_up
            elif metric_value < rule.threshold_down:
                action = rule.action_down
                threshold = rule.threshold_down
            
            if action != ScalingAction.NO_ACTION:
                decisions.append({
                    'rule': rule,
                    'action': action,
                    'metric_value': metric_value,
                    'threshold': threshold,
                    'reason': f"{rule.metric.value} {metric_value:.1f} vs threshold {threshold:.1f}"
                })
                
                rule.last_triggered = current_time
        
        return decisions
    
    async def _execute_scaling_action(self, decision: Dict[str, Any]):
        """Execute a scaling action."""
        action = decision['action']
        rule = decision['rule']
        
        start_time = time.time()
        active_instances = self.instance_manager.get_active_instances()
        instances_before = len(active_instances)
        
        success = True
        instances_after = instances_before
        
        try:
            if action == ScalingAction.SCALE_OUT:
                # Add instance
                if instances_before < self.max_instances:
                    instance_type = self._select_optimal_instance_type()
                    await self.instance_manager.launch_instance(instance_type)
                    instances_after = instances_before + 1
                    logger.info(f"Scaled out: added {instance_type.value} instance")
                else:
                    logger.warning("Cannot scale out: at maximum instances")
                    success = False
            
            elif action == ScalingAction.SCALE_IN:
                # Remove instance
                if instances_before > self.min_instances:
                    # Select instance to terminate (least loaded)
                    least_loaded = min(active_instances, key=lambda i: i.current_load)
                    await self.instance_manager.terminate_instance(least_loaded.instance_id)
                    instances_after = instances_before - 1
                    logger.info(f"Scaled in: removed instance {least_loaded.instance_id}")
                else:
                    logger.warning("Cannot scale in: at minimum instances")
                    success = False
            
            duration = time.time() - start_time
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=start_time,
                action=action,
                metric=rule.metric,
                metric_value=decision['metric_value'],
                threshold=decision['threshold'],
                instances_before=instances_before,
                instances_after=instances_after,
                reason=decision['reason'],
                duration=duration,
                success=success
            )
            
            self.scaling_events.append(event)
            
        except Exception as e:
            logger.error(f"Scaling action failed: {e}")
            
            # Record failed event
            event = ScalingEvent(
                timestamp=start_time,
                action=action,
                metric=rule.metric,
                metric_value=decision['metric_value'],
                threshold=decision['threshold'],
                instances_before=instances_before,
                instances_after=instances_before,
                reason=f"Failed: {e}",
                duration=time.time() - start_time,
                success=False
            )
            
            self.scaling_events.append(event)
    
    async def _prediction_loop(self):
        """Background prediction and proactive scaling loop."""
        while self.running:
            try:
                # Collect current metrics for training
                current_metrics = await self._get_current_metrics()
                
                # Train predictive model periodically
                if len(self.predictive_model.training_data['features']) % 100 == 0:
                    self.predictive_model.train_model()
                
                # Make predictions
                prediction = self.predictive_model.predict(current_metrics, time_horizon=600.0)  # 10 minutes
                
                # Proactive scaling based on predictions
                if prediction.confidence > 0.7:
                    await self._proactive_scaling(prediction)
                
                await asyncio.sleep(300)  # Predict every 5 minutes
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(600)
    
    async def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        return {
            'cpu_utilization': self.metrics_collector.get_average(ScalingMetric.CPU_UTILIZATION, 60.0) or 0.0,
            'memory_utilization': self.metrics_collector.get_average(ScalingMetric.MEMORY_UTILIZATION, 60.0) or 0.0,
            'queue_length': self.metrics_collector.get_average(ScalingMetric.QUEUE_LENGTH, 60.0) or 0.0,
            'response_time': self.metrics_collector.get_average(ScalingMetric.RESPONSE_TIME, 60.0) or 0.0,
            'throughput': self.metrics_collector.get_average(ScalingMetric.THROUGHPUT, 60.0) or 0.0,
            'error_rate': self.metrics_collector.get_average(ScalingMetric.ERROR_RATE, 60.0) or 0.0
        }
    
    async def _proactive_scaling(self, prediction: ResourcePrediction):
        """Perform proactive scaling based on predictions."""
        if prediction.predicted_cpu_usage > 85.0 or prediction.predicted_memory_usage > 85.0:
            # Preemptively scale out
            active_instances = self.instance_manager.get_active_instances()
            if len(active_instances) < self.max_instances:
                instance_type = self._select_optimal_instance_type()
                await self.instance_manager.launch_instance(instance_type)
                logger.info(f"Proactive scale out based on prediction: CPU {prediction.predicted_cpu_usage:.1f}%")
        
        elif prediction.predicted_cpu_usage < 30.0 and prediction.predicted_memory_usage < 30.0:
            # Consider scaling in
            active_instances = self.instance_manager.get_active_instances()
            if len(active_instances) > self.min_instances:
                # Wait for actual low utilization before scaling in
                pass  # Conservative approach for scale-in
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules[rule.rule_id] = rule
        logger.info(f"Added scaling rule: {rule.rule_id}")
    
    def remove_scaling_rule(self, rule_id: str) -> bool:
        """Remove a scaling rule."""
        if rule_id in self.scaling_rules:
            del self.scaling_rules[rule_id]
            logger.info(f"Removed scaling rule: {rule_id}")
            return True
        return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status."""
        active_instances = self.instance_manager.get_active_instances()
        capacity = self.instance_manager.get_total_capacity()
        
        # Recent scaling events
        recent_events = list(self.scaling_events)[-10:]
        
        # Current metrics
        current_metrics = {}
        for metric in ScalingMetric:
            value = self.metrics_collector.get_average(metric, 300.0)
            if value is not None:
                current_metrics[metric.value] = value
        
        return {
            'running': self.running,
            'instances': {
                'active_count': len(active_instances),
                'min_instances': self.min_instances,
                'max_instances': self.max_instances,
                'capacity': capacity
            },
            'metrics': current_metrics,
            'scaling_rules': {
                'total_rules': len(self.scaling_rules),
                'enabled_rules': len([r for r in self.scaling_rules.values() if r.enabled])
            },
            'recent_events': [
                {
                    'timestamp': e.timestamp,
                    'action': e.action.value,
                    'metric': e.metric.value,
                    'reason': e.reason,
                    'success': e.success
                }
                for e in recent_events
            ],
            'cost_per_hour': capacity.get('total_cost_per_hour', 0.0)
        }
    
    async def optimize_for_cost(self):
        """Optimize the cluster for cost efficiency."""
        active_instances = self.instance_manager.get_active_instances()
        
        # Sort by cost efficiency (performance per dollar)
        def cost_efficiency(instance):
            performance = instance.performance_score
            cost = instance.cost_per_hour
            return performance / (cost + 0.01)  # Avoid division by zero
        
        sorted_instances = sorted(active_instances, key=cost_efficiency, reverse=True)
        
        # Identify underutilized expensive instances
        for instance in sorted_instances:
            if (instance.current_load < 30.0 and 
                instance.cost_per_hour > 0.5 and 
                len(active_instances) > self.min_instances):
                
                # Consider replacing with cheaper instance
                await self.instance_manager.terminate_instance(instance.instance_id)
                
                # Launch cheaper alternative
                cheaper_type = InstanceType.BALANCED
                await self.instance_manager.launch_instance(cheaper_type)
                
                logger.info(f"Cost optimization: replaced expensive instance {instance.instance_id}")
                break  # One at a time


# Example usage
async def demonstrate_auto_scaling():
    """Demonstrate auto-scaling capabilities."""
    print("⚡ Auto-Scaling Orchestrator Demonstration")
    print("=" * 60)
    
    # Create auto-scaling orchestrator
    config = {
        'min_instances': 2,
        'max_instances': 8,
        'target_utilization': 70.0,
        'cost_optimization': True
    }
    
    orchestrator = AutoScalingOrchestrator(config)
    
    try:
        # Start orchestrator
        print("🚀 Starting auto-scaling orchestrator...")
        await orchestrator.start()
        
        # Simulate workload
        print("📈 Simulating variable workload...")
        
        for i in range(20):
            # Simulate increasing load
            cpu_load = 50 + i * 2  # Increasing CPU load
            memory_load = 40 + i * 1.5  # Increasing memory load
            queue_length = i // 2  # Increasing queue
            
            # Record metrics
            current_time = time.time()
            orchestrator.metrics_collector.record_metric(ScalingMetric.CPU_UTILIZATION, cpu_load, current_time)
            orchestrator.metrics_collector.record_metric(ScalingMetric.MEMORY_UTILIZATION, memory_load, current_time)
            orchestrator.metrics_collector.record_metric(ScalingMetric.QUEUE_LENGTH, queue_length, current_time)
            
            # Check status periodically
            if i % 5 == 0:
                status = orchestrator.get_scaling_status()
                print(f"   Step {i}: Instances: {status['instances']['active_count']}, "
                      f"CPU: {cpu_load:.1f}%, Memory: {memory_load:.1f}%, "
                      f"Cost: ${status['cost_per_hour']:.2f}/hour")
            
            await asyncio.sleep(0.5)  # Fast simulation
        
        # Let auto-scaling react
        print("⏳ Waiting for auto-scaling reactions...")
        await asyncio.sleep(5)
        
        # Final status
        final_status = orchestrator.get_scaling_status()
        print(f"\n📊 Final Status:")
        print(f"   Active instances: {final_status['instances']['active_count']}")
        print(f"   Total CPU cores: {final_status['instances']['capacity']['total_cpu_cores']}")
        print(f"   Total memory: {final_status['instances']['capacity']['total_memory_gb']:.1f} GB")
        print(f"   Cost per hour: ${final_status['instances']['capacity']['total_cost_per_hour']:.2f}")
        
        print(f"\n🎯 Recent Scaling Events:")
        for event in final_status['recent_events'][-5:]:
            print(f"   {event['action']}: {event['reason']} ({'✅' if event['success'] else '❌'})")
        
    finally:
        # Stop orchestrator
        print("\n🛑 Stopping auto-scaling orchestrator...")
        await orchestrator.stop()
    
    print("\n" + "=" * 60)
    return orchestrator


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_auto_scaling())