"""
Comprehensive Monitoring and Observability System
Provides real-time monitoring, metrics collection, and alerting capabilities.
"""

import asyncio
import logging
import time
import json
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from collections import defaultdict, deque
from pathlib import Path
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricValue:
    """Individual metric measurement."""
    timestamp: datetime
    value: Union[float, int]
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition and storage."""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_value(self, value: Union[float, int], labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Add a new value to the metric."""
        metric_value = MetricValue(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        self.values.append(metric_value)
    
    def get_latest_value(self) -> Optional[MetricValue]:
        """Get the most recent metric value."""
        return self.values[-1] if self.values else None
    
    def get_values_in_range(self, start_time: datetime, end_time: datetime) -> List[MetricValue]:
        """Get metric values within a time range."""
        return [
            value for value in self.values
            if start_time <= value.timestamp <= end_time
        ]


@dataclass
class Alert:
    """Alert definition."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    metric_name: str
    duration: timedelta
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class AlertEvent:
    """Alert event instance."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold: float
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    description: str
    check_function: Callable
    interval: float
    timeout: float
    critical: bool = False
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_status: Optional[HealthStatus] = None
    last_result: Optional[Dict[str, Any]] = None
    failure_count: int = 0
    success_count: int = 0


class ComprehensiveMonitoringSystem:
    """Advanced monitoring and observability system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Metrics storage
        self.metrics: Dict[str, Metric] = {}
        self.metric_lock = threading.RLock()
        
        # Alerting system
        self.alerts: Dict[str, Alert] = {}
        self.alert_events: List[AlertEvent] = []
        self.alert_handlers: List[Callable] = []
        
        # Health monitoring
        self.health_checks: Dict[str, HealthCheck] = {}
        self.overall_health: HealthStatus = HealthStatus.HEALTHY
        self.health_history: deque = deque(maxlen=100)
        
        # Performance tracking
        self.performance_baselines: Dict[str, float] = {}
        self.performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # System metrics collection
        self.system_metrics_enabled = True
        self.custom_collectors: List[Callable] = []
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.alert_check_task: Optional[asyncio.Task] = None
        
        # Initialize default metrics and health checks
        self._initialize_default_monitoring()
    
    def _initialize_default_monitoring(self):
        """Initialize default metrics and health checks."""
        
        # System metrics
        self.create_metric("system_cpu_percent", MetricType.GAUGE, "CPU usage percentage", "%")
        self.create_metric("system_memory_percent", MetricType.GAUGE, "Memory usage percentage", "%")
        self.create_metric("system_disk_usage", MetricType.GAUGE, "Disk usage percentage", "%")
        self.create_metric("system_load_average", MetricType.GAUGE, "System load average", "")
        
        # Application metrics
        self.create_metric("app_request_count", MetricType.COUNTER, "Total requests processed", "requests")
        self.create_metric("app_request_duration", MetricType.HISTOGRAM, "Request processing time", "seconds")
        self.create_metric("app_error_count", MetricType.COUNTER, "Total errors encountered", "errors")
        self.create_metric("app_active_connections", MetricType.GAUGE, "Active connections", "connections")
        
        # Optimization metrics
        self.create_metric("optimization_duration", MetricType.HISTOGRAM, "Optimization execution time", "seconds")
        self.create_metric("optimization_iterations", MetricType.HISTOGRAM, "Optimization iterations", "iterations")
        self.create_metric("optimization_convergence", MetricType.GAUGE, "Optimization convergence quality", "")
        self.create_metric("optimization_success_rate", MetricType.GAUGE, "Optimization success rate", "%")
        
        # Hardware metrics
        self.create_metric("hardware_temperature", MetricType.GAUGE, "Hardware temperature", "°C")
        self.create_metric("hardware_power_consumption", MetricType.GAUGE, "Power consumption", "W")
        self.create_metric("hardware_response_time", MetricType.HISTOGRAM, "Hardware response time", "ms")
        
        # Default alerts
        self.create_alert(
            "high_cpu_usage", "High CPU Usage", "CPU usage exceeds threshold",
            AlertSeverity.WARNING, "system_cpu_percent > 80", 80.0,
            "system_cpu_percent", timedelta(minutes=5)
        )
        
        self.create_alert(
            "high_memory_usage", "High Memory Usage", "Memory usage exceeds threshold",
            AlertSeverity.CRITICAL, "system_memory_percent > 90", 90.0,
            "system_memory_percent", timedelta(minutes=2)
        )
        
        self.create_alert(
            "optimization_failures", "Optimization Failures", "High optimization failure rate",
            AlertSeverity.WARNING, "optimization_success_rate < 70", 70.0,
            "optimization_success_rate", timedelta(minutes=10)
        )
        
        # Default health checks
        self.add_health_check(
            "system_resources", "System Resource Check",
            self._check_system_resources, 30.0, 5.0, critical=True
        )
        
        self.add_health_check(
            "application_health", "Application Health Check",
            self._check_application_health, 60.0, 10.0, critical=True
        )
        
        self.add_health_check(
            "database_connectivity", "Database Connectivity Check",
            self._check_database_connectivity, 120.0, 15.0, critical=False
        )
    
    def create_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str,
        labels: Dict[str, str] = None
    ) -> Metric:
        """Create a new metric."""
        
        with self.metric_lock:
            metric = Metric(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit,
                labels=labels or {}
            )
            self.metrics[name] = metric
            
            logger.info(f"Created metric: {name} ({metric_type.value})")
            return metric
    
    def record_metric(
        self,
        name: str,
        value: Union[float, int],
        labels: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Record a metric value."""
        
        with self.metric_lock:
            if name in self.metrics:
                self.metrics[name].add_value(value, labels, metadata)
            else:
                logger.warning(f"Metric {name} not found, creating gauge metric")
                self.create_metric(name, MetricType.GAUGE, f"Auto-created metric {name}", "")
                self.metrics[name].add_value(value, labels, metadata)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        
        with self.metric_lock:
            if name in self.metrics and self.metrics[name].metric_type == MetricType.COUNTER:
                current_value = 0
                if self.metrics[name].values:
                    current_value = self.metrics[name].get_latest_value().value
                self.record_metric(name, current_value + value, labels)
            else:
                self.record_metric(name, value, labels)
    
    def time_operation(self, metric_name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations."""
        
        class TimingContext:
            def __init__(self, monitoring_system, name, labels):
                self.monitoring_system = monitoring_system
                self.name = name
                self.labels = labels
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.monitoring_system.record_metric(self.name, duration, self.labels)
        
        return TimingContext(self, metric_name, labels)
    
    def create_alert(
        self,
        alert_id: str,
        name: str,
        description: str,
        severity: AlertSeverity,
        condition: str,
        threshold: float,
        metric_name: str,
        duration: timedelta,
        labels: Dict[str, str] = None
    ) -> Alert:
        """Create a new alert."""
        
        alert = Alert(
            id=alert_id,
            name=name,
            description=description,
            severity=severity,
            condition=condition,
            threshold=threshold,
            metric_name=metric_name,
            duration=duration,
            labels=labels or {}
        )
        
        self.alerts[alert_id] = alert
        logger.info(f"Created alert: {name} ({severity.value})")
        return alert
    
    def add_alert_handler(self, handler: Callable[[AlertEvent], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info("Added alert handler")
    
    def add_health_check(
        self,
        name: str,
        description: str,
        check_function: Callable,
        interval: float,
        timeout: float,
        critical: bool = False
    ) -> HealthCheck:
        """Add a health check."""
        
        health_check = HealthCheck(
            name=name,
            description=description,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            critical=critical
        )
        
        self.health_checks[name] = health_check
        logger.info(f"Added health check: {name} (critical: {critical})")
        return health_check
    
    def add_custom_collector(self, collector: Callable[[], Dict[str, float]]):
        """Add a custom metrics collector function."""
        self.custom_collectors.append(collector)
        logger.info("Added custom metrics collector")
    
    async def start_monitoring(self):
        """Start all monitoring tasks."""
        
        logger.info("Starting comprehensive monitoring system")
        
        # Start system metrics collection
        if self.system_metrics_enabled:
            self.monitoring_task = asyncio.create_task(self._collect_system_metrics_loop())
        
        # Start health checks
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start alert evaluation
        self.alert_check_task = asyncio.create_task(self._alert_check_loop())
        
        logger.info("Monitoring system started successfully")
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks."""
        
        logger.info("Stopping monitoring system")
        
        tasks = [self.monitoring_task, self.health_check_task, self.alert_check_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Monitoring system stopped")
    
    async def _collect_system_metrics_loop(self):
        """Background task for collecting system metrics."""
        
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self.record_metric("system_cpu_percent", cpu_percent)
                self.record_metric("system_memory_percent", memory.percent)
                self.record_metric("system_disk_usage", disk.percent)
                
                # Load average (if available)
                if hasattr(psutil, 'getloadavg'):
                    load_avg = psutil.getloadavg()[0]  # 1-minute load average
                    self.record_metric("system_load_average", load_avg)
                
                # Collect custom metrics
                for collector in self.custom_collectors:
                    try:
                        custom_metrics = await asyncio.wait_for(
                            asyncio.create_task(collector()) if asyncio.iscoroutinefunction(collector) else asyncio.to_thread(collector),
                            timeout=10.0
                        )
                        for metric_name, value in custom_metrics.items():
                            self.record_metric(metric_name, value)
                    except Exception as e:
                        logger.error(f"Error in custom metrics collector: {e}")
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _health_check_loop(self):
        """Background task for running health checks."""
        
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                for health_check in self.health_checks.values():
                    if not health_check.enabled:
                        continue
                    
                    # Check if it's time to run this health check
                    if (health_check.last_run is None or 
                        (current_time - health_check.last_run).total_seconds() >= health_check.interval):
                        
                        await self._run_health_check(health_check)
                
                # Update overall health status
                self._update_overall_health()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _run_health_check(self, health_check: HealthCheck):
        """Run a single health check."""
        
        try:
            start_time = time.time()
            
            # Run the health check with timeout
            if asyncio.iscoroutinefunction(health_check.check_function):
                result = await asyncio.wait_for(
                    health_check.check_function(),
                    timeout=health_check.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(health_check.check_function),
                    timeout=health_check.timeout
                )
            
            duration = time.time() - start_time
            
            # Determine status from result
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'healthy'))
                health_check.last_result = result
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                health_check.last_result = {'status': status.value, 'success': result}
            else:
                status = HealthStatus.HEALTHY
                health_check.last_result = {'status': status.value, 'result': result}
            
            health_check.last_status = status
            health_check.last_run = datetime.now(timezone.utc)
            
            if status == HealthStatus.HEALTHY:
                health_check.success_count += 1
                health_check.failure_count = max(0, health_check.failure_count - 1)
            else:
                health_check.failure_count += 1
                health_check.success_count = 0
            
            # Record health check metric
            self.record_metric(
                f"health_check_{health_check.name}_duration",
                duration,
                {"status": status.value}
            )
            
            self.record_metric(
                f"health_check_{health_check.name}_status",
                1.0 if status == HealthStatus.HEALTHY else 0.0
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Health check {health_check.name} timed out")
            health_check.last_status = HealthStatus.UNHEALTHY
            health_check.last_result = {'status': 'unhealthy', 'error': 'timeout'}
            health_check.failure_count += 1
            health_check.last_run = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Health check {health_check.name} failed: {e}")
            health_check.last_status = HealthStatus.UNHEALTHY
            health_check.last_result = {'status': 'unhealthy', 'error': str(e)}
            health_check.failure_count += 1
            health_check.last_run = datetime.now(timezone.utc)
    
    def _update_overall_health(self):
        """Update the overall system health status."""
        
        critical_checks = [hc for hc in self.health_checks.values() if hc.critical and hc.enabled]
        non_critical_checks = [hc for hc in self.health_checks.values() if not hc.critical and hc.enabled]
        
        # Check critical health checks
        critical_unhealthy = any(
            hc.last_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
            for hc in critical_checks
            if hc.last_status is not None
        )
        
        critical_degraded = any(
            hc.last_status == HealthStatus.DEGRADED
            for hc in critical_checks
            if hc.last_status is not None
        )
        
        # Check non-critical health checks
        non_critical_issues = sum(
            1 for hc in non_critical_checks
            if hc.last_status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED, HealthStatus.CRITICAL]
            and hc.last_status is not None
        )
        
        # Determine overall health
        if critical_unhealthy:
            new_status = HealthStatus.CRITICAL
        elif critical_degraded:
            new_status = HealthStatus.DEGRADED
        elif non_critical_issues >= len(non_critical_checks) * 0.5:  # More than 50% non-critical issues
            new_status = HealthStatus.DEGRADED
        elif non_critical_issues > 0:
            new_status = HealthStatus.DEGRADED
        else:
            new_status = HealthStatus.HEALTHY
        
        if new_status != self.overall_health:
            logger.info(f"Overall health status changed: {self.overall_health.value} -> {new_status.value}")
            self.overall_health = new_status
        
        # Record health status
        self.health_history.append({
            'timestamp': datetime.now(timezone.utc),
            'status': self.overall_health.value,
            'critical_checks': len(critical_checks),
            'non_critical_issues': non_critical_issues
        })
        
        self.record_metric("system_health_status", {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.7,
            HealthStatus.UNHEALTHY: 0.3,
            HealthStatus.CRITICAL: 0.0
        }[self.overall_health])
    
    async def _alert_check_loop(self):
        """Background task for evaluating alerts."""
        
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                for alert in self.alerts.values():
                    if not alert.enabled:
                        continue
                    
                    await self._evaluate_alert(alert, current_time)
                
                await asyncio.sleep(30)  # Check alerts every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alert check loop: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_alert(self, alert: Alert, current_time: datetime):
        """Evaluate a single alert."""
        
        try:
            # Get metric values for the alert duration
            start_time = current_time - alert.duration
            
            if alert.metric_name not in self.metrics:
                return
            
            metric = self.metrics[alert.metric_name]
            recent_values = metric.get_values_in_range(start_time, current_time)
            
            if not recent_values:
                return
            
            # Evaluate alert condition
            triggered = self._evaluate_alert_condition(alert, recent_values)
            
            if triggered and (alert.last_triggered is None or 
                            (current_time - alert.last_triggered) > alert.duration):
                
                # Create alert event
                latest_value = recent_values[-1]
                alert_event = AlertEvent(
                    alert_id=alert.id,
                    timestamp=current_time,
                    severity=alert.severity,
                    message=f"{alert.name}: {alert.description}",
                    metric_value=latest_value.value,
                    threshold=alert.threshold,
                    labels=alert.labels
                )
                
                self.alert_events.append(alert_event)
                alert.last_triggered = current_time
                alert.trigger_count += 1
                
                # Notify alert handlers
                for handler in self.alert_handlers:
                    try:
                        await asyncio.create_task(handler(alert_event))
                    except Exception as e:
                        logger.error(f"Error in alert handler: {e}")
                
                logger.warning(f"Alert triggered: {alert.name} (value: {latest_value.value}, threshold: {alert.threshold})")
                
        except Exception as e:
            logger.error(f"Error evaluating alert {alert.id}: {e}")
    
    def _evaluate_alert_condition(self, alert: Alert, values: List[MetricValue]) -> bool:
        """Evaluate if alert condition is met."""
        
        if not values:
            return False
        
        latest_value = values[-1].value
        
        # Simple threshold evaluation (can be extended for complex conditions)
        if ">" in alert.condition:
            return latest_value > alert.threshold
        elif "<" in alert.condition:
            return latest_value < alert.threshold
        elif "==" in alert.condition:
            return abs(latest_value - alert.threshold) < 0.001
        elif "!=" in alert.condition:
            return abs(latest_value - alert.threshold) >= 0.001
        
        return False
    
    # Default health check functions
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                issues.append("CPU usage critical")
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                issues.append("CPU usage high")
            
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
                issues.append("Memory usage critical")
            elif memory.percent > 85:
                status = HealthStatus.DEGRADED
                issues.append("Memory usage high")
            
            if disk.percent > 95:
                status = HealthStatus.CRITICAL
                issues.append("Disk usage critical")
            elif disk.percent > 90:
                status = HealthStatus.DEGRADED
                issues.append("Disk usage high")
            
            return {
                'status': status.value,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'error': str(e)
            }
    
    async def _check_application_health(self) -> Dict[str, Any]:
        """Check application-specific health."""
        try:
            # Check if main components are responsive
            components_status = {}
            
            # Simulate component checks
            components_status['optimization_engine'] = 'healthy'
            components_status['hardware_interface'] = 'healthy'
            components_status['api_server'] = 'healthy'
            components_status['database'] = 'healthy'
            
            # Check error rates
            error_rate = 0.0  # Would calculate actual error rate
            
            status = HealthStatus.HEALTHY
            if error_rate > 0.1:  # 10% error rate
                status = HealthStatus.DEGRADED
            if error_rate > 0.2:  # 20% error rate
                status = HealthStatus.UNHEALTHY
            
            return {
                'status': status.value,
                'components': components_status,
                'error_rate': error_rate
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'error': str(e)
            }
    
    async def _check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            # Simulate database check
            # In real implementation, this would test actual database connection
            
            connection_time = 0.05  # seconds
            query_time = 0.02  # seconds
            
            status = HealthStatus.HEALTHY
            if connection_time > 1.0:
                status = HealthStatus.DEGRADED
            if connection_time > 5.0:
                status = HealthStatus.UNHEALTHY
            
            return {
                'status': status.value,
                'connection_time': connection_time,
                'query_time': query_time,
                'connected': True
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'error': str(e),
                'connected': False
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        
        summary = {}
        
        with self.metric_lock:
            for name, metric in self.metrics.items():
                latest_value = metric.get_latest_value()
                
                summary[name] = {
                    'type': metric.metric_type.value,
                    'description': metric.description,
                    'unit': metric.unit,
                    'latest_value': latest_value.value if latest_value else None,
                    'latest_timestamp': latest_value.timestamp.isoformat() if latest_value else None,
                    'value_count': len(metric.values),
                    'labels': metric.labels
                }
        
        return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status and individual check results."""
        
        health_checks_status = {}
        for name, health_check in self.health_checks.items():
            health_checks_status[name] = {
                'status': health_check.last_status.value if health_check.last_status else 'unknown',
                'last_run': health_check.last_run.isoformat() if health_check.last_run else None,
                'result': health_check.last_result,
                'failure_count': health_check.failure_count,
                'success_count': health_check.success_count,
                'critical': health_check.critical,
                'enabled': health_check.enabled
            }
        
        return {
            'overall_status': self.overall_health.value,
            'health_checks': health_checks_status,
            'last_update': datetime.now(timezone.utc).isoformat()
        }
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert configuration and recent events."""
        
        active_alerts = [
            alert for alert in self.alerts.values()
            if alert.enabled and alert.last_triggered and
            (datetime.now(timezone.utc) - alert.last_triggered) < timedelta(hours=1)
        ]
        
        recent_events = [
            {
                'alert_id': event.alert_id,
                'timestamp': event.timestamp.isoformat(),
                'severity': event.severity.value,
                'message': event.message,
                'metric_value': event.metric_value,
                'threshold': event.threshold,
                'resolved': event.resolved
            }
            for event in self.alert_events[-50:]  # Last 50 events
        ]
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'recent_events': recent_events,
            'alert_handlers': len(self.alert_handlers)
        }
    
    def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in specified format."""
        
        if format_type == "prometheus":
            return self._export_prometheus_format()
        elif format_type == "json":
            return json.dumps(self.get_metrics_summary(), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        
        lines = []
        
        with self.metric_lock:
            for name, metric in self.metrics.items():
                # Add help and type comments
                lines.append(f"# HELP {name} {metric.description}")
                lines.append(f"# TYPE {name} {metric.metric_type.value}")
                
                # Add latest value
                latest_value = metric.get_latest_value()
                if latest_value:
                    labels_str = ""
                    if latest_value.labels or metric.labels:
                        all_labels = {**metric.labels, **latest_value.labels}
                        labels_list = [f'{k}="{v}"' for k, v in all_labels.items()]
                        labels_str = "{" + ",".join(labels_list) + "}"
                    
                    lines.append(f"{name}{labels_str} {latest_value.value}")
                
                lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)


# Example usage and demonstration
async def demonstrate_monitoring():
    """Demonstrate monitoring system capabilities."""
    
    print("📊 Comprehensive Monitoring System Demonstration")
    print("=" * 60)
    
    # Initialize monitoring system
    monitoring = ComprehensiveMonitoringSystem()
    
    # Add custom alert handler
    async def log_alert_handler(alert_event: AlertEvent):
        print(f"🚨 ALERT: {alert_event.message} (value: {alert_event.metric_value}, threshold: {alert_event.threshold})")
    
    monitoring.add_alert_handler(log_alert_handler)
    
    # Add custom metrics collector
    def custom_metrics_collector():
        return {
            "custom_metric_1": 42.0,
            "custom_metric_2": 3.14
        }
    
    monitoring.add_custom_collector(custom_metrics_collector)
    
    # Start monitoring
    await monitoring.start_monitoring()
    
    # Simulate some activity
    print("\n📈 Recording sample metrics...")
    
    for i in range(5):
        monitoring.record_metric("app_request_count", i * 10)
        monitoring.record_metric("app_request_duration", 0.1 + i * 0.05)
        monitoring.record_metric("optimization_success_rate", 95 - i * 2)
        
        # Simulate high CPU to trigger alert
        if i == 3:
            monitoring.record_metric("system_cpu_percent", 85.0)
        
        await asyncio.sleep(1)
    
    # Wait for monitoring cycles
    await asyncio.sleep(5)
    
    # Display results
    print("\n📊 Metrics Summary:")
    metrics_summary = monitoring.get_metrics_summary()
    for name, info in list(metrics_summary.items())[:5]:  # Show first 5 metrics
        print(f"  {name}: {info['latest_value']} {info['unit']}")
    
    print("\n🏥 Health Status:")
    health_status = monitoring.get_health_status()
    print(f"  Overall: {health_status['overall_status']}")
    
    print("\n🚨 Alert Summary:")
    alert_summary = monitoring.get_alert_summary()
    print(f"  Total alerts: {alert_summary['total_alerts']}")
    print(f"  Recent events: {len(alert_summary['recent_events'])}")
    
    # Stop monitoring
    await monitoring.stop_monitoring()
    
    print("\n" + "=" * 60)
    return monitoring


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_monitoring())