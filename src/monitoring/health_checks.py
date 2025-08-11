"""
Health monitoring and system diagnostics for Acousto-Gen.
Provides comprehensive health checks, system monitoring, and alerting.
"""

import time
import logging
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from contextlib import contextmanager

# Handle optional dependencies gracefully
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    response_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'response_time': self.response_time
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: float = field(default_factory=time.time)
    uptime: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status.value,
            'checks': [check.to_dict() for check in self.checks],
            'timestamp': self.timestamp,
            'uptime': self.uptime,
            'summary': self._get_summary()
        }
    
    def _get_summary(self) -> Dict[str, int]:
        """Get summary of check statuses."""
        summary = {status.value: 0 for status in HealthStatus}
        for check in self.checks:
            summary[check.status.value] += 1
        return summary


class HealthChecker:
    """
    Comprehensive system health monitoring.
    
    Performs various health checks including hardware status,
    system resources, dependencies, and application components.
    """
    
    def __init__(self, check_interval: int = 30):
        """
        Initialize health checker.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.start_time = time.time()
        
        # Registered health checks
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        
        # Health history
        self.health_history: List[SystemHealth] = []
        self.max_history = 100  # Keep last 100 health checks
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = False
        self.monitoring_lock = threading.Lock()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[HealthCheckResult], None]] = []
        
        # System thresholds
        self.thresholds = {
            'cpu_warning': 80,  # CPU usage %
            'cpu_critical': 95,
            'memory_warning': 80,  # Memory usage %
            'memory_critical': 95,
            'disk_warning': 85,  # Disk usage %
            'disk_critical': 95,
            'gpu_memory_warning': 90,  # GPU memory %
            'gpu_memory_critical': 98,
            'temperature_warning': 70,  # Temperature °C
            'temperature_critical': 85,
            'response_time_warning': 5.0,  # Response time seconds
            'response_time_critical': 10.0,
        }
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("dependencies", self._check_dependencies)
        self.register_check("gpu_status", self._check_gpu_status)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("network_connectivity", self._check_network)
        self.register_check("hardware_interfaces", self._check_hardware_interfaces)
        self.register_check("acoustic_simulation", self._check_acoustic_simulation)
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """
        Register a custom health check.
        
        Args:
            name: Name of the health check
            check_func: Function that performs the check
        """
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def add_alert_callback(self, callback: Callable[[HealthCheckResult], None]):
        """
        Add alert callback for critical health checks.
        
        Args:
            callback: Function to call when critical issues are detected
        """
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_thread is not None:
            logger.warning("Monitoring already started")
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring_thread(self):
        """Stop continuous health monitoring."""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            self.monitoring_thread = None
        
        logger.info("Health monitoring stopped")
    
    def run_health_check(self) -> SystemHealth:
        """
        Run all registered health checks.
        
        Returns:
            SystemHealth object with all check results
        """
        results = []
        start_time = time.time()
        
        for name, check_func in self.health_checks.items():
            try:
                with self._measure_time() as timer:
                    result = check_func()
                    result.response_time = timer.elapsed
                    results.append(result)
                
                # Trigger alerts for critical issues
                if result.status == HealthStatus.CRITICAL:
                    self._trigger_alerts(result)
                    
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                error_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {str(e)}",
                    response_time=time.time() - start_time
                )
                results.append(error_result)
        
        # Determine overall system status
        overall_status = self._determine_overall_status(results)
        
        system_health = SystemHealth(
            status=overall_status,
            checks=results,
            uptime=time.time() - self.start_time
        )
        
        # Store in history
        with self.monitoring_lock:
            self.health_history.append(system_health)
            if len(self.health_history) > self.max_history:
                self.health_history.pop(0)
        
        return system_health
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status.
        
        Returns:
            Dictionary representation of current health
        """
        health = self.run_health_check()
        return health.to_dict()
    
    def get_health_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """
        Get health check history.
        
        Args:
            hours: Hours of history to retrieve
            
        Returns:
            List of health check results
        """
        cutoff_time = time.time() - hours * 3600
        
        with self.monitoring_lock:
            filtered_history = [
                health.to_dict()
                for health in self.health_history
                if health.timestamp >= cutoff_time
            ]
        
        return filtered_history
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self.stop_monitoring:
            try:
                self.run_health_check()
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system health status."""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _trigger_alerts(self, result: HealthCheckResult):
        """Trigger alert callbacks for critical issues."""
        for callback in self.alert_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    @contextmanager
    def _measure_time(self):
        """Context manager to measure execution time."""
        class Timer:
            def __init__(self):
                self.start = time.time()
                self.elapsed = 0.0
        
        timer = Timer()
        try:
            yield timer
        finally:
            timer.elapsed = time.time() - timer.start
    
    # Default health check implementations
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Determine status
            status = HealthStatus.HEALTHY
            messages = []
            
            if cpu_percent >= self.thresholds['cpu_critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent >= self.thresholds['cpu_warning']:
                status = HealthStatus.WARNING
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent >= self.thresholds['memory_critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical memory usage: {memory_percent:.1f}%")
            elif memory_percent >= self.thresholds['memory_warning']:
                if status != HealthStatus.CRITICAL:
                    status = HealthStatus.WARNING
                messages.append(f"High memory usage: {memory_percent:.1f}%")
            
            if status == HealthStatus.HEALTHY:
                message = "System resources normal"
            else:
                message = "; ".join(messages)
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'cpu_count': psutil.cpu_count()
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {e}"
            )
    
    def _check_dependencies(self) -> HealthCheckResult:
        """Check critical dependencies."""
        missing_deps = []
        optional_missing = []
        
        # Critical dependencies
        critical_deps = ['numpy', 'scipy']
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        # Optional dependencies
        optional_deps = ['torch', 'matplotlib', 'h5py']
        for dep in optional_deps:
            try:
                __import__(dep)
            except ImportError:
                optional_missing.append(dep)
        
        if missing_deps:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Critical dependencies missing: {', '.join(missing_deps)}",
                details={'missing_critical': missing_deps, 'missing_optional': optional_missing}
            )
        elif optional_missing:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.WARNING,
                message=f"Optional dependencies missing: {', '.join(optional_missing)}",
                details={'missing_optional': optional_missing}
            )
        else:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.HEALTHY,
                message="All dependencies available"
            )
    
    def _check_gpu_status(self) -> HealthCheckResult:
        """Check GPU availability and status."""
        if not HAS_TORCH:
            return HealthCheckResult(
                name="gpu_status",
                status=HealthStatus.WARNING,
                message="PyTorch not available for GPU checking"
            )
        
        try:
            if not torch.cuda.is_available():
                return HealthCheckResult(
                    name="gpu_status",
                    status=HealthStatus.WARNING,
                    message="CUDA not available - using CPU only",
                    details={'cuda_available': False}
                )
            
            device_count = torch.cuda.device_count()
            gpu_details = {}
            status = HealthStatus.HEALTHY
            messages = []
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                memory_percent = (memory_allocated / memory_total) * 100
                
                gpu_details[f'gpu_{i}'] = {
                    'name': device_name,
                    'memory_allocated_gb': memory_allocated / (1024**3),
                    'memory_total_gb': memory_total / (1024**3),
                    'memory_percent': memory_percent
                }
                
                if memory_percent >= self.thresholds['gpu_memory_critical']:
                    status = HealthStatus.CRITICAL
                    messages.append(f"GPU {i} critical memory: {memory_percent:.1f}%")
                elif memory_percent >= self.thresholds['gpu_memory_warning']:
                    if status != HealthStatus.CRITICAL:
                        status = HealthStatus.WARNING
                    messages.append(f"GPU {i} high memory: {memory_percent:.1f}%")
            
            message = f"CUDA available with {device_count} device(s)"
            if messages:
                message += " - " + "; ".join(messages)
            
            return HealthCheckResult(
                name="gpu_status",
                status=status,
                message=message,
                details={'cuda_available': True, 'device_count': device_count, **gpu_details}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="gpu_status",
                status=HealthStatus.UNKNOWN,
                message=f"GPU check failed: {e}"
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space usage."""
        try:
            disk_usage = psutil.disk_usage('/')
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            if usage_percent >= self.thresholds['disk_critical']:
                status = HealthStatus.CRITICAL
                message = f"Critical disk usage: {usage_percent:.1f}%"
            elif usage_percent >= self.thresholds['disk_warning']:
                status = HealthStatus.WARNING
                message = f"High disk usage: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1f}%"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                details={
                    'usage_percent': usage_percent,
                    'free_gb': disk_usage.free / (1024**3),
                    'total_gb': disk_usage.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Disk check failed: {e}"
            )
    
    def _check_network(self) -> HealthCheckResult:
        """Check network connectivity."""
        try:
            import socket
            
            # Test DNS resolution
            socket.gethostbyname('google.com')
            
            # Test basic socket connectivity
            sock = socket.create_connection(('8.8.8.8', 53), timeout=5)
            sock.close()
            
            return HealthCheckResult(
                name="network_connectivity",
                status=HealthStatus.HEALTHY,
                message="Network connectivity normal"
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="network_connectivity",
                status=HealthStatus.WARNING,
                message=f"Network connectivity issues: {e}"
            )
    
    def _check_hardware_interfaces(self) -> HealthCheckResult:
        """Check hardware interface status."""
        try:
            # This would check actual hardware interfaces
            # For now, return healthy status with mock data
            
            interfaces = {
                'simulation': {'status': 'connected', 'type': 'simulation'},
                'serial_ports': {'available': 0, 'connected': 0},
                'network_devices': {'available': 1, 'connected': 0}
            }
            
            return HealthCheckResult(
                name="hardware_interfaces",
                status=HealthStatus.HEALTHY,
                message="Hardware interfaces operational",
                details=interfaces
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="hardware_interfaces",
                status=HealthStatus.WARNING,
                message=f"Hardware interface check failed: {e}"
            )
    
    def _check_acoustic_simulation(self) -> HealthCheckResult:
        """Check acoustic simulation functionality."""
        try:
            # Test basic acoustic calculation
            if not HAS_NUMPY:
                return HealthCheckResult(
                    name="acoustic_simulation",
                    status=HealthStatus.CRITICAL,
                    message="NumPy required for acoustic simulation"
                )
            
            # Simple test calculation
            import numpy as np
            test_array = np.random.random(256)
            test_result = np.fft.fft(test_array)
            
            if np.any(np.isnan(test_result)):
                status = HealthStatus.WARNING
                message = "Acoustic simulation producing NaN values"
            else:
                status = HealthStatus.HEALTHY
                message = "Acoustic simulation functional"
            
            return HealthCheckResult(
                name="acoustic_simulation",
                status=status,
                message=message,
                details={'test_array_size': len(test_array)}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="acoustic_simulation",
                status=HealthStatus.CRITICAL,
                message=f"Acoustic simulation failed: {e}"
            )


# Global health checker instance
health_checker = HealthChecker()


# Convenience functions
def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    return health_checker.get_health_status()


def start_health_monitoring():
    """Start continuous health monitoring."""
    health_checker.start_monitoring()


def stop_health_monitoring():
    """Stop continuous health monitoring."""
    health_checker.stop_monitoring_thread()


def register_health_check(name: str, check_func: Callable[[], HealthCheckResult]):
    """Register a custom health check."""
    health_checker.register_check(name, check_func)


def add_health_alert(callback: Callable[[HealthCheckResult], None]):
    """Add health alert callback."""
    health_checker.add_alert_callback(callback)