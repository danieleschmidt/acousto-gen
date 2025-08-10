"""
Monitoring and metrics collection for Acousto-Gen.
Provides Prometheus metrics and OpenTelemetry tracing.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager

# Handle optional dependencies gracefully
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
    HAS_PROMETHEUS = True
except ImportError:
    print("⚠️ Prometheus client not available - metrics will be mocked")
    HAS_PROMETHEUS = False
    
    # Mock Prometheus metrics
    class MockMetric:
        def __init__(self, name, doc, labels=None):
            self.name = name
            self.doc = doc
            self.labels_list = labels or []
            self.value = 0
        
        def inc(self, amount=1):
            self.value += amount
            return self
        
        def set(self, value):
            self.value = value
            return self
        
        def observe(self, value):
            self.value = value
            return self
        
        def labels(self, **kwargs):
            return self  # Return self for chaining
        
        def time(self):
            return self
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, *args):
            self.value = time.time() - self.start_time
    
    def Counter(name, doc, labelnames=None, **kwargs):
        return MockMetric(name, doc, labelnames)
    
    def Histogram(name, doc, labelnames=None, buckets=None, **kwargs):
        return MockMetric(name, doc, labelnames)
    
    def Gauge(name, doc, labelnames=None, **kwargs):
        return MockMetric(name, doc, labelnames)
    
    def Summary(name, doc, labelnames=None, **kwargs):
        return MockMetric(name, doc, labelnames)
    
    def generate_latest():
        return "# Mock metrics\ntest_metric 1.0\n"

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    HAS_OPENTELEMETRY = True
except ImportError:
    print("⚠️ OpenTelemetry not available - tracing will be mocked")
    HAS_OPENTELEMETRY = False
    
    # Mock OpenTelemetry
    class MockTrace:
        @staticmethod
        def get_tracer(name):
            class MockTracer:
                @contextmanager
                def start_as_current_span(self, name):
                    yield None
            return MockTracer()
        
        @staticmethod
        def set_tracer_provider(provider):
            pass
    
    class MockResource:
        @staticmethod
        def create(attrs):
            return None
    
    class MockTracerProvider:
        def __init__(self, resource=None):
            pass
        
        def get_tracer(self, name):
            return MockTrace.get_tracer(name)
    
    class MockMeterProvider:
        def __init__(self, metric_readers=None, resource=None):
            pass
        
        def get_meter(self, name):
            return None
    
    class MockMetrics:
        @staticmethod
        def set_meter_provider(provider):
            pass
        
        @staticmethod
        def get_meter(name):
            class MockMeter:
                def create_counter(self, name, description=None, unit=None):
                    return MockMetric(name, description)
                
                def create_histogram(self, name, description=None, unit=None):
                    return MockMetric(name, description)
                
                def create_gauge(self, name, description=None, unit=None):
                    return MockMetric(name, description)
                
                def create_up_down_counter(self, name, description=None, unit=None):
                    return MockMetric(name, description)
            
            return MockMeter()
    
    class MockPrometheusMetricReader:
        pass
    
    trace = MockTrace()
    metrics = MockMetrics()
    Resource = MockResource()
    TracerProvider = MockTracerProvider
    MeterProvider = MockMeterProvider  
    PrometheusMetricReader = MockPrometheusMetricReader


logger = logging.getLogger(__name__)


# Prometheus metrics
optimization_counter = Counter(
    'acousto_gen_optimizations_total',
    'Total number of hologram optimizations performed',
    ['method', 'status']
)

optimization_duration = Histogram(
    'acousto_gen_optimization_duration_seconds',
    'Duration of hologram optimization in seconds',
    ['method'],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60)
)

particles_gauge = Gauge(
    'acousto_gen_particles_active',
    'Number of actively levitated particles'
)

field_computation_time = Summary(
    'acousto_gen_field_computation_seconds',
    'Time spent computing acoustic fields'
)

hardware_status = Gauge(
    'acousto_gen_hardware_connected',
    'Hardware connection status (1=connected, 0=disconnected)',
    ['interface_type']
)

api_requests = Counter(
    'acousto_gen_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

websocket_connections = Gauge(
    'acousto_gen_websocket_connections',
    'Active WebSocket connections'
)

# Performance metrics
gpu_memory_usage = Gauge(
    'acousto_gen_gpu_memory_bytes',
    'GPU memory usage in bytes',
    ['device']
)

cpu_usage_percent = Gauge(
    'acousto_gen_cpu_usage_percent',
    'CPU usage percentage'
)


class MetricsCollector:
    """Central metrics collection and management."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = time.time()
        self.optimization_count = 0
        self.total_computation_time = 0
        self.error_count = 0
        
        # Setup OpenTelemetry
        self._setup_telemetry()
        
        # Custom metrics storage
        self.custom_metrics: Dict[str, Any] = {}
    
    def _setup_telemetry(self):
        """Setup OpenTelemetry providers."""
        # Create resource
        resource = Resource.create({
            "service.name": "acousto-gen",
            "service.version": "1.0.0"
        })
        
        # Setup tracing
        trace.set_tracer_provider(TracerProvider(resource=resource))
        self.tracer = trace.get_tracer(__name__)
        
        # Setup metrics
        reader = PrometheusMetricReader()
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        self.meter = metrics.get_meter(__name__)
        
        # Create meters
        self._create_meters()
    
    def _create_meters(self):
        """Create OpenTelemetry meters."""
        self.optimization_meter = self.meter.create_counter(
            "optimization.count",
            description="Number of optimizations"
        )
        
        self.particle_meter = self.meter.create_up_down_counter(
            "particle.count",
            description="Number of particles"
        )
        
        self.latency_meter = self.meter.create_histogram(
            "request.latency",
            description="Request latency",
            unit="ms"
        )
    
    def record_optimization(
        self,
        method: str,
        duration: float,
        success: bool,
        final_loss: Optional[float] = None
    ):
        """
        Record optimization metrics.
        
        Args:
            method: Optimization method used
            duration: Optimization duration in seconds
            success: Whether optimization succeeded
            final_loss: Final loss value
        """
        status = "success" if success else "failure"
        
        # Prometheus metrics
        optimization_counter.labels(method=method, status=status).inc()
        optimization_duration.labels(method=method).observe(duration)
        
        # OpenTelemetry metrics
        self.optimization_meter.add(1, {"method": method, "status": status})
        
        # Internal tracking
        self.optimization_count += 1
        self.total_computation_time += duration
        
        if not success:
            self.error_count += 1
        
        logger.info(
            f"Optimization recorded: method={method}, duration={duration:.2f}s, "
            f"success={success}, final_loss={final_loss}"
        )
    
    def record_particle_change(self, delta: int):
        """
        Record particle count change.
        
        Args:
            delta: Change in particle count (+1 for add, -1 for remove)
        """
        current_count = particles_gauge._value.get()
        particles_gauge.set(current_count + delta)
        self.particle_meter.add(delta)
    
    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float
    ):
        """
        Record API request metrics.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            status_code: Response status code
            duration: Request duration in seconds
        """
        api_requests.labels(
            endpoint=endpoint,
            method=method,
            status=str(status_code)
        ).inc()
        
        self.latency_meter.record(
            duration * 1000,  # Convert to milliseconds
            {"endpoint": endpoint, "method": method}
        )
    
    def record_hardware_status(self, interface_type: str, connected: bool):
        """
        Record hardware connection status.
        
        Args:
            interface_type: Type of hardware interface
            connected: Connection status
        """
        hardware_status.labels(interface_type=interface_type).set(
            1 if connected else 0
        )
    
    def record_websocket_connection(self, delta: int):
        """
        Record WebSocket connection change.
        
        Args:
            delta: Change in connection count
        """
        websocket_connections.inc(delta)
    
    @contextmanager
    def measure_time(self, metric_name: str):
        """
        Context manager to measure execution time.
        
        Args:
            metric_name: Name of the metric to record
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            if metric_name == "field_computation":
                field_computation_time.observe(duration)
            else:
                # Store in custom metrics
                if metric_name not in self.custom_metrics:
                    self.custom_metrics[metric_name] = []
                self.custom_metrics[metric_name].append(duration)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            Dictionary of system metrics
        """
        uptime = time.time() - self.start_time
        
        # Try to get GPU metrics
        gpu_metrics = self._get_gpu_metrics()
        
        # Try to get CPU metrics
        cpu_metrics = self._get_cpu_metrics()
        
        return {
            "uptime_seconds": uptime,
            "optimizations_total": self.optimization_count,
            "total_computation_time": self.total_computation_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.optimization_count, 1),
            "active_particles": particles_gauge._value.get(),
            "websocket_connections": websocket_connections._value.get(),
            **gpu_metrics,
            **cpu_metrics,
            "custom_metrics": self.custom_metrics
        }
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if available."""
        metrics = {}
        
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    
                    gpu_memory_usage.labels(device=f"cuda:{i}").set(memory_allocated)
                    
                    metrics[f"gpu_{i}_memory_allocated"] = memory_allocated
                    metrics[f"gpu_{i}_memory_reserved"] = memory_reserved
                    metrics[f"gpu_{i}_utilization"] = torch.cuda.utilization(i)
        except Exception as e:
            logger.debug(f"Could not get GPU metrics: {e}")
        
        return metrics
    
    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU metrics."""
        metrics = {}
        
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_usage_percent.set(cpu_percent)
            
            metrics["cpu_usage_percent"] = cpu_percent
            metrics["cpu_count"] = psutil.cpu_count()
            metrics["memory_usage_percent"] = psutil.virtual_memory().percent
            metrics["memory_available_gb"] = psutil.virtual_memory().available / (1024**3)
            
        except ImportError:
            logger.debug("psutil not available for CPU metrics")
        except Exception as e:
            logger.debug(f"Could not get CPU metrics: {e}")
        
        return metrics
    
    def export_prometheus_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus formatted metrics
        """
        return generate_latest()


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Decorators for automatic metric collection
def track_optimization(method: str):
    """
    Decorator to track optimization metrics.
    
    Args:
        method: Optimization method name
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            final_loss = None
            
            try:
                result = func(*args, **kwargs)
                success = True
                
                # Try to extract final loss from result
                if hasattr(result, 'final_loss'):
                    final_loss = result.final_loss
                elif isinstance(result, dict) and 'final_loss' in result:
                    final_loss = result['final_loss']
                
                return result
                
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                raise
                
            finally:
                duration = time.time() - start_time
                metrics_collector.record_optimization(
                    method=method,
                    duration=duration,
                    success=success,
                    final_loss=final_loss
                )
        
        return wrapper
    return decorator


def track_api_request(func: Callable) -> Callable:
    """Decorator to track API request metrics."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request') or (args[0] if args else None)
        endpoint = request.url.path if request else "unknown"
        method = request.method if request else "unknown"
        
        start_time = time.time()
        status_code = 500  # Default to error
        
        try:
            response = await func(*args, **kwargs)
            status_code = getattr(response, 'status_code', 200)
            return response
            
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
            
        finally:
            duration = time.time() - start_time
            metrics_collector.record_api_request(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                duration=duration
            )
    
    return wrapper


class SpanTracer:
    """OpenTelemetry span tracing helper."""
    
    @staticmethod
    @contextmanager
    def trace(name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Create a trace span.
        
        Args:
            name: Span name
            attributes: Optional span attributes
        """
        tracer = trace.get_tracer(__name__)
        
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise