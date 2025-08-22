"""
Adaptive Performance Optimization System for Acousto-Gen Generation 3.
Implements intelligent caching, GPU acceleration, and dynamic resource management.
"""

import time
import threading
import psutil
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import numpy as np
import json
import hashlib
from pathlib import Path
import pickle
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ComputeDevice(Enum):
    """Available computation devices."""
    CPU = "cpu"
    CUDA = "cuda"
    OPENCL = "opencl"
    AUTO = "auto"


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: float
    gpu_usage: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    computation_time: Optional[float] = None
    throughput: Optional[float] = None
    cache_hit_rate: Optional[float] = None


@dataclass
class ComputationProfile:
    """Profile for different computation patterns."""
    operation_type: str
    input_size: int
    preferred_device: ComputeDevice
    memory_requirement: float
    typical_duration: float
    optimization_hints: Dict[str, Any] = field(default_factory=dict)


class IntelligentCache:
    """Intelligent caching system with LRU eviction and compression."""
    
    def __init__(self, max_size_mb: int = 1000, compression_threshold: int = 10):
        """Initialize cache with size limit and compression."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.compression_threshold = compression_threshold * 1024 * 1024
        self.cache = OrderedDict()
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "current_size": 0
        }
        self._lock = threading.RLock()
    
    def _generate_key(self, operation: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key from operation and parameters."""
        # Create deterministic hash of parameters
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        return f"{operation}:{param_hash}"
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_size(item) for item in data)
        elif isinstance(data, dict):
            return sum(self._estimate_size(v) for v in data.values())
        else:
            return len(str(data).encode('utf-8'))
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress large data objects."""
        import gzip
        serialized = pickle.dumps(data)
        if len(serialized) > self.compression_threshold:
            return gzip.compress(serialized)
        return serialized
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress cached data."""
        import gzip
        try:
            # Try decompression first
            decompressed = gzip.decompress(compressed_data)
            return pickle.loads(decompressed)
        except:
            # If decompression fails, assume it's uncompressed
            return pickle.loads(compressed_data)
    
    def get(self, operation: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """Get cached result."""
        key = self._generate_key(operation, parameters)
        
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.cache_stats["hits"] += 1
                
                # Decompress if needed
                return self._decompress_data(value["data"])
            else:
                self.cache_stats["misses"] += 1
                return None
    
    def put(self, operation: str, parameters: Dict[str, Any], result: Any):
        """Cache result with automatic eviction."""
        key = self._generate_key(operation, parameters)
        
        with self._lock:
            # Compress and store
            compressed_data = self._compress_data(result)
            data_size = len(compressed_data)
            
            cache_entry = {
                "data": compressed_data,
                "size": data_size,
                "timestamp": time.time(),
                "access_count": 1
            }
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache.pop(key)
                self.cache_stats["current_size"] -= old_entry["size"]
            
            # Evict entries if necessary
            while (self.cache_stats["current_size"] + data_size > self.max_size_bytes and 
                   len(self.cache) > 0):
                oldest_key = next(iter(self.cache))
                old_entry = self.cache.pop(oldest_key)
                self.cache_stats["current_size"] -= old_entry["size"]
                self.cache_stats["evictions"] += 1
            
            # Add new entry
            self.cache[key] = cache_entry
            self.cache_stats["current_size"] += data_size
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.cache_stats = {
                "hits": 0,
                "misses": 0, 
                "evictions": 0,
                "current_size": 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = self.cache_stats["hits"] / max(1, total_requests)
            
            return {
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "cache_size_mb": self.cache_stats["current_size"] / (1024 * 1024),
                "entries": len(self.cache),
                "evictions": self.cache_stats["evictions"]
            }


class ResourceMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize resource monitor."""
        self.update_interval = update_interval
        self.metrics_history = []
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start resource monitoring thread."""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    # Keep only last 1000 entries
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            memory_available=memory.available / (1024 * 1024)  # MB
        )
        
        # GPU metrics if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization()
                gpu_memory_used = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
                
                metrics.gpu_usage = gpu_usage
                metrics.gpu_memory_used = gpu_memory_used
                metrics.gpu_memory_total = gpu_memory_total
            except:
                pass
        
        return metrics
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics."""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            return None
    
    def get_average_metrics(self, time_window: float = 60.0) -> Optional[PerformanceMetrics]:
        """Get average metrics over time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self._lock:
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return None
            
            # Calculate averages
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory_avail = sum(m.memory_available for m in recent_metrics) / len(recent_metrics)
            
            avg_gpu = None
            avg_gpu_mem = None
            avg_gpu_total = None
            
            gpu_metrics = [m for m in recent_metrics if m.gpu_usage is not None]
            if gpu_metrics:
                avg_gpu = sum(m.gpu_usage for m in gpu_metrics) / len(gpu_metrics)
                avg_gpu_mem = sum(m.gpu_memory_used for m in gpu_metrics) / len(gpu_metrics)
                avg_gpu_total = sum(m.gpu_memory_total for m in gpu_metrics) / len(gpu_metrics)
            
            return PerformanceMetrics(
                timestamp=current_time,
                cpu_usage=avg_cpu,
                memory_usage=avg_memory,
                memory_available=avg_memory_avail,
                gpu_usage=avg_gpu,
                gpu_memory_used=avg_gpu_mem,
                gpu_memory_total=avg_gpu_total
            )


class AdaptiveDeviceSelector:
    """Intelligently selects optimal computation device."""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        """Initialize device selector."""
        self.resource_monitor = resource_monitor
        self.device_profiles = {}
        self.benchmark_cache = {}
    
    def benchmark_devices(self, operation_types: List[str]) -> Dict[str, Dict[ComputeDevice, float]]:
        """Benchmark different devices for various operations."""
        results = {}
        
        for operation in operation_types:
            results[operation] = {}
            
            # CPU benchmark
            cpu_time = self._benchmark_cpu_operation(operation)
            results[operation][ComputeDevice.CPU] = cpu_time
            
            # GPU benchmark if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_time = self._benchmark_gpu_operation(operation)
                results[operation][ComputeDevice.CUDA] = gpu_time
        
        self.benchmark_cache = results
        logger.info(f"Benchmarked {len(operation_types)} operations across available devices")
        return results
    
    def _benchmark_cpu_operation(self, operation: str) -> float:
        """Benchmark CPU performance for operation."""
        if operation == "matrix_multiply":
            # Simple matrix multiplication benchmark
            size = 512
            a = np.random.random((size, size)).astype(np.float32)
            b = np.random.random((size, size)).astype(np.float32)
            
            start_time = time.time()
            np.dot(a, b)
            return time.time() - start_time
        
        elif operation == "fft":
            # FFT benchmark
            size = 2**16
            data = np.random.random(size).astype(np.complex64)
            
            start_time = time.time()
            np.fft.fft(data)
            return time.time() - start_time
        
        elif operation == "field_propagation":
            # Simulated field propagation
            size = (64, 64, 64)
            field = np.random.random(size).astype(np.complex64)
            
            start_time = time.time()
            # Simulate wave propagation computation
            result = np.fft.fftn(field) * np.exp(1j * np.random.random(size))
            result = np.fft.ifftn(result)
            return time.time() - start_time
        
        return 1.0  # Default fallback
    
    def _benchmark_gpu_operation(self, operation: str) -> float:
        """Benchmark GPU performance for operation."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return float('inf')
        
        device = torch.device('cuda')
        
        if operation == "matrix_multiply":
            size = 512
            a = torch.random((size, size), device=device, dtype=torch.float32)
            b = torch.random((size, size), device=device, dtype=torch.float32)
            
            # Warm up
            torch.mm(a, b)
            torch.cuda.synchronize()
            
            start_time = time.time()
            torch.mm(a, b)
            torch.cuda.synchronize()
            return time.time() - start_time
        
        elif operation == "fft":
            size = 2**16
            data = torch.random(size, device=device, dtype=torch.complex64)
            
            # Warm up
            torch.fft.fft(data)
            torch.cuda.synchronize()
            
            start_time = time.time()
            torch.fft.fft(data)
            torch.cuda.synchronize()
            return time.time() - start_time
        
        elif operation == "field_propagation":
            size = (64, 64, 64)
            field = torch.random(size, device=device, dtype=torch.complex64)
            
            # Warm up
            result = torch.fft.fftn(field) * torch.exp(1j * torch.random(size, device=device))
            result = torch.fft.ifftn(result)
            torch.cuda.synchronize()
            
            start_time = time.time()
            result = torch.fft.fftn(field) * torch.exp(1j * torch.random(size, device=device))
            result = torch.fft.ifftn(result)
            torch.cuda.synchronize()
            return time.time() - start_time
        
        return 1.0
    
    def select_optimal_device(self, operation: str, data_size: int, 
                            current_load: Optional[PerformanceMetrics] = None) -> ComputeDevice:
        """Select optimal device for operation."""
        # Get current system metrics
        if current_load is None:
            current_load = self.resource_monitor.get_current_metrics()
        
        # Check benchmark results
        if operation in self.benchmark_cache:
            benchmarks = self.benchmark_cache[operation]
            
            # Adjust for current system load
            cpu_penalty = 1.0
            gpu_penalty = 1.0
            
            if current_load:
                # Penalize heavily loaded devices
                if current_load.cpu_usage > 80:
                    cpu_penalty = 2.0
                elif current_load.cpu_usage > 50:
                    cpu_penalty = 1.5
                
                if (current_load.gpu_usage and current_load.gpu_usage > 80):
                    gpu_penalty = 2.0
                elif (current_load.gpu_usage and current_load.gpu_usage > 50):
                    gpu_penalty = 1.5
                
                # Consider memory constraints
                if (current_load.gpu_memory_used and current_load.gpu_memory_total and
                    current_load.gpu_memory_used / current_load.gpu_memory_total > 0.8):
                    gpu_penalty = 3.0
            
            # Calculate adjusted times
            adjusted_times = {}
            for device, base_time in benchmarks.items():
                if device == ComputeDevice.CPU:
                    adjusted_times[device] = base_time * cpu_penalty
                elif device == ComputeDevice.CUDA:
                    adjusted_times[device] = base_time * gpu_penalty
            
            # Select device with minimum adjusted time
            optimal_device = min(adjusted_times.keys(), key=lambda d: adjusted_times[d])
            return optimal_device
        
        # Fallback logic based on operation type and data size
        if data_size > 1000000:  # Large data - prefer GPU if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                if not current_load or not current_load.gpu_usage or current_load.gpu_usage < 70:
                    return ComputeDevice.CUDA
        
        return ComputeDevice.CPU


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        """Initialize performance optimizer."""
        self.optimization_level = optimization_level
        self.cache = IntelligentCache(max_size_mb=self._get_cache_size())
        self.resource_monitor = ResourceMonitor()
        self.device_selector = AdaptiveDeviceSelector(self.resource_monitor)
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        # Benchmark common operations
        self.device_selector.benchmark_devices([
            "matrix_multiply", "fft", "field_propagation"
        ])
        
        logger.info(f"Performance optimizer initialized with {optimization_level.value} level")
    
    def _get_cache_size(self) -> int:
        """Get cache size based on optimization level."""
        sizes = {
            OptimizationLevel.CONSERVATIVE: 500,   # 500 MB
            OptimizationLevel.BALANCED: 1000,     # 1 GB
            OptimizationLevel.AGGRESSIVE: 2000,   # 2 GB
            OptimizationLevel.EXTREME: 4000       # 4 GB
        }
        return sizes[self.optimization_level]
    
    def optimize_computation(self, operation: str, compute_func: Callable, 
                           parameters: Dict[str, Any], 
                           force_recompute: bool = False) -> Any:
        """Optimize computation with caching and device selection."""
        # Check cache first
        if not force_recompute:
            cached_result = self.cache.get(operation, parameters)
            if cached_result is not None:
                logger.debug(f"Cache hit for {operation}")
                return cached_result
        
        # Estimate data size for device selection
        data_size = self._estimate_data_size(parameters)
        
        # Select optimal device
        optimal_device = self.device_selector.select_optimal_device(operation, data_size)
        
        # Update parameters with optimal device
        optimized_params = parameters.copy()
        optimized_params['device'] = optimal_device.value
        
        # Execute computation
        start_time = time.time()
        try:
            result = compute_func(**optimized_params)
            computation_time = time.time() - start_time
            
            # Cache result
            self.cache.put(operation, parameters, result)
            
            # Log performance
            logger.debug(f"Computed {operation} in {computation_time:.3f}s on {optimal_device.value}")
            
            return result
            
        except Exception as e:
            # Fallback to CPU if GPU computation fails
            if optimal_device != ComputeDevice.CPU:
                logger.warning(f"GPU computation failed, falling back to CPU: {e}")
                optimized_params['device'] = ComputeDevice.CPU.value
                result = compute_func(**optimized_params)
                computation_time = time.time() - start_time
                self.cache.put(operation, parameters, result)
                return result
            else:
                raise
    
    def _estimate_data_size(self, parameters: Dict[str, Any]) -> int:
        """Estimate total data size from parameters."""
        total_size = 0
        
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                total_size += value.nbytes
            elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                total_size += value.numel() * value.element_size()
            elif isinstance(value, (list, tuple)):
                total_size += len(value) * 8  # Rough estimate
            elif key in ['resolution', 'shape', 'size']:
                if isinstance(value, (int, float)):
                    total_size += int(value ** 3) * 8  # Assume 3D field
                elif isinstance(value, (list, tuple)):
                    total_size += np.prod(value) * 8
        
        return total_size
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_metrics = self.resource_monitor.get_current_metrics()
        cache_stats = self.cache.get_stats()
        
        summary = {
            "optimization_level": self.optimization_level.value,
            "cache_stats": cache_stats,
            "system_metrics": {
                "cpu_usage": current_metrics.cpu_usage if current_metrics else None,
                "memory_usage": current_metrics.memory_usage if current_metrics else None,
                "gpu_usage": current_metrics.gpu_usage if current_metrics else None,
            },
            "available_devices": self._get_available_devices(),
            "benchmark_results": self.device_selector.benchmark_cache
        }
        
        return summary
    
    def _get_available_devices(self) -> List[str]:
        """Get list of available computation devices."""
        devices = [ComputeDevice.CPU.value]
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            devices.append(ComputeDevice.CUDA.value)
        
        if CUPY_AVAILABLE:
            devices.append("cupy")
        
        return devices
    
    def cleanup(self):
        """Cleanup resources."""
        self.resource_monitor.stop_monitoring()
        self.cache.clear()
        logger.info("Performance optimizer cleaned up")


# Global performance optimizer instance
global_optimizer = PerformanceOptimizer()


def optimized_computation(operation: str, **kwargs):
    """Decorator for optimized computation with automatic caching and device selection."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **func_kwargs):
            # Extract parameters for caching
            cache_params = {
                "args": str(args)[:200],  # Truncate for cache key
                "kwargs": {k: v for k, v in func_kwargs.items() if isinstance(v, (int, float, str, bool))}
            }
            
            return global_optimizer.optimize_computation(
                operation=operation,
                compute_func=lambda **params: func(*args, **func_kwargs),
                parameters=cache_params,
                **kwargs
            )
        
        return wrapper
    return decorator