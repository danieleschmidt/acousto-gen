"""
Advanced caching system for acoustic holography computations.
Provides intelligent caching with cache invalidation, memory management, and performance optimization.
"""

import hashlib
import pickle
import time
import threading
from typing import Any, Dict, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import weakref
from pathlib import Path
import json
import logging

# Handle optional dependencies gracefully
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    from diskcache import Cache as DiskCache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False


logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for different storage strategies."""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def should_evict(self, max_idle_time: float) -> bool:
        """Check if entry should be evicted due to inactivity."""
        return time.time() > (self.last_accessed + max_idle_time)
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """Thread-safe LRU cache with size limits and expiration."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[float] = 3600
    ):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._current_memory_usage = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = False
        self._start_cleanup_thread()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access
            entry.update_access()
            
            # Move to end of access order (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            self.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Ensure space
            self._ensure_space(size_bytes)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            # Add to cache
            self._cache[key] = entry
            self._access_order.append(key)
            self._current_memory_usage += size_bytes
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key not in self._cache:
                return False
            
            self._remove_entry(key)
            return True
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_memory_usage = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_bytes': self._current_memory_usage,
                'memory_usage_mb': self._current_memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024)
            }
    
    def _remove_entry(self, key: str):
        """Remove entry and update bookkeeping."""
        if key in self._cache:
            entry = self._cache[key]
            del self._cache[key]
            self._current_memory_usage -= entry.size_bytes
            
            if key in self._access_order:
                self._access_order.remove(key)
    
    def _ensure_space(self, needed_bytes: int):
        """Ensure enough space by evicting LRU entries."""
        # Check size limit
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order[0]
            self._remove_entry(oldest_key)
            self.evictions += 1
        
        # Check memory limit
        while (self._current_memory_usage + needed_bytes > self.max_memory_bytes 
               and self._access_order):
            oldest_key = self._access_order[0]
            self._remove_entry(oldest_key)
            self.evictions += 1
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if HAS_NUMPY and isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                # Rough estimation using pickle
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback estimation
            return 1024  # 1KB default
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while not self._stop_cleanup:
                try:
                    self._cleanup_expired()
                    time.sleep(60)  # Cleanup every minute
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    time.sleep(60)
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def shutdown(self):
        """Shutdown cache and cleanup thread."""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1.0)


class AcousticComputationCache:
    """
    Specialized cache for acoustic holography computations.
    
    Provides intelligent caching of field calculations, optimization results,
    and other expensive computations with automatic invalidation.
    """
    
    def __init__(
        self,
        cache_level: CacheLevel = CacheLevel.HYBRID,
        memory_cache_size: int = 500,
        memory_cache_mb: int = 200,
        disk_cache_dir: Optional[str] = None,
        redis_url: Optional[str] = None
    ):
        """
        Initialize acoustic computation cache.
        
        Args:
            cache_level: Caching strategy to use
            memory_cache_size: Maximum entries in memory cache
            memory_cache_mb: Maximum memory usage in MB
            disk_cache_dir: Directory for disk cache
            redis_url: Redis connection URL
        """
        self.cache_level = cache_level
        
        # Memory cache (always present)
        self.memory_cache = LRUCache(
            max_size=memory_cache_size,
            max_memory_mb=memory_cache_mb,
            default_ttl=3600  # 1 hour default
        )
        
        # Disk cache
        self.disk_cache = None
        if cache_level in [CacheLevel.DISK, CacheLevel.HYBRID] and HAS_DISKCACHE:
            cache_dir = disk_cache_dir or "cache"
            Path(cache_dir).mkdir(exist_ok=True)
            self.disk_cache = DiskCache(cache_dir, size_limit=1000 * 1024 * 1024)  # 1GB
        
        # Redis cache
        self.redis_cache = None
        if cache_level in [CacheLevel.REDIS, CacheLevel.HYBRID] and HAS_REDIS:
            try:
                self.redis_cache = redis.from_url(redis_url or "redis://localhost:6379")
                self.redis_cache.ping()  # Test connection
            except Exception as e:
                logger.warning(f"Redis cache unavailable: {e}")
                self.redis_cache = None
        
        # Cache key generators
        self._key_generators: Dict[str, Callable] = {
            'field_calculation': self._generate_field_cache_key,
            'optimization_result': self._generate_optimization_cache_key,
            'array_response': self._generate_array_response_key,
            'propagation_matrix': self._generate_propagation_key
        }
        
        logger.info(f"Initialized acoustic cache with level: {cache_level.value}")
    
    def get_cached_field(
        self,
        transducer_positions: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
        frequency: float,
        field_bounds: List[Tuple[float, float]],
        resolution: float,
        medium_properties: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Get cached field calculation result."""
        cache_key = self._generate_field_cache_key(
            transducer_positions, phases, amplitudes, frequency,
            field_bounds, resolution, medium_properties
        )
        
        return self._get_from_cache('field_calculation', cache_key)
    
    def cache_field_result(
        self,
        transducer_positions: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
        frequency: float,
        field_bounds: List[Tuple[float, float]],
        resolution: float,
        medium_properties: Dict[str, Any],
        field_result: np.ndarray,
        ttl: Optional[float] = None
    ) -> bool:
        """Cache field calculation result."""
        cache_key = self._generate_field_cache_key(
            transducer_positions, phases, amplitudes, frequency,
            field_bounds, resolution, medium_properties
        )
        
        return self._put_in_cache('field_calculation', cache_key, field_result, ttl)
    
    def get_cached_optimization(
        self,
        target_field: np.ndarray,
        transducer_config: Dict[str, Any],
        optimization_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached optimization result."""
        cache_key = self._generate_optimization_cache_key(
            target_field, transducer_config, optimization_params
        )
        
        return self._get_from_cache('optimization_result', cache_key)
    
    def cache_optimization_result(
        self,
        target_field: np.ndarray,
        transducer_config: Dict[str, Any],
        optimization_params: Dict[str, Any],
        result: Dict[str, Any],
        ttl: Optional[float] = None
    ) -> bool:
        """Cache optimization result."""
        cache_key = self._generate_optimization_cache_key(
            target_field, transducer_config, optimization_params
        )
        
        return self._put_in_cache('optimization_result', cache_key, result, ttl)
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        # Implementation would depend on cache backend
        # For now, just clear memory cache matching pattern
        with self.memory_cache._lock:
            keys_to_remove = [
                key for key in self.memory_cache._cache.keys()
                if pattern in key
            ]
            
            for key in keys_to_remove:
                self.memory_cache._remove_entry(key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'memory_cache': self.memory_cache.get_stats(),
            'cache_level': self.cache_level.value,
            'disk_cache_available': self.disk_cache is not None,
            'redis_cache_available': self.redis_cache is not None
        }
        
        # Add disk cache stats if available
        if self.disk_cache:
            stats['disk_cache'] = {
                'size': len(self.disk_cache),
                'volume': self.disk_cache.volume()
            }
        
        return stats
    
    def _get_from_cache(self, cache_type: str, key: str) -> Optional[Any]:
        """Get value from appropriate cache level."""
        # Try memory cache first
        result = self.memory_cache.get(key)
        if result is not None:
            return result
        
        # Try disk cache
        if self.disk_cache and self.cache_level in [CacheLevel.DISK, CacheLevel.HYBRID]:
            try:
                result = self.disk_cache.get(key)
                if result is not None:
                    # Promote to memory cache
                    self.memory_cache.put(key, result, ttl=3600)
                    return result
            except Exception as e:
                logger.warning(f"Disk cache error: {e}")
        
        # Try Redis cache
        if self.redis_cache and self.cache_level in [CacheLevel.REDIS, CacheLevel.HYBRID]:
            try:
                serialized = self.redis_cache.get(key)
                if serialized:
                    result = pickle.loads(serialized)
                    # Promote to memory cache
                    self.memory_cache.put(key, result, ttl=3600)
                    return result
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        return None
    
    def _put_in_cache(self, cache_type: str, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in appropriate cache levels."""
        success = False
        
        # Always cache in memory
        if self.memory_cache.put(key, value, ttl):
            success = True
        
        # Cache in disk if configured
        if self.disk_cache and self.cache_level in [CacheLevel.DISK, CacheLevel.HYBRID]:
            try:
                self.disk_cache.set(key, value, expire=ttl)
                success = True
            except Exception as e:
                logger.warning(f"Disk cache error: {e}")
        
        # Cache in Redis if configured
        if self.redis_cache and self.cache_level in [CacheLevel.REDIS, CacheLevel.HYBRID]:
            try:
                serialized = pickle.dumps(value)
                if ttl:
                    self.redis_cache.setex(key, int(ttl), serialized)
                else:
                    self.redis_cache.set(key, serialized)
                success = True
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        return success
    
    def _generate_field_cache_key(
        self,
        positions: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
        frequency: float,
        bounds: List[Tuple[float, float]],
        resolution: float,
        medium: Dict[str, Any]
    ) -> str:
        """Generate cache key for field calculation."""
        # Create hash of all parameters
        hasher = hashlib.sha256()
        
        if HAS_NUMPY:
            hasher.update(positions.tobytes())
            hasher.update(phases.tobytes())
            hasher.update(amplitudes.tobytes())
        
        hasher.update(str(frequency).encode())
        hasher.update(str(bounds).encode())
        hasher.update(str(resolution).encode())
        hasher.update(json.dumps(medium, sort_keys=True).encode())
        
        return f"field_calc:{hasher.hexdigest()}"
    
    def _generate_optimization_cache_key(
        self,
        target: np.ndarray,
        transducer_config: Dict[str, Any],
        opt_params: Dict[str, Any]
    ) -> str:
        """Generate cache key for optimization result."""
        hasher = hashlib.sha256()
        
        if HAS_NUMPY:
            hasher.update(target.tobytes())
        
        hasher.update(json.dumps(transducer_config, sort_keys=True).encode())
        hasher.update(json.dumps(opt_params, sort_keys=True).encode())
        
        return f"optimization:{hasher.hexdigest()}"
    
    def _generate_array_response_key(self, array_config: Dict[str, Any]) -> str:
        """Generate cache key for array response."""
        hasher = hashlib.sha256()
        hasher.update(json.dumps(array_config, sort_keys=True).encode())
        return f"array_response:{hasher.hexdigest()}"
    
    def _generate_propagation_key(
        self,
        source_points: np.ndarray,
        field_points: np.ndarray,
        frequency: float,
        medium: Dict[str, Any]
    ) -> str:
        """Generate cache key for propagation matrix."""
        hasher = hashlib.sha256()
        
        if HAS_NUMPY:
            hasher.update(source_points.tobytes())
            hasher.update(field_points.tobytes())
        
        hasher.update(str(frequency).encode())
        hasher.update(json.dumps(medium, sort_keys=True).encode())
        
        return f"propagation:{hasher.hexdigest()}"
    
    def shutdown(self):
        """Shutdown all cache systems."""
        self.memory_cache.shutdown()
        
        if self.disk_cache:
            self.disk_cache.close()
        
        if self.redis_cache:
            try:
                self.redis_cache.close()
            except Exception:
                pass


# Global cache instance
_global_cache: Optional[AcousticComputationCache] = None


def get_global_cache() -> AcousticComputationCache:
    """Get or create global cache instance."""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = AcousticComputationCache(
            cache_level=CacheLevel.HYBRID,
            memory_cache_size=200,
            memory_cache_mb=100
        )
    
    return _global_cache


def cache_computation(cache_type: str, ttl: Optional[float] = None):
    """Decorator to automatically cache computation results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            cache = get_global_cache()
            
            # Generate cache key from function name and arguments
            hasher = hashlib.sha256()
            hasher.update(func.__name__.encode())
            hasher.update(str(args).encode())
            hasher.update(str(sorted(kwargs.items())).encode())
            
            cache_key = f"{cache_type}:{hasher.hexdigest()}"
            
            # Try to get from cache
            cached_result = cache._get_from_cache(cache_type, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache._put_in_cache(cache_type, cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator