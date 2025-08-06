"""
Intelligent caching system for acoustic holography computations.
Provides field computation caching, phase pattern caching, and adaptive cache management.
"""

import hashlib
import pickle
import time
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field
from threading import RLock
import numpy as np
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    cost: float = 1.0  # Computation cost saved
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at
    
    def idle_time(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed


class AdaptiveCache:
    """
    Adaptive cache with intelligent eviction policies.
    Combines LRU, LFU, and cost-aware eviction strategies.
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        max_memory_mb: float = 512.0,
        ttl_seconds: Optional[float] = 3600.0
    ):
        """
        Initialize adaptive cache.
        
        Args:
            max_entries: Maximum number of cache entries
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for entries (None for no expiration)
        """
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = RLock()
        self._total_memory = 0
        self._hit_count = 0
        self._miss_count = 0
        
        # Adaptive parameters
        self._access_weight = 0.3
        self._frequency_weight = 0.3
        self._cost_weight = 0.4
    
    def _hash_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create deterministic hash from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        # Handle numpy arrays specially
        processed_data = self._process_for_hashing(key_data)
        key_string = pickle.dumps(processed_data, protocol=pickle.HIGHEST_PROTOCOL)
        
        return hashlib.sha256(key_string).hexdigest()[:16]
    
    def _process_for_hashing(self, obj: Any) -> Any:
        """Process objects for consistent hashing."""
        if isinstance(obj, np.ndarray):
            return {
                'type': 'ndarray',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'data_hash': hashlib.sha256(obj.tobytes()).hexdigest()[:16]
            }
        elif isinstance(obj, dict):
            return {k: self._process_for_hashing(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._process_for_hashing(item) for item in obj]
        else:
            return obj
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if self.ttl_seconds and entry.age() > self.ttl_seconds:
                    self.remove(key)
                    self._miss_count += 1
                    return None
                
                entry.update_access()
                self._hit_count += 1
                logger.debug(f"Cache hit for key {key}")
                return entry.data
            
            self._miss_count += 1
            logger.debug(f"Cache miss for key {key}")
            return None
    
    def put(self, key: str, data: Any, cost: float = 1.0) -> bool:
        """Put value into cache."""
        with self._lock:
            # Calculate data size
            try:
                size_bytes = len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                size_bytes = 1024  # Fallback estimate
            
            # Check if we need to evict entries
            if not self._make_space(size_bytes):
                logger.warning(f"Could not make space for cache entry of {size_bytes} bytes")
                return False
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size_bytes,
                cost=cost
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self.remove(key)
            
            # Add new entry
            self._cache[key] = entry
            self._total_memory += size_bytes
            
            logger.debug(f"Cached entry {key} ({size_bytes} bytes, cost={cost})")
            return True
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._total_memory -= entry.size_bytes
                del self._cache[key]
                logger.debug(f"Removed cache entry {key}")
                return True
            return False
    
    def _make_space(self, required_bytes: int) -> bool:
        """Make space for new entry by evicting old ones."""
        # Check if we have space
        available_memory = self.max_memory_bytes - self._total_memory
        available_entries = self.max_entries - len(self._cache)
        
        if available_memory >= required_bytes and available_entries > 0:
            return True
        
        # Calculate how many entries to evict
        entries_to_evict = []
        memory_to_free = max(0, required_bytes - available_memory)
        entries_needed = max(0, 1 - available_entries)
        
        # Score entries for eviction (lower score = more likely to evict)
        scored_entries = []
        for entry in self._cache.values():
            # Normalize factors
            age_score = min(entry.age() / 3600, 1.0)  # Age factor (0-1)
            idle_score = min(entry.idle_time() / 1800, 1.0)  # Idle factor (0-1)
            freq_score = 1.0 / (1.0 + entry.access_count)  # Frequency factor
            cost_score = 1.0 / (1.0 + entry.cost)  # Cost factor
            
            # Combined eviction score
            eviction_score = (
                self._access_weight * idle_score +
                self._frequency_weight * freq_score +
                self._cost_weight * cost_score
            )
            
            scored_entries.append((eviction_score, entry))
        
        # Sort by eviction score (highest first)
        scored_entries.sort(reverse=True)
        
        # Select entries to evict
        memory_freed = 0
        for score, entry in scored_entries:
            entries_to_evict.append(entry.key)
            memory_freed += entry.size_bytes
            
            if (memory_freed >= memory_to_free and 
                len(entries_to_evict) >= entries_needed):
                break
        
        # Evict selected entries
        for key in entries_to_evict:
            self.remove(key)
        
        logger.debug(f"Evicted {len(entries_to_evict)} entries, freed {memory_freed} bytes")
        
        # Check if we now have enough space
        available_memory = self.max_memory_bytes - self._total_memory
        available_entries = self.max_entries - len(self._cache)
        
        return available_memory >= required_bytes and available_entries > 0
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_memory = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / max(total_requests, 1)
            
            return {
                'entries': len(self._cache),
                'max_entries': self.max_entries,
                'memory_usage_mb': self._total_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': self._total_memory / self.max_memory_bytes,
                'hit_count': self._hit_count,
                'miss_count': self._miss_count,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
    
    def optimize_parameters(self):
        """Optimize cache parameters based on access patterns."""
        with self._lock:
            if not self._cache:
                return
            
            # Analyze access patterns
            entries = list(self._cache.values())
            avg_access_count = np.mean([e.access_count for e in entries])
            avg_cost = np.mean([e.cost for e in entries])
            
            # Adjust weights based on patterns
            if avg_access_count > 5:  # High frequency access
                self._frequency_weight = min(0.5, self._frequency_weight + 0.1)
            
            if avg_cost > 10:  # High computation costs
                self._cost_weight = min(0.6, self._cost_weight + 0.1)
            
            # Normalize weights
            total_weight = (self._access_weight + 
                           self._frequency_weight + 
                           self._cost_weight)
            self._access_weight /= total_weight
            self._frequency_weight /= total_weight
            self._cost_weight /= total_weight
            
            logger.debug(f"Optimized cache weights: access={self._access_weight:.2f}, "
                        f"frequency={self._frequency_weight:.2f}, cost={self._cost_weight:.2f}")


class FieldCache(AdaptiveCache):
    """Specialized cache for acoustic field computations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def cache_field_computation(
        self,
        positions: np.ndarray,
        phases: np.ndarray,
        frequency: float,
        field: np.ndarray,
        computation_time: float
    ) -> str:
        """Cache field computation result."""
        key = self._hash_key(
            'field',
            positions=positions,
            phases=phases,
            frequency=frequency
        )
        
        # Cost is proportional to computation time
        cost = max(1.0, computation_time)
        
        self.put(key, field, cost=cost)
        return key
    
    def get_cached_field(
        self,
        positions: np.ndarray,
        phases: np.ndarray,
        frequency: float
    ) -> Optional[np.ndarray]:
        """Get cached field computation."""
        key = self._hash_key(
            'field',
            positions=positions,
            phases=phases,
            frequency=frequency
        )
        
        return self.get(key)


class OptimizationCache(AdaptiveCache):
    """Specialized cache for optimization results."""
    
    def cache_optimization_result(
        self,
        target_field: np.ndarray,
        method: str,
        iterations: int,
        result_phases: np.ndarray,
        final_loss: float,
        computation_time: float
    ) -> str:
        """Cache optimization result."""
        key = self._hash_key(
            'optimization',
            target=target_field,
            method=method,
            iterations=iterations
        )
        
        result = {
            'phases': result_phases,
            'final_loss': final_loss,
            'computation_time': computation_time
        }
        
        # Cost based on computation time and quality
        cost = computation_time * (1.0 + 1.0 / max(final_loss, 0.001))
        
        self.put(key, result, cost=cost)
        return key
    
    def get_cached_optimization(
        self,
        target_field: np.ndarray,
        method: str,
        iterations: int
    ) -> Optional[Dict[str, Any]]:
        """Get cached optimization result."""
        key = self._hash_key(
            'optimization',
            target=target_field,
            method=method,
            iterations=iterations
        )
        
        return self.get(key)


# Global cache instances
_field_cache: Optional[FieldCache] = None
_optimization_cache: Optional[OptimizationCache] = None


def get_field_cache() -> FieldCache:
    """Get global field cache instance."""
    global _field_cache
    if _field_cache is None:
        _field_cache = FieldCache(max_entries=500, max_memory_mb=256)
    return _field_cache


def get_optimization_cache() -> OptimizationCache:
    """Get global optimization cache instance."""
    global _optimization_cache
    if _optimization_cache is None:
        _optimization_cache = OptimizationCache(max_entries=200, max_memory_mb=128)
    return _optimization_cache


def cached_field_computation(func):
    """Decorator for caching field computations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache = get_field_cache()
        
        # Try to get from cache first
        key = cache._hash_key('func', func.__name__, *args, **kwargs)
        cached_result = cache.get(key)
        
        if cached_result is not None:
            logger.debug(f"Using cached result for {func.__name__}")
            return cached_result
        
        # Compute result
        start_time = time.time()
        result = func(*args, **kwargs)
        computation_time = time.time() - start_time
        
        # Cache result
        cost = max(1.0, computation_time * 10)  # Weight by computation time
        cache.put(key, result, cost=cost)
        
        logger.debug(f"Cached result for {func.__name__} (took {computation_time:.3f}s)")
        return result
    
    return wrapper


def cached_optimization(func):
    """Decorator for caching optimization results."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache = get_optimization_cache()
        
        # Try to get from cache first
        key = cache._hash_key('opt_func', func.__name__, *args, **kwargs)
        cached_result = cache.get(key)
        
        if cached_result is not None:
            logger.debug(f"Using cached optimization result for {func.__name__}")
            return cached_result
        
        # Compute result
        start_time = time.time()
        result = func(*args, **kwargs)
        computation_time = time.time() - start_time
        
        # Cache result with high cost due to expensive computation
        cost = max(10.0, computation_time * 100)
        cache.put(key, result, cost=cost)
        
        logger.debug(f"Cached optimization result for {func.__name__} (took {computation_time:.3f}s)")
        return result
    
    return wrapper


def clear_all_caches():
    """Clear all global caches."""
    get_field_cache().clear()
    get_optimization_cache().clear()
    logger.info("All caches cleared")


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches."""
    return {
        'field_cache': get_field_cache().get_stats(),
        'optimization_cache': get_optimization_cache().get_stats()
    }


def optimize_all_caches():
    """Optimize parameters for all caches."""
    get_field_cache().optimize_parameters()
    get_optimization_cache().optimize_parameters()
    logger.info("Cache parameters optimized")