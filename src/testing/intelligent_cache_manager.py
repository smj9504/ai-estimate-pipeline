# src/testing/intelligent_cache_manager.py
"""
Intelligent Cache Manager for AI Estimation Pipeline
Implements sophisticated caching strategies for test data and computation results
"""
import asyncio
import json
import hashlib
import pickle
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import redis
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import threading

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheLevel(Enum):
    """Cache storage levels with different characteristics"""
    MEMORY = "memory"        # In-process memory (fastest, limited capacity)
    LOCAL_DISK = "local_disk" # Local SSD storage (fast, larger capacity)
    DISTRIBUTED = "distributed" # Redis/distributed cache (shared, network latency)
    ARCHIVE = "archive"      # Long-term storage (slowest, unlimited capacity)


class CacheStrategy(Enum):
    """Cache eviction and management strategies"""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    PRIORITY = "priority"    # Priority-based eviction
    HYBRID = "hybrid"        # Combination of strategies


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    priority: int = 1        # 1=low, 5=high priority
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def touch(self):
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class IntelligentCacheManager:
    """
    Multi-level intelligent cache manager for AI estimation pipeline
    
    Features:
    - Multi-level cache hierarchy (memory → disk → distributed → archive)
    - Intelligent eviction policies
    - Automatic cache warming
    - Content-aware compression
    - Cache analytics and optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.base_path = Path("test_outputs/cache")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Cache levels configuration
        self.memory_cache = {}
        self.memory_lock = threading.RLock()
        self.max_memory_entries = self.config.get('max_memory_entries', 1000)
        self.max_memory_size_mb = self.config.get('max_memory_size_mb', 512)
        
        # Local disk cache
        self.disk_cache_path = self.base_path / "disk_cache"
        self.disk_cache_path.mkdir(exist_ok=True)
        
        # SQLite for cache metadata
        self.metadata_db_path = self.base_path / "cache_metadata.db"
        self._initialize_metadata_db()
        
        # Redis connection (optional)
        self.redis_client = self._initialize_redis()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_memory_mb': 0,
            'size_disk_mb': 0
        }
    
    def _initialize_metadata_db(self):
        """Initialize SQLite database for cache metadata"""
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    priority INTEGER DEFAULT 1,
                    size_bytes INTEGER DEFAULT 0,
                    ttl_seconds INTEGER,
                    tags TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed);
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_priority ON cache_entries(priority);
            ''')
    
    def _initialize_redis(self) -> Optional['redis.Redis']:
        """Initialize Redis client if available"""
        try:
            redis_config = self.config.get('redis', {})
            if redis_config.get('enabled', False):
                import redis
                client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    decode_responses=False  # We handle binary data
                )
                client.ping()  # Test connection
                logger.info("Redis cache initialized")
                return client
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
        
        return None
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with intelligent level traversal
        
        Args:
            key: Cache key
            default: Default value if not found
        
        Returns:
            Cached value or default
        """
        cache_key = self._normalize_key(key)
        
        # Check memory cache first
        memory_result = await self._get_from_memory(cache_key)
        if memory_result is not None:
            self.stats['hits'] += 1
            return memory_result
        
        # Check disk cache
        disk_result = await self._get_from_disk(cache_key)
        if disk_result is not None:
            self.stats['hits'] += 1
            # Promote to memory cache
            await self._set_to_memory(cache_key, disk_result, priority=2)
            return disk_result
        
        # Check distributed cache (Redis)
        if self.redis_client:
            redis_result = await self._get_from_redis(cache_key)
            if redis_result is not None:
                self.stats['hits'] += 1
                # Promote to disk and memory
                await self._set_to_disk(cache_key, redis_result, priority=2)
                await self._set_to_memory(cache_key, redis_result, priority=2)
                return redis_result
        
        self.stats['misses'] += 1
        return default
    
    async def set(self, 
                  key: str, 
                  value: Any, 
                  ttl_seconds: Optional[int] = None,
                  priority: int = 1,
                  tags: List[str] = None,
                  force_levels: List[CacheLevel] = None) -> bool:
        """
        Set value in cache with intelligent level distribution
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            priority: Priority (1=low, 5=high)
            tags: Tags for grouping and invalidation
            force_levels: Force specific cache levels
        
        Returns:
            Success status
        """
        cache_key = self._normalize_key(key)
        tags = tags or []
        
        # Determine cache levels based on value characteristics
        if force_levels:
            target_levels = force_levels
        else:
            target_levels = await self._determine_optimal_levels(value, priority)
        
        success = True
        
        # Store in appropriate levels
        for level in target_levels:
            try:
                if level == CacheLevel.MEMORY:
                    await self._set_to_memory(cache_key, value, ttl_seconds, priority, tags)
                elif level == CacheLevel.LOCAL_DISK:
                    await self._set_to_disk(cache_key, value, ttl_seconds, priority, tags)
                elif level == CacheLevel.DISTRIBUTED and self.redis_client:
                    await self._set_to_redis(cache_key, value, ttl_seconds, priority, tags)
                elif level == CacheLevel.ARCHIVE:
                    await self._set_to_archive(cache_key, value, ttl_seconds, priority, tags)
            except Exception as e:
                logger.error(f"Failed to set cache at level {level}: {e}")
                success = False
        
        return success
    
    async def _determine_optimal_levels(self, value: Any, priority: int) -> List[CacheLevel]:
        """Determine optimal cache levels based on value characteristics"""
        levels = []
        
        # Estimate value size
        size_estimate = len(pickle.dumps(value))
        
        # High priority or small values go to memory
        if priority >= 4 or size_estimate < 1024 * 100:  # < 100KB
            levels.append(CacheLevel.MEMORY)
        
        # Most values go to disk
        levels.append(CacheLevel.LOCAL_DISK)
        
        # Large or shared values go to distributed cache
        if size_estimate > 1024 * 1024 or priority >= 3:  # > 1MB or medium+ priority
            if self.redis_client:
                levels.append(CacheLevel.DISTRIBUTED)
        
        # Long-term storage for valuable computations
        if priority >= 2:
            levels.append(CacheLevel.ARCHIVE)
        
        return levels
    
    async def _get_from_memory(self, key: str) -> Any:
        """Get value from memory cache"""
        with self.memory_lock:
            entry = self.memory_cache.get(key)
            if entry and not entry.is_expired():
                entry.touch()
                return entry.value
            elif entry and entry.is_expired():
                del self.memory_cache[key]
        
        return None
    
    async def _set_to_memory(self, 
                             key: str, 
                             value: Any, 
                             ttl_seconds: Optional[int] = None,
                             priority: int = 1,
                             tags: List[str] = None) -> bool:
        """Set value in memory cache"""
        tags = tags or []
        
        # Check memory limits before adding
        await self._enforce_memory_limits()
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            priority=priority,
            size_bytes=len(pickle.dumps(value)),
            ttl_seconds=ttl_seconds,
            tags=tags
        )
        
        with self.memory_lock:
            self.memory_cache[key] = entry
            self._update_memory_stats()
        
        return True
    
    async def _get_from_disk(self, key: str) -> Any:
        """Get value from disk cache"""
        cache_file = self.disk_cache_path / f"{key}.gz"
        
        if not cache_file.exists():
            return None
        
        try:
            with gzip.open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check TTL
            metadata = await self._get_entry_metadata(key, CacheLevel.LOCAL_DISK)
            if metadata and metadata.get('ttl_seconds'):
                created_at = datetime.fromisoformat(metadata['created_at'])
                if (datetime.now() - created_at).total_seconds() > metadata['ttl_seconds']:
                    cache_file.unlink()  # Remove expired file
                    await self._delete_entry_metadata(key, CacheLevel.LOCAL_DISK)
                    return None
            
            # Update access metadata
            await self._update_entry_metadata(key, CacheLevel.LOCAL_DISK)
            
            return data
            
        except Exception as e:
            logger.error(f"Error reading disk cache {key}: {e}")
            if cache_file.exists():
                cache_file.unlink()  # Remove corrupted file
            return None
    
    async def _set_to_disk(self, 
                           key: str, 
                           value: Any,
                           ttl_seconds: Optional[int] = None,
                           priority: int = 1,
                           tags: List[str] = None) -> bool:
        """Set value in disk cache"""
        cache_file = self.disk_cache_path / f"{key}.gz"
        
        try:
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Store metadata
            await self._store_entry_metadata(key, CacheLevel.LOCAL_DISK, ttl_seconds, priority, tags)
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing disk cache {key}: {e}")
            return False
    
    async def _get_from_redis(self, key: str) -> Any:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(f"ai_estimate:{key}")
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error reading Redis cache {key}: {e}")
        
        return None
    
    async def _set_to_redis(self,
                            key: str,
                            value: Any,
                            ttl_seconds: Optional[int] = None,
                            priority: int = 1,
                            tags: List[str] = None) -> bool:
        """Set value in Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            data = pickle.dumps(value)
            redis_key = f"ai_estimate:{key}"
            
            if ttl_seconds:
                self.redis_client.setex(redis_key, ttl_seconds, data)
            else:
                self.redis_client.set(redis_key, data)
            
            # Store metadata
            await self._store_entry_metadata(key, CacheLevel.DISTRIBUTED, ttl_seconds, priority, tags)
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing Redis cache {key}: {e}")
            return False
    
    async def _set_to_archive(self,
                              key: str,
                              value: Any,
                              ttl_seconds: Optional[int] = None,
                              priority: int = 1,
                              tags: List[str] = None) -> bool:
        """Set value in archive storage"""
        archive_path = self.base_path / "archive"
        archive_path.mkdir(exist_ok=True)
        
        # Use date-based partitioning for archives
        date_partition = datetime.now().strftime("%Y/%m/%d")
        partition_path = archive_path / date_partition
        partition_path.mkdir(parents=True, exist_ok=True)
        
        archive_file = partition_path / f"{key}.gz"
        
        try:
            with gzip.open(archive_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Store metadata
            await self._store_entry_metadata(key, CacheLevel.ARCHIVE, ttl_seconds, priority, tags)
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing archive cache {key}: {e}")
            return False
    
    async def _store_entry_metadata(self,
                                    key: str,
                                    level: CacheLevel,
                                    ttl_seconds: Optional[int],
                                    priority: int,
                                    tags: List[str]) -> bool:
        """Store cache entry metadata in SQLite"""
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (key, level, created_at, last_accessed, priority, ttl_seconds, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    key,
                    level.value,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    priority,
                    ttl_seconds,
                    json.dumps(tags)
                ))
            return True
        except Exception as e:
            logger.error(f"Error storing metadata for {key}: {e}")
            return False
    
    async def _get_entry_metadata(self, key: str, level: CacheLevel) -> Optional[Dict]:
        """Get cache entry metadata"""
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                cursor = conn.execute(
                    'SELECT * FROM cache_entries WHERE key = ? AND level = ?',
                    (key, level.value)
                )
                row = cursor.fetchone()
                
                if row:
                    return {
                        'key': row[0],
                        'level': row[1],
                        'created_at': row[2],
                        'last_accessed': row[3],
                        'access_count': row[4],
                        'priority': row[5],
                        'size_bytes': row[6],
                        'ttl_seconds': row[7],
                        'tags': json.loads(row[8]) if row[8] else [],
                        'metadata': json.loads(row[9]) if row[9] else {}
                    }
        except Exception as e:
            logger.error(f"Error getting metadata for {key}: {e}")
        
        return None
    
    async def _update_entry_metadata(self, key: str, level: CacheLevel):
        """Update entry access metadata"""
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                conn.execute('''
                    UPDATE cache_entries 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE key = ? AND level = ?
                ''', (datetime.now().isoformat(), key, level.value))
        except Exception as e:
            logger.error(f"Error updating metadata for {key}: {e}")
    
    async def _delete_entry_metadata(self, key: str, level: CacheLevel):
        """Delete entry metadata"""
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                conn.execute(
                    'DELETE FROM cache_entries WHERE key = ? AND level = ?',
                    (key, level.value)
                )
        except Exception as e:
            logger.error(f"Error deleting metadata for {key}: {e}")
    
    def _normalize_key(self, key: str) -> str:
        """Normalize cache key for consistency"""
        if isinstance(key, dict):
            # Convert dict to deterministic string
            key_str = json.dumps(key, sort_keys=True)
        else:
            key_str = str(key)
        
        # Create hash for long keys
        if len(key_str) > 200:
            return hashlib.sha256(key_str.encode()).hexdigest()
        
        # Sanitize key for filesystem
        return key_str.replace('/', '_').replace('\\', '_')
    
    async def _enforce_memory_limits(self):
        """Enforce memory cache size limits"""
        with self.memory_lock:
            # Check entry count limit
            if len(self.memory_cache) >= self.max_memory_entries:
                await self._evict_from_memory(count=max(1, len(self.memory_cache) // 10))
            
            # Check size limit
            total_size_mb = sum(entry.size_bytes for entry in self.memory_cache.values()) / (1024 * 1024)
            if total_size_mb > self.max_memory_size_mb:
                await self._evict_from_memory(target_size_mb=self.max_memory_size_mb * 0.8)
    
    async def _evict_from_memory(self, count: Optional[int] = None, target_size_mb: Optional[float] = None):
        """Evict entries from memory cache using LRU strategy"""
        if not self.memory_cache:
            return
        
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: (x[1].priority, x[1].last_accessed)  # Priority first, then LRU
        )
        
        evicted = 0
        current_size_mb = sum(entry.size_bytes for entry in self.memory_cache.values()) / (1024 * 1024)
        
        for key, entry in sorted_entries:
            if count and evicted >= count:
                break
            
            if target_size_mb and current_size_mb <= target_size_mb:
                break
            
            # Remove entry
            del self.memory_cache[key]
            current_size_mb -= entry.size_bytes / (1024 * 1024)
            evicted += 1
            self.stats['evictions'] += 1
        
        logger.info(f"Evicted {evicted} entries from memory cache")
    
    def _update_memory_stats(self):
        """Update memory cache statistics"""
        self.stats['size_memory_mb'] = sum(
            entry.size_bytes for entry in self.memory_cache.values()
        ) / (1024 * 1024)
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags"""
        invalidated = 0
        
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                # Find entries with matching tags
                cursor = conn.execute('SELECT key, level, tags FROM cache_entries')
                entries_to_invalidate = []
                
                for row in cursor.fetchall():
                    entry_tags = json.loads(row[2]) if row[2] else []
                    if any(tag in entry_tags for tag in tags):
                        entries_to_invalidate.append((row[0], CacheLevel(row[1])))
                
                # Invalidate found entries
                for key, level in entries_to_invalidate:
                    await self._invalidate_entry(key, level)
                    invalidated += 1
        
        except Exception as e:
            logger.error(f"Error invalidating by tags: {e}")
        
        logger.info(f"Invalidated {invalidated} cache entries by tags: {tags}")
        return invalidated
    
    async def _invalidate_entry(self, key: str, level: CacheLevel):
        """Invalidate specific cache entry"""
        try:
            # Remove from memory
            with self.memory_lock:
                if key in self.memory_cache:
                    del self.memory_cache[key]
            
            # Remove from disk
            if level == CacheLevel.LOCAL_DISK:
                cache_file = self.disk_cache_path / f"{key}.gz"
                if cache_file.exists():
                    cache_file.unlink()
            
            # Remove from Redis
            elif level == CacheLevel.DISTRIBUTED and self.redis_client:
                self.redis_client.delete(f"ai_estimate:{key}")
            
            # Remove metadata
            await self._delete_entry_metadata(key, level)
            
        except Exception as e:
            logger.error(f"Error invalidating entry {key} at {level}: {e}")
    
    async def warm_cache(self, 
                         dataset_generator: callable,
                         warm_count: int = 100,
                         priority: int = 3) -> Dict[str, Any]:
        """Warm cache with commonly accessed data"""
        logger.info(f"Starting cache warming with {warm_count} entries")
        
        warmed = 0
        errors = 0
        
        try:
            # Generate warming dataset
            warming_data = await dataset_generator(warm_count)
            
            # Store in cache with high priority
            for item in warming_data:
                try:
                    key = self._generate_warming_key(item)
                    await self.set(
                        key=key,
                        value=item,
                        priority=priority,
                        tags=['cache_warming'],
                        force_levels=[CacheLevel.MEMORY, CacheLevel.LOCAL_DISK]
                    )
                    warmed += 1
                except Exception as e:
                    logger.error(f"Error warming cache item: {e}")
                    errors += 1
        
        except Exception as e:
            logger.error(f"Error in cache warming: {e}")
            errors += 1
        
        result = {
            'warmed_entries': warmed,
            'errors': errors,
            'success_rate': warmed / (warmed + errors) if (warmed + errors) > 0 else 0
        }
        
        logger.info(f"Cache warming completed: {result}")
        return result
    
    def _generate_warming_key(self, item: Dict) -> str:
        """Generate cache key for warming items"""
        # Generate key based on item characteristics
        key_components = []
        
        if 'rooms' in item:
            key_components.append(f"rooms_{len(item['rooms'])}")
        
        if 'data' in item and 'rooms' in item['data']:
            total_sqft = sum(
                room.get('measurements', {}).get('sqft', 0) 
                for room in item['data']['rooms']
            )
            key_components.append(f"sqft_{int(total_sqft)}")
        
        return f"warming_{'_'.join(key_components)}_{hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()[:8]}"
    
    async def get_cache_analytics(self) -> Dict[str, Any]:
        """Get comprehensive cache analytics"""
        analytics = {
            'statistics': self.stats.copy(),
            'hit_rate': self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0,
            'memory_cache': {
                'entries': len(self.memory_cache),
                'size_mb': self.stats['size_memory_mb'],
                'utilization': len(self.memory_cache) / self.max_memory_entries
            }
        }
        
        # Get database analytics
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                # Count by level
                cursor = conn.execute('SELECT level, COUNT(*) FROM cache_entries GROUP BY level')
                level_counts = dict(cursor.fetchall())
                
                # Top accessed entries
                cursor = conn.execute(
                    'SELECT key, access_count FROM cache_entries ORDER BY access_count DESC LIMIT 10'
                )
                top_accessed = cursor.fetchall()
                
                analytics['database'] = {
                    'level_distribution': level_counts,
                    'top_accessed_keys': top_accessed
                }
        
        except Exception as e:
            logger.error(f"Error getting database analytics: {e}")
            analytics['database'] = {'error': str(e)}
        
        return analytics
    
    async def cleanup_expired_entries(self) -> Dict[str, int]:
        """Clean up expired cache entries"""
        logger.info("Starting cache cleanup")
        
        cleaned = {'memory': 0, 'disk': 0, 'redis': 0, 'archive': 0}
        
        # Clean memory cache
        with self.memory_lock:
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self.memory_cache[key]
                cleaned['memory'] += 1
        
        # Clean disk cache using metadata
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                # Find expired entries
                cursor = conn.execute('''
                    SELECT key, level FROM cache_entries 
                    WHERE ttl_seconds IS NOT NULL 
                    AND (julianday('now') - julianday(created_at)) * 24 * 3600 > ttl_seconds
                ''')
                
                expired_entries = cursor.fetchall()
                
                for key, level_str in expired_entries:
                    level = CacheLevel(level_str)
                    await self._invalidate_entry(key, level)
                    cleaned[level.value] += 1
        
        except Exception as e:
            logger.error(f"Error cleaning expired entries: {e}")
        
        logger.info(f"Cache cleanup completed: {cleaned}")
        return cleaned


# Example usage and testing
async def main():
    """Example usage of intelligent cache manager"""
    cache = IntelligentCacheManager({
        'max_memory_entries': 100,
        'max_memory_size_mb': 50
    })
    
    # Test data
    test_data = {
        'rooms': [
            {'name': 'living_room', 'sqft': 200},
            {'name': 'bedroom', 'sqft': 150}
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    # Store in cache
    await cache.set('test_scenario_1', test_data, priority=4, tags=['test', 'phase1'])
    
    # Retrieve from cache
    cached_data = await cache.get('test_scenario_1')
    print(f"Retrieved from cache: {cached_data is not None}")
    
    # Get analytics
    analytics = await cache.get_cache_analytics()
    print(f"Cache analytics: {analytics}")


if __name__ == "__main__":
    asyncio.run(main())