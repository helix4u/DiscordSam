import asyncio
import logging
import random
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple
import json
import os

from config import config

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limit Header Parser
# ============================================================================

@dataclass
class RateLimitInfo:
    """Parsed rate limit information from response headers."""
    limit: Optional[int] = None
    remaining: Optional[int] = None
    reset_timestamp: Optional[float] = None
    reset_seconds: Optional[float] = None
    retry_after: Optional[float] = None
    tokens_limit: Optional[int] = None
    tokens_remaining: Optional[int] = None
    tokens_reset_seconds: Optional[float] = None
    
    def get_wait_time(self) -> Optional[float]:
        """Get recommended wait time based on headers."""
        if self.retry_after is not None:
            return self.retry_after
        
        if self.remaining is not None and self.remaining == 0:
            if self.reset_seconds is not None:
                return self.reset_seconds
            if self.reset_timestamp is not None:
                return max(0.0, self.reset_timestamp - time.time())
        
        return None


def parse_rate_limit_headers(headers: Mapping[str, str]) -> RateLimitInfo:
    """Parse rate limit information from response headers."""
    info = RateLimitInfo()
    normalized = {k.lower(): v for k, v in headers.items()}
    
    # Standard headers
    if "retry-after" in normalized:
        try:
            info.retry_after = float(normalized["retry-after"])
        except ValueError:
            pass
    
    # OpenAI-style headers
    if "x-ratelimit-limit-requests" in normalized:
        try:
            info.limit = int(normalized["x-ratelimit-limit-requests"])
        except ValueError:
            pass
    
    if "x-ratelimit-remaining-requests" in normalized:
        try:
            info.remaining = int(normalized["x-ratelimit-remaining-requests"])
        except ValueError:
            pass
    
    if "x-ratelimit-reset-requests" in normalized:
        try:
            # Can be seconds or timestamp
            value = normalized["x-ratelimit-reset-requests"]
            if value.endswith("ms"):
                info.reset_seconds = float(value[:-2]) / 1000.0
            elif value.endswith("s"):
                info.reset_seconds = float(value[:-1])
            else:
                val = float(value)
                if val > 1e9:  # Unix timestamp
                    info.reset_timestamp = val
                else:  # Seconds
                    info.reset_seconds = val
        except ValueError:
            pass
    
    # Token limits
    if "x-ratelimit-limit-tokens" in normalized:
        try:
            info.tokens_limit = int(normalized["x-ratelimit-limit-tokens"])
        except ValueError:
            pass
    
    if "x-ratelimit-remaining-tokens" in normalized:
        try:
            info.tokens_remaining = int(normalized["x-ratelimit-remaining-tokens"])
        except ValueError:
            pass
    
    if "x-ratelimit-reset-tokens" in normalized:
        try:
            value = normalized["x-ratelimit-reset-tokens"]
            if value.endswith("ms"):
                info.tokens_reset_seconds = float(value[:-2]) / 1000.0
            elif value.endswith("s"):
                info.tokens_reset_seconds = float(value[:-1])
            else:
                info.tokens_reset_seconds = float(value)
        except ValueError:
            pass
    
    # Anthropic-style headers
    if "anthropic-ratelimit-requests-limit" in normalized:
        try:
            info.limit = int(normalized["anthropic-ratelimit-requests-limit"])
        except ValueError:
            pass
    
    if "anthropic-ratelimit-requests-remaining" in normalized:
        try:
            info.remaining = int(normalized["anthropic-ratelimit-requests-remaining"])
        except ValueError:
            pass
    
    if "anthropic-ratelimit-requests-reset" in normalized:
        try:
            info.reset_timestamp = float(normalized["anthropic-ratelimit-requests-reset"])
        except ValueError:
            pass
    
    return info


# ============================================================================
# Enhanced Rate Limiter
# ============================================================================

@dataclass
class ProviderRateLimits:
    """Rate limit configuration for a provider."""
    requests_per_minute: float = 60.0
    tokens_per_minute: int = 100000
    concurrent_requests: int = 10
    
    # Proactive safety margins (percentage below limit to target)
    safety_margin: float = 0.9  # Use only 90% of capacity
    
    # Adaptive fields (updated from headers)
    actual_limit: Optional[int] = None
    actual_remaining: Optional[int] = None
    next_reset: Optional[float] = None


class RateLimiter:
    """Enhanced async rate limiter with comprehensive rate limit compliance.
    
    Features:
    1. Proactive rate limiting: Stay under limits proactively
    2. Rate limit header compliance: Parse and respect all provider headers
    3. Per-provider configurations: Different limits for different APIs
    4. Concurrent request tracking: Limit parallel requests
    5. Token-based limiting: Track token usage when available
    6. Adaptive limits: Learn from response headers
    """

    def __init__(
        self,
        *,
        requests_per_minute: float = 16.0,
        jitter_seconds: float = 1.5,
        failure_backoff_seconds: float = 3.0,
        fallback_window_seconds: float = 90.0,
    ) -> None:
        self._default_rpm = max(1.0, requests_per_minute)
        self._window_seconds = 60.0
        self._jitter_seconds = max(0.0, jitter_seconds)
        self._failure_backoff_seconds = max(0.0, failure_backoff_seconds)
        self._fallback_window_seconds = max(0.0, fallback_window_seconds)
        self._lock = asyncio.Lock()
        
        # Reactive cooldowns (from 429 responses)
        self._next_available: Dict[str, float] = {}
        
        # Proactive rate limiting: track request timestamps per key
        self._request_timestamps: Dict[str, deque] = {}
        
        # Token tracking per key
        self._token_usage: Dict[str, deque] = {}
        
        # Concurrent request tracking
        self._active_requests: Dict[str, int] = defaultdict(int)
        self._max_concurrent: Dict[str, int] = {}
        
        # Provider-specific configurations
        self._provider_limits: Dict[str, ProviderRateLimits] = {}
        
        # Statistics
        self._stats = {
            "requests_blocked": 0,
            "requests_allowed": 0,
            "cooldowns_triggered": 0,
            "rate_limit_hits": 0,
        }
        
        # Initialize default provider limits
        self._init_default_provider_limits()

    def _init_default_provider_limits(self) -> None:
        """Initialize default rate limits for known providers."""
        self._provider_limits = {
            "api.openai.com": ProviderRateLimits(
                requests_per_minute=60,
                tokens_per_minute=90000,
                concurrent_requests=25,
            ),
            "api.anthropic.com": ProviderRateLimits(
                requests_per_minute=50,
                tokens_per_minute=100000,
                concurrent_requests=10,
            ),
            "generativelanguage.googleapis.com": ProviderRateLimits(
                requests_per_minute=60,
                tokens_per_minute=1000000,
                concurrent_requests=20,
            ),
            "api.mistral.ai": ProviderRateLimits(
                requests_per_minute=100,
                tokens_per_minute=100000,
                concurrent_requests=15,
            ),
            "openrouter.ai": ProviderRateLimits(
                requests_per_minute=20,
                tokens_per_minute=50000,
                concurrent_requests=5,
            ),
            "localhost": ProviderRateLimits(
                requests_per_minute=1000,
                tokens_per_minute=10000000,
                concurrent_requests=50,
            ),
            "default": ProviderRateLimits(
                requests_per_minute=self._default_rpm,
                tokens_per_minute=100000,
                concurrent_requests=10,
            ),
        }

    def _get_provider_limits(self, key: str) -> ProviderRateLimits:
        """Get rate limits for a provider key."""
        # Try exact match
        if key in self._provider_limits:
            return self._provider_limits[key]
        
        # Try partial match (for host names)
        for provider_key, limits in self._provider_limits.items():
            if provider_key in key or key in provider_key:
                return limits
        
        return self._provider_limits["default"]

    def set_provider_limits(
        self,
        key: str,
        requests_per_minute: Optional[float] = None,
        tokens_per_minute: Optional[int] = None,
        concurrent_requests: Optional[int] = None,
    ) -> None:
        """Configure rate limits for a specific provider."""
        if key not in self._provider_limits:
            self._provider_limits[key] = ProviderRateLimits()
        
        limits = self._provider_limits[key]
        if requests_per_minute is not None:
            limits.requests_per_minute = requests_per_minute
        if tokens_per_minute is not None:
            limits.tokens_per_minute = tokens_per_minute
        if concurrent_requests is not None:
            limits.concurrent_requests = concurrent_requests

    async def await_slot(self, key: str, estimated_tokens: int = 0) -> None:
        """Wait until the given key is allowed to make a request.
        
        This enforces:
        1. Reactive cooldowns from 429 responses
        2. Proactive RPM limits
        3. Token-per-minute limits (if estimated_tokens provided)
        4. Concurrent request limits
        """
        if not key:
            key = "default"
        
        limits = self._get_provider_limits(key)
        effective_rpm = limits.requests_per_minute * limits.safety_margin
        
        while True:
            async with self._lock:
                now = time.monotonic()
                
                # Check reactive cooldown first
                ready_at = self._next_available.get(key, 0.0)
                if now < ready_at:
                    wait_for = ready_at - now
                    logger.debug(
                        "Rate limiter: %s in reactive cooldown, waiting %.2fs",
                        key,
                        wait_for,
                    )
                    self._stats["requests_blocked"] += 1
                
                # Check concurrent request limit
                max_concurrent = limits.concurrent_requests
                if self._active_requests[key] >= max_concurrent:
                    wait_for = 0.1  # Short wait for concurrent slot
                    logger.debug(
                        "Rate limiter: %s at concurrent limit (%d/%d), waiting",
                        key,
                        self._active_requests[key],
                        max_concurrent,
                    )
                    self._stats["requests_blocked"] += 1
                    await asyncio.sleep(wait_for)
                    continue
                
                # Check proactive rate limit (sliding window)
                if key not in self._request_timestamps:
                    self._request_timestamps[key] = deque()
                
                timestamps = self._request_timestamps[key]
                
                # Remove timestamps outside the sliding window
                cutoff = now - self._window_seconds
                while timestamps and timestamps[0] < cutoff:
                    timestamps.popleft()
                
                # Check if we're under the rate limit
                if len(timestamps) < effective_rpm:
                    # Check token limits if applicable
                    if estimated_tokens > 0:
                        if key not in self._token_usage:
                            self._token_usage[key] = deque()
                        
                        token_queue = self._token_usage[key]
                        while token_queue and token_queue[0][0] < cutoff:
                            token_queue.popleft()
                        
                        current_tokens = sum(t[1] for t in token_queue)
                        effective_tpm = limits.tokens_per_minute * limits.safety_margin
                        
                        if current_tokens + estimated_tokens > effective_tpm:
                            # Calculate wait time for token reset
                            if token_queue:
                                oldest_token_time = token_queue[0][0]
                                wait_for = (oldest_token_time + self._window_seconds) - now
                                if wait_for > 0:
                                    logger.debug(
                                        "Rate limiter: %s at token limit (%d + %d > %d TPM), waiting %.2fs",
                                        key,
                                        current_tokens,
                                        estimated_tokens,
                                        int(effective_tpm),
                                        wait_for,
                                    )
                                    self._stats["requests_blocked"] += 1
                                    await asyncio.sleep(wait_for)
                                    continue
                    
                    # We're good to go! Record this request
                    timestamps.append(now)
                    self._active_requests[key] += 1
                    self._stats["requests_allowed"] += 1
                    return
                
                # We've hit the rate limit, calculate wait time
                oldest = timestamps[0]
                wait_for = (oldest + self._window_seconds) - now
                
                if wait_for > 0:
                    logger.debug(
                        "Rate limiter: %s at limit (%d/%d requests in %.0fs window), waiting %.2fs",
                        key,
                        len(timestamps),
                        int(effective_rpm),
                        self._window_seconds,
                        wait_for,
                    )
                    self._stats["rate_limit_hits"] += 1
            
            # Sleep outside the lock
            if wait_for > 0:
                await asyncio.sleep(wait_for + random.uniform(0, self._jitter_seconds))

    async def release_slot(self, key: str) -> None:
        """Release a concurrent request slot."""
        if not key:
            key = "default"
        
        async with self._lock:
            if self._active_requests[key] > 0:
                self._active_requests[key] -= 1

    async def record_tokens(self, key: str, tokens: int) -> None:
        """Record token usage for a completed request."""
        if not key:
            key = "default"
        
        async with self._lock:
            if key not in self._token_usage:
                self._token_usage[key] = deque()
            
            self._token_usage[key].append((time.monotonic(), tokens))

    async def record_response(
        self,
        key: str,
        status: int,
        headers: Mapping[str, str],
    ) -> None:
        """Update limiter state using HTTP response headers."""
        if not key:
            key = "default"

        rate_info = parse_rate_limit_headers(headers)
        now_monotonic = time.monotonic()
        
        # Update adaptive limits from headers
        limits = self._get_provider_limits(key)
        if rate_info.limit is not None:
            limits.actual_limit = rate_info.limit
        if rate_info.remaining is not None:
            limits.actual_remaining = rate_info.remaining
        
        # Determine retry/wait time
        retry_seconds: Optional[float] = rate_info.get_wait_time()
        
        if retry_seconds is None and status == 429:
            retry_seconds = self._failure_backoff_seconds
            self._stats["cooldowns_triggered"] += 1

        if retry_seconds is None:
            if status in (500, 502, 503, 504):
                retry_seconds = min(self._fallback_window_seconds, self._failure_backoff_seconds)

        if retry_seconds is None:
            # Check if remaining is low and preemptively slow down
            if rate_info.remaining is not None and rate_info.remaining < 5:
                if rate_info.reset_seconds is not None:
                    # Spread remaining requests over reset window
                    retry_seconds = rate_info.reset_seconds / max(1, rate_info.remaining)
                    logger.debug(
                        "Rate limiter: %s proactively slowing down (remaining: %d), spacing by %.2fs",
                        key,
                        rate_info.remaining,
                        retry_seconds,
                    )
            return

        jitter = random.uniform(0.0, self._jitter_seconds) if self._jitter_seconds else 0.0
        next_ready = now_monotonic + retry_seconds + jitter

        async with self._lock:
            current_ready = self._next_available.get(key, 0.0)
            if next_ready > current_ready:
                self._next_available[key] = next_ready
                logger.info(
                    "Rate limiter set cooldown for %s: %.2fs (status %s).",
                    key,
                    retry_seconds + jitter,
                    status,
                )

    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            **self._stats,
            "provider_limits": {
                k: {
                    "rpm": v.requests_per_minute,
                    "tpm": v.tokens_per_minute,
                    "concurrent": v.concurrent_requests,
                    "actual_remaining": v.actual_remaining,
                }
                for k, v in self._provider_limits.items()
            },
            "active_requests": dict(self._active_requests),
        }


# ============================================================================
# Channel Output Rate Limiter (Discord-specific)
# ============================================================================

class ChannelOutputLimiter:
    """Rate limiter for Discord channel output to prevent overlapping messages.
    
    Features:
    1. Global edits per second limiting
    2. Per-channel output serialization
    3. Message queue management
    """

    def __init__(
        self,
        edits_per_second: float = 1.3,
        max_queue_size: int = 100,
    ):
        self._edits_per_second = max(0.1, edits_per_second)
        self._max_queue_size = max_queue_size
        self._lock = asyncio.Lock()
        
        # Per-channel locks for output serialization
        self._channel_locks: Dict[int, asyncio.Lock] = {}
        
        # Global edit timing
        self._last_edit_time: float = 0.0
        self._edit_interval = 1.0 / self._edits_per_second
        
        # Message queues per channel
        self._message_queues: Dict[int, deque] = {}
        
        # Active operations per channel
        self._active_operations: Dict[int, str] = {}
        
        # Statistics
        self._stats = {
            "edits_throttled": 0,
            "operations_serialized": 0,
        }

    def get_channel_lock(self, channel_id: int) -> asyncio.Lock:
        """Get or create a lock for a specific channel."""
        if channel_id not in self._channel_locks:
            self._channel_locks[channel_id] = asyncio.Lock()
        return self._channel_locks[channel_id]

    async def acquire_channel(
        self,
        channel_id: int,
        operation_id: str,
        timeout: float = 30.0,
    ) -> bool:
        """Acquire exclusive access to a channel for output.
        
        Returns True if acquired, False if timed out.
        """
        lock = self.get_channel_lock(channel_id)
        
        try:
            acquired = await asyncio.wait_for(lock.acquire(), timeout=timeout)
            if acquired:
                async with self._lock:
                    self._active_operations[channel_id] = operation_id
                    self._stats["operations_serialized"] += 1
                return True
        except asyncio.TimeoutError:
            logger.warning(
                "Channel %s output lock acquisition timed out for operation %s",
                channel_id,
                operation_id,
            )
        return False

    async def release_channel(self, channel_id: int) -> None:
        """Release exclusive access to a channel."""
        lock = self._channel_locks.get(channel_id)
        if lock and lock.locked():
            async with self._lock:
                self._active_operations.pop(channel_id, None)
            lock.release()

    async def throttle_edit(self) -> None:
        """Throttle to respect global edits per second limit."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_edit_time
            
            if elapsed < self._edit_interval:
                wait_time = self._edit_interval - elapsed
                self._stats["edits_throttled"] += 1
                await asyncio.sleep(wait_time)
            
            self._last_edit_time = time.monotonic()

    def is_channel_busy(self, channel_id: int) -> bool:
        """Check if a channel has an active operation."""
        return channel_id in self._active_operations

    def get_active_operation(self, channel_id: int) -> Optional[str]:
        """Get the active operation ID for a channel."""
        return self._active_operations.get(channel_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get output limiter statistics."""
        return {
            **self._stats,
            "active_channels": len(self._active_operations),
            "active_operations": dict(self._active_operations),
        }


# ============================================================================
# Global Instances
# ============================================================================

_global_rate_limiter = RateLimiter(
    requests_per_minute=config.RATE_LIMIT_REQUESTS_PER_MINUTE,
    jitter_seconds=config.RATE_LIMIT_JITTER_SECONDS,
    failure_backoff_seconds=config.RATE_LIMIT_FAILURE_BACKOFF_SECONDS,
    fallback_window_seconds=config.RATE_LIMIT_FALLBACK_WINDOW_SECONDS,
)

_channel_output_limiter: Optional[ChannelOutputLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Return the shared rate limiter instance."""
    return _global_rate_limiter


def get_channel_output_limiter() -> ChannelOutputLimiter:
    """Return the shared channel output limiter."""
    global _channel_output_limiter
    if _channel_output_limiter is None:
        edits_per_second = getattr(config, "EDITS_PER_SECOND", 1.3)
        _channel_output_limiter = ChannelOutputLimiter(
            edits_per_second=edits_per_second
        )
    return _channel_output_limiter


# ============================================================================
# Context Managers
# ============================================================================

class RateLimitedRequest:
    """Context manager for rate-limited requests."""
    
    def __init__(
        self,
        key: str,
        estimated_tokens: int = 0,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.key = key
        self.estimated_tokens = estimated_tokens
        self._limiter = rate_limiter or get_rate_limiter()
    
    async def __aenter__(self):
        await self._limiter.await_slot(self.key, self.estimated_tokens)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._limiter.release_slot(self.key)
        return False


class ChannelOutput:
    """Context manager for serialized channel output."""
    
    def __init__(
        self,
        channel_id: int,
        operation_id: str,
        timeout: float = 30.0,
        output_limiter: Optional[ChannelOutputLimiter] = None,
    ):
        self.channel_id = channel_id
        self.operation_id = operation_id
        self.timeout = timeout
        self._limiter = output_limiter or get_channel_output_limiter()
        self._acquired = False
    
    async def __aenter__(self):
        self._acquired = await self._limiter.acquire_channel(
            self.channel_id,
            self.operation_id,
            self.timeout,
        )
        if not self._acquired:
            raise TimeoutError(
                f"Could not acquire channel {self.channel_id} for output"
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            await self._limiter.release_channel(self.channel_id)
        return False
    
    async def throttle_edit(self) -> None:
        """Throttle before editing a message."""
        await self._limiter.throttle_edit()

