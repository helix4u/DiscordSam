import asyncio
import logging
import random
import time
from collections import deque
from typing import Mapping, Optional

from config import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Proactive async rate limiter with sliding window and per-key cooldowns.
    
    This limiter enforces both:
    1. Proactive rate limiting: max requests per time window (e.g., 16 req/min)
    2. Reactive cooldowns: respects 429 responses and backoff headers
    """

    def __init__(
        self,
        *,
        requests_per_minute: float = 16.0,
        jitter_seconds: float = 1.5,
        failure_backoff_seconds: float = 3.0,
        fallback_window_seconds: float = 90.0,
    ) -> None:
        self._requests_per_minute = max(1.0, requests_per_minute)
        self._window_seconds = 60.0  # Always use 60 second window for RPM calculation
        self._jitter_seconds = max(0.0, jitter_seconds)
        self._failure_backoff_seconds = max(0.0, failure_backoff_seconds)
        self._fallback_window_seconds = max(0.0, fallback_window_seconds)
        self._lock = asyncio.Lock()
        
        # Reactive cooldowns (from 429 responses)
        self._next_available: dict[str, float] = {}
        
        # Proactive rate limiting: track request timestamps per key
        self._request_timestamps: dict[str, deque[float]] = {}

    async def await_slot(self, key: str) -> None:
        """Wait until the given key is allowed to make a request.
        
        This enforces both:
        1. Any reactive cooldown from previous 429 responses
        2. Proactive rate limit based on requests per minute
        """
        if not key:
            key = "default"
        
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
                if len(timestamps) < self._requests_per_minute:
                    # We're good to go! Record this request timestamp
                    timestamps.append(now)
                    return
                
                # We've hit the rate limit, calculate wait time
                oldest = timestamps[0]
                wait_for = (oldest + self._window_seconds) - now
                
                if wait_for > 0:
                    logger.debug(
                        "Rate limiter: %s at limit (%d/%d requests in %.0fs window), waiting %.2fs",
                        key,
                        len(timestamps),
                        int(self._requests_per_minute),
                        self._window_seconds,
                        wait_for,
                    )
            
            # Sleep outside the lock to allow other coroutines to proceed
            if wait_for > 0:
                await asyncio.sleep(wait_for)

    async def record_response(
        self,
        key: str,
        status: int,
        headers: Mapping[str, str],
    ) -> None:
        """Update limiter state using HTTP response headers."""
        if not key:
            key = "default"

        # Normalize headers for consistent lookups.
        normalized = {k.lower(): v for k, v in headers.items()}

        retry_seconds: Optional[float] = None
        now_monotonic = time.monotonic()

        retry_after = normalized.get("retry-after")
        if retry_after:
            try:
                retry_seconds = float(retry_after)
            except ValueError:
                logger.debug("retry-after header not numeric for %s: %s", key, retry_after)

        if retry_seconds is None:
            rate_reset = normalized.get("x-rate-limit-reset")
            rate_remaining = normalized.get("x-rate-limit-remaining")
            if rate_reset and rate_remaining == "0":
                try:
                    reset_epoch = float(rate_reset)
                    delta = max(0.0, reset_epoch - time.time())
                    retry_seconds = delta
                except ValueError:
                    logger.debug("Invalid x-rate-limit-reset header for %s: %s", key, rate_reset)

        if retry_seconds is None and status == 429:
            retry_seconds = self._failure_backoff_seconds

        if retry_seconds is None:
            # When headers do not contain explicit signals but the status hints at limits,
            # provide a conservative fallback.
            if status in (500, 503):
                retry_seconds = min(self._fallback_window_seconds, self._failure_backoff_seconds)

        if retry_seconds is None:
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


_global_rate_limiter = RateLimiter(
    requests_per_minute=config.RATE_LIMIT_REQUESTS_PER_MINUTE,
    jitter_seconds=config.RATE_LIMIT_JITTER_SECONDS,
    failure_backoff_seconds=config.RATE_LIMIT_FAILURE_BACKOFF_SECONDS,
    fallback_window_seconds=config.RATE_LIMIT_FALLBACK_WINDOW_SECONDS,
)

# Discord message edits: per-channel throttle so we stay under Discord's limit
# and respect 429 retry_after instead of hammering the API.
_discord_edit_limiter = RateLimiter(
    requests_per_minute=config.DISCORD_EDIT_REQUESTS_PER_MINUTE,
    jitter_seconds=0.5,
    failure_backoff_seconds=5.0,
    fallback_window_seconds=60.0,
)


def get_rate_limiter() -> RateLimiter:
    """Return the shared rate limiter instance."""
    return _global_rate_limiter


def get_discord_edit_limiter() -> RateLimiter:
    """Return the rate limiter for Discord message edits (per channel)."""
    return _discord_edit_limiter

