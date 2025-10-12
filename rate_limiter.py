import asyncio
import logging
import random
import time
from typing import Mapping, Optional

from config import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple async rate limiter that tracks per-key cooldowns."""

    def __init__(
        self,
        *,
        jitter_seconds: float = 1.5,
        failure_backoff_seconds: float = 30.0,
        fallback_window_seconds: float = 60.0,
    ) -> None:
        self._jitter_seconds = max(0.0, jitter_seconds)
        self._failure_backoff_seconds = max(0.0, failure_backoff_seconds)
        self._fallback_window_seconds = max(0.0, fallback_window_seconds)
        self._lock = asyncio.Lock()
        self._next_available: dict[str, float] = {}

    async def await_slot(self, key: str) -> None:
        """Wait until the given key is allowed to make a request."""
        if not key:
            key = "default"
        while True:
            async with self._lock:
                ready_at = self._next_available.get(key, 0.0)
            now = time.monotonic()
            if now >= ready_at:
                return
            wait_for = ready_at - now
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
    jitter_seconds=config.RATE_LIMIT_JITTER_SECONDS,
    failure_backoff_seconds=config.RATE_LIMIT_FAILURE_BACKOFF_SECONDS,
    fallback_window_seconds=config.RATE_LIMIT_FALLBACK_WINDOW_SECONDS,
)


def get_rate_limiter() -> RateLimiter:
    """Return the shared rate limiter instance."""
    return _global_rate_limiter

