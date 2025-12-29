import asyncio
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class GlobalDiscordRateLimiter:
    def __init__(self, requests_per_second: float = 5.0):
        self.delay = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.delay:
                wait_time = self.delay - elapsed
                logger.debug(f"Global Discord Rate Limit: Waiting {wait_time:.3f}s")
                await asyncio.sleep(wait_time)
            self.last_request_time = time.monotonic()

# Global instance with a safe default (e.g., 5 requests per second)
# Discord allows more, but staying safe is better.
discord_limiter = GlobalDiscordRateLimiter(requests_per_second=4.0)
