"""Enhanced Rate Limiter with global edits-per-second limiting and per-channel tracking."""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class GlobalEditRateLimiter:
    """Global rate limiter for Discord message edits per second."""
    
    def __init__(self, edits_per_second: float = 1.3):
        self._edits_per_second = max(0.1, edits_per_second)
        self._min_interval = 1.0 / self._edits_per_second
        self._lock = asyncio.Lock()
        self._edit_timestamps: deque[float] = deque()
        self._window_seconds = 1.0
    
    async def await_edit_slot(self) -> None:
        """Wait until an edit slot is available."""
        async with self._lock:
            now = time.monotonic()
            
            # Remove timestamps outside the window
            cutoff = now - self._window_seconds
            while self._edit_timestamps and self._edit_timestamps[0] < cutoff:
                self._edit_timestamps.popleft()
            
            # Check if we're at the limit
            if len(self._edit_timestamps) >= self._edits_per_second:
                # Wait until the oldest edit is outside the window
                oldest = self._edit_timestamps[0]
                wait_time = (oldest + self._window_seconds) - now
                if wait_time > 0:
                    logger.debug(
                        f"Global edit rate limit: waiting {wait_time:.3f}s "
                        f"({len(self._edit_timestamps)}/{self._edits_per_second} edits in window)"
                    )
                    await asyncio.sleep(wait_time)
                    # Re-check after sleep
                    return await self.await_edit_slot()
            
            # Record this edit
            self._edit_timestamps.append(time.monotonic())
    
    def get_current_rate(self) -> float:
        """Get the current edit rate (edits per second)."""
        async def _get():
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self._window_seconds
                while self._edit_timestamps and self._edit_timestamps[0] < cutoff:
                    self._edit_timestamps.popleft()
                return len(self._edit_timestamps) / self._window_seconds
        
        # This is a sync method, so we can't await
        # Return approximate rate based on current state
        return len(self._edit_timestamps) / self._window_seconds


class ChannelCommandTracker:
    """Tracks active commands per channel to prevent overlapping outputs."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._active_commands: Dict[int, Set[str]] = defaultdict(set)
        self._command_start_times: Dict[Tuple[int, str], datetime] = {}
    
    async def register_command(
        self,
        channel_id: int,
        command_id: str,
    ) -> bool:
        """Register a command as active in a channel.
        
        Returns:
            True if registered successfully, False if another command is already active.
        """
        async with self._lock:
            if self._active_commands[channel_id]:
                # Check if any existing command is still active
                now = datetime.now()
                active_cmds = list(self._active_commands[channel_id])
                for cmd_id in active_cmds:
                    start_time = self._command_start_times.get((channel_id, cmd_id))
                    if start_time:
                        # Consider command inactive if it's been running for more than 1 hour
                        if (now - start_time).total_seconds() > 3600:
                            await self.unregister_command(channel_id, cmd_id)
                
                # Check again after cleanup
                if self._active_commands[channel_id]:
                    logger.debug(
                        f"Channel {channel_id} already has active commands: "
                        f"{list(self._active_commands[channel_id])}"
                    )
                    return False
            
            self._active_commands[channel_id].add(command_id)
            self._command_start_times[(channel_id, command_id)] = datetime.now()
            logger.debug(f"Registered command {command_id} for channel {channel_id}")
            return True
    
    async def unregister_command(
        self,
        channel_id: int,
        command_id: str,
    ) -> None:
        """Unregister a command from a channel."""
        async with self._lock:
            self._active_commands[channel_id].discard(command_id)
            self._command_start_times.pop((channel_id, command_id), None)
            if not self._active_commands[channel_id]:
                del self._active_commands[channel_id]
            logger.debug(f"Unregistered command {command_id} from channel {channel_id}")
    
    async def is_channel_busy(self, channel_id: int) -> bool:
        """Check if a channel has active commands."""
        async with self._lock:
            return bool(self._active_commands.get(channel_id))
    
    async def get_active_commands(self, channel_id: int) -> Set[str]:
        """Get the set of active command IDs for a channel."""
        async with self._lock:
            return set(self._active_commands.get(channel_id, set()))
    
    async def clear_channel(self, channel_id: int) -> None:
        """Clear all active commands for a channel."""
        async with self._lock:
            if channel_id in self._active_commands:
                cmd_ids = list(self._active_commands[channel_id])
                for cmd_id in cmd_ids:
                    self._command_start_times.pop((channel_id, cmd_id), None)
                del self._active_commands[channel_id]
            logger.debug(f"Cleared all commands for channel {channel_id}")


# Global instances
_global_edit_limiter: Optional[GlobalEditRateLimiter] = None
_channel_tracker: Optional[ChannelCommandTracker] = None


def initialize_enhanced_rate_limiters(edits_per_second: float = 1.3) -> None:
    """Initialize global rate limiters."""
    global _global_edit_limiter, _channel_tracker
    _global_edit_limiter = GlobalEditRateLimiter(edits_per_second)
    _channel_tracker = ChannelCommandTracker()
    logger.info(f"Initialized enhanced rate limiters (edits/sec: {edits_per_second})")


def get_global_edit_limiter() -> Optional[GlobalEditRateLimiter]:
    """Get the global edit rate limiter."""
    return _global_edit_limiter


def get_channel_tracker() -> Optional[ChannelCommandTracker]:
    """Get the channel command tracker."""
    return _channel_tracker
