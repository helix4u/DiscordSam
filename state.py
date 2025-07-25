import asyncio
from collections import defaultdict
from datetime import datetime, timedelta # Added timedelta
from typing import List, Tuple, Any, Optional # Added Optional

class BotState:
    """An async-safe container for the bot's shared, mutable state."""
    def __init__(self):
        self._lock = asyncio.Lock()
        # Stores MsgNode objects for short-term conversational context
        self.message_history = defaultdict(list)
        # Stores reminder tuples: (datetime, channel_id, user_id, message, time_str)
        self.reminders: List[Tuple[datetime, int, int, str, str]] = []
        # Stores the last time a Playwright-dependent command was initiated
        self.last_playwright_usage_time: Optional[datetime] = None
        # Locks to ensure only one LLM stream per channel at a time
        self.channel_locks = defaultdict(asyncio.Lock)
        # Global lock to serialize scraping operations across commands
        self.scrape_lock = asyncio.Lock()

    async def update_last_playwright_usage_time(self):
        """Updates the timestamp for the last Playwright usage."""
        async with self._lock:
            self.last_playwright_usage_time = datetime.now()
            # logger.debug(f"Last Playwright usage time updated to: {self.last_playwright_usage_time}") # Optional: for debugging

    async def get_last_playwright_usage_time(self) -> Optional[datetime]:
        """Gets the last Playwright usage time."""
        async with self._lock:
            return self.last_playwright_usage_time

    async def append_history(self, channel_id: int, msg_node: Any, max_len: int):
        """Appends a message node to a channel's history and trims it."""
        async with self._lock:
            self.message_history[channel_id].append(msg_node)
            # Ensure the total number of items doesn't exceed max_len
            if len(self.message_history[channel_id]) > max_len:
                self.message_history[channel_id] = self.message_history[channel_id][-max_len:]

    async def get_history(self, channel_id: int) -> List[Any]:
        """Gets a copy of a channel's message history."""
        async with self._lock:
            # Return a copy to prevent mutation outside the lock
            return list(self.message_history[channel_id])

    async def clear_channel_history(self, channel_id: int):
        """Clears the short-term message history for a specific channel."""
        async with self._lock:
            if channel_id in self.message_history:
                self.message_history[channel_id].clear()

    async def add_reminder(self, entry: Tuple[datetime, int, int, str, str]):
        """Adds a new reminder and keeps the list sorted by due time."""
        async with self._lock:
            self.reminders.append(entry)
            self.reminders.sort(key=lambda r: r[0])

    async def pop_due_reminders(self, now: datetime) -> List[Tuple[datetime, int, int, str, str]]:
        """Atomically gets and removes all reminders that are currently due."""
        async with self._lock:
            due = [r for r in self.reminders if r[0] <= now]
            self.reminders = [r for r in self.reminders if r[0] > now]
            return due

    def get_channel_lock(self, channel_id: int) -> asyncio.Lock:
        """Retrieve (or create) the asyncio.Lock for a specific channel."""
        return self.channel_locks[channel_id]

    def get_scrape_lock(self) -> asyncio.Lock:
        """Retrieve the global lock used for serializing scraping tasks."""
        return self.scrape_lock
