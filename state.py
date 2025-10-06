import asyncio
from collections import defaultdict
from datetime import datetime, timedelta # Added timedelta
from typing import List, Tuple, Any, Optional # Added Optional
from typing import Dict
import json
import os

from config import config

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
        # Per-channel toggle: auto "podcast that shit" after RSS/allrss chunks
        self._podcast_after_rss_by_channel: Dict[int, bool] = {}
        # Persistent schedules for background tasks
        self.schedules_file = os.path.join(os.path.dirname(__file__), "schedules.json")
        self._schedules: List[Dict[str, Any]] = []
        # Track long-running per-channel tasks
        self._active_tasks: Dict[int, asyncio.Task[Any]] = {}
        # TTS delivery preferences per guild
        self.tts_delivery_file = os.path.join(os.path.dirname(__file__), "tts_delivery_modes.json")
        self._tts_delivery_by_guild: Dict[int, str] = {}
        try:
            self._load_schedules()
        except Exception:
            # Start with empty if load fails; errors are not fatal
            self._schedules = []
        try:
            self._load_tts_delivery_modes()
        except Exception:
            self._tts_delivery_by_guild = {}

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

    async def get_history_counts(self) -> Dict[int, int]:
        """Return the number of cached messages per channel."""
        async with self._lock:
            return {channel_id: len(messages) for channel_id, messages in self.message_history.items()}

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

    async def get_reminder_count(self) -> int:
        """Return the number of scheduled reminders."""
        async with self._lock:
            return len(self.reminders)

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

    async def set_podcast_after_rss_enabled(self, channel_id: int, enabled: bool) -> None:
        """Enable/disable auto-"podcast that shit" after RSS/allrss chunks for a channel."""
        async with self._lock:
            self._podcast_after_rss_by_channel[channel_id] = bool(enabled)

    async def is_podcast_after_rss_enabled(self, channel_id: int) -> bool:
        """Check if auto-podcast-after-RSS is enabled for the given channel."""
        async with self._lock:
            return bool(self._podcast_after_rss_by_channel.get(channel_id, False))

    async def get_podcast_after_rss_channels(self) -> Dict[int, bool]:
        """Return a snapshot of channels with auto-podcast enabled."""
        async with self._lock:
            return dict(self._podcast_after_rss_by_channel)

    # --- Scheduling support ---
    def _load_schedules(self) -> None:
        if os.path.exists(self.schedules_file):
            with open(self.schedules_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self._schedules = data

    def _save_schedules(self) -> None:
        tmp_path = self.schedules_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._schedules, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.schedules_file)

    async def list_schedules(self, channel_id: Optional[int] = None) -> List[Dict[str, Any]]:
        async with self._lock:
            if channel_id is None:
                return list(self._schedules)
            return [s for s in self._schedules if int(s.get("channel_id", -1)) == int(channel_id)]

    async def add_schedule(self, schedule: Dict[str, Any]) -> None:
        async with self._lock:
            self._schedules.append(schedule)
            self._save_schedules()

    async def remove_schedule(self, schedule_id: str) -> bool:
        async with self._lock:
            before = len(self._schedules)
            self._schedules = [s for s in self._schedules if s.get("id") != schedule_id]
            changed = len(self._schedules) != before
            if changed:
                self._save_schedules()
            return changed

    async def update_schedule_last_run(self, schedule_id: str, when: datetime) -> None:
        async with self._lock:
            for s in self._schedules:
                if s.get("id") == schedule_id:
                    s["last_run"] = when.isoformat()
                    break
            self._save_schedules()

    # --- Active task tracking ---
    async def set_active_task(self, channel_id: int, task: asyncio.Task[Any]) -> None:
        async with self._lock:
            self._active_tasks[channel_id] = task

    async def clear_active_task(self, channel_id: int, task: Optional[asyncio.Task[Any]] = None) -> None:
        async with self._lock:
            current = self._active_tasks.get(channel_id)
            if current and (task is None or current is task):
                self._active_tasks.pop(channel_id, None)

    async def cancel_active_task(self, channel_id: int) -> bool:
        async with self._lock:
            task = self._active_tasks.get(channel_id)
            if not task:
                return False
            task.cancel()
            return True

    # --- TTS delivery mode persistence ---
    def _load_tts_delivery_modes(self) -> None:
        if os.path.exists(self.tts_delivery_file):
            with open(self.tts_delivery_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    loaded: Dict[int, str] = {}
                    for key, value in data.items():
                        try:
                            loaded[int(key)] = str(value)
                        except (ValueError, TypeError):
                            continue
                    self._tts_delivery_by_guild = loaded

    def _save_tts_delivery_modes(self) -> None:
        tmp_path = self.tts_delivery_file + ".tmp"
        serializable = {str(gid): mode for gid, mode in self._tts_delivery_by_guild.items()}
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.tts_delivery_file)

    async def set_tts_delivery_mode(self, guild_id: int, mode: str) -> None:
        normalized = mode.lower()
        if normalized not in {"off", "audio", "video", "both"}:
            normalized = config.TTS_DELIVERY_DEFAULT
        async with self._lock:
            self._tts_delivery_by_guild[guild_id] = normalized
            self._save_tts_delivery_modes()

    async def get_tts_delivery_mode(self, guild_id: int) -> str:
        async with self._lock:
            return self._tts_delivery_by_guild.get(guild_id, config.TTS_DELIVERY_DEFAULT)

    async def clear_tts_delivery_mode(self, guild_id: int) -> None:
        async with self._lock:
            if guild_id in self._tts_delivery_by_guild:
                self._tts_delivery_by_guild.pop(guild_id, None)
                self._save_tts_delivery_modes()
