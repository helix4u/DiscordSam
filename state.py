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
        base_dir = os.path.dirname(__file__)
        self.schedules_file = os.path.join(base_dir, "schedules.json")
        self.scheduler_status_file = os.path.join(base_dir, "scheduler_status.json")
        self._schedules: List[Dict[str, Any]] = []
        self._schedules_paused: bool = False
        self._schedules_paused_reason: str = ""
        self._schedules_paused_by: Optional[str] = None
        self._schedules_paused_at: Optional[str] = None
        # Track long-running per-channel tasks
        self._active_tasks: Dict[int, asyncio.Task[Any]] = {}
        # TTS delivery preferences per guild
        self.tts_delivery_file = os.path.join(os.path.dirname(__file__), "tts_delivery_modes.json")
        self._tts_delivery_by_guild: Dict[int, str] = {}
        # Custom Twitter lists per guild (or per-admin DM scope) for scheduled/alltweets commands
        self.twitter_lists_file = os.path.join(os.path.dirname(__file__), "twitter_lists.json")
        self._twitter_lists: Dict[str, Dict[str, List[str]]] = {}
        # Per-guild: show raw tweet embeds in scheduled alltweets (default False = summary only)
        self._scheduled_alltweets_show_tweets_file = os.path.join(
            os.path.dirname(__file__), "scheduled_alltweets_show_tweets.json"
        )
        self._scheduled_alltweets_show_tweets_by_guild: Dict[int, bool] = {}
        try:
            self._load_schedules()
        except Exception:
            # Start with empty if load fails; errors are not fatal
            self._schedules = []
        try:
            self._load_scheduler_status()
        except Exception:
            self._schedules_paused = False
            self._schedules_paused_reason = ""
            self._schedules_paused_by = None
            self._schedules_paused_at = None
        try:
            self._load_tts_delivery_modes()
        except Exception:
            self._tts_delivery_by_guild = {}
        try:
            self._load_twitter_lists()
        except Exception:
            self._twitter_lists = {}
        try:
            self._load_scheduled_alltweets_show_tweets()
        except Exception:
            self._scheduled_alltweets_show_tweets_by_guild = {}

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

    def _load_scheduler_status(self) -> None:
        if os.path.exists(self.scheduler_status_file):
            with open(self.scheduler_status_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._schedules_paused = bool(data.get("paused", False))
                    self._schedules_paused_reason = str(data.get("reason") or "")
                    paused_by = data.get("paused_by")
                    self._schedules_paused_by = str(paused_by) if paused_by is not None else None
                    paused_at = data.get("paused_at")
                    self._schedules_paused_at = paused_at if isinstance(paused_at, str) else None

    def _save_scheduler_status(self) -> None:
        tmp_path = self.scheduler_status_file + ".tmp"
        payload = {
            "paused": self._schedules_paused,
            "reason": self._schedules_paused_reason,
            "paused_by": self._schedules_paused_by,
            "paused_at": self._schedules_paused_at,
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.scheduler_status_file)

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

    async def set_schedules_paused(
        self,
        paused: bool,
        *,
        reason: str = "",
        user_id: Optional[int] = None,
    ) -> None:
        """Toggle the global pause state for scheduled jobs."""
        async with self._lock:
            if paused:
                trimmed_reason = reason.strip()
                if len(trimmed_reason) > 200:
                    trimmed_reason = trimmed_reason[:200] + "…"
                self._schedules_paused = True
                self._schedules_paused_reason = trimmed_reason
                self._schedules_paused_by = str(user_id) if user_id is not None else None
                self._schedules_paused_at = datetime.now().astimezone().isoformat()
            else:
                self._schedules_paused = False
                self._schedules_paused_reason = ""
                self._schedules_paused_by = None
                self._schedules_paused_at = None
            self._save_scheduler_status()

    async def get_schedules_pause_state(self) -> Dict[str, Any]:
        """Return the current pause metadata for scheduled jobs."""
        async with self._lock:
            return {
                "paused": self._schedules_paused,
                "reason": self._schedules_paused_reason,
                "paused_by": self._schedules_paused_by,
                "paused_at": self._schedules_paused_at,
            }

    async def are_schedules_paused(self) -> bool:
        async with self._lock:
            return self._schedules_paused

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

    # --- Scheduled alltweets: show raw tweet data (per guild) ---
    def _load_scheduled_alltweets_show_tweets(self) -> None:
        if os.path.exists(self._scheduled_alltweets_show_tweets_file):
            with open(self._scheduled_alltweets_show_tweets_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    loaded: Dict[int, bool] = {}
                    for key, value in data.items():
                        try:
                            loaded[int(key)] = bool(value)
                        except (ValueError, TypeError):
                            continue
                    self._scheduled_alltweets_show_tweets_by_guild = loaded

    def _save_scheduled_alltweets_show_tweets(self) -> None:
        tmp_path = self._scheduled_alltweets_show_tweets_file + ".tmp"
        serializable = {
            str(gid): enabled
            for gid, enabled in self._scheduled_alltweets_show_tweets_by_guild.items()
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._scheduled_alltweets_show_tweets_file)

    async def get_scheduled_alltweets_show_tweets(self, guild_id: Optional[int]) -> bool:
        """Whether to post raw tweet embeds in scheduled alltweets for this scope. Default False.
        Scope: guild_id in servers; negative user id (-user_id) in DMs."""
        if guild_id is None:
            return False
        async with self._lock:
            return self._scheduled_alltweets_show_tweets_by_guild.get(int(guild_id), False)

    async def set_scheduled_alltweets_show_tweets(self, guild_id: int, enabled: bool) -> None:
        async with self._lock:
            self._scheduled_alltweets_show_tweets_by_guild[int(guild_id)] = bool(enabled)
            self._save_scheduled_alltweets_show_tweets()

    # --- Twitter list persistence ---
    def _load_twitter_lists(self) -> None:
        if os.path.exists(self.twitter_lists_file):
            with open(self.twitter_lists_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    parsed: Dict[str, Dict[str, List[str]]] = {}
                    for scope_key, lists in data.items():
                        if not isinstance(lists, dict):
                            continue
                        normalized_scope = str(scope_key)
                        if ":" not in normalized_scope and normalized_scope.isdigit():
                            normalized_scope = f"guild:{normalized_scope}"
                        parsed_lists: Dict[str, List[str]] = {}
                        for list_name, handles in lists.items():
                            if isinstance(handles, list):
                                normalized = sorted(
                                    {str(h).lstrip("@").lower() for h in handles if h}
                                )
                                parsed_lists[str(list_name)] = normalized
                        if parsed_lists:
                            parsed[normalized_scope] = parsed_lists
                    self._twitter_lists = parsed

    def _save_twitter_lists(self) -> None:
        tmp_path = self.twitter_lists_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._twitter_lists, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.twitter_lists_file)

    @staticmethod
    def _twitter_scope_key(guild_id: Optional[int], user_id: Optional[int]) -> str:
        if guild_id is not None:
            return f"guild:{int(guild_id)}"
        if user_id is not None:
            return f"dm:{int(user_id)}"
        return "global"

    async def add_twitter_list_handle(
        self,
        guild_id: Optional[int],
        list_name: str,
        handle: str,
        *,
        user_id: Optional[int] = None,
    ) -> bool:
        """Add a handle to a named Twitter list for a guild."""
        normalized_handle = handle.lstrip("@").lower()
        if not normalized_handle:
            return False
        normalized_list = list_name.strip().lower()
        if not normalized_list:
            return False
        async with self._lock:
            scope_key = self._twitter_scope_key(guild_id, user_id)
            lists = self._twitter_lists.setdefault(scope_key, {})
            handles = lists.setdefault(normalized_list, [])
            if normalized_handle in handles:
                return False
            handles.append(normalized_handle)
            handles.sort()
            self._save_twitter_lists()
            return True

    async def set_twitter_list(
        self,
        guild_id: Optional[int],
        list_name: str,
        handles: List[str],
        *,
        user_id: Optional[int] = None,
    ) -> None:
        """Replace an entire twitter list with provided handles."""
        normalized_list = list_name.strip().lower()
        if not normalized_list:
            return
        cleaned_handles = sorted({h.lstrip("@").lower() for h in handles if h})
        async with self._lock:
            scope_key = self._twitter_scope_key(guild_id, user_id)
            lists = self._twitter_lists.setdefault(scope_key, {})
            lists[normalized_list] = cleaned_handles
            if not cleaned_handles:
                lists.pop(normalized_list, None)
            if not lists:
                self._twitter_lists.pop(scope_key, None)
            self._save_twitter_lists()

    async def remove_twitter_list_handle(
        self,
        guild_id: Optional[int],
        list_name: str,
        handle: str,
        *,
        user_id: Optional[int] = None,
    ) -> bool:
        """Remove a handle from a named Twitter list. Deletes the list if empty."""
        normalized_handle = handle.lstrip("@").lower()
        normalized_list = list_name.strip().lower()
        if not normalized_handle or not normalized_list:
            return False
        async with self._lock:
            scope_key = self._twitter_scope_key(guild_id, user_id)
            lists = self._twitter_lists.get(scope_key)
            if not lists:
                return False
            handles = lists.get(normalized_list)
            if not handles or normalized_handle not in handles:
                return False
            handles = [h for h in handles if h != normalized_handle]
            if handles:
                lists[normalized_list] = handles
            else:
                lists.pop(normalized_list, None)
            if not lists:
                self._twitter_lists.pop(scope_key, None)
            self._save_twitter_lists()
            return True

    async def get_twitter_list_handles(
        self,
        guild_id: Optional[int],
        list_name: str,
        *,
        user_id: Optional[int] = None,
    ) -> List[str]:
        """Return handles for a specific Twitter list."""
        normalized_list = list_name.strip().lower()
        if not normalized_list:
            return []
        async with self._lock:
            lists = self._twitter_lists.get(self._twitter_scope_key(guild_id, user_id), {})
            return list(lists.get(normalized_list, []))

    async def list_twitter_lists(
        self,
        guild_id: Optional[int],
        *,
        user_id: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """Return all Twitter lists for the guild."""
        async with self._lock:
            lists = self._twitter_lists.get(self._twitter_scope_key(guild_id, user_id), {})
            return {name: list(handles) for name, handles in lists.items()}

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
