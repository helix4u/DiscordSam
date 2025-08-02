import asyncio
import re
from datetime import timedelta
from typing import List, Optional, Tuple, Any, Coroutine
import psutil
import logging
import discord

logger = logging.getLogger(__name__)

# Assuming config is imported from config.py where it's defined
# from .config import config # If in a package
# For standalone scripts, you might need to adjust import paths or pass config
# For this refactor, we'll assume config can be imported if placed correctly.
# If main_bot.py initializes config, then other modules import it from there or config.py
from config import config


def chunk_text(text: str, max_length: int = config.EMBED_MAX_LENGTH) -> List[str]:
    if not text: return [""]
    chunks = []
    current_chunk = ""
    for line in text.splitlines(keepends=True):
        if len(current_chunk) + len(line) > max_length:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = line
            while len(current_chunk) > max_length: # Handle very long lines
                chunks.append(current_chunk[:max_length])
                current_chunk = current_chunk[max_length:]
        else:
            current_chunk += line
    if current_chunk: chunks.append(current_chunk)
    return chunks if chunks else [""]

def detect_urls(message_text: str) -> List[str]:
    if not message_text: return []
    # Basic URL detection, can be improved for more complex cases
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(message_text)

def clean_text_for_tts(text: str) -> str:
    if not text: return ""
    # Remove Markdown-like characters
    text = re.sub(r'[\*#_~\<\>\[\]\(\)]+', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove <think> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def parse_time_string_to_delta(time_str: str) -> Tuple[Optional[timedelta], Optional[str]]:
    patterns = {
        'd': r'(\d+)\s*d(?:ay(?:s)?)?',
        'h': r'(\d+)\s*h(?:our(?:s)?|r(?:s)?)?',
        'm': r'(\d+)\s*m(?:inute(?:s)?|in(?:s)?)?',
        's': r'(\d+)\s*s(?:econd(?:s)?|ec(?:s)?)?'
    }
    delta_args = {'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
    original_parts = []
    time_str_processed = time_str.lower() # Process a copy

    for key, pattern_regex in patterns.items():
        for match in re.finditer(pattern_regex, time_str_processed):
            value = int(match.group(1))
            unit_full = {'d': 'days', 'h': 'hours', 'm': 'minutes', 's': 'seconds'}[key]
            delta_args[unit_full] += value
            original_parts.append(f"{value} {unit_full.rstrip('s') if value == 1 else unit_full}")
        # Remove matched parts from the string to avoid re-matching
        time_str_processed = re.sub(pattern_regex, "", time_str_processed)
    
    if not any(val > 0 for val in delta_args.values()):
        return None, None # No valid time units found

    time_delta = timedelta(**delta_args)
    descriptive_str = ", ".join(original_parts) if original_parts else "immediately"
    
    # Fallback descriptive string if delta_args had values but original_parts somehow didn't form
    if not descriptive_str and time_delta.total_seconds() > 0 :
        descriptive_str = "a duration" # Should ideally not be reached if parsing is correct

    return time_delta, descriptive_str


def cleanup_playwright_processes() -> int:
    """Kill lingering Playwright/Chromium processes.

    Returns the number of processes terminated."""
    killed = 0
    markers = [".pw-", "playwright"]
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if any(m in cmdline for m in markers):
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed


async def safe_followup_send(
    interaction: discord.Interaction, *, error_hint: str = "", **kwargs
) -> discord.Message:
    """Send a followup message with fallback to channel.send if the token expired."""
    try:
        return await interaction.followup.send(**kwargs)
    except discord.HTTPException as e:
        if e.status == 401 and getattr(e, "code", None) == 50027:
            logger.warning(
                "Interaction token expired%s; falling back to channel.send", " " + error_hint if error_hint else ""
            )
            if interaction.channel:
                kwargs_fallback = dict(kwargs)
                # Remove parameters not supported by channel.send
                kwargs_fallback.pop("ephemeral", None)
                kwargs_fallback.pop("wait", None)
                return await interaction.channel.send(**kwargs_fallback)
        raise


async def safe_message_edit(
    message: discord.Message,
    channel: discord.abc.Messageable,
    *,
    cleanup_old: bool = True,
    **kwargs: Any,
) -> discord.Message:
    """Edit a message safely.

    Attempts to edit ``message`` with ``kwargs``. If the underlying webhook
    has expired (HTTP 401 with code 50027), a new message is sent to
    ``channel`` and, optionally, the old message is removed.

    Parameters
    ----------
    message : :class:`discord.Message`
        The message to edit.
    channel : :class:`discord.abc.Messageable`
        Channel used to send a replacement message if editing fails.
    cleanup_old : bool, optional
        Whether to delete ``message`` when a new one is sent due to webhook
        expiration, by default ``True``.

    Returns
    -------
    :class:`discord.Message`
        The edited message, or the newly-sent replacement.
    """
    try:
        await message.edit(**kwargs)
        return message
    except discord.HTTPException as e:
        if e.status == 401 and getattr(e, "code", None) == 50027:
            logger.warning(
                "Webhook token expired during edit; sending new message"
            )
            new_msg = await channel.send(**kwargs)
            if cleanup_old:
                try:
                    await message.delete()
                except discord.HTTPException:
                    pass
            return new_msg
        raise


def start_post_processing_task(
    coro: Coroutine[Any, Any, Any],
    *,
    progress_message: Optional[discord.Message] = None,
) -> asyncio.Task:
    """Run ``coro`` in the background and handle progress cleanup.

    Parameters
    ----------
    coro:
        The coroutine performing the post-processing work.
    progress_message:
        Optional message indicating progress. It will be deleted when
        ``coro`` finishes.
    """

    async def _runner() -> None:
        try:
            await coro
        finally:
            if progress_message:
                try:
                    await progress_message.delete()
                except discord.HTTPException:
                    pass

    return asyncio.create_task(_runner())
