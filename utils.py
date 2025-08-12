import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Coroutine, List, Optional, Tuple
import logging
import psutil
import discord
from dateparser.search import search_dates

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

RELATIVE_DATE_PATTERN = re.compile(
    r'\b('
    r'yesterday|today|tomorrow|tonight|'
    r'this\s+(?:morning|afternoon|evening)|'
    r'last\s+\w+|next\s+\w+|'
    r'\d+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+(?:ago|from\\s+now)|'
    r'in\s+\d+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)'
    r')\b',
    re.IGNORECASE,
)

def append_absolute_dates(
    text: str, current_time: Optional[datetime] = None
) -> str:
    """Annotate relative date phrases in ``text`` with absolute timestamps.

    Parameters
    ----------
    text:
        The input text potentially containing relative date expressions.
    current_time:
        Reference time used to resolve relative phrases. Defaults to the
        current UTC time.

    Returns
    -------
    str
        Text where each detected relative date phrase is followed by its
        absolute representation, e.g. ``"tomorrow (2024-05-10 00:00 UTC)"``.
        Phrases already containing explicit dates are left unmodified.
    """

    if not text:
        return text

    current_time = current_time or datetime.now(timezone.utc)
    results = search_dates(text, settings={"RELATIVE_BASE": current_time})
    if not results:
        return text

    for phrase, dt in results:
        if "(" in phrase:
            continue
        if not RELATIVE_DATE_PATTERN.search(phrase):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=current_time.tzinfo)
        pattern = re.compile(
            rf"({re.escape(phrase)})(?!\s*\()",
            re.IGNORECASE,
        )
        text = pattern.sub(
            f"{phrase} ({dt.astimezone().strftime('%Y-%m-%d %H:%M %Z')})",
            text,
        )

    return text

def clean_text_for_tts(text: str) -> str:
    if not text:
        return ""

    # 1. Normalize special characters to their ASCII equivalents.
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("—", "--")
    text = text.replace("‑", "-")

    # 2. Remove URLs and <think> tags first, as they can contain complex characters.
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 3. Define a whitelist of characters to keep.
    # This includes letters (unicode), numbers, basic punctuation, and whitespace.
    # This is safer than a blacklist for preventing unknown "special characters".
    # \w includes unicode letters, numbers, and underscore.
    # We add common punctuation and the hyphen.
    allowed_chars = re.compile(r"[^\w\s.,!?'\"-]")

    # 4. Remove all characters that are not in the whitelist.
    text = allowed_chars.sub("", text)

    # 5. Clean up excessive whitespace, preserving line breaks.
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
    text = '\n'.join(lines)

    return text

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
