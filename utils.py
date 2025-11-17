import asyncio
import re
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Coroutine, List, Optional, Tuple, Callable, Awaitable, Dict, AsyncIterator
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

    # Explicitly replace directional quotes and backticks first for robustness.
    text = text.replace('’', "'")  # Right single quote
    text = text.replace('‘', "'")  # Left single quote
    text = text.replace('“', '"')  # Left double quote
    text = text.replace('”', '"')  # Right double quote
    text = text.replace('`', "'")  # Backtick

    # 1. Normalize unicode characters to their closest ASCII equivalents.
    # NFKC is aggressive and handles many "compatibility" characters like smart quotes.
    text = unicodedata.normalize('NFKC', text)

    # 2. Manually replace any remaining common special characters.
    text = text.replace("—", "--")  # Em-dash

    # 3. Remove URLs and <think> tags.
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 4. Whitelist allowed characters.
    # This strips out any remaining non-standard characters after normalization.
    # We allow basic letters, numbers, punctuation, and whitespace.
    allowed_chars = re.compile(r"[^a-zA-Z0-9\s.,!?'\"-]")
    text = allowed_chars.sub('', text)

    # 5. Clean up whitespace with care for newlines.
    # Collapse horizontal whitespace on each line.
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove leading whitespace from each line, but preserve blank lines.
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
    # Remove trailing whitespace from each line.
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # Reduce more than two consecutive newlines down to two.
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove any leading/trailing whitespace from the whole block.
    text = text.strip()

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


def is_admin_user(user_id: int) -> bool:
    """Return True if ``user_id`` is configured as an admin."""

    if not config.ADMIN_USER_IDS:
        return False
    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        return False
    return user_id_int in config.ADMIN_USER_IDS


def cleanup_playwright_processes() -> int:
    """Kill lingering Playwright/Chromium processes.

    Returns the number of processes terminated."""
    killed = 0
    # Only target processes associated with this bot's Playwright profile.
    # Using the more specific ".pw-" avoids killing other unrelated Playwright
    # instances the user might be running elsewhere.
    markers = [".pw-"]
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
        status = getattr(e, "status", None)
        code = getattr(e, "code", None)
        token_lost = (status == 401 and code == 50027) or (status == 404 and code == 10062)
        if token_lost:
            logger.warning(
                "Interaction token unavailable%s; falling back to alternate delivery",
                " " + error_hint if error_hint else "",
            )
            if kwargs.get("ephemeral"):
                logger.warning(
                    "Skipping public fallback for ephemeral message%s to avoid leaking response.",
                    " " + error_hint if error_hint else "",
                )
                content = kwargs.get("content")
                if content and interaction.user:
                    try:
                        return await interaction.user.send(content)
                    except Exception as dm_exc:
                        logger.warning("Failed to DM user after token loss: %s", dm_exc)
                raise
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


@asynccontextmanager
async def temporary_status_message(
    *,
    interaction: Optional[discord.Interaction] = None,
    channel: Optional[discord.abc.Messageable] = None,
    initial_text: str,
    send_kwargs: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Callable[[str], Awaitable[None]]]:
    """Post a temporary status message that can be updated and cleaned up automatically."""

    message: Optional[discord.Message] = None
    send_available = True
    send_kwargs = dict(send_kwargs or {})

    if "allowed_mentions" not in send_kwargs:
        send_kwargs["allowed_mentions"] = discord.AllowedMentions.none()

    async def update(text: str) -> None:
        nonlocal message, send_available
        if not send_available:
            return
        if message is None:
            try:
                if interaction is not None:
                    message = await safe_followup_send(
                        interaction,
                        content=text,
                        **send_kwargs,
                    )
                elif channel is not None:
                    message = await channel.send(content=text, **send_kwargs)
                else:
                    send_available = False
            except Exception as send_exc:  # noqa: BLE001
                logger.debug("Failed to send status indicator: %s", send_exc, exc_info=True)
                send_available = False
        else:
            target_channel: Optional[discord.abc.Messageable]
            if interaction is not None:
                target_channel = interaction.channel
            else:
                target_channel = channel

            if not target_channel:
                return

            try:
                message = await safe_message_edit(
                    message,
                    target_channel,
                    content=text,
                )
            except Exception as edit_exc:  # noqa: BLE001
                logger.debug("Failed to update status indicator: %s", edit_exc, exc_info=True)

    await update(initial_text)

    if not send_available:
        async def noop(_: str) -> None:
            return None

        yield noop
        return

    try:
        yield update
    finally:
        if message:
            try:
                await message.delete()
            except discord.HTTPException:
                pass
