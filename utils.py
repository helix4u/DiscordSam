import re
from datetime import timedelta
from typing import List, Optional, Tuple, Any
import psutil

# Assuming config is imported from config.py where it's defined
# from .config import config # If in a package
# For standalone scripts, you might need to adjust import paths or pass config
# For this refactor, we'll assume config can be imported if placed correctly.
# If main_bot.py initializes config, then other modules import it from there or config.py
from config import config
from common_models import MsgNode


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


async def call_llm_api(
    llm_client: Any,
    messages: List[MsgNode],
    model: str,
    *,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = config.MAX_COMPLETION_TOKENS,
) -> Any:
    """Call either the Responses API or Chat Completions depending on config."""

    msg_dicts = [m.to_dict() for m in messages]

    use_responses = config.USE_RESPONSES_API and hasattr(llm_client, "responses")

    if use_responses:
        instructions_parts = [
            d["content"]
            for d in msg_dicts
            if d.get("role") == "system" and isinstance(d.get("content"), str)
        ]
        instructions = "\n".join(instructions_parts) if instructions_parts else None
        input_messages = [d for d in msg_dicts if d.get("role") != "system"]
        return await llm_client.responses.create(
            model=model,
            instructions=instructions,
            input=input_messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
            stream=stream,
            service_tier=config.SERVICE_TIER,
        )
    else:
        return await llm_client.chat.completions.create(
            model=model,
            messages=msg_dicts,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            service_tier=config.SERVICE_TIER,
        )


def extract_text_from_response(response: Any) -> Optional[str]:
    """Return the assistant text from either API response."""

    if hasattr(response, "choices"):
        choice = response.choices[0] if response.choices else None
        if choice and getattr(choice, "message", None):
            return choice.message.content or ""

    if hasattr(response, "output_text"):
        return response.output_text or ""

    return None
