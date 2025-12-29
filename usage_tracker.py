from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from config import config

logger = logging.getLogger(__name__)


# Best-effort pricing map (USD per 1M tokens).
# Note: Providers frequently change pricing. Unknown models will be reported with cost=None.
_PRICE_PER_1M_TOKENS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    # OpenAI (examples; adjust as needed for your environment)
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
}


def _host_from_base_url(llm_client: Any) -> str:
    try:
        base_url = getattr(llm_client, "base_url", None)
        if base_url:
            parsed = urlparse(str(base_url))
            return parsed.netloc or "default"
    except Exception:
        pass
    return "default"


def _extract_usage_tokens(response: Any, use_responses_api: bool) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (prompt_tokens, completion_tokens, total_tokens) if present."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None

    def _get_int(obj: Any, name: str) -> Optional[int]:
        value = getattr(obj, name, None)
        if value is None and isinstance(obj, dict):
            value = obj.get(name)
        try:
            return int(value) if value is not None else None
        except Exception:
            return None

    # Chat Completions usage naming
    prompt = _get_int(usage, "prompt_tokens")
    completion = _get_int(usage, "completion_tokens")
    total = _get_int(usage, "total_tokens")

    # Responses API usage naming (varies by SDK versions; try common keys)
    if use_responses_api and prompt is None and completion is None and total is None:
        prompt = _get_int(usage, "input_tokens")
        completion = _get_int(usage, "output_tokens")
        total = _get_int(usage, "total_tokens")
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion

    return prompt, completion, total


def estimate_cost_usd(model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
    if prompt_tokens is None or completion_tokens is None:
        return None
    pricing = _PRICE_PER_1M_TOKENS.get(model)
    if not pricing:
        return None
    price_in, price_out = pricing
    if price_in is None or price_out is None:
        return None
    return (prompt_tokens / 1_000_000.0) * price_in + (completion_tokens / 1_000_000.0) * price_out


@dataclass(frozen=True)
class UsageTotals:
    requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


_lock = asyncio.Lock()


async def record_usage(
    *,
    llm_client: Any,
    model: str,
    use_responses_api: bool,
    stream: bool,
    response: Any,
) -> None:
    """Append a usage record to a JSONL file (best-effort, never raises)."""

    try:
        prompt, completion, total = _extract_usage_tokens(response, use_responses_api)
        cost = estimate_cost_usd(model, prompt, completion)

        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "host": _host_from_base_url(llm_client),
            "model": model,
            "use_responses_api": bool(use_responses_api),
            "stream": bool(stream),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "cost_usd": cost,
        }

        path = getattr(config, "USAGE_LOG_PATH", "./usage_log.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        async with _lock:
            await asyncio.to_thread(_append_jsonl, path, payload)
    except Exception as exc:
        logger.debug("Usage logging failed: %s", exc, exc_info=True)


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


async def summarize_usage(
    *,
    since: datetime,
) -> UsageTotals:
    """Summarize usage since a UTC datetime."""

    path = getattr(config, "USAGE_LOG_PATH", "./usage_log.jsonl")
    if not os.path.exists(path):
        return UsageTotals(0, 0, 0, 0, 0.0)

    cutoff = since.astimezone(timezone.utc)

    def _read_lines() -> list[str]:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()

    lines = await asyncio.to_thread(_read_lines)
    requests = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cost_usd = 0.0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ts = rec.get("ts")
        try:
            ts_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if ts_dt < cutoff:
            continue
        requests += 1
        prompt_tokens += int(rec.get("prompt_tokens") or 0)
        completion_tokens += int(rec.get("completion_tokens") or 0)
        total_tokens += int(rec.get("total_tokens") or 0)
        cost_usd += float(rec.get("cost_usd") or 0.0)

    return UsageTotals(
        requests=requests,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


def timeframe_to_since(timeframe: str) -> datetime:
    now = datetime.now(timezone.utc)
    tf = (timeframe or "").lower().strip()
    if tf in {"hour", "1h"}:
        return now - timedelta(hours=1)
    if tf in {"day", "24h"}:
        return now - timedelta(days=1)
    if tf in {"week", "7d"}:
        return now - timedelta(days=7)
    if tf in {"month", "30d"}:
        return now - timedelta(days=30)
    if tf in {"year", "365d"}:
        return now - timedelta(days=365)
    return datetime.min.replace(tzinfo=timezone.utc)


def list_known_pricing() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    return dict(_PRICE_PER_1M_TOKENS)

