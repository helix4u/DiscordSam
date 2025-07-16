import json
import os
import logging
from typing import Set

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.path.dirname(__file__), "ground_news_seen.json")


def load_seen_links() -> Set[str]:
    """Load the set of previously processed Ground News article URLs."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return set(data)
        except Exception as e:  # pragma: no cover - simple cache
            logger.error("Failed to load Ground News cache from %s: %s", CACHE_FILE, e)
    return set()


def save_seen_links(urls: Set[str]) -> None:
    """Persist the set of processed Ground News article URLs."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(list(urls), f)
    except Exception as e:  # pragma: no cover - simple cache
        logger.error("Failed to save Ground News cache to %s: %s", CACHE_FILE, e)
