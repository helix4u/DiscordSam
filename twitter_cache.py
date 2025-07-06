import json
import os
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.path.dirname(__file__), "twitter_seen_ids.json")


def load_seen_tweet_ids() -> Dict[str, Set[str]]:
    """Loads seen tweet IDs from the cache file.

    Returns:
        Dict[str, Set[str]]: A dictionary where keys are usernames and
                             values are sets of seen tweet IDs for that user.
                             Returns an empty dict if cache doesn't exist or is invalid.
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Convert lists back to sets
                    return {user: set(ids) for user, ids in data.items()}
        except Exception as e:
            logger.error(f"Failed to load Twitter cache from {CACHE_FILE}: {e}")
    return {}


def save_seen_tweet_ids(data: Dict[str, Set[str]]) -> None:
    """Saves seen tweet IDs to the cache file.

    Args:
        data (Dict[str, Set[str]]): A dictionary where keys are usernames
                                     and values are sets of seen tweet IDs.
    """
    try:
        # Convert sets to lists for JSON serialization
        serializable_data = {user: list(ids) for user, ids in data.items()}
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2) # Add indent for readability
    except Exception as e:
        logger.error(f"Failed to save Twitter cache to {CACHE_FILE}: {e}")
