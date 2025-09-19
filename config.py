import os
import logging
import discord
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the bot."""
    def __init__(self):
        load_dotenv()

        def _get_int(env_var: str, default: int) -> int:
            value = os.getenv(env_var, str(default))
            try:
                return int(value)
            except (TypeError, ValueError):
                logger.warning("Invalid value for %s=%s; using %s", env_var, value, default)
                return default

        def _get_float(env_var: str, default: float) -> float:
            value = os.getenv(env_var, str(default))
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.warning("Invalid value for %s=%s; using %s", env_var, value, default)
                return default

        def _get_bool(env_var: str, default: bool) -> bool:
            value = os.getenv(env_var)
            if value is None:
                return default
            lowered = value.lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
            logger.warning("Invalid value for %s=%s; using %s", env_var, value, default)
            return default

        def _get_choice(
            env_var: str, choices: set[str], default: str | None = None
        ) -> str | None:
            """Return a string choice from a set or a default if unset/invalid."""

            value = os.getenv(env_var)
            if value is None:
                return default
            if value not in choices:
                logger.warning(
                    "Invalid value for %s=%s; using %s", env_var, value, default
                )
                return default
            return value

        def _parse_int_list(env_var: str) -> list[int]:
            parts = os.getenv(env_var, "").split(",")
            values: list[int] = []
            for part in parts:
                if not part:
                    continue
                try:
                    values.append(int(part))
                except ValueError:
                    logger.warning("Invalid integer '%s' in %s; skipping", part, env_var)
            return values
        self.DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
        # Token validation moved to ``require_bot_token`` so that modules which
        # don't need the bot token (e.g. ``open_chatgpt_login.py``) can import
        # the configuration without raising an exception.

        self.LOCAL_SERVER_URL = os.getenv("LOCAL_SERVER_URL", "http://localhost:1234/v1")
        self.LLM_MODEL = os.getenv("LLM", "local-model")
        self.VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "llava")
        self.FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", self.LLM_MODEL)
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")
        self.LLM_SUPPORTS_JSON_MODE = _get_bool("LLM_SUPPORTS_JSON_MODE", False) # New Flag
        self.IS_GOOGLE_MODEL = _get_bool("IS_GOOGLE_MODEL", False)
        self.USE_RESPONSES_API = _get_bool("USE_RESPONSES_API", False)
        self.LLM_USE_RESPONSES_API = _get_bool(
            "LLM_USE_RESPONSES_API", self.USE_RESPONSES_API
        )
        self.FAST_LLM_USE_RESPONSES_API = _get_bool(
            "FAST_LLM_USE_RESPONSES_API", self.USE_RESPONSES_API
        )
        self.VISION_LLM_USE_RESPONSES_API = _get_bool(
            "VISION_LLM_USE_RESPONSES_API", self.USE_RESPONSES_API
        )
        self.LLM_STREAMING = _get_bool("LLM_STREAMING", True)
        self.LLM_REQUEST_TIMEOUT_SECONDS = _get_float(
            "LLM_REQUEST_TIMEOUT_SECONDS", 900.0
        )

        # GPT-5 mode: adapt Chat Completions for GPT-5 models
        # - Force temperature to 1.0
        # - Remove logit_bias
        # - Map system role to developer
        self.GPT5_MODE = _get_bool("GPT5_MODE", False)

        # Whisper ASR model settings
        self.WHISPER_DEVICE = os.getenv("WHISPER_DEVICE") # e.g. "cuda", "cpu". None for auto-detect
        self.RESPONSES_REASONING_EFFORT = _get_choice(
            "RESPONSES_REASONING_EFFORT",
            {"minimal", "low", "medium", "high"},
        )
        self.RESPONSES_VERBOSITY = _get_choice(
            "RESPONSES_VERBOSITY",
            {"low", "medium", "high"},
        )
        self.RESPONSES_SERVICE_TIER = _get_choice(
            "RESPONSES_SERVICE_TIER",
            {"auto", "default", "flex", "priority"},
        )
        self.SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.md")

        self.ALLOWED_CHANNEL_IDS = _parse_int_list("ALLOWED_CHANNEL_IDS")
        self.ALLOWED_ROLE_IDS = _parse_int_list("ALLOWED_ROLE_IDS")

        # IDs of users allowed to run privileged admin-only commands.
        self.ADMIN_USER_IDS = _parse_int_list("ADMIN_USER_IDS")

        self.MAX_IMAGES_PER_MESSAGE = _get_int("MAX_IMAGES_PER_MESSAGE", 1)
        self.MAX_MESSAGE_HISTORY = _get_int("MAX_MESSAGE_HISTORY", 10)
        self.MAX_COMPLETION_TOKENS = _get_int("MAX_COMPLETION_TOKENS", 2048)

        self.TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:8880/v1/audio/speech")
        self.TTS_VOICE = os.getenv("TTS_VOICE", "af_sky+af+af_nicole")
        self.TTS_ENABLED_DEFAULT = _get_bool("TTS_ENABLED_DEFAULT", True)
        self.TTS_INCLUDE_THOUGHTS = _get_bool("TTS_INCLUDE_THOUGHTS", False)
        # If true, re-enable TTS globally after the /podcastthatshit command completes
        self.PODCAST_ENABLE_TTS_AFTER = _get_bool("PODCAST_ENABLE_TTS_AFTER", True)
        # Discord limits attachments from bots to 8MB on most servers.
        # Use 8MB as the default so TTS audio gets split automatically if needed.
        self.TTS_MAX_AUDIO_BYTES = _get_int("TTS_MAX_AUDIO_BYTES", 8 * 1024 * 1024)
        self.TTS_SPEED = _get_float("TTS_SPEED", 1.3)
        self.TTS_REQUEST_TIMEOUT_SECONDS = _get_int("TTS_REQUEST_TIMEOUT_SECONDS", 180)

        self.SEARX_URL = os.getenv("SEARX_URL", "http://192.168.1.3:9092/search")
        # Changed default for SEARX_PREFERENCES to an empty string.
        # Users should set complex preferences via the environment variable.
        self.SEARX_PREFERENCES = os.getenv("SEARX_PREFERENCES", "")

        def _parse_color(env_var: str, default: int) -> int:
            value = os.getenv(env_var)
            if value:
                value = value.strip()
                if value.startswith("#"):
                    value = value[1:]
                if value.lower().startswith("0x"):
                    value = value[2:]
                try:
                    return int(value, 16)
                except ValueError:
                    logger.warning("Invalid color value for %s=%s; using default", env_var, value)
            return default

        self.EMBED_COLOR = {
            "incomplete": _parse_color(
                "EMBED_COLOR_INCOMPLETE", discord.Color.orange().value
            ),
            "complete": _parse_color(
                "EMBED_COLOR_COMPLETE", discord.Color.green().value
            ),
            "error": _parse_color("EMBED_COLOR_ERROR", discord.Color.red().value),
        }
        self.EMBED_MAX_LENGTH = _get_int("EMBED_MAX_LENGTH", 4096)
        self.EDITS_PER_SECOND = 1.3
        self.STREAM_EDIT_THROTTLE_SECONDS = _get_float("STREAM_EDIT_THROTTLE_SECONDS", 0.1)

        self.CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")
        self.CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "long_term_memory")
        self.CHROMA_DISTILLED_COLLECTION_NAME = os.getenv("CHROMA_DISTILLED_COLLECTION_NAME", "distilled_chat_summaries")
        self.CHROMA_NEWS_SUMMARY_COLLECTION_NAME = os.getenv("CHROMA_NEWS_SUMMARY_COLLECTION_NAME", "news_summaries")
        self.CHROMA_TIMELINE_SUMMARY_COLLECTION_NAME = os.getenv("CHROMA_TIMELINE_SUMMARY_COLLECTION_NAME", "timeline_summaries")
        self.CHROMA_ENTITIES_COLLECTION_NAME = os.getenv("CHROMA_ENTITIES_COLLECTION_NAME", "entities_collection")
        self.CHROMA_RELATIONS_COLLECTION_NAME = os.getenv("CHROMA_RELATIONS_COLLECTION_NAME", "relations_collection")
        self.CHROMA_OBSERVATIONS_COLLECTION_NAME = os.getenv("CHROMA_OBSERVATIONS_COLLECTION_NAME", "observations_collection")
        self.CHROMA_RSS_SUMMARY_COLLECTION_NAME = os.getenv("CHROMA_RSS_SUMMARY_COLLECTION_NAME", "rss_summaries") # New collection for RSS summaries
        self.CHROMA_TWEETS_COLLECTION_NAME = os.getenv("CHROMA_TWEETS_COLLECTION_NAME", "tweets_collection") # New collection for tweets

        self.USER_PROVIDED_CONTEXT = os.getenv("USER_PROVIDED_CONTEXT", "")

        self.MAX_IMAGE_BYTES_FOR_PROMPT = _get_int("MAX_IMAGE_BYTES_FOR_PROMPT", 4 * 1024 * 1024)
        self.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT = _get_int("MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT", 8000)
        self.RAG_NUM_DISTILLED_SENTENCES_TO_FETCH = _get_int("RAG_NUM_DISTILLED_SENTENCES_TO_FETCH", 3)
        self.RAG_NUM_COLLECTION_DOCS_TO_FETCH = _get_int("RAG_NUM_COLLECTION_DOCS_TO_FETCH", 3)
        self.RAG_MAX_FULL_CONVO_CHARS = _get_int("RAG_MAX_FULL_CONVO_CHARS", 20000)
        self.RAG_MAX_DATE_RANGE_DOCS = _get_int("RAG_MAX_DATE_RANGE_DOCS", 15)
        self.ENABLE_MEMORY_MERGE = _get_bool("ENABLE_MEMORY_MERGE", False)

        self.NEWS_MAX_LINKS_TO_PROCESS = _get_int("NEWS_MAX_LINKS_TO_PROCESS", 15)

        self.TIMELINE_PRUNE_DAYS = _get_int("TIMELINE_PRUNE_DAYS", 365)

        self.HEADLESS_PLAYWRIGHT = _get_bool("HEADLESS_PLAYWRIGHT", True)
        self.PLAYWRIGHT_MAX_CONCURRENCY = _get_int("PLAYWRIGHT_MAX_CONCURRENCY", 1)
        self.SCRAPE_SCROLL_ATTEMPTS = _get_int("SCRAPE_SCROLL_ATTEMPTS", 5)
        self.GROUND_NEWS_SEE_MORE_CLICKS = _get_int("GROUND_NEWS_SEE_MORE_CLICKS", 10)
        self.GROUND_NEWS_CLICK_DELAY_SECONDS = _get_float(
            "GROUND_NEWS_CLICK_DELAY_SECONDS", 1.0
        )
        self.GROUND_NEWS_ARTICLE_DELAY_SECONDS = _get_float(
            "GROUND_NEWS_ARTICLE_DELAY_SECONDS", 5.0
        )

        # Configuration for Playwright cleanup task
        self.PLAYWRIGHT_CLEANUP_INTERVAL_MINUTES = _get_int("PLAYWRIGHT_CLEANUP_INTERVAL_MINUTES", 5) # How often the cleanup task runs
        self.PLAYWRIGHT_IDLE_CLEANUP_THRESHOLD_MINUTES = _get_int("PLAYWRIGHT_IDLE_CLEANUP_THRESHOLD_MINUTES", 10) # How long Playwright must be idle before cleanup

        self.SCRAPE_LOCK_TIMEOUT_SECONDS = _get_int("SCRAPE_LOCK_TIMEOUT_SECONDS", 60) # Timeout for acquiring scrape lock


# Global config instance
config = Config()


def require_bot_token() -> str:
    """Return the Discord bot token or raise if it's missing."""
    if not config.DISCORD_BOT_TOKEN:
        raise ValueError("DISCORD_BOT_TOKEN environment variable is missing")
    return config.DISCORD_BOT_TOKEN
