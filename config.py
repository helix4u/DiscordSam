import os
import logging
import platform
from dataclasses import dataclass
from typing import Dict
import discord
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LLMApiConfig:
    """Provider configuration for an LLM role."""

    role: str
    model: str
    api_base_url: str
    api_key: str | None
    temperature: float
    use_responses_api: bool
    supports_logit_bias: bool
    supports_json_mode: bool
    is_google_model: bool


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
        def _default_font_path() -> str:
            system = platform.system().lower()
            if system == "windows":
                root = os.getenv("SYSTEMROOT", "C:/Windows")
                return os.path.join(root, "Fonts", "arial.ttf")
            if system == "darwin":
                return "/Library/Fonts/Arial.ttf"
            return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
        # Token validation moved to ``require_bot_token`` so that modules which
        # don't need the bot token (e.g. ``open_chatgpt_login.py``) can import
        # the configuration without raising an exception.

        self.LOCAL_SERVER_URL = os.getenv("LOCAL_SERVER_URL", "http://localhost:1234/v1")
        self.LLM_MODEL = os.getenv("LLM", "local-model")
        self.VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "llava")
        self.FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", self.LLM_MODEL)
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")
        self.FAST_LLM_API_KEY = os.getenv("FAST_LLM_API_KEY", self.LLM_API_KEY)
        self.VISION_LLM_API_KEY = os.getenv("VISION_LLM_API_KEY", self.LLM_API_KEY)
        self.LLM_COMPLETIONS_URL = os.getenv("LLM_COMPLETIONS_URL", self.LOCAL_SERVER_URL)
        self.FAST_LLM_COMPLETIONS_URL = os.getenv("FAST_LLM_COMPLETIONS_URL", self.LLM_COMPLETIONS_URL)
        self.VISION_LLM_COMPLETIONS_URL = os.getenv("VISION_LLM_COMPLETIONS_URL", self.LLM_COMPLETIONS_URL)
        self.LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.7)
        self.FAST_LLM_TEMPERATURE = _get_float("FAST_LLM_TEMPERATURE", self.LLM_TEMPERATURE)
        self.VISION_LLM_TEMPERATURE = _get_float("VISION_LLM_TEMPERATURE", self.LLM_TEMPERATURE)
        self.LLM_SUPPORTS_JSON_MODE = _get_bool("LLM_SUPPORTS_JSON_MODE", False)
        self.FAST_LLM_SUPPORTS_JSON_MODE = _get_bool(
            "FAST_LLM_SUPPORTS_JSON_MODE", self.LLM_SUPPORTS_JSON_MODE
        )
        self.VISION_LLM_SUPPORTS_JSON_MODE = _get_bool(
            "VISION_LLM_SUPPORTS_JSON_MODE", self.LLM_SUPPORTS_JSON_MODE
        )
        self.IS_GOOGLE_MODEL = _get_bool("IS_GOOGLE_MODEL", False)
        self.FAST_IS_GOOGLE_MODEL = _get_bool("FAST_IS_GOOGLE_MODEL", self.IS_GOOGLE_MODEL)
        self.VISION_IS_GOOGLE_MODEL = _get_bool(
            "VISION_IS_GOOGLE_MODEL", self.IS_GOOGLE_MODEL
        )
        self.LLM_SUPPORTS_LOGIT_BIAS = _get_bool(
            "LLM_SUPPORTS_LOGIT_BIAS", not self.IS_GOOGLE_MODEL
        )
        self.FAST_LLM_SUPPORTS_LOGIT_BIAS = _get_bool(
            "FAST_LLM_SUPPORTS_LOGIT_BIAS", self.LLM_SUPPORTS_LOGIT_BIAS
        )
        self.VISION_LLM_SUPPORTS_LOGIT_BIAS = _get_bool(
            "VISION_LLM_SUPPORTS_LOGIT_BIAS", self.LLM_SUPPORTS_LOGIT_BIAS
        )
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
        self.OPENAI_RETRY_MAX_ATTEMPTS = max(
            1, _get_int("OPENAI_RETRY_MAX_ATTEMPTS", 6)
        )
        self.OPENAI_BACKOFF_BASE_SECONDS = max(
            0.1, _get_float("OPENAI_BACKOFF_BASE_SECONDS", 30.0)
        )
        self.OPENAI_BACKOFF_MAX_SECONDS = max(
            self.OPENAI_BACKOFF_BASE_SECONDS,
            _get_float("OPENAI_BACKOFF_MAX_SECONDS", 360.0),
        )
        self.OPENAI_BACKOFF_JITTER_SECONDS = max(
            0.0, _get_float("OPENAI_BACKOFF_JITTER_SECONDS", 3.0)
        )
        self.OPENAI_CLIENT_MAX_RETRIES = max(
            0, _get_int("OPENAI_CLIENT_MAX_RETRIES", 0)
        )

        # GPT-5 mode: adapt Chat Completions for GPT-5 models
        # - Force temperature to 1.0
        # - Remove logit_bias
        # - Map system role to developer
        self.GPT5_MODE = _get_bool("GPT5_MODE", False)

        self.LLM_PROVIDERS: Dict[str, LLMApiConfig] = {
            "main": self._build_llm_provider(
                role="main",
                model=self.LLM_MODEL,
                api_base_url=self.LLM_COMPLETIONS_URL,
                api_key=self.LLM_API_KEY,
                temperature=self.LLM_TEMPERATURE,
                use_responses_api=self.LLM_USE_RESPONSES_API,
                supports_logit_bias=self.LLM_SUPPORTS_LOGIT_BIAS,
                supports_json_mode=self.LLM_SUPPORTS_JSON_MODE,
                is_google_model=self.IS_GOOGLE_MODEL,
            ),
            "fast": self._build_llm_provider(
                role="fast",
                model=self.FAST_LLM_MODEL,
                api_base_url=self.FAST_LLM_COMPLETIONS_URL,
                api_key=self.FAST_LLM_API_KEY,
                temperature=self.FAST_LLM_TEMPERATURE,
                use_responses_api=self.FAST_LLM_USE_RESPONSES_API,
                supports_logit_bias=self.FAST_LLM_SUPPORTS_LOGIT_BIAS,
                supports_json_mode=self.FAST_LLM_SUPPORTS_JSON_MODE,
                is_google_model=self.FAST_IS_GOOGLE_MODEL,
            ),
            "vision": self._build_llm_provider(
                role="vision",
                model=self.VISION_LLM_MODEL,
                api_base_url=self.VISION_LLM_COMPLETIONS_URL,
                api_key=self.VISION_LLM_API_KEY,
                temperature=self.VISION_LLM_TEMPERATURE,
                use_responses_api=self.VISION_LLM_USE_RESPONSES_API,
                supports_logit_bias=self.VISION_LLM_SUPPORTS_LOGIT_BIAS,
                supports_json_mode=self.VISION_LLM_SUPPORTS_JSON_MODE,
                is_google_model=self.VISION_IS_GOOGLE_MODEL,
            ),
        }

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
        self.TTS_VOICE = os.getenv("TTS_VOICE", "af_sky+af_v0+af_nicole")
        self.TTS_ENABLED_DEFAULT = _get_bool("TTS_ENABLED_DEFAULT", True)
        self.TTS_INCLUDE_THOUGHTS = _get_bool("TTS_INCLUDE_THOUGHTS", False)
        self.TTS_DELIVERY_DEFAULT = _get_choice(
            "TTS_DELIVERY_DEFAULT",
            {"off", "audio", "video", "both"},
            "audio",
        ) or "audio"
        # If true, re-enable TTS globally after the /podcastthatshit command completes
        self.PODCAST_ENABLE_TTS_AFTER = _get_bool("PODCAST_ENABLE_TTS_AFTER", True)
        # Discord limits attachments from bots to 8MB on most servers.
        # Use 8MB as the default so TTS audio gets split automatically if needed.
        self.TTS_MAX_AUDIO_BYTES = _get_int("TTS_MAX_AUDIO_BYTES", 8 * 1024 * 1024)
        self.TTS_SPEED = _get_float("TTS_SPEED", 1.3)
        self.TTS_REQUEST_TIMEOUT_SECONDS = _get_int("TTS_REQUEST_TIMEOUT_SECONDS", 180)
        self.TTS_VIDEO_WIDTH = _get_int("TTS_VIDEO_WIDTH", 1280)
        self.TTS_VIDEO_HEIGHT = _get_int("TTS_VIDEO_HEIGHT", 720)
        self.TTS_VIDEO_FPS = _get_int("TTS_VIDEO_FPS", 30)
        self.TTS_VIDEO_BACKGROUND_COLOR = os.getenv(
            "TTS_VIDEO_BACKGROUND_COLOR",
            "#111827",
        )
        self.TTS_VIDEO_TEXT_COLOR = os.getenv(
            "TTS_VIDEO_TEXT_COLOR",
            "#F8FAFC",
        )
        self.TTS_VIDEO_TEXT_BOX_COLOR = os.getenv(
            "TTS_VIDEO_TEXT_BOX_COLOR",
            "#000000AA",
        )
        self.TTS_VIDEO_TEXT_BOX_PADDING = _get_int(
            "TTS_VIDEO_TEXT_BOX_PADDING",
            56,
        )
        self.TTS_VIDEO_LINE_SPACING = _get_int(
            "TTS_VIDEO_LINE_SPACING",
            16,
        )
        self.TTS_VIDEO_MARGIN = _get_int("TTS_VIDEO_MARGIN", 96)
        self.TTS_VIDEO_WRAP_CHARS = _get_int("TTS_VIDEO_WRAP_CHARS", 60)
        self.TTS_VIDEO_BLUR_SIGMA = _get_float("TTS_VIDEO_BLUR_SIGMA", 28.0)
        self.TTS_VIDEO_NOISE_OPACITY = _get_float("TTS_VIDEO_NOISE_OPACITY", 0.35)
        self.TTS_VIDEO_FONT_PATH = os.getenv(
            "TTS_VIDEO_FONT_PATH",
            _default_font_path(),
        )
        self.TTS_VIDEO_FONT_SIZE = _get_int("TTS_VIDEO_FONT_SIZE", 38)
        self.TTS_MAX_VIDEO_BYTES = _get_int("TTS_MAX_VIDEO_BYTES", 8 * 1024 * 1024)  # Match Discord 8MB limit

        self.SEARX_URL = os.getenv("SEARX_URL", "http://192.168.1.3:9092/search")
        # Changed default for SEARX_PREFERENCES to an empty string.
        # Users should set complex preferences via the environment variable.
        self.SEARX_PREFERENCES = os.getenv("SEARX_PREFERENCES", "")

        # Moltbook integration
        self.MOLTBOOK_BASE_URL = os.getenv(
            "MOLTBOOK_BASE_URL",
            "https://www.moltbook.com/api/v1",
        )
        # Sanitize key: strip, remove surrounding quotes, drop CR/LF and other control chars (Windows .env)
        _raw_key = os.getenv("MOLTBOOK_API_KEY", "")
        if _raw_key:
            _raw_key = _raw_key.strip().strip('"').strip("'")
            self.MOLTBOOK_API_KEY = "".join(c for c in _raw_key if ord(c) >= 32 and ord(c) != 127)
        else:
            self.MOLTBOOK_API_KEY = ""
        self.MOLTBOOK_AGENT_NAME = (os.getenv("MOLTBOOK_AGENT_NAME", "") or "").strip().strip('"').strip("'")

        # Shared rate limiter configuration for outbound HTTP requests.
        # Proactive rate limiting: max requests per minute (default conservative for OpenRouter)
        self.RATE_LIMIT_REQUESTS_PER_MINUTE = _get_float("RATE_LIMIT_REQUESTS_PER_MINUTE", 15.0)
        self.RATE_LIMIT_JITTER_SECONDS = _get_float("RATE_LIMIT_JITTER_SECONDS", 1.5)
        self.RATE_LIMIT_FAILURE_BACKOFF_SECONDS = _get_float(
            "RATE_LIMIT_FAILURE_BACKOFF_SECONDS", 3.0
        )
        self.RATE_LIMIT_FALLBACK_WINDOW_SECONDS = _get_float(
            "RATE_LIMIT_FALLBACK_WINDOW_SECONDS", 90.0
        )

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
        self.MAX_CHARS_PER_EDIT = _get_float("MAX_CHARS_PER_EDIT", float('inf'))  # Infinite by default to stay current with generation speed

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

        self.RSS_FETCH_HOURS = _get_int("RSS_FETCH_HOURS", 24)

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

        # Archive service configuration for paywalled sites
        self.USE_ARCHIVE_SERVICE = _get_bool("USE_ARCHIVE_SERVICE", False)  # Disabled by default
        self.ARCHIVE_SERVICE = os.getenv("ARCHIVE_SERVICE", "archive.is")  # Options: archive.is, archive.today, archive.ph, web.archive.org, none
        self.ARCHIVE_FALLBACK_TO_ORIGINAL = _get_bool("ARCHIVE_FALLBACK_TO_ORIGINAL", True)  # Fallback to original URL if archive fails

        # Configuration for Playwright cleanup task
        self.PLAYWRIGHT_CLEANUP_INTERVAL_MINUTES = _get_int("PLAYWRIGHT_CLEANUP_INTERVAL_MINUTES", 5) # How often the cleanup task runs
        self.PLAYWRIGHT_IDLE_CLEANUP_THRESHOLD_MINUTES = _get_int("PLAYWRIGHT_IDLE_CLEANUP_THRESHOLD_MINUTES", 10) # How long Playwright must be idle before cleanup

        self.SCRAPE_LOCK_TIMEOUT_SECONDS = _get_int("SCRAPE_LOCK_TIMEOUT_SECONDS", 60) # Timeout for acquiring scrape lock

    def _build_llm_provider(
        self,
        *,
        role: str,
        model: str,
        api_base_url: str,
        api_key: str | None,
        temperature: float,
        use_responses_api: bool,
        supports_logit_bias: bool,
        supports_json_mode: bool,
        is_google_model: bool,
    ) -> LLMApiConfig:
        """Construct a provider record for a specific LLM role."""

        base_url = api_base_url.strip() if api_base_url else self.LLM_COMPLETIONS_URL
        key = api_key.strip() if api_key else None
        return LLMApiConfig(
            role=role,
            model=model,
            api_base_url=base_url,
            api_key=key,
            temperature=temperature,
            use_responses_api=use_responses_api,
            supports_logit_bias=supports_logit_bias,
            supports_json_mode=supports_json_mode,
            is_google_model=is_google_model,
        )


# Global config instance
config = Config()


def require_bot_token() -> str:
    """Return the Discord bot token or raise if it's missing."""
    if not config.DISCORD_BOT_TOKEN:
        raise ValueError("DISCORD_BOT_TOKEN environment variable is missing")
    return config.DISCORD_BOT_TOKEN
