import os
import discord
from dotenv import load_dotenv

class Config:
    """Configuration class for the bot."""
    def __init__(self):
        load_dotenv()
        self.DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
        # Token validation moved to ``require_bot_token`` so that modules which
        # don't need the bot token (e.g. ``open_chatgpt_login.py``) can import
        # the configuration without raising an exception.

        self.LOCAL_SERVER_URL = os.getenv("LOCAL_SERVER_URL", "http://localhost:1234/v1")
        self.LLM_MODEL = os.getenv("LLM", "local-model")
        self.VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "llava")
        self.FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", self.LLM_MODEL)
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")
        self.SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.md")

        self.ALLOWED_CHANNEL_IDS = [int(i) for i in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",") if i]
        self.ALLOWED_ROLE_IDS = [int(i) for i in os.getenv("ALLOWED_ROLE_IDS", "").split(",") if i]
        
        self.MAX_IMAGES_PER_MESSAGE = int(os.getenv("MAX_IMAGES_PER_MESSAGE", 1))
        self.MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", 10))
        self.MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", 2048))
        
        self.TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:8880/v1/audio/speech")
        self.TTS_VOICE = os.getenv("TTS_VOICE", "af_sky+af+af_nicole")
        self.TTS_ENABLED_DEFAULT = os.getenv("TTS_ENABLED_DEFAULT", "true").lower() == "true"

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
                    pass
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
        self.EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", 4096))
        self.EDITS_PER_SECOND = 1.3
        self.STREAM_EDIT_THROTTLE_SECONDS = float(os.getenv("STREAM_EDIT_THROTTLE_SECONDS", 0.1))

        self.CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")
        self.CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "long_term_memory")
        self.CHROMA_DISTILLED_COLLECTION_NAME = os.getenv("CHROMA_DISTILLED_COLLECTION_NAME", "distilled_chat_summaries")
        self.CHROMA_NEWS_SUMMARY_COLLECTION_NAME = os.getenv("CHROMA_NEWS_SUMMARY_COLLECTION_NAME", "news_summaries")
        
        self.USER_PROVIDED_CONTEXT = os.getenv("USER_PROVIDED_CONTEXT", "")

        self.MAX_IMAGE_BYTES_FOR_PROMPT = int(os.getenv("MAX_IMAGE_BYTES_FOR_PROMPT", 4 * 1024 * 1024))
        self.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT = int(os.getenv("MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT", 8000))
        self.RAG_NUM_DISTILLED_SENTENCES_TO_FETCH = int(os.getenv("RAG_NUM_DISTILLED_SENTENCES_TO_FETCH", 3))
        
        self.NEWS_MAX_LINKS_TO_PROCESS = int(os.getenv("NEWS_MAX_LINKS_TO_PROCESS", 5))

        self.HEADLESS_PLAYWRIGHT = os.getenv("HEADLESS_PLAYWRIGHT", "true").lower() == "true"
        self.PLAYWRIGHT_MAX_CONCURRENCY = int(os.getenv("PLAYWRIGHT_MAX_CONCURRENCY", 2))


# Global config instance
config = Config()


def require_bot_token() -> str:
    """Return the Discord bot token or raise if it's missing."""
    if not config.DISCORD_BOT_TOKEN:
        raise ValueError("DISCORD_BOT_TOKEN environment variable is missing")
    return config.DISCORD_BOT_TOKEN
