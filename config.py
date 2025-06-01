import os
import discord
from dotenv import load_dotenv

class Config:
    """Configuration class for the bot."""
    def __init__(self):
        load_dotenv()
        self.DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
        if not self.DISCORD_BOT_TOKEN:
            raise ValueError("DISCORD_BOT_TOKEN environment variable is missing")

        self.LOCAL_SERVER_URL = os.getenv("LOCAL_SERVER_URL", "http://localhost:1234/v1")
        self.LLM_MODEL = os.getenv("LLM", "local-model")
        self.VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "llava")
        self.FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", self.LLM_MODEL) 

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

        self.EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green(), "error": discord.Color.red()}
        self.EMBED_MAX_LENGTH = 4096
        self.EDITS_PER_SECOND = 1.3
        self.STREAM_EDIT_THROTTLE_SECONDS = float(os.getenv("STREAM_EDIT_THROTTLE_SECONDS", 0.1))

        self.CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")
        self.CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "long_term_memory")
        self.CHROMA_DISTILLED_COLLECTION_NAME = os.getenv("CHROMA_DISTILLED_COLLECTION_NAME", "distilled_chat_summaries")
        
        self.USER_PROVIDED_CONTEXT = os.getenv("USER_PROVIDED_CONTEXT", "")

        self.MAX_IMAGE_BYTES_FOR_PROMPT = int(os.getenv("MAX_IMAGE_BYTES_FOR_PROMPT", 4 * 1024 * 1024))
        self.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT = int(os.getenv("MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT", 8000))
        self.RAG_NUM_DISTILLED_SENTENCES_TO_FETCH = int(os.getenv("RAG_NUM_DISTILLED_SENTENCES_TO_FETCH", 3))
        
        self.NEWS_MAX_LINKS_TO_PROCESS = int(os.getenv("NEWS_MAX_LINKS_TO_PROCESS", 5))


# Global config instance
config = Config()
