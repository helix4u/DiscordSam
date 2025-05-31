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
        self.FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", self.LLM_MODEL) # For distillation/synthesis tasks

        self.ALLOWED_CHANNEL_IDS = [int(i) for i in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",") if i]
        self.ALLOWED_ROLE_IDS = [int(i) for i in os.getenv("ALLOWED_ROLE_IDS", "").split(",") if i]
        
        self.MAX_IMAGES_PER_MESSAGE = int(os.getenv("MAX_IMAGES_PER_MESSAGE", 1))
        self.MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", 10))
        self.MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", 2048))
        
        self.TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:8880/v1/audio/speech")
        self.TTS_VOICE = os.getenv("TTS_VOICE", "af_sky+af+af_nicole")
        self.TTS_ENABLED_DEFAULT = os.getenv("TTS_ENABLED_DEFAULT", "true").lower() == "true"

        self.SEARX_URL = os.getenv("SEARX_URL", "http://192.168.1.3:9092/search")
        # Ensure SEARX_PREFERENCES is correctly formatted for string replacement if it contains %s
        # If it's a fixed string, no changes needed. If it expects a query to be inserted via %s,
        # then the usage in web_utils.py will handle that.
        self.SEARX_PREFERENCES = os.getenv("SEARX_PREFERENCES", "eJx1V8uy2zYM_Zp6o4mnaRadLrzqTLftTLvXQCQsISIJhg_bul9f0JIsyrpZRNcEQRA4AA4YBQl7DoTx0qPDAOaX3_50eI_yh5J8oiJ0CssvVgSmsagJTgZcn6HHC-TEJ8MKDF7QncpSsfUGE1565t7giawotj7wY7r8BSbiyWIaWF_--fvf_04RrhgRghouv57SgBYvkYqBU8CYTYotu1Y8ahN0y3HN1MommxuGC4Mszxz603ysjWkyi18KXcLQgqHeWfm9nAd9A4lJt8u9s_RHxjC15NpESQzMQnJXcpTEqgpszOoAReiMGEDXkxPs_uihb9saIQEMnIPYlBvohm17JYMFVvBjYykEDrVMPG_k28TEoVaW040hlx_NnUZq22dSIDzoJve9ctNR6rIaMS0KXdLU95sZpdSXJCdupJGfgsAxBrzWRkR21zcSqNutFib09VLlYAhriUb8EIxbmyOpstZ9o_GJGrGLO1UWF0Mz5G5xE-VG0m3LkvdQFCSG8q_n5lk0cnr--4LIghehfMtpy9_JF7A3C0nSMe38wzRZdgIhblpfH5Xhqw5cnFixugbEJvI13SFgoymgkpxMi8vXQG4kUJUBuSsxL_u9FA-s8SnW2GHo171nPzRRDWwg1NAvO97AVKogbt7UO5YlO7HKYu9Z6xr8AboA5bPcODxPLwtyUJ0lJz-Jc_xctlocSY0QK4cMyRVhagrCkaqNjy_LVqXLXUx4DnGt2w8zBVKVeWECUB7couC8XX9NAJshCwI6rlGxRweSFw0JIgpDVEiuez53hhQsNbjfl12uHPcgIfYU1-bxusP6iKcgLNlBFZhYP2u8rQdyJz2_OzL5tVd_3MGluiCfgqZwbNsuTBsFhJGFza6G7ytWccxddikv65g9hhxfIMzcVSqsKZ9XeYKFkgd8qQl2idRnVSfXhuQLOVfuJR4nThwHHkta1oiFbUlLHwQp9gq7CYZS-hVfFME-ujt1U60io-IBTofSt5u00FvHPMZ34ZutkX5kTviuFTkHdZAKpJHSdBDz9Bb112_ffn9sYems0W18EfHDga31hXoQx1ri4FZyswlC7qYe7Vr7HjGkXCrr1W7PQSFXjIV079hVWyHbztS6d37QyE6qromTYzfZis_id3_298rhEqQVCq79kd687WL20ntmLqOfiI8sLKW1tfOM-ZDTxm8vQydpw92IXOZgQ7sESwFSzaWdFIYC6zeGKKEsJF_FQq5_Xx-dfUp35fOUzJhW2BZOKJWC1VkZdFLqatqPuhtJM4nPlWLxrxDRTvE1xz6XNncseQ9bwq5CVmPYDxWqeVjmyjY3-4XWN9vLmDhAsMh3ICyyAwxkdbd59J2EArblSvt8FbpxvTy_9mXh-tsOAksPZTjrLYKCckwyXJOM73WG-0G4xq2rmRoPQczig78yADSlw7MroidYy_qlLLUqlbX3KN0plffhu4XsoszbOFQuTJzfmvcleT18gIw8NcrEqdRuZJHrPqaxaDzH5Ktb2cy87YfqKaNJpQ92uzRbaXsrj5EmBXDRyFjStZmgHY2VIKVwppUWtlerN1l6Ml4KSI_zsjoPHJOwHMrTWqKfB8ROgaGdX973IC_iw7YMh1YNqMbDjmDQSp5HnOKLFn7iQ0F9XhysLCkVdNX8n5ZJ3u9GHmefaJqrPOSvfPQxQHmCtvKKlThsGbknGXhSmJf_AWSO7BY=")

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

# Global config instance
config = Config()
