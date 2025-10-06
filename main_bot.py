import logging

import discord
from discord.ext import commands

# Import configurations and state
from config import config, require_bot_token  # ``require_bot_token`` enforces that the bot token is set
from state import BotState # Assuming state.py is in the same directory
from llm_clients import initialize_llm_clients, get_llm_client

# Import utility and manager modules
from rag_chroma_manager import initialize_chromadb  # For initializing ChromaDB
# audio_utils handles Whisper model loading on demand when needed

# Import setup functions for commands and events
from discord_commands import setup_commands
from discord_events import setup_events_and_tasks

# --- Logging Setup ---
# BasicConfig should ideally be called only once.
# If other modules also call it, it might lead to unexpected behavior or multiple handlers.
# It's best to configure logging once at the entry point of the application.
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s.%(msecs)03d %(levelname)s:%(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__) # Get a logger for this main file

# Announce GPT-5 mode if active (after logging is configured)
try:
    if config.GPT5_MODE:
        logger.info(
            "GPT-5 mode enabled: temperature=1.0, system->developer mapping, logit_bias disabled"
        )
except Exception:
    pass

# --- Bot Initialization ---
intents = discord.Intents.default()
intents.message_content = True # Required for on_message to read content
intents.reactions = True     # Required for on_raw_reaction_add
intents.guilds = True        # Often needed for various guild-related events/info

# The bot instance
bot = commands.Bot(command_prefix=commands.when_mentioned_or("!"), intents=intents)

# --- Client Initializations ---
# Create and share LLM clients via the central registry so each role can
# target its own completions endpoint and API key.
initialize_llm_clients()
llm_client = get_llm_client("main")

# Bot State Manager
bot_state = BotState()

# --- Main Execution Block ---
if __name__ == "__main__":
    # Validate essential configurations before trying to run
    try:
        bot_token = require_bot_token()
    except ValueError:
        logger.critical(
            "DISCORD_BOT_TOKEN is not set in the environment or .env file. Bot cannot start."
        )
        exit(1)

    # Initialize ChromaDB
    # This function now resides in rag_chroma_manager and sets up global chroma clients there
    if not initialize_chromadb():
        logger.critical("ChromaDB initialization failed. RAG capabilities will be severely limited or non-functional. Check ChromaDB setup and path.")
        # Depending on desired behavior, you might choose to exit or run with degraded functionality.
        # For now, it will attempt to run but log the critical failure.

    # Setup commands and events by calling the setup functions from their respective modules
    # Pass necessary initialized objects like bot, llm_client, bot_state
    try:
        logger.info("Setting up Discord slash commands...")
        setup_commands(bot, llm_client, bot_state) # Pass the initialized llm_client and bot_state
        logger.info("Discord slash commands setup complete.")

        logger.info("Setting up Discord events and tasks...")
        setup_events_and_tasks(bot, llm_client, bot_state) # Pass the initialized llm_client and bot_state
        logger.info("Discord events and tasks setup complete.")
    except Exception as e_setup:
        logger.critical(f"An error occurred during command or event setup: {e_setup}", exc_info=True)
        exit(1) # Exit if setup fails critically

    # Run the bot
    try:
        logger.info("Starting Discord bot...")
        # log_handler=None prevents discord.py from setting up its own root logger if we've already configured one.
        bot.run(bot_token, log_handler=None)
    except discord.LoginFailure:
        logger.critical("Failed to log in to Discord. Check the DISCORD_BOT_TOKEN.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred while trying to run the bot: {e}", exc_info=True)
    finally:
        logger.info("Bot process is shutting down.")
        # Any cleanup tasks can go here if needed, though bot.run() is blocking.
