DiscordSam: The Sentient Hyper-Intelligent Discord Bot
DiscordSam is an advanced, context-aware Discord bot designed to provide intelligent, detailed, and rational responses. It leverages local Large Language Models (LLMs), Retrieval Augmented Generation (RAG) with ChromaDB for long-term memory, web scraping capabilities, Text-to-Speech (TTS), and a variety of interactive slash commands.

Features
Intelligent Conversations: Powered by local LLMs (configurable, e.g., LM Studio compatible).

Web Capabilities:

Scrape website content and YouTube transcripts.

Perform web searches using a local SearXNG instance and summarize results.

Fetch and summarize recent tweets from X/Twitter users.

Multimedia Interaction:

Analyze and describe attached images (with a creative "AP Photo" twist).

Transcribe attached audio files using a local Whisper model.

Text-to-Speech (TTS): Voice responses for bot messages, including separate TTS for "thoughts" vs. main response.

Slash Commands: A suite of commands for various functionalities:

/remindme: Set reminders.

/roast <url>: Generate a comedy routine based on a webpage.

/search <query>: Perform a web search and summarize.

/pol <statement>: Generate sarcastic political commentary.

/gettweets <username>: Fetch and summarize tweets.

/ap <image>: Describe an image with an AP Photo twist.

/clearhistory: Clear short-term channel history.

/ingest_chatgpt_export: Import conversations from a ChatGPT export file into ChromaDB.

Configurable: Most settings are managed via a .env file.

Modular Codebase: Refactored into multiple Python files for better organization and maintainability.

Prerequisites
Python 3.10+

pip (Python package installer)

A local LLM server: Compatible with the OpenAI API format (e.g., LM Studio, Ollama with an OpenAI compatible endpoint).

A local TTS server: Compatible with the OpenAI TTS API format (or modify audio_utils.py for your service).

(Optional) A local SearXNG instance: For the /search command.

(Optional) CUDA-enabled GPU: For faster Whisper model transcription (fp16).

Configuration
Create a .env file in the root directory of the project.

Populate the .env file with your specific configurations. See the "Environment Variables" section below for details.

Environment Variables (.env file)
# Discord Bot Token (Required)
DISCORD_BOT_TOKEN="YOUR_DISCORD_BOT_TOKEN_HERE"

# --- LLM Configuration ---
# URL for your local LLM server (OpenAI API compatible)
LOCAL_SERVER_URL="http://localhost:1234/v1"
# Default LLM model name (as recognized by your server)
LLM="your-default-llm-model-name"
# Vision-capable LLM model name
VISION_LLM_MODEL="your-vision-llm-model-name" # e.g., llava
# Fast LLM model for distillation/synthesis tasks (can be same as LLM)
FAST_LLM_MODEL="your-fast-llm-model-name"

# --- Discord Bot Behavior ---
# Comma-separated list of channel IDs where the bot should respond without being mentioned (leave empty to allow all)
ALLOWED_CHANNEL_IDS=""
# Comma-separated list of role IDs that can interact with the bot without mentioning it in allowed channels (leave empty for no role restriction)
ALLOWED_ROLE_IDS=""

MAX_IMAGES_PER_MESSAGE="1"      # Max images to process from a single user message
MAX_MESSAGE_HISTORY="10"        # Max number of (user + assistant) messages for short-term context
MAX_COMPLETION_TOKENS="2048"    # Max tokens for LLM completion

# --- Text-to-Speech (TTS) ---
# URL for your TTS server (OpenAI API compatible)
TTS_API_URL="http://localhost:8880/v1/audio/speech" # Example for a local XTTSv2 server
TTS_VOICE="your-chosen-tts-voice" # Voice model name for TTS
TTS_ENABLED_DEFAULT="true"      # Enable TTS by default ("true" or "false")

# --- Web Scraping & Search ---
# URL for your local SearXNG instance (for /search command)
SEARX_URL="http://localhost:9092/search" # Example
# (Optional) SearXNG preferences string (can be exported from SearXNG UI)
SEARX_PREFERENCES="your_searx_preferences_string_if_any"

# --- RAG & ChromaDB ---
CHROMA_DB_PATH="./chroma_data"  # Path to store ChromaDB data
CHROMA_COLLECTION_NAME="long_term_memory"
CHROMA_DISTILLED_COLLECTION_NAME="distilled_chat_summaries"
RAG_NUM_DISTILLED_SENTENCES_TO_FETCH="3" # Number of distilled summaries to fetch for RAG

# --- Miscellaneous ---
# (Optional) Global context string to always provide to the LLM
USER_PROVIDED_CONTEXT=""
# Max image size (bytes) to accept for vision prompts
MAX_IMAGE_BYTES_FOR_PROMPT="4194304" # 4MB
# Max length of scraped text to include in prompts
MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT="8000"
# Throttle for editing streamed messages (seconds)
STREAM_EDIT_THROTTLE_SECONDS="0.1"

Installation
Clone the repository:

git clone https://github.com/your-username/DiscordSam.git
cd DiscordSam

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
Refer to the requirements.txt file.

pip install -r requirements.txt

Note: For torch, especially with GPU support (CUDA for Whisper), it's often best to install it separately by following the official PyTorch instructions: https://pytorch.org/get-started/locally/

Install Playwright browsers (if not already installed):
This is required for web scraping features like /roast and /gettweets.

playwright install

Running the Bot
Ensure your local LLM server, TTS server (if TTS_ENABLED_DEFAULT="true"), and SearXNG instance (if using /search) are running and accessible at the URLs specified in your .env file.

Make sure your .env file is correctly configured with your DISCORD_BOT_TOKEN and other settings.

Run the main bot script from the root directory of the project:

python main_bot.py
