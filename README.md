# DiscordSam: The Sentient Hyper-Intelligent Discord Bot

DiscordSam is an advanced, context-aware Discord bot designed to provide intelligent, detailed, and rational responses. It leverages local Large Language Models (LLMs), Retrieval Augmented Generation (RAG) with ChromaDB for long-term memory, web scraping capabilities, Text-to-Speech (TTS), and a variety of interactive slash commands.

---

## Features

**Intelligent Conversations:**
Powered by local LLMs (configurable, e.g., LM Studio compatible).

**Web Capabilities:**

* Scrape website content and YouTube transcripts.
* Perform web searches using a local SearXNG instance and summarize results.
* Fetch and summarize recent tweets from X/Twitter users.

**Multimedia Interaction:**

* Analyze and describe attached images (with a creative "AP Photo" twist).
* Transcribe attached audio files using a local Whisper model.

**Text-to-Speech (TTS):**
Voice responses for bot messages, including separate TTS for "thoughts" vs. main response.

**Slash Commands:**
Suite of commands for various functionalities:

* `/remindme`: Set reminders.
* `/roast <url>`: Generate a comedy routine based on a webpage.
* `/search <query>`: Perform a web search and summarize.
* `/pol <statement>`: Generate sarcastic political commentary.
* `/gettweets <username>`: Fetch and summarize tweets.
* `/ap <image>`: Describe an image with an AP Photo twist.
* `/clearhistory`: Clear short-term channel history.
* `/ingest_chatgpt_export`: Import conversations from a ChatGPT export file into ChromaDB.

**Configurable:**
Most settings are managed via a `.env` file.

**Modular Codebase:**
Refactored into multiple Python files for better organization and maintainability.

---

## Prerequisites

* Python 3.10+
* pip (Python package installer)
* A local LLM server (OpenAI API compatible, e.g., LM Studio, Ollama)
* A local TTS server (OpenAI TTS API compatible, or edit `audio_utils.py` for your own)
* (Optional) A local SearXNG instance (for `/search`)
* (Optional) CUDA-enabled GPU (for faster Whisper transcription, fp16)

---

## Configuration

Create a `.env` file in the root directory.

Fill in your specific configurations. See below for all environment variables.

---

## Environment Variables (`.env` file)

```
# Discord Bot Token (Required)
DISCORD_BOT_TOKEN="YOUR_DISCORD_BOT_TOKEN_HERE"

# --- LLM Configuration ---
LOCAL_SERVER_URL="http://localhost:1234/v1"
LLM="your-default-llm-model-name"
VISION_LLM_MODEL="your-vision-llm-model-name"
FAST_LLM_MODEL="your-fast-llm-model-name"

# --- Discord Bot Behavior ---
ALLOWED_CHANNEL_IDS=""
ALLOWED_ROLE_IDS=""
MAX_IMAGES_PER_MESSAGE="1"
MAX_MESSAGE_HISTORY="10"
MAX_COMPLETION_TOKENS="2048"

# --- Text-to-Speech (TTS) ---
TTS_API_URL="http://localhost:8880/v1/audio/speech"
TTS_VOICE="your-chosen-tts-voice"
TTS_ENABLED_DEFAULT="true"

# --- Web Scraping & Search ---
SEARX_URL="http://localhost:9092/search"
SEARX_PREFERENCES="your_searx_preferences_string_if_any"

# --- RAG & ChromaDB ---
CHROMA_DB_PATH="./chroma_data"
CHROMA_COLLECTION_NAME="long_term_memory"
CHROMA_DISTILLED_COLLECTION_NAME="distilled_chat_summaries"
RAG_NUM_DISTILLED_SENTENCES_TO_FETCH="3"

# --- Miscellaneous ---
USER_PROVIDED_CONTEXT=""
MAX_IMAGE_BYTES_FOR_PROMPT="4194304"
MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT="8000"
STREAM_EDIT_THROTTLE_SECONDS="0.1"
```

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/DiscordSam.git
cd DiscordSam
```

Create a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

Install dependencies (see `requirements.txt`):

```
pip install -r requirements.txt
```

> **Note:** For torch (especially with GPU support), install it separately by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

Install Playwright browsers for web scraping:

```
playwright install
```

---

## Running the Bot

1. Ensure your local LLM server, TTS server (if enabled), and SearXNG instance (if using `/search`) are running and accessible at the URLs specified in your `.env`.
2. Make sure your `.env` is correctly configured with your `DISCORD_BOT_TOKEN` and other settings.
3. Run the main bot script from the root of the project:

```
python main_bot.py
```

---

## Support / Issues

Open an issue on GitHub or submit a pull request if you want to contribute or need help.

---
