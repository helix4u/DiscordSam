# DiscordSam: The Sentient Hyper-Intelligent Discord Bot

## 1. Project Overview

DiscordSam is an advanced, context-aware Discord bot designed to provide intelligent, detailed, and rational responses. Its primary purpose is to serve as a highly interactive and knowledgeable assistant within a Discord server, capable of engaging in complex conversations, accessing and processing external information, and performing various tasks through slash commands. The bot leverages local Large Language Models (LLMs), Retrieval Augmented Generation (RAG) for long-term memory, web scraping, multimedia understanding, and Text-to-Speech (TTS) to create a rich and dynamic user experience.

---

## 2. Features

*   **Intelligent Conversations:** Powered by local LLMs (e.g., LM Studio, Ollama compatible), allowing for nuanced and context-aware dialogue.
*   **Long-Term Memory (RAG):** Utilizes ChromaDB to store and retrieve conversation history, enabling the bot to recall past interactions and provide more informed responses. Conversations are "distilled" into keyword-rich sentences for efficient semantic search.
*   **Web Capabilities:**
    *   Scrape website content and YouTube transcripts.
    *   Perform web searches (optionally via a local SearXNG instance) and summarize results.
    *   Fetch and summarize recent tweets from X/Twitter users.
*   **Multimedia Interaction:**
    *   Analyze and describe attached images (with a creative "AP Photo" twist).
    *   Transcribe attached audio files using a local Whisper model.
*   **Text-to-Speech (TTS):**
    *   Voice responses for bot messages.
    *   Separate TTS for "thoughts" vs. main response if configured.
*   **Slash Commands:** A comprehensive suite of commands for various functionalities:
    *   `/remindme`: Set reminders for yourself.
    *   `/roast <url>`: Generate a comedy routine based on the content of a webpage.
    *   `/search <query>`: Perform a web search (requires SearXNG) and summarize the findings.
    *   `/pol <statement>`: Generate sarcastic political commentary on a given statement.
    *   `/gettweets <username>`: Fetch and summarize the most recent tweets from a specified X/Twitter user.
    *   `/ap <image>`: Analyze an attached image and provide a description in the style of an AP Photo caption.
    *   `/clearhistory`: Clear the bot's short-term conversational history for the current channel.
    *   `/ingest_chatgpt_export`: Import conversations from a ChatGPT export file into the bot's long-term memory (ChromaDB).
*   **High Configurability:** Most settings are managed via a `.env` file, allowing for easy customization of LLM endpoints, API keys, and bot behavior.
*   **Modular Codebase:** Refactored into multiple Python files for better organization, maintainability, and scalability.

---

## 3. Key Technologies Used

*   **Python 3.10+**
*   **discord.py:** The primary library for creating Discord bots in Python.
*   **OpenAI API (compatible):** Standard interface for interacting with local Large Language Models.
*   **ChromaDB:** An open-source vector database used for implementing Retrieval Augmented Generation (RAG) for long-term memory.
*   **Playwright:** A library for browser automation, used here for web scraping capabilities.
*   **Whisper (OpenAI):** A state-of-the-art speech-to-text model for audio transcription.
*   **Asyncio:** Python's framework for asynchronous programming, crucial for a responsive Discord bot.
*   **Supporting LLM Servers (Examples):** LM Studio, Ollama (these run the actual language models).
*   **SearXNG (Optional):** A local metasearch engine for the `/search` command.
*   **Local TTS Server (Optional but Recommended):** An OpenAI TTS API compatible server for voice output.

---

## 4. Prerequisites

*   **Python 3.10+**
*   **pip** (Python package installer)
*   **A local LLM server:**
    *   Must be OpenAI API compatible (e.g., LM Studio, Ollama with OpenAI compatible endpoint).
    *   You will need the server URL and potentially model names.
*   **Git:** For cloning the repository.

**Optional (but Recommended for Full Functionality):**

*   **A local TTS server:**
    *   OpenAI TTS API compatible (e.g., a local instance of an OpenAI TTS compatible server).
    *   Alternatively, you can edit `audio_utils.py` to integrate your own TTS solution.
*   **A local SearXNG instance:**
    *   Required for the `/search` command to function.
    *   The URL of this instance is needed in the `.env` file.
*   **CUDA-enabled GPU:**
    *   Strongly recommended for faster Whisper audio transcription (especially with fp16).
    *   Also beneficial for running local LLMs if they support GPU acceleration.

---

## 5. Configuration

1.  **Create a `.env` file** in the root directory of the project. You can copy `example.env` to `.env` to get started.
2.  **Fill in your specific configurations.** See the section below for details on all environment variables.

**Essential Environment Variables:**

*   `DISCORD_BOT_TOKEN`: Your Discord bot's token.
*   `LOCAL_SERVER_URL`: The base URL for your local LLM server (e.g., `http://localhost:1234/v1`).
*   `LLM_MODEL`: The default model name your LLM server uses for text generation.
*   `VISION_LLM_MODEL`: The model name for vision-capable tasks (image analysis).
*   `FAST_LLM_MODEL`: A model name for faster, less complex tasks like summarization or distillation.

**Full Environment Variable List (`.env` file):**

```env
# Discord Bot Token (Required)
DISCORD_BOT_TOKEN="YOUR_DISCORD_BOT_TOKEN_HERE"

# --- LLM Configuration ---
LOCAL_SERVER_URL="http://localhost:1234/v1" # Example for LM Studio
LLM_MODEL="your-default-llm-model-name"
VISION_LLM_MODEL="your-vision-llm-model-name" # e.g., a LLaVA model
FAST_LLM_MODEL="your-fast-llm-model-name" # For tasks like distillation

# --- Discord Bot Behavior ---
ALLOWED_CHANNEL_IDS="" # Optional: Comma-separated list of channel IDs where the bot is allowed
ALLOWED_ROLE_IDS=""    # Optional: Comma-separated list of role IDs allowed to use the bot
MAX_IMAGES_PER_MESSAGE="1"
MAX_MESSAGE_HISTORY="10" # Number of recent messages to keep in short-term memory per channel
MAX_COMPLETION_TOKENS="2048" # Max tokens the LLM should generate

# --- Text-to-Speech (TTS) ---
# If you don't have a local OpenAI TTS compatible server, set TTS_ENABLED_DEFAULT to "false"
TTS_API_URL="http://localhost:8880/v1/audio/speech" # Example TTS server URL
TTS_VOICE="your-chosen-tts-voice" # Voice model for TTS
TTS_ENABLED_DEFAULT="true" # Set to "false" to disable TTS by default

# --- Web Scraping & Search ---
# Required only if using /search command
SEARX_URL="http://localhost:9092/search" # URL of your SearXNG instance
SEARX_PREFERENCES="your_searx_preferences_string_if_any" # Optional SearXNG preferences

# --- RAG & ChromaDB ---
CHROMA_DB_PATH="./chroma_data" # Path to store ChromaDB data
CHROMA_COLLECTION_NAME="long_term_memory"
CHROMA_DISTILLED_COLLECTION_NAME="distilled_chat_summaries"
RAG_NUM_DISTILLED_SENTENCES_TO_FETCH="3" # How many relevant past conversation snippets to fetch

# --- Miscellaneous ---
USER_PROVIDED_CONTEXT="" # Optional: Global context/persona string to always include in prompts
MAX_IMAGE_BYTES_FOR_PROMPT="4194304" # Max size of image data for vision prompts
MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT="8000" # Max length of scraped text to include in prompts
STREAM_EDIT_THROTTLE_SECONDS="0.1" # How frequently to edit messages when streaming LLM responses
EDITS_PER_SECOND="5" # Target edits per second for streaming (inverse of throttle)
# Embed colors are hex integers (e.g., 0xEDA439 or #EDA439)
EMBED_COLOR_INCOMPLETE="0xEDA439" # Embed color for in-progress messages
EMBED_COLOR_COMPLETE="0x4CAF50" # Embed color for completed messages
EMBED_COLOR_ERROR="0xF44336" # Embed color for error messages
EMBED_MAX_LENGTH="4000" # Max characters per Discord embed description field
```

---

## 6. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/username/repository.git # Replace with your repository's URL or skip if already cloned
    cd DiscordSam
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The `requirements.txt` file lists all necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```
    > **Important Note for PyTorch (used by Whisper):** For optimal performance, especially with GPU support, it's often best to install PyTorch separately by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`. The `requirements.txt` might install a CPU-only version if a specific PyTorch version isn't already present.

4.  **Install Playwright browsers:**
    Playwright requires browser binaries for web scraping.
    ```bash
    playwright install
    ```

---

## 7. Running the Bot

1.  **Ensure your local servers are running:**
    *   Your LLM server (e.g., LM Studio) must be active and accessible at the `LOCAL_SERVER_URL`.
    *   If `TTS_ENABLED_DEFAULT` is `true`, your TTS server must be running at `TTS_API_URL`.
    *   If you plan to use the `/search` command, your SearXNG instance must be running at `SEARX_URL`.

2.  **Verify your `.env` file:** Double-check that it's correctly configured, especially the `DISCORD_BOT_TOKEN` and server URLs.

3.  **Run the main bot script:**
    Execute this command from the root directory of the project:
    ```bash
    python main_bot.py
    ```

---

## 8. Potential Future Directions

DiscordSam is a project with significant potential for growth. Here are some ideas for future enhancements (see `PROJECT_OVERVIEW.md` for more details):

*   **Enhanced RAG Strategies:** More sophisticated summarization, knowledge graph integration, hybrid search.
*   **Expanded LLM/Multimodal Support:** Support for more LLM backends, true multimodal inputs/outputs, advanced agentic capabilities.
*   **Slash Command Enhancements:** New creative tools, personalization features, integration with external services.
*   **Improved Error Handling & Resilience:** Clearer error reporting, graceful degradation of services.
*   **Sophisticated Context Management:** User-specific context, thread-specific context, dynamic context adjustment.
*   **UI/UX Enhancements:** Interactive embeds/buttons, potential web UI for management.
*   **Observability & Logging:** Structured logging, performance metrics, admin commands for bot health.
*   **Security & Permissions:** Fine-grained command permissions, data privacy controls for RAG.

---

## 9. Support / Issues / Contributing

Found a bug? Have a feature request? Want to contribute?

*   Please **open an issue** on the GitHub repository for bugs or suggestions.
*   If you'd like to contribute code, please **fork the repository** and submit a **pull request** with your changes.

We welcome contributions to improve and expand DiscordSam!
```
