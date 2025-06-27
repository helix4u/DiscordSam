# DiscordSam: The Hyper-Intelligent Discord Bot

## 1. Project Overview

DiscordSam is an advanced, context-aware Discord bot designed to provide intelligent, detailed, and rational responses. Its primary purpose is to serve as a highly interactive and knowledgeable assistant within a Discord server, capable of engaging in complex conversations, accessing and processing external information, and performing various tasks through slash commands. The bot leverages local Large Language Models (LLMs), Retrieval Augmented Generation (RAG) for long-term memory, web scraping, multimedia understanding, and Text-to-Speech (TTS) to create a rich and dynamic user experience.

---

## 2. Features

*   **Intelligent Conversations:** Powered by local LLMs (e.g., LM Studio, Ollama compatible), allowing for nuanced and context-aware dialogue.
*   **Long-Term Memory (RAG):** Utilizes ChromaDB to store and retrieve conversation history, enabling the bot to recall past interactions and provide more informed responses. Conversations are "distilled" into keyword-rich sentences for efficient semantic search.
*   **Web Capabilities:**
    *   Scrape website content and YouTube transcripts. If Playwright yields little text, the bot falls back to a simple BeautifulSoup request with a Googlebot user agent.
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
*   **BeautifulSoup:** Lightweight HTML parsing for fallback scraping when Playwright results are sparse.
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

### Key Environment Variables

Some of the most important settings you can tweak via the `.env` file are listed below. Refer to `config.py` and `example.env` for defaults and additional options. Numeric values must be valid integers (or floats where noted) and booleans should be expressed as `true` or `false` (case-insensitive).

* `DISCORD_BOT_TOKEN` – token for your Discord bot.
* `LOCAL_SERVER_URL` – base URL of the OpenAI-compatible LLM server.
* `LLM` – default language model name.
* `FAST_LLM_MODEL` – model used for quick responses.
* `LLM_API_KEY` – API key used with your LLM server.
* `SYSTEM_PROMPT_FILE` – path to a text/markdown file containing the system prompt.
* `VISION_LLM_MODEL` – model used for image analysis.
* `MAX_IMAGES_PER_MESSAGE` – number of images the bot will process per message.
* `MAX_MESSAGE_HISTORY` – how many messages are kept in short-term history.
* `MAX_COMPLETION_TOKENS` – limit on tokens when generating a reply.
* `TTS_API_URL` and `TTS_VOICE` – endpoint and voice for text-to-speech.
* `TTS_ENABLED_DEFAULT` – whether TTS is enabled by default.
* `SEARX_URL` – URL of your SearXNG instance for `/search`.
* `SEARX_PREFERENCES` – optional JSON string with engine preferences.
* `CHROMA_DB_PATH` – location of the ChromaDB database directory.
* `CHROMA_COLLECTION_NAME` – name of the collection for chat history.
* `CHROMA_DISTILLED_COLLECTION_NAME` – name of the distilled summary collection.
* `CHROMA_NEWS_SUMMARY_COLLECTION_NAME` – collection used to store summarized news articles.
* `USER_PROVIDED_CONTEXT` – additional context prepended to every prompt.
* `RAG_NUM_DISTILLED_SENTENCES_TO_FETCH` – how many distilled sentences to include from history.
* `MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT` – limit for scraped text added to prompts.
* `NEWS_MAX_LINKS_TO_PROCESS` – number of news links to read with `/gettweets`.
* `SCRAPE_SCROLL_ATTEMPTS` – how many times to scroll when scraping a webpage to load dynamic content.

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
    The list now includes `psutil`, which lets the bot clean up stray Playwright processes automatically.
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
## 8. Troubleshooting

If Chromium or Playwright processes remain running after scraping, the bot will attempt to clean them up every 10 minutes using `psutil`.


## 9. Potential Future Directions

DiscordSam is a project with significant potential for growth. Here are some ideas for future enhancements:

*   **Enhanced RAG Strategies:** More sophisticated summarization, knowledge graph integration, hybrid search.
*   **Expanded LLM/Multimodal Support:** Support for more LLM backends, true multimodal inputs/outputs, advanced agentic capabilities.
*   **Slash Command Enhancements:** New creative tools, personalization features, integration with external services.
*   **Improved Error Handling & Resilience:** Clearer error reporting, graceful degradation of services.
*   **Sophisticated Context Management:** User-specific context, thread-specific context, dynamic context adjustment.
*   **UI/UX Enhancements:** Interactive embeds/buttons, potential web UI for management.
*   **Observability & Logging:** Structured logging, performance metrics, admin commands for bot health.
*   **Security & Permissions:** Fine-grained command permissions, data privacy controls for RAG.

---

## 10. Support / Issues / Contributing

Found a bug? Have a feature request? Want to contribute?

*   Please **open an issue** on the GitHub repository for bugs or suggestions.
*   If you'd like to contribute code, please **fork the repository** and submit a **pull request** with your changes.

We welcome contributions to improve and expand DiscordSam!
