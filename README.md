# DiscordSam: The Hyper-Intelligent Discord Bot

## 1. Project Overview

DiscordSam is an advanced, context-aware Discord bot designed to provide intelligent, detailed, and rational responses. Its primary purpose is to serve as a highly interactive and knowledgeable assistant within a Discord server, capable of engaging in complex conversations, accessing and processing external information, and performing various tasks through slash commands. The bot leverages local Large Language Models (LLMs), Retrieval Augmented Generation (RAG) for long-term memory (including structured data extraction), web scraping, multimedia understanding, and Text-to-Speech (TTS) to create a rich and dynamic user experience.

---

## 2. Features

*   **Intelligent Conversations:** Powered by local LLMs (e.g., LM Studio, Ollama compatible), allowing for nuanced and context-aware dialogue.
*   **Long-Term Memory (RAG):** Utilizes ChromaDB to store and retrieve conversation history, enabling the bot to recall past interactions and provide more informed responses.
    *   Conversations are "distilled" into keyword-rich sentences for efficient semantic search.
    *   Extracts structured data (entities, relations, observations) from conversations to build a knowledge base.
    *   Can ingest ChatGPT export files to import prior conversation history.
    *   Automatically annotates relative date expressions with exact timestamps to improve date-based retrieval.
*   **Web Capabilities:**
    *   Scrape website content (with Playwright, falling back to BeautifulSoup) and YouTube transcripts.
    *   Generate textual descriptions of webpage screenshots using a vision LLM.
    *   Perform web searches (optionally via a local SearXNG instance) and summarize results.
    *   Fetch and summarize recent tweets from X/Twitter users.
    *   Process RSS feeds, summarizing new articles.
*   **Multimedia Interaction:**
    *   Analyze and describe attached images (with a creative "AP Photo" twist using a vision LLM).
    *   Transcribe attached audio files using a local Whisper model.
*   **Text-to-Speech (TTS):**
    *   Voice responses for bot messages via an OpenAI TTS API compatible server.
    *   Separate TTS for "thoughts" (content within `<think>...</think>` tags) vs. main response if configured.
    *   TTS operations are queued so only one audio clip plays at a time.
*   **Slash Commands:** A comprehensive suite of commands for various functionalities (detailed in section 8).
*   **High Configurability:** Most settings are managed via a `.env` file, allowing for easy customization of LLM endpoints, API keys, and bot behavior.
*   **Modular Codebase:** Refactored into multiple Python files for better organization, maintainability, and scalability.
*   **Automated Maintenance:** Includes background tasks for checking reminders, cleaning up idle Playwright processes, and pruning/summarizing old chat history from ChromaDB.

---

## 3. Key Technologies Used

*   **Python 3.10+**
*   **discord.py:** The primary library for creating Discord bots in Python.
*   **OpenAI API (compatible):** Standard interface for interacting with local Large Language Models (text and vision).
*   **ChromaDB:** An open-source vector database used for implementing Retrieval Augmented Generation (RAG) and storing structured knowledge.
*   **Playwright:** A library for browser automation, used for web scraping and tweet fetching.
*   **BeautifulSoup:** Lightweight HTML parsing for fallback web scraping.
*   **Whisper (OpenAI):** A state-of-the-art speech-to-text model for audio transcription.
*   **Asyncio:** Python's framework for asynchronous programming, crucial for a responsive Discord bot.
*   **psutil:** Used for managing and cleaning up system processes (e.g., stray Playwright instances).
*   **Supporting LLM Servers (Examples):** LM Studio, Ollama (these run the actual language models).
*   **SearXNG (Optional):** A local metasearch engine for the `/search` and `/news` commands.
*   **Local TTS Server (Optional but Recommended):** An OpenAI TTS API compatible server for voice output.

---

## 4. Project Architecture

DiscordSam is a modular Python application designed for extensibility and maintainability. The core components interact as follows:

1.  **Main Bot (`main_bot.py`)**:
    *   The entry point of the application.
    *   Initializes the Discord client (`discord.py.Bot`), global configurations (`config.py`), and shared state (`state.py.BotState`).
    *   Sets up logging for the entire application.
    *   Initializes the primary LLM client (OpenAI compatible, e.g., for LM Studio or Ollama).
    *   Initializes the RAG (Retrieval Augmented Generation) system by calling `rag_chroma_manager.initialize_chromadb()`.
    *   Loads and registers all slash commands from `discord_commands.py` via `setup_commands()`.
    *   Loads all event handlers and background tasks from `discord_events.py` via `setup_events_and_tasks()`.
    *   Connects to Discord and starts the event loop.

2.  **Configuration (`config.py`)**:
    *   Provides a centralized `Config` class that loads all settings from environment variables (typically from an `.env` file).
    *   This `config` object is imported and used by most other modules to access settings like API keys, model names, file paths, and behavior flags.

3.  **Shared State (`state.py`)**:
    *   The `BotState` class manages shared, mutable data in an asynchronous-safe manner. This includes:
        *   Short-term message history for each channel.
        *   Scheduled reminders.
        *   Timestamp of the last Playwright usage (for automated cleanup).
        *   Per-channel locks to prevent concurrent LLM streaming operations within the same channel.
        *   A global scrape lock ensures web scraping tasks (e.g., RSS fetching) run one at a time. If a command must wait for the lock, the bot now notifies the user with an ephemeral "queued" message before starting a new public response once scraping begins.
    *   Instances of `BotState` are passed to modules that need to access or modify this shared information (e.g., command handlers, event processors).

4.  **Discord Event Handlers (`discord_events.py`)**:
    *   Contains listeners for Discord gateway events (e.g., `on_ready`, `on_message`, `on_raw_reaction_add`).
    *   The `on_message` handler is crucial for general interaction:
        *   It filters messages based on configured criteria (mentions, DMs, allowed channels/roles).
        *   Processes message content, including text, audio attachments (transcription via `audio_utils`), and image attachments.
        *   Detects URLs and uses `web_utils` to scrape content (webpages, YouTube transcripts) and generate descriptions for screenshots (via `llm_handling`).
        *   Interacts with `llm_handling.py` to get responses from the LLM, incorporating RAG context from `rag_chroma_manager.py`.
    *   Manages background tasks (`@tasks.loop`) for reminders, Playwright process cleanup, and timeline pruning.

5.  **Discord Slash Commands (`discord_commands.py`)**:
    *   Defines and implements all application commands (slash commands).
    *   Each command handler typically:
        *   Parses user input.
        *   Interacts with various utility modules (`web_utils` for scraping, `audio_utils` for TTS, `rag_chroma_manager` for RAG/DB operations).
        *   Uses `llm_handling.py` to communicate with the LLM for generating responses or performing tasks (e.g., summarization, creative generation).
        *   Manages interaction responses (deferring, sending embeds, streaming updates).

6.  **LLM Handling (`llm_handling.py`)**:
    *   Abstracts communication with the Large Language Model.
    *   `get_system_prompt()`: Loads the bot's persona and instructions.
    *   `_build_initial_prompt_messages()`: Constructs the complete prompt (system message, RAG context, conversation history, user query) to be sent to the LLM.
    *   `stream_llm_response_to_interaction()` and `stream_llm_response_to_message()`: Manage the streaming of LLM responses back to Discord, updating messages in real-time.
    *   Handles post-response actions like updating conversation history in `BotState`, triggering TTS via `audio_utils`, and **asynchronously** ingesting the conversation into ChromaDB via `rag_chroma_manager`.
    *   Shows a short "Post-processing" notification while conversations and other ingestion tasks are archived and analyzed. These tasks now run in the background so replies are never blocked. Once finished, the progress message is simply removed, leaving no additional ephemeral notice.
    *   `get_description_for_image()`: Uses a vision-capable LLM to describe images.

7.  **RAG and ChromaDB Management (`rag_chroma_manager.py`)**:
    *   Manages the Retrieval Augmented Generation pipeline and all interactions with the ChromaDB vector store.
    *   `initialize_chromadb()`: Sets up connections to various data collections (raw history, distilled summaries, news, timeline summaries, entities, relations, observations).
    *   `ingest_conversation_to_chromadb()`: Stores new conversations, extracts structured data (entities, relations, observations), distills key sentences, and saves them for future retrieval. Distilled summaries now include a `Conversation recorded at:` header so timestamps persist when memories are merged.
    *   `retrieve_and_prepare_rag_context()`: Given a query, searches relevant collections for pertinent information and synthesizes it into a context string for the LLM. The synthesis step now receives the current date so it can express how long ago memories occurred. When a relative date phrase (e.g. "yesterday" or "last week") is detected, the system searches stored tweets, chat history, timeline events, news, RSS summaries, entities, relations, and observations for that range and uses only those results in the RAG context.
    *   `update_retrieved_memories()`: Merges retrieved memory snippets with the latest conversation summary and stores updated memories for future use. The merge prompt now includes both the snippet's original date and the current date, giving the LLM clearer temporal context. This behavior can be disabled via the `ENABLE_MEMORY_MERGE` setting.
    *   Includes functions for importing data (e.g., ChatGPT exports) and storing specific data types (e.g., news summaries).

8.  **Utility Modules**:
    *   **`audio_utils.py`**: Handles Text-to-Speech (TTS) generation (via an external API) and Speech-to-Text (STT) for audio attachments (via a local Whisper model).
    *   **`web_utils.py`**: Provides functions for web scraping (using Playwright and BeautifulSoup), querying SearXNG, fetching YouTube transcripts, and parsing RSS feeds. Manages a semaphore for Playwright concurrency.
    *   **`utils.py`**: Contains general helper functions like text chunking, URL detection, text cleaning for TTS, time string parsing, and Playwright process cleanup.
    *   **`common_models.py`**: Defines common data structures like `MsgNode` used throughout the application.
    *   **`rss_cache.py`**: Manages a local cache of seen RSS feed items to avoid reprocessing.
    *   **`timeline_pruner.py`**: Contains logic for the background task that prunes and summarizes old chat history from ChromaDB.
    *   **`rag_cleanup.py`**: Utility script that removes distilled summary entries referencing conversation IDs that no longer exist in the main chat history collection.
    *   **`open_chatgpt_login.py`**: Launches a persistent Playwright browser so you can log into ChatGPT (or other services) once and reuse the saved cookies.
    *   **`llm_request_processor.py`**: Optional worker for processing queued LLM requests separately from the Discord event loop.

**Data Flow Example (User sends a message mentioning the bot):**

1.  `discord_events.on_message` receives the message.
2.  It processes attachments (e.g., transcribes audio via `audio_utils`, notes images).
3.  If URLs are present, `web_utils.scrape_website` (and potentially `llm_handling.get_description_for_image` for screenshots) is called.
4.  `rag_chroma_manager.retrieve_and_prepare_rag_context` is called to get relevant past information.
5.  `llm_handling._build_initial_prompt_messages` assembles the full prompt using system prompt, RAG context, current `BotState` history, and the user's processed message.
6.  `llm_handling.stream_llm_response_to_message` sends the prompt to the LLM and streams the response back.
    *   The Discord message is updated live.
7.  After the stream:
    *   The user's message and the bot's full response are added to `BotState.message_history`.
    *   `audio_utils.send_tts_audio` is called (as a background task) if TTS is enabled.
    *   `rag_chroma_manager.ingest_conversation_to_chromadb` stores **only** this final user question and bot answer (not the full prompt) and distills it for RAG.

This modular architecture allows for easier development, testing, and modification of individual components without impacting the entire system.

---

## 5. Prerequisites

*   **Python 3.10+**
*   **pip** (Python package installer)
*   **A local LLM server:**
    *   Must be OpenAI API compatible (e.g., LM Studio, Ollama with OpenAI compatible endpoint).
    *   You will need the server URL and potentially model names for text (`LLM`, `FAST_LLM_MODEL`) and vision (`VISION_LLM_MODEL`).
*   **Git:** For cloning the repository.

**Optional (but Recommended for Full Functionality):**

*   **A local TTS server:**
    *   OpenAI TTS API compatible (e.g., a local instance of an OpenAI TTS compatible server).
    *   The URL for this server is set via `TTS_API_URL`.
*   **A local SearXNG instance:**
    *   Required for the `/search` and `/news` commands to function.
    *   The URL of this instance is needed in the `.env` file (`SEARX_URL`).
*   **CUDA-enabled GPU:**
    *   Strongly recommended for faster Whisper audio transcription (especially with fp16).
    *   Also beneficial for running local LLMs if they support GPU acceleration.
*   **Playwright login profile (optional):**
    *   Some scraping features (e.g., ChatGPT transcript imports, Ground News, Twitter home timeline) require a logged-in browser profile.
    *   Run `python open_chatgpt_login.py` once to open a persistent Playwright browser (`.pw-profile`) and sign in to the necessary services.

---

## 6. Configuration

Configuration is primarily managed through an `.env` file located in the root directory of the project. You can create this file by copying `example.env` to `.env` and then modifying the values to suit your setup.

Numeric values must be valid integers (or floats where specified). Boolean values should be expressed as `true` or `false` (case-insensitive).

### Environment Variables

Below is a comprehensive list of environment variables used by DiscordSam, along with their purpose and default values where applicable.

**Core Bot Settings:**

*   `DISCORD_BOT_TOKEN` (Required): Your Discord bot token. The bot cannot start without this.
*   `ALLOWED_CHANNEL_IDS` (Optional): A comma-separated list of Discord channel IDs where the bot is allowed to respond to general messages (i.e., messages not directly mentioning it or in DMs). If empty, the bot will respond in any channel it has access to (respecting role permissions).
    *   Example: `123456789012345678,987654321098765432`
*   `ALLOWED_ROLE_IDS` (Optional): A comma-separated list of Discord role IDs. If set, users must have at least one of these roles for the bot to respond to their general messages in allowed channels. Does not affect DMs or direct mentions. If empty, no role restrictions apply (beyond channel/mention checks).
    *   Example: `112233445566778899,009988776655443322`
*   `SYSTEM_PROMPT_FILE` (Default: `system_prompt.md`): Path to a text or markdown file containing the main system prompt that defines the bot's persona and core instructions.
*   `USER_PROVIDED_CONTEXT` (Optional): Additional global context that will be prepended to the system prompt for every LLM interaction. Useful for providing persistent high-level instructions or information.

**LLM Configuration:**

*   `LOCAL_SERVER_URL` (Default: `http://localhost:1234/v1`): The base URL of your OpenAI-compatible LLM server (e.g., LM Studio, Ollama with OpenAI endpoint).
*   `LLM_API_KEY` (Optional, Default: `""` or `lm-studio`): The API key for your LLM server. Often not strictly required for local servers but can be set if needed.
*   `LLM` (Default: `local-model`): The name/identifier of the default language model to be used for most text generation tasks.
*   `FAST_LLM_MODEL` (Default: Same as `LLM`): The model used for tasks where speed is preferred over maximum quality, such as intermediate summarizations or quick classifications.
*   `VISION_LLM_MODEL` (Default: `llava`): The model used for tasks involving image understanding (e.g., the `/ap` command or describing screenshots).
*   `LLM_SUPPORTS_JSON_MODE` (Default: `false`): Set to `true` if your LLM server and the selected model support JSON mode for structured output (e.g., for entity extraction).
*   `USE_RESPONSES_API` (Default: `false`): When `true`, use OpenAI's Responses API instead of legacy Chat Completions. System prompts are passed via the `instructions` field and model names may require the orchestrator variant (e.g., `gpt-4o`).
*   `MAX_MESSAGE_HISTORY` (Default: `10`): The maximum number of recent messages (user and assistant turns) to include in the short-term context sent to the LLM.
*   `MAX_COMPLETION_TOKENS` (Default: `2048`): The maximum number of tokens the LLM is allowed to generate in a single response.

**Text-to-Speech (TTS) Settings:**

*   `TTS_ENABLED_DEFAULT` (Default: `true`): Whether TTS is enabled by default for bot responses.
*   `TTS_API_URL` (Default: `http://localhost:8880/v1/audio/speech`): The endpoint of your OpenAI TTS API compatible server.
*   `TTS_VOICE` (Default: `af_sky+af+af_nicole`): The voice to be used for TTS generation. The format may depend on your TTS server.
*   `TTS_MAX_AUDIO_BYTES` (Default: `8388608` (8MB)): Maximum size of a single generated TTS audio clip. Longer audio will be split into parts before uploading to stay under Discord's attachment limits.
*   `TTS_SPEED` (Default: `1.3`): Playback speed multiplier for TTS audio. Use `1.0` for normal speed.
*   `TTS_INCLUDE_THOUGHTS` (Default: `false`): If `true`, content within `<think>...</think>` tags will also be spoken using TTS. When `false`, only the user-facing portion of the response is processed.

**Web Features & Scraping:**

*   `SEARX_URL` (Default: `http://192.168.1.3:9092/search`): The URL of your SearXNG instance, required for the `/search` and `/news` commands.
*   `SEARX_PREFERENCES` (Optional, Default: `""`): Optional JSON string or formatted string for SearXNG engine preferences. Example: `{"engines" : ["google", "wikipedia"]}` or for dynamic query insertion: `!google %s`.
*   `MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT` (Default: `8000`): Maximum number of characters from a single scraped webpage or YouTube transcript to include in the LLM prompt.
*   `MAX_IMAGE_BYTES_FOR_PROMPT` (Default: `4194304` (4MB)): Maximum size in bytes for an image to be processed and sent to the vision LLM.
*   `NEWS_MAX_LINKS_TO_PROCESS` (Default: `5`): The maximum number of news links or search results the bot will attempt to scrape and process for commands like `/news` or `/search`.
*   `HEADLESS_PLAYWRIGHT` (Default: `true`): Set to `false` to run Playwright browser instances in non-headless mode (visible window), useful for debugging scraping.
*   `PLAYWRIGHT_MAX_CONCURRENCY` (Default: `2`): Maximum number of concurrent Playwright browser instances/contexts allowed.
*   `SCRAPE_SCROLL_ATTEMPTS` (Default: `3`): How many times the bot will attempt to scroll down a webpage when scraping to load dynamically loaded content.
*   `GROUND_NEWS_SEE_MORE_CLICKS` (Default: `3`): How many times to click Ground News's "See more stories" button before scraping.
*   `GROUND_NEWS_CLICK_DELAY_SECONDS` (Default: `1.0`): Seconds to wait after each "See more stories" click when scraping Ground News.
*   `GROUND_NEWS_ARTICLE_DELAY_SECONDS` (Default: `5.0`): Seconds to wait between scraping each Ground News article.
*   `PLAYWRIGHT_CLEANUP_INTERVAL_MINUTES` (Default: `5`): How often the background task runs to check for and clean up idle Playwright processes.
*   `PLAYWRIGHT_IDLE_CLEANUP_THRESHOLD_MINUTES` (Default: `10`): How long Playwright must be idle (no scraping activity) before the cleanup task will terminate its processes.
*   `SCRAPE_LOCK_TIMEOUT_SECONDS` (Default: `60`): How long to wait when acquiring the scraping lock before giving up.

**ChromaDB (RAG & Long-Term Memory):**

*   `CHROMA_DB_PATH` (Default: `./chroma_data`): The local file system path where ChromaDB will store its data.
*   `CHROMA_COLLECTION_NAME` (Default: `long_term_memory`): Name of the ChromaDB collection for storing recent conversation exchanges (user prompt and bot response).
*   `CHROMA_DISTILLED_COLLECTION_NAME` (Default: `distilled_chat_summaries`): Name of the collection for storing concise, keyword-rich distilled summaries of conversations (used for primary RAG retrieval).
*   `CHROMA_NEWS_SUMMARY_COLLECTION_NAME` (Default: `news_summaries`): Collection for storing summaries of news articles processed by the `/news` command.
*   `CHROMA_RSS_SUMMARY_COLLECTION_NAME` (Default: `rss_summaries`): Collection for storing summaries of RSS feed items processed by the `/rss` command.
*   `CHROMA_TIMELINE_SUMMARY_COLLECTION_NAME` (Default: `timeline_summaries`): Collection for storing summaries of pruned, older chat history.
*   `CHROMA_TWEETS_COLLECTION_NAME` (Default: `tweets_collection`): Collection for storing tweets fetched for summarization.
*   `CHROMA_ENTITIES_COLLECTION_NAME` (Default: `entities_collection`): Collection for storing extracted entities (persons, organizations, etc.) from conversations.
*   `CHROMA_RELATIONS_COLLECTION_NAME` (Default: `relations_collection`): Collection for storing extracted relationships between entities.
*   `CHROMA_OBSERVATIONS_COLLECTION_NAME` (Default: `observations_collection`): Collection for storing extracted key observations or facts.
*   `RAG_NUM_DISTILLED_SENTENCES_TO_FETCH` (Default: `3`): How many relevant distilled sentences/summaries to fetch from `CHROMA_DISTILLED_COLLECTION_NAME` for RAG context.
*   `RAG_NUM_COLLECTION_DOCS_TO_FETCH` (Default: `3`): How many relevant documents to fetch from other ChromaDB collections (news, timeline, entities, etc.) for RAG context.
*   `RAG_MAX_FULL_CONVO_CHARS` (Default: `20000`): When retrieving full conversation logs for RAG context, only the last N characters of each log will be used, trimming the oldest text from the beginning.
*   `RAG_MAX_DATE_RANGE_DOCS` (Default: `100`): Maximum number of RSS documents to include when using date-range retrieval. Other collections use `RAG_NUM_COLLECTION_DOCS_TO_FETCH` as their limit.
*   `ENABLE_MEMORY_MERGE` (Default: `false`): Set to `true` to merge retrieved memory snippets with new conversation summaries after each response.
*   `TIMELINE_PRUNE_DAYS` (Default: `30`): How many days of chat history to retain in the main `CHROMA_COLLECTION_NAME` before the daily `timeline_pruner_task` summarizes and moves it to `CHROMA_TIMELINE_SUMMARY_COLLECTION_NAME`.
    *   The summarized timeline entries are stored in a separate collection so the main history stays small while older context remains searchable.

**Discord Embed & Streaming Settings:**

*   `EMBED_COLOR_INCOMPLETE` (Default: `discord.Color.orange().value` e.g., `0xFFA500` or `16753920`): Hex color code (e.g., `#FFA500` or `0xFFA500`) or integer value for embeds indicating an incomplete or in-progress operation.
*   `EMBED_COLOR_COMPLETE` (Default: `discord.Color.green().value` e.g., `0x00FF00` or `65280`): Hex color code or integer value for embeds indicating a successful or completed operation.
*   `EMBED_COLOR_ERROR` (Default: `discord.Color.red().value` e.g., `0xFF0000` or `16711680`): Hex color code or integer value for embeds indicating an error.
*   `EMBED_MAX_LENGTH` (Default: `4096`): Maximum character length for a single Discord embed description. Text will be chunked if it exceeds this.
*   `STREAM_EDIT_THROTTLE_SECONDS` (Default: `0.1`): Minimum time in seconds to wait between edits when streaming LLM responses. This is related to `EDITS_PER_SECOND` (hardcoded at 1.3 in `config.py`, meaning throttle is roughly 1/1.3). This variable directly controls the sleep duration.

**Image Processing (Attachments & Vision):**

*   `MAX_IMAGES_PER_MESSAGE` (Default: `1`): The maximum number of images attached to a single Discord message that the bot will process and send to the vision LLM.

---

## 7. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/helix4u/DiscordSam.git
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
    This list includes `psutil`, which allows the bot to automatically clean up stray Playwright processes.
    > **Important Note for PyTorch (used by Whisper):** For optimal performance, especially with GPU support, it's often best to install PyTorch separately by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/) *before* running `pip install -r requirements.txt`. The `requirements.txt` might install a CPU-only version if a specific PyTorch version isn't already present.

4.  **Install Playwright browsers:**
    Playwright requires browser binaries for web scraping.
    ```bash
    playwright install
    ```

---

## 8. Slash Commands

DiscordSam offers a variety of slash commands for diverse functionalities. Here's a detailed breakdown:

*   **`/news <topic>`**
    *   **Purpose:** Generates a news briefing on a specified topic.
    *   **Arguments:**
        *   `topic` (Required): The news topic you want a briefing on (e.g., "AI advancements", "local city council decisions").
    *   **Behavior:**
        1.  Performs a web search (via SearXNG, if configured) for news articles related to the `topic`.
        2.  Scrapes the content of the top few articles (up to `NEWS_MAX_LINKS_TO_PROCESS`).
        3.  Uses a fast LLM (`FAST_LLM_MODEL`) to summarize each scraped article individually.
        4.  Stores these individual summaries in ChromaDB (`CHROMA_NEWS_SUMMARY_COLLECTION_NAME`).
        5.  Sends all individual summaries to the main LLM (`LLM`) to synthesize a final, coherent news briefing.
        6.  Streams the briefing back to the Discord channel.
        7.  Provides TTS for the final briefing if enabled.
    *   **Output:** An embed message containing the news briefing, updated live as the process unfolds (gathering articles, summarizing, final briefing).

*   **`/ingest_chatgpt_export <file_path>`**
    *   **Purpose:** Ingests conversations from a ChatGPT data export file (`conversations.json`) into the bot's long-term memory (ChromaDB).
    *   **Arguments:**
        *   `file_path` (Required): The full local path to your `conversations.json` file.
    *   **Permissions:** Requires 'Manage Messages' permission.
    *   **Behavior:**
        1.  Parses the `conversations.json` file.
        2.  For each conversation, it stores the full text in the main chat history collection (`CHROMA_COLLECTION_NAME`).
        3.  It then distills each conversation (typically the last user/assistant turn, or full text as fallback) into keyword-rich sentences using an LLM.
        4.  These distilled sentences are stored in `CHROMA_DISTILLED_COLLECTION_NAME` (prefixed with a `Conversation recorded at:` timestamp) and linked to the full conversation document, enabling RAG.
    *   **Output:** An ephemeral message confirming the number of conversations successfully processed and stored.

*   **`/remindme <time_duration> <reminder_message>`**
    *   **Purpose:** Sets a reminder for yourself.
    *   **Arguments:**
        *   `time_duration` (Required): How long from now to set the reminder (e.g., "10m", "2h30m", "1d").
        *   `reminder_message` (Required): The message for your reminder.
    *   **Behavior:**
        1.  Parses the `time_duration` string into a `timedelta`.
        2.  Calculates the reminder time.
        3.  Stores the reminder (time, channel ID, user ID, message, original time string) in `BotState`.
        4.  A background task (`check_reminders_task` in `discord_events.py`) periodically checks for due reminders.
    *   **Output:**
        *   Immediate: A confirmation message that the reminder has been set.
        *   Later: When due, the bot sends a message in the original channel, mentioning the user with the reminder message and providing TTS.

*   **`/roast <url>`**
    *   **Purpose:** Generates a short, witty, and biting comedy roast routine based on the content of a given webpage.
    *   **Arguments:**
        *   `url` (Required): The URL of the webpage to roast.
    *   **Behavior:**
        1.  Scrapes the content of the provided `url` using `web_utils.scrape_website`.
        2.  Sends the scraped text to the LLM with a specific prompt instructing it to create a comedy roast.
        3.  Streams the LLM's roast back to the Discord channel.
        4.  Provides TTS for the roast if enabled.
    *   **Output:** An embed message, updated live, containing the comedy roast.

*   **`/search <query>`**
    *   **Purpose:** Performs a web search using a configured SearXNG instance and summarizes the findings.
    *   **Arguments:**
        *   `query` (Required): Your search query.
    *   **Behavior:**
        1.  Queries the configured SearXNG instance (`SEARX_URL`).
        2.  Scrapes the content of the top few search results (up to `NEWS_MAX_LINKS_TO_PROCESS`).
        3.  Uses a fast LLM (`FAST_LLM_MODEL`) to summarize each scraped page individually, focusing on relevance to the query.
        4.  Sends these individual summaries to the main LLM (`LLM`) with a prompt to synthesize a single, integrated summary that directly addresses the user's query.
        5.  Streams this final summary back to the Discord channel as a new message flow (using `force_new_followup_flow=True`).
        6.  Provides TTS for the summary if enabled.
    *   **Output:**
        *   Initial embed updating progress (searching, processing results).
        *   A new set of messages (embeds) containing the final synthesized search summary.

*   **`/pol <statement>`**
    *   **Purpose:** Generates a sarcastic, snarky, and somewhat troll-like political commentary on a given statement.
    *   **Arguments:**
        *   `statement` (Required): The political statement to comment on.
    *   **Behavior:**
        1.  Sends the user's `statement` to the LLM with a specialized system prompt instructing it to generate sarcastic commentary, focusing on wit, irony, and highlighting logical fallacies humorously.
        2.  Streams the LLM's response back to the Discord channel.
        3.  Provides TTS for the commentary if enabled.
    *   **Output:** An embed message, updated live, containing the sarcastic political commentary.

*   **`/rss <feed_url> [limit]`**
    *   **Purpose:** Fetches new entries from a specified RSS feed, scrapes the linked articles, summarizes them, and displays the summaries.
    *   **Arguments:**
        *   `feed_url` (Required): The URL of the RSS feed. Can be selected from a preset list or provided directly.
        *   `limit` (Optional, Default: 15): The maximum number of new entries to fetch and process (max 20).
    *   **Behavior:**
        1.  Fetches the RSS feed using `web_utils.fetch_rss_entries`.
        2.  Skips CBS News entries from `https://www.cbsnews.com/video/` as these pages lack article text.
        3.  Compares entries against a local cache (`rss_seen.json`) to identify new ones.
        4.  For each new entry (up to `limit`):
            *   Scrapes the content of the article linked in the entry.
            *   Uses a fast LLM (`FAST_LLM_MODEL`) to summarize the scraped article.
        5.  Displays the summaries (title, publication date in your local time, link, summary) in Discord embeds. If the content is long, it's chunked into multiple embeds.
        6.  Updates the `rss_seen.json` cache.
        7.  Provides TTS for the combined summaries if enabled.
        8.  The user's command and the bot's full summarized response are added to short-term history and ingested into ChromaDB. A brief ephemeral "Post-processing..." notice appears while this occurs.
        9.  If no new entries are found, the bot replies with an ephemeral message instead of posting publicly.
    *   **Output:** One or more embed messages containing summaries of new RSS feed entries, each showing the title, local publication date, link, and summary.

*   **`/allrss [limit]`**
    *   **Purpose:** Fetches recent articles from all default RSS feeds in chronological order.
    *   **Arguments:**
        *   `limit` (Optional, Default: 15): Maximum number of entries to pull from each feed (max 20).
    *   **Behavior:**
        1.  Prefetches entries from every preset RSS feed (up to `limit` per feed).
        2.  Combines and sorts all new entries by publication time using the user's local timezone.
        3.  Processes articles in batches of `limit`, scraping and summarizing each just like `/rss`.
        4.  After each batch, sends an embed with the summaries and optional TTS before moving to the next batch.
        5.  Each batch's summary is immediately archived to ChromaDB so RAG stays updated throughout the run, showing a short ephemeral "Post-processing..." indicator.
    *   **Output:** Summaries are delivered in batches of `limit` entries until all new articles are processed.

*   **`/gettweets [username] [preset_user] [limit]`**
    *   **Purpose:** Fetches and summarizes recent tweets from a specified X/Twitter user.
    *   **Arguments:**
        *   `username` (Optional): The X/Twitter username (without the '@').
        *   `preset_user` (Optional): Choose a username from a predefined list. If both `username` and `preset_user` are provided, `username` takes precedence. One must be provided.
        *   `limit` (Optional, Default: 25): The maximum number of tweets to fetch (max 100).
    *   **Behavior:**
        1.  Uses `web_utils.scrape_latest_tweets` (Playwright with JS execution) to scrape recent tweets from the user's profile (specifically their "with_replies" timeline).
        2.  Displays the raw fetched tweets (timestamp, author, content, link) in Discord embeds, chunked if necessary. Any images are downloaded and described via the vision LLM, and these descriptions are included with the tweet text.
        3.  Sends the text of these tweets to the LLM with a prompt to analyze and summarize the main themes, topics, and overall sentiment.
        4.  Streams this summary back to the Discord channel as a new message flow.
        5.  Provides TTS for the summary if enabled.
        6.  Stores newly fetched tweets in the `CHROMA_TWEETS_COLLECTION_NAME` collection for future retrieval and records their IDs so they aren't resummarized later. A short ephemeral "Post-processing..." message indicates when this archival happens.
        7.  If another scraping task is running, the bot sends an ephemeral "waiting for other scraping tasks" notice while it queues your request.
    *   **Output:**
        *   Embed(s) containing the raw fetched tweets.
        *   A new set of messages (embeds) containing the LLM-generated summary of the tweets. Progress updates are sent during scraping.

*   **`/homefeed [limit]`**
    *   **Purpose:** Fetches and summarizes tweets from the logged-in X/Twitter home timeline.
    *   **Arguments:**
        *   `limit` (Optional, Default: 25): The maximum number of tweets to fetch (max 200).
    *   **Behavior:**
        1.  Uses `web_utils.scrape_home_timeline` (Playwright with JS execution) to scrape tweets from `https://x.com/home`.
        2.  Displays the raw fetched tweets (timestamp, author, content, link) in Discord embeds, chunked if necessary. Any images are downloaded and described via the vision LLM, and these descriptions are included with the tweet text.
        3.  Sends the text of these tweets to the LLM with a prompt to analyze and summarize the main themes, topics, and overall sentiment.
        4.  Streams this summary back to the Discord channel as a new message flow.
        5.  Provides TTS for the summary if enabled.
        6.  Stores newly fetched tweets in the `CHROMA_TWEETS_COLLECTION_NAME` collection for future retrieval and records their IDs so they aren't resummarized later. A short ephemeral "Post-processing..." message indicates when this archival happens.
        7.  If another scraping task is running, the bot sends an ephemeral "waiting for other scraping tasks" notice while it queues your request.
    *   **Output:**
        *   Embed(s) containing the raw fetched tweets.
        *   A new set of messages (embeds) containing the LLM-generated summary of the tweets. Progress updates are sent during scraping.

*   **`/alltweets [limit]`**
    *   **Purpose:** Fetches and summarizes tweets from all default X/Twitter accounts.
    *   **Arguments:**
        *   `limit` (Optional, Default: 25): The maximum number of tweets to fetch per account (max 50).
    *   **Behavior:**
        1.  Iterates through each preset Twitter account defined for the bot.
        2.  Uses `web_utils.scrape_latest_tweets` to scrape recent tweets for each account.
        3.  Displays the raw fetched tweets in Discord embeds, chunked if necessary. Any images are downloaded and described via the vision LLM, and these descriptions are included with the tweet text.
        4.  Sends these tweets to the LLM to generate a brief summary, streamed back to the channel.
        5.  Provides TTS for each summary if enabled.
        6.  Stores newly fetched tweets in the `CHROMA_TWEETS_COLLECTION_NAME` collection for future retrieval. A short ephemeral "Post-processing..." message indicates when this archival happens.
    *   **Output:** A series of embeds and summaries for each default account. If no new tweets are found for any account, the bot replies with an ephemeral message.

*   **`/groundnews [limit]`**
    *   **Purpose:** Scrapes the Ground News "My Feed" page and summarizes new articles.
    *   **Arguments:**
        *   `limit` (Optional, Default: 20): Maximum number of articles to process (max 50).
    *   **Behavior:**
        1.  Uses `web_utils.scrape_ground_news_my` (Playwright) to extract "See the Story" links, scrolling the page to load more items if necessary.
        2.  Skips any links already recorded in `ground_news_seen.json`.
        3.  Scrapes each new article with `web_utils.scrape_website` and summarizes it using the fast LLM.
        4.  Displays the summaries (title, link, short summary) in Discord embeds and updates the cache.
        5.  Each article summary is ingested into ChromaDB individually as it is processed, with a brief ephemeral "Post-processing..." message shown during ingestion.
        6.  If another scraping task is running, you will see an ephemeral "waiting for other scraping tasks" notice until the global scrape lock is free.
    *   **Requirements:** You must already be logged in to Ground News using Playwright's persistent profile (`.pw-profile`). If not logged in, the command will likely return no articles.
    *   **Output:** Embeds containing summaries for each newly found Ground News article.

*   **`/groundtopic <topic> [limit]`**
    *   **Purpose:** Scrapes a specified Ground News topic page and summarizes new articles.
    *   **Arguments:**
        *   `topic` (Required): The topic to fetch, chosen from a preset list.
        *   `limit` (Optional, Default: 20): Maximum number of articles to process (max 50).
    *   **Behavior:**
        1.  Uses `web_utils.scrape_ground_news_topic` to extract "See the Story" links from the selected topic page.
        2.  Skips links already recorded in `ground_news_seen.json`.
        3.  Scrapes and summarizes each new article via the fast LLM.
        4.  Displays summaries in Discord embeds and updates the cache.
        5.  Each article summary is ingested into ChromaDB individually as it is processed, accompanied by a brief ephemeral "Post-processing..." message.
        6.  If another scraping task is running, you will see an ephemeral "waiting for other scraping tasks" notice until the global scrape lock is free.
    *   **Requirements:** Same as `/groundnews` &mdash; you must be logged in with Playwright's persistent profile.
    *   **Output:** Embeds containing summaries for each new topic article.

*   **`/ap <image> [user_prompt]`**
    *   **Purpose:** Describes an attached image in the style of an Associated Press (AP) photo caption, with a humorous twist: a randomly chosen celebrity is creatively inserted as the main subject.
    *   **Arguments:**
        *   `image` (Required): The image file to describe.
        *   `user_prompt` (Optional): Additional context or a specific theme for the description.
    *   **Behavior:**
        1.  Reads the attached image bytes and converts it to base64.
        2.  Constructs a prompt for a vision-capable LLM (`VISION_LLM_MODEL`). The prompt instructs the LLM to act as an AP photo writer, describe the image vividly, and seamlessly incorporate a randomly chosen celebrity (e.g., Keanu Reeves, Zendaya) as the main subject.
        3.  If `user_prompt` is provided, it's included in the instructions to the LLM.
        4.  Sends the image data and prompt to the LLM.
        5.  Streams the LLM's creative description back to the Discord channel.
        6.  Provides TTS for the description if enabled.
    *   **Output:** An embed message, updated live, containing the AP-style photo description featuring the celebrity.

*   **`/clearhistory`**
    *   **Purpose:** Clears the bot's short-term conversational memory (message history) for the current channel. This does not affect long-term memory in ChromaDB.
    *   **Arguments:** None.
    *   **Permissions:** Requires 'Manage Messages' permission.
    *   **Behavior:**
        1.  Calls `BotState.clear_channel_history()` for the current channel ID.
    *   **Output:** An ephemeral message confirming that the short-term history for the channel has been cleared.

---

## 9. Running the Bot

1.  **Ensure your local servers are running:**
    *   Your LLM server (e.g., LM Studio, Ollama) must be active and accessible at the `LOCAL_SERVER_URL`. Ensure the correct models specified in your `.env` file (e.g., `LLM`, `VISION_LLM_MODEL`) are loaded and available on this server.
    *   If `TTS_ENABLED_DEFAULT` is `true`, your TTS server must be running and accessible at `TTS_API_URL`.
    *   If you plan to use the `/search` or `/news` commands, your SearXNG instance must be running and accessible at `SEARX_URL`.

2.  **Verify your `.env` file:** Double-check that it's correctly configured, especially the `DISCORD_BOT_TOKEN` and all server URLs and model names.

3.  **Run the main bot script:**
    Execute this command from the root directory of the project:
    ```bash
    python main_bot.py
    ```

---

## 10. Logging

DiscordSam utilizes Python's built-in `logging` module to provide information about its operations, warnings, and errors.

*   **Configuration:** Basic logging is configured at the entry point of the application in `main_bot.py`. It sets the logging level (defaulting to `INFO`) and a format that includes timestamps, log level, filename, line number, and the message.
*   **Output:** By default, logs are directed to the console (standard output/standard error) where the bot is running. This means you will see log messages appearing in the terminal window from which you launched `python main_bot.py`.
*   **Log Levels:**
    *   `INFO`: General information about the bot's operations (e.g., connecting to Discord, commands being invoked, tasks starting).
    *   `WARNING`: Indicates potential issues or unexpected situations that don't necessarily stop the bot but might be worth investigating.
    *   `ERROR`: Signifies errors that occurred during operation, which might affect functionality.
    *   `CRITICAL`: Severe errors that likely prevent the bot from starting or continuing to run correctly.
    *   `DEBUG`: More verbose information useful for development and troubleshooting. This level is typically not enabled by default but can be changed in `main_bot.py` if needed for deeper inspection.

If you encounter issues, the console output containing these logs is the first place to look for more details about what the bot was doing and any errors it might have encountered.

---

## 11. Maintenance & Auxiliary Scripts

Several helper scripts are provided for upkeep and optional functionality:

* **`timeline_pruner.py`** – Summarizes chat logs older than `TIMELINE_PRUNE_DAYS` and stores the results in `CHROMA_TIMELINE_SUMMARY_COLLECTION_NAME`. This runs daily via a background task but can also be executed manually.
* **`rag_cleanup.py`** – Removes distilled summaries that reference conversations which no longer exist in the main history collection.
* **`open_chatgpt_login.py`** – Launches a persistent Playwright browser (`.pw-profile`) so you can log into ChatGPT or other sites once and reuse the saved cookies.
* **`llm_request_processor.py`** – Experimental worker that processes queued LLM requests separately from the Discord bot. Start it with `python llm_request_processor.py` if you wish to try concurrent request handling.

---

## 12. Troubleshooting

*   **Playwright Processes:** If Chromium or Playwright processes remain running after scraping, the bot has a built-in task (`cleanup_playwright_task` in `discord_events.py`) that attempts to clean them up periodically based on `PLAYWRIGHT_CLEANUP_INTERVAL_MINUTES` and `PLAYWRIGHT_IDLE_CLEANUP_THRESHOLD_MINUTES`. You can also manually kill these processes if needed.
*   **Model Not Found Errors:** Ensure the model names specified in your `.env` file (e.g., `LLM`, `VISION_LLM_MODEL`, `FAST_LLM_MODEL`) exactly match the models loaded and available on your LLM server at `LOCAL_SERVER_URL`.
*   **Connection Errors:** Verify that all specified server URLs (`LOCAL_SERVER_URL`, `TTS_API_URL`, `SEARX_URL`) are correct and that the respective services are running and accessible from where the bot is running.
*   **Permissions:** If slash commands don't appear or certain commands fail, check the bot's permissions in your Discord server settings. It generally needs permissions to read messages, send messages, use embeds, attach files, and use slash commands. Some commands like `/clearhistory` or `/ingest_chatgpt_export` might require 'Manage Messages'.

---

## 13. Potential Future Directions

DiscordSam is a project with significant potential for growth. Here are some ideas for future enhancements:

*   **Enhanced RAG Strategies:** More sophisticated summarization, knowledge graph integration, hybrid search, advanced query understanding for RAG.
*   **Expanded LLM/Multimodal Support:** Support for more LLM backends, true multimodal inputs/outputs beyond simple image description, advanced agentic capabilities (e.g., function calling, planning).
*   **Slash Command Enhancements:** New creative tools, personalization features (user-specific settings), integration with more external services (e.g., calendar, project management).
*   **Improved Error Handling & Resilience:** More granular error reporting to users, graceful degradation of services when components are unavailable.
*   **Sophisticated Context Management:** User-specific context windows, thread-specific context, dynamic adjustment of context length based on LLM limits.
*   **UI/UX Enhancements:** More interactive embeds with buttons and select menus, potential web UI for bot management and RAG database exploration.
*   **Observability & Logging:** Structured logging (e.g., JSON logs), performance metrics dashboard, admin commands for bot health checks and stats.
*   **Security & Permissions:** Fine-grained command permissions based on roles/users, data privacy controls for RAG content.
*   **Codebase Improvements:** Further type hinting, more comprehensive unit and integration tests.

---

## 14. Support / Issues / Contributing

Found a bug? Have a feature request? Want to contribute?

*   Please **open an issue** on the GitHub repository for bugs or suggestions. Provide as much detail as possible, including steps to reproduce, error messages, and your configuration.
*   If you'd like to contribute code, please **fork the repository** and submit a **pull request** with your changes. Ensure your code follows existing style conventions and consider adding tests for new features.

We welcome contributions to improve and expand DiscordSam!
