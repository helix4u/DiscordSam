# DiscordSam: The Hyper-Intelligent Discord Bot

## 1. Project Overview

i test in main. lol. deal with it.

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
    *   Validate user-supplied URLs before scraping to block localhost and private-network targets for safety.
*   **Multimedia Interaction:**
    *   Analyze and describe attached images (with a creative "AP Photo" twist using a vision LLM).
    *   Transcribe attached audio files using a local Whisper model.
*   **Text-to-Speech (TTS):**
    *   Voice responses for bot messages via an OpenAI TTS API compatible server.
    *   Separate TTS for "thoughts" (content within `<think>...</think>` tags) vs. main response if configured.
    *   Optional hardsubbed MP4 replies when enabled via `/tts_delivery` (requires `ffmpeg` on the bot's PATH).
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
*   **FFmpeg (Required for video TTS mode):** Used to package MP3 audio into hardsubbed MP4 responses when the feature is enabled.

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
        *   Per-channel toggle for automatically running the "podcast that shit" workflow after RSS batches.
        *   Per-guild TTS delivery preferences (audio / video / both / off) persisted to `tts_delivery_modes.json`.
    *   Instances of `BotState` are passed to modules that need to access or modify this shared information (e.g., command handlers, event processors).
    *   It also manages active scheduled tasks and handles task cancellation requests.

4.  **Discord Event Handlers (`discord_events.py`)**:
    *   Contains listeners for Discord gateway events (e.g., `on_ready`, `on_message`, `on_raw_reaction_add`).
    *   The `on_message` handler is crucial for general interaction:
        *   It filters messages based on configured criteria (mentions, DMs, allowed channels/roles).
        *   Processes message content, including text, audio attachments (transcription via `audio_utils`), and image attachments.
        *   Detects URLs and uses `web_utils` to scrape content (webpages, YouTube transcripts) and generate descriptions for screenshots (via `llm_handling`).
        *   Interacts with `llm_handling.py` to get responses from the LLM, incorporating RAG context from `rag_chroma_manager.py`.
    *   Manages background tasks (`@tasks.loop`) for reminders, scheduled jobs (like recurring `/allrss`), Playwright process cleanup, and timeline pruning.

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
    *   **`audio_utils.py`**: Handles Text-to-Speech (TTS) generation and Speech-to-Text (STT).
    *   **`web_utils.py`**: Provides functions for web scraping, querying SearXNG, and parsing RSS feeds.
    *   **`utils.py`**: Contains general helper functions like text chunking, URL detection, and time string parsing.
    *   **`common_models.py`**: Defines common data structures like `MsgNode`, `TweetData`, and `GroundNewsArticle`.
    *   **`rss_cache.py`**: Manages a cache of seen RSS feed items.
    *   **`twitter_cache.py`**: Manages a cache of seen tweet IDs to avoid reprocessing.
    *   **`ground_news_cache.py`**: Manages a cache of seen Ground News article links.
    *   **`timeline_pruner.py`**: Contains logic for pruning and summarizing old chat history.
    *   **`rag_cleanup.py`**: A utility script to remove stale distilled summaries from ChromaDB.
    *   **`open_chatgpt_login.py`**: A helper script to launch a persistent Playwright browser for logging into services.
    *   **`llm_request_processor.py`**: An optional worker for processing LLM requests separately.

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
    *   https://github.com/remsky/Kokoro-FastAPI OpenAI TTS API compatible.
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
*   `ADMIN_USER_IDS` (Optional but recommended): Comma-separated Discord user IDs that are allowed to run privileged commands. These commands are disabled if this variable is not set.
*   `CHATGPT_EXPORT_IMPORT_ROOT` (Optional, Default: current working directory): Absolute or relative path that bounds where `/ingest_chatgpt_export` is allowed to read `conversations.json` files from. Paths outside this directory are rejected for safety.
*   `SYSTEM_PROMPT_FILE` (Default: `system_prompt.md`): Path to a text or markdown file containing the main system prompt that defines the bot's persona and core instructions.
*   `USER_PROVIDED_CONTEXT` (Optional): Additional global context that will be prepended to the system prompt for every LLM interaction. Useful for providing persistent high-level instructions or information.

**LLM Configuration (Provider Architecture):**

The bot now uses a provider-based architecture to manage different LLM roles (main, fast, vision). You can configure each role independently.

*   `LOCAL_SERVER_URL` (Default: `http://localhost:1234/v1`): A general-purpose base URL for your OpenAI-compatible LLM server. This is used as a fallback if a role-specific URL is not provided.
*   `LLM_API_KEY` (Optional, Default: `""`): A general-purpose API key for your LLM server. This is used as a fallback if a role-specific key is not provided.

***Main Conversational Role (`main`):***
*   `LLM_MODEL` (Default: `local-model`): The model name for the main conversational LLM.
*   `LLM_COMPLETIONS_URL` (Default: `LOCAL_SERVER_URL`): The specific API endpoint for the main LLM.
*   `LLM_API_KEY` (Optional, Default: falls back to global `LLM_API_KEY`): The specific API key for the main LLM.
*   `LLM_TEMPERATURE` (Default: `0.7`): Sampling temperature for the main LLM.
*   `LLM_SUPPORTS_JSON_MODE` (Default: `false`): Set to `true` if the main LLM supports JSON mode.
*   `LLM_SUPPORTS_LOGIT_BIAS` (Default: `true` unless `IS_GOOGLE_MODEL` is true): Set to `false` to disable logit bias.
*   `LLM_USE_RESPONSES_API` (Default: `false`): Set to `true` to use the OpenAI Responses API for this role.

***Fast/Summarization Role (`fast`):***
*   `FAST_LLM_MODEL` (Default: Same as `LLM_MODEL`): The model name for the fast/summarization LLM.
*   `FAST_LLM_COMPLETIONS_URL` (Default: `LLM_COMPLETIONS_URL`): The specific API endpoint for the fast LLM.
*   `FAST_LLM_API_KEY` (Optional, Default: falls back to global `LLM_API_KEY`): The specific API key for the fast LLM.
*   `FAST_LLM_TEMPERATURE` (Default: `LLM_TEMPERATURE`): Sampling temperature for the fast LLM.
*   `FAST_LLM_SUPPORTS_JSON_MODE` (Default: `LLM_SUPPORTS_JSON_MODE`): Set to `true` if the fast LLM supports JSON mode.
*   `FAST_LLM_SUPPORTS_LOGIT_BIAS` (Default: `LLM_SUPPORTS_LOGIT_BIAS`): Set to `false` to disable logit bias.
*   `FAST_LLM_USE_RESPONSES_API` (Default: `false`): Set to `true` to use the OpenAI Responses API for this role.

***Vision Role (`vision`):***
*   `VISION_LLM_MODEL` (Default: `llava`): The model name for the vision-capable LLM.
*   `VISION_LLM_COMPLETIONS_URL` (Default: `LLM_COMPLETIONS_URL`): The specific API endpoint for the vision LLM.
*   `VISION_LLM_API_KEY` (Optional, Default: falls back to global `LLM_API_KEY`): The specific API key for the vision LLM.
*   `VISION_LLM_TEMPERATURE` (Default: `LLM_TEMPERATURE`): Sampling temperature for the vision LLM.
*   `VISION_LLM_SUPPORTS_JSON_MODE` (Default: `LLM_SUPPORTS_JSON_MODE`): Set to `true` if the vision LLM supports JSON mode.
*   `VISION_LLM_SUPPORTS_LOGIT_BIAS` (Default: `LLM_SUPPORTS_LOGIT_BIAS`): Set to `false` to disable logit bias.
*   `VISION_LLM_USE_RESPONSES_API` (Default: `false`): Set to `true` to use the OpenAI Responses API for this role.

***General LLM Settings:***
*   `IS_GOOGLE_MODEL` (Default: `false`): Set to `true` if you are using a Google Gemini model via an OpenAI-compatible endpoint. This disables unsupported features like `logit_bias` to prevent errors. You can set this per-role as well (e.g., `FAST_IS_GOOGLE_MODEL`).
*   `GPT5_MODE` (Default: `false`): Adapts requests for GPT-5 models: forces `temperature=1.0`, removes `logit_bias`, and maps `system` messages to `developer` role.
*   `USE_RESPONSES_API` (Default: `false`): A global default for whether to use OpenAI's Responses API. Can be overridden per-role.
*   `LLM_STREAMING` (Default: `true`): Set to `false` to disable token-by-token streaming responses.
*   `LLM_REQUEST_TIMEOUT_SECONDS` (Default: `900.0`): Timeout in seconds for waiting on a response from the LLM server.
*   `RESPONSES_REASONING_EFFORT` (Optional): Controls reasoning effort for the Responses API. Options: `minimal`, `low`, `medium`, `high`.
*   `RESPONSES_VERBOSITY` (Optional): Controls verbosity for the Responses API. Options: `low`, `medium`, `high`.
*   `RESPONSES_SERVICE_TIER` (Optional): Selects service tier for the Responses API. Options: `auto`, `default`, `flex`, `priority`.
*   `OPENAI_RETRY_MAX_ATTEMPTS` (Default: `6`): Maximum retry attempts for a failed API request.
*   `OPENAI_BACKOFF_BASE_SECONDS` (Default: `1.5`): Initial delay for exponential backoff on retries.
*   `OPENAI_BACKOFF_MAX_SECONDS` (Default: `60`): Maximum delay between retries.
*   `OPENAI_BACKOFF_JITTER_SECONDS` (Default: `0.5`): Random jitter added to retry delays.
*   `MAX_MESSAGE_HISTORY` (Default: `10`): The maximum number of recent messages to include in the LLM context.
*   `MAX_COMPLETION_TOKENS` (Default: `2048`): The maximum number of tokens the LLM is allowed to generate.

**Whisper (Audio Transcription) Settings:**

*   `WHISPER_DEVICE` (Optional): Specify the device for Whisper to run on (e.g., `cuda`, `cpu`). If not set, it will auto-detect.

**Text-to-Speech (TTS) Settings:**

*   `TTS_ENABLED_DEFAULT` (Default: `true`): Whether TTS is enabled by default.
*   `TTS_API_URL` (Default: `http://localhost:8880/v1/audio/speech`): The endpoint of your OpenAI TTS API compatible server.
*   `TTS_VOICE` (Default: `af_sky+af_v0+af_nicole`): The voice to be used for TTS.
*   `TTS_SPEED` (Default: `1.3`): Playback speed multiplier for TTS audio.
*   `TTS_MAX_AUDIO_BYTES` (Default: `8388608` (8MB)): Maximum size of a generated audio clip to stay under Discord's limit.
*   `TTS_REQUEST_TIMEOUT_SECONDS` (Default: `180`): Timeout in seconds for waiting on a response from the TTS server.
*   `TTS_INCLUDE_THOUGHTS` (Default: `false`): If `true`, content within `<think>...</think>` tags will also be spoken.
*   `PODCAST_ENABLE_TTS_AFTER` (Default: `true`): If `true`, re-enables global TTS after the `/podcastthatshit` command.
*   `TTS_DELIVERY_DEFAULT` (Default: `audio`): Default delivery mode for voice replies. Options: `audio`, `video`, `both`, `off`.
*   `TTS_MAX_VIDEO_BYTES` (Default: `8388608` (8MB)): Maximum size of a generated MP4 video clip.
*   **Video Styling:** Video appearance is controlled by `TTS_VIDEO_WIDTH`, `TTS_VIDEO_HEIGHT`, `TTS_VIDEO_FPS`, `TTS_VIDEO_BACKGROUND_COLOR`, `TTS_VIDEO_TEXT_COLOR`, `TTS_VIDEO_TEXT_BOX_COLOR`, `TTS_VIDEO_TEXT_BOX_PADDING`, `TTS_VIDEO_LINE_SPACING`, `TTS_VIDEO_MARGIN`, `TTS_VIDEO_WRAP_CHARS`, `TTS_VIDEO_BLUR_SIGMA`, `TTS_VIDEO_NOISE_OPACITY`, `TTS_VIDEO_FONT_PATH`, and `TTS_VIDEO_FONT_SIZE`.

**Web Features & Scraping:**

*   `SEARX_URL` (Default: `http://192.168.1.3:9092/search`): The URL of your SearXNG instance.
*   `SEARX_PREFERENCES` (Optional, Default: `""`): Optional preferences for SearXNG engine selection. Example: `{"engines" : ["google", "wikipedia"]}`.
*   `MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT` (Default: `8000`): Maximum characters from a scraped webpage to include in the LLM prompt.
*   `MAX_IMAGE_BYTES_FOR_PROMPT` (Default: `4194304` (4MB)): Maximum size for an image to be sent to the vision LLM.
*   `NEWS_MAX_LINKS_TO_PROCESS` (Default: `15`): Maximum number of links to process for commands like `/news` or `/search`.
*   `RSS_FETCH_HOURS` (Default: `24`): How many hours back to fetch RSS entries.
*   `HEADLESS_PLAYWRIGHT` (Default: `true`): Set to `false` to run Playwright in a visible window for debugging.
*   `PLAYWRIGHT_MAX_CONCURRENCY` (Default: `1`): Maximum number of concurrent Playwright browser instances.
*   `SCRAPE_SCROLL_ATTEMPTS` (Default: `5`): How many times to scroll down a page to load dynamic content.
*   `GROUND_NEWS_SEE_MORE_CLICKS` (Default: `10`): How many times to click "See more stories" on Ground News.
*   `GROUND_NEWS_CLICK_DELAY_SECONDS` (Default: `1.0`): Delay between "See more" clicks.
*   `GROUND_NEWS_ARTICLE_DELAY_SECONDS` (Default: `5.0`): Delay between scraping each Ground News article.
*   `PLAYWRIGHT_CLEANUP_INTERVAL_MINUTES` (Default: `5`): How often the background task runs to clean up idle Playwright processes.
*   `PLAYWRIGHT_IDLE_CLEANUP_THRESHOLD_MINUTES` (Default: `10`): How long Playwright must be idle before its processes are terminated.
*   `SCRAPE_LOCK_TIMEOUT_SECONDS` (Default: `60`): How long to wait when acquiring the scraping lock before giving up.

**ChromaDB (RAG & Long-Term Memory):**

*   `CHROMA_DB_PATH` (Default: `./chroma_data`): The path where ChromaDB will store its data.
*   `CHROMA_COLLECTION_NAME` (Default: `long_term_memory`): Collection for raw conversation exchanges.
*   `CHROMA_DISTILLED_COLLECTION_NAME` (Default: `distilled_chat_summaries`): Collection for distilled conversation summaries used in RAG.
*   `CHROMA_NEWS_SUMMARY_COLLECTION_NAME` (Default: `news_summaries`): Collection for news article summaries.
*   `CHROMA_RSS_SUMMARY_COLLECTION_NAME` (Default: `rss_summaries`): Collection for RSS feed item summaries.
*   `CHROMA_TIMELINE_SUMMARY_COLLECTION_NAME` (Default: `timeline_summaries`): Collection for summaries of pruned, older chat history.
*   `CHROMA_TWEETS_COLLECTION_NAME` (Default: `tweets_collection`): Collection for fetched tweets.
*   `CHROMA_ENTITIES_COLLECTION_NAME` (Default: `entities_collection`): Collection for extracted entities.
*   `CHROMA_RELATIONS_COLLECTION_NAME` (Default: `relations_collection`): Collection for extracted relationships.
*   `CHROMA_OBSERVATIONS_COLLECTION_NAME` (Default: `observations_collection`): Collection for extracted observations/facts.
*   `RAG_NUM_DISTILLED_SENTENCES_TO_FETCH` (Default: `3`): Number of distilled summaries to fetch for RAG context.
*   `RAG_NUM_COLLECTION_DOCS_TO_FETCH` (Default: `3`): Number of documents to fetch from other collections (news, etc.) for RAG.
*   `RAG_MAX_FULL_CONVO_CHARS` (Default: `20000`): Character limit for full conversation logs used in RAG.
*   `RAG_MAX_DATE_RANGE_DOCS` (Default: `15`): Maximum documents to include for date-range RAG retrieval.
*   `ENABLE_MEMORY_MERGE` (Default: `false`): Set to `true` to enable the experimental memory merging feature.
*   `TIMELINE_PRUNE_DAYS` (Default: `365`): Age in days at which chat history is pruned and summarized.

**Discord Embed & Streaming Settings:**

*   `EMBED_COLOR_INCOMPLETE` (Default: `0xFFA500` - Orange): Hex color code for embeds indicating an in-progress operation.
*   `EMBED_COLOR_COMPLETE` (Default: `0x00FF00` - Green): Hex color code for embeds indicating a completed operation.
*   `EMBED_COLOR_ERROR` (Default: `0xFF0000` - Red): Hex color code for embeds indicating an error.
*   `EMBED_MAX_LENGTH` (Default: `4096`): Maximum character length for a single Discord embed.
*   `STREAM_EDIT_THROTTLE_SECONDS` (Default: `0.1`): Minimum time between edits when streaming LLM responses. This directly controls the `sleep` duration and is related to the hardcoded `EDITS_PER_SECOND` of 1.3.

**Image Processing (Attachments & Vision):**

*   `MAX_IMAGES_PER_MESSAGE` (Default: `1`): The maximum number of images per message to be processed by the vision LLM.

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
    npx playwright install
    ```

---

## 8. Slash Commands

DiscordSam offers a variety of slash commands for diverse functionalities, grouped by category.

### Content & Web Commands

*   **`/news <topic>`**
    *   **Purpose:** Generates a news briefing on a specified topic.
    *   **Arguments:** `topic` (Required).
    *   **Behavior:** Searches for news articles, scrapes the top results, summarizes each, and then synthesizes a final briefing.

*   **`/search <query>`**
    *   **Purpose:** Performs a web search and summarizes the findings.
    *   **Arguments:** `query` (Required).
    *   **Behavior:** Queries SearXNG, scrapes top results, summarizes each, and synthesizes a final summary.

*   **`/rss [feed_url] [feed_url_manual] [limit]`**
    *   **Purpose:** Fetches and summarizes new entries from an RSS feed.
    *   **Arguments:** `feed_url` (Optional, from preset list), `feed_url_manual` (Optional, for any URL), `limit` (Optional, Default: 20).
    *   **Behavior:** Fetches new entries from a feed, scrapes and summarizes the linked articles, and posts them. If the channel has auto-podcast enabled, it will follow up with a podcast-style narration.

*   **`/allrss [limit]`**
    *   **Purpose:** Fetches new entries from all default RSS feeds.
    *   **Arguments:** `limit` (Optional, Default: 20).
    *   **Behavior:** Processes all preset RSS feeds, posting summaries in batches. Can trigger auto-podcast for each batch.

*   **`/gettweets [username] [preset_user] [limit]`**
    *   **Purpose:** Fetches and summarizes recent tweets from an X/Twitter user.
    *   **Arguments:** `username` or `preset_user` (Required), `limit` (Optional, Default: 50).
    *   **Behavior:** Scrapes tweets, displays them raw (including descriptions of any images), and then generates a summary of the user's recent activity.

*   **`/homefeed [limit]`**
    *   **Purpose:** Fetches and summarizes tweets from the logged-in X/Twitter home timeline.
    *   **Arguments:** `limit` (Optional, Default: 30).
    *   **Behavior:** Scrapes the home timeline, displays raw tweets (with image descriptions), and generates a summary. Requires a logged-in Playwright profile.

*   **`/alltweets [limit]`**
    *   **Purpose:** Fetches and summarizes tweets from all default X/Twitter accounts.
    *   **Arguments:** `limit` (Optional, Default: 50).
    *   **Behavior:** Iterates through all preset accounts, scraping and summarizing tweets for each.

*   **`/groundnews [limit]`**
    *   **Purpose:** Scrapes and summarizes new articles from your Ground News "My Feed".
    *   **Arguments:** `limit` (Optional, Default: 50).
    *   **Behavior:** Scrapes your personal feed, summarizes new articles, and posts them. Requires a logged-in Playwright profile.

*   **`/groundtopic <topic> [limit]`**
    *   **Purpose:** Scrapes and summarizes new articles from a specified Ground News topic page.
    *   **Arguments:** `topic` (Required, from preset list), `limit` (Optional, Default: 50).
    *   **Behavior:** Scrapes the selected topic page for new articles and posts summaries.

### Creative & Fun Commands

*   **`/ap <image> [user_prompt]`**
    *   **Purpose:** Describes an image in the style of an AP photo caption, with a humorous celebrity twist.
    *   **Arguments:** `image` (Required), `user_prompt` (Optional).
    *   **Behavior:** Uses a vision LLM to write a creative caption, inserting a randomly chosen celebrity as the subject.

*   **`/roast <url>`**
    *   **Purpose:** Generates a comedy roast of a webpage.
    *   **Arguments:** `url` (Required).
    *   **Behavior:** Scrapes the URL's content and uses an LLM to generate a witty, biting roast.

*   **`/pol <statement>`**
    *   **Purpose:** Generates a sarcastic, troll-like political commentary.
    *   **Arguments:** `statement` (Required).
    *   **Behavior:** Uses an LLM with a specialized prompt to generate a snarky, humorous take on the provided statement.

*   **`/podcastthatshit`**
    *   **Purpose:** Generates a podcast-style monologue based on the recent conversation history.
    *   **Arguments:** None.
    *   **Behavior:** Injects "Podcast that shit" into the conversation history and uses the LLM to create a spoken-word style narration of the preceding context.

### Utility & Admin Commands

*   **`/remindme <time_duration> <reminder_message>`**
    *   **Purpose:** Sets a personal reminder.
    *   **Arguments:** `time_duration` (Required, e.g., "1h30m"), `reminder_message` (Required).
    *   **Behavior:** The bot will mention you with your reminder message in the same channel after the specified duration.

*   **`/clearhistory`**
    *   **Purpose:** Clears the bot's short-term memory for the current channel.
    *   **Arguments:** None.
    *   **Access:** Requires 'Manage Messages' permission.
    *   **Behavior:** Deletes the recent conversation history from the bot's state, but does not affect the long-term RAG database.

*   **`/ingest_chatgpt_export <file_path>`**
    *   **Purpose:** Ingests a `conversations.json` file from a ChatGPT data export into the bot's RAG memory.
    *   **Arguments:** `file_path` (Required).
    *   **Access:** Admin-only (`ADMIN_USER_IDS`).

*   **`/analytics`**
    *   **Purpose:** Displays a snapshot of the bot's operational metrics.
    *   **Arguments:** None.
    *   **Access:** Admin-only (`ADMIN_USER_IDS`).
    *   **Output:** An embed showing message cache counts, active reminders, Playwright usage, and ChromaDB collection sizes.

*   **`/memoryinspector <scope> [limit]`**
    *   **Purpose:** Allows admins to review the most recent memories stored in ChromaDB for the current channel.
    *   **Arguments:** `scope` (Required: "Distilled summaries" or "Full conversation logs"), `limit` (Optional, Default: 5).
    *   **Access:** Admin-only (`ADMIN_USER_IDS`).

*   **`/dbcounts`**
    *   **Purpose:** Displays the number of documents in each ChromaDB collection.
    *   **Arguments:** None.
    *   **Access:** Admin-only.

*   **`/pruneitems <limit>`**
    *   **Purpose:** Manually triggers the summarization and pruning of the oldest chat history entries.
    *   **Arguments:** `limit` (Required, 1-10).
    *   **Access:** Admin-only.

### Scheduling Commands (Admin-only)

*   **`/schedule_allrss <interval_minutes> [limit]`**
    *   **Purpose:** Schedules a recurring job to run `/allrss` in the current channel.
    *   **Arguments:** `interval_minutes` (Required, min 15), `limit` (Optional, Default: 10).

*   **`/schedules`**
    *   **Purpose:** Lists all active scheduled jobs for the current channel.

*   **`/unschedule <schedule_id>`**
    *   **Purpose:** Removes a scheduled job by its ID.

*   **`/cancel`**
    *   **Purpose:** Cancels the current long-running task (like `/allrss`) in the channel.

### TTS & Voice Commands

*   **`/tts_delivery <mode>`**
    *   **Purpose:** Sets the TTS delivery preference for the server.
    *   **Arguments:** `mode` (Required: `audio`, `video`, `both`, or `off`).
    *   **Behavior:** Changes how voice replies are sent (MP3, MP4 with subtitles, etc.).

*   **`/tts_thoughts <enabled>`**
    *   **Purpose:** Controls whether the bot's internal `<think>` messages are included in TTS output.
    *   **Arguments:** `enabled` (Required: `true` or `false`).

*   **`/rss_podcast <enabled>`**
    *   **Purpose:** Toggles whether the bot automatically runs `/podcastthatshit` after RSS commands.
    *   **Arguments:** `enabled` (Required: `true` or `false`).

---

## 9. Running the Bot

1.  **Ensure your local servers are running:**
    *   Your LLM server (e.g., LM Studio, Ollama) must be active and accessible at the `LOCAL_SERVER_URL`. Ensure the correct models specified in your `.env` file (e.g., `LLM`, `VISION_LLM_MODEL`) are loaded and available on this server.
    *   If `TTS_ENABLED_DEFAULT` is `true`, your TTS server must be running and accessible at `TTS_API_URL`.
    *   If you enable MP4 TTS delivery (via `/tts_delivery video` or `both`), verify that `ffmpeg` is installed and available on the system PATH.
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

*   **`timeline_pruner.py`**: Summarizes chat logs older than `TIMELINE_PRUNE_DAYS` and stores the results in the timeline summary collection. This runs daily as a background task but can also be executed manually.
*   **`rag_cleanup.py`**: Removes distilled summary entries that reference conversation IDs no longer present in the main history collection.
*   **`rag_cleanup_entities.py`**: A destructive utility to completely delete the entities collection (`CHROMA_ENTITIES_COLLECTION_NAME`) from ChromaDB. Use with caution.
*   **`open_chatgpt_login.py`**: Launches a persistent Playwright browser (`.pw-profile`) so you can log into services like ChatGPT or Ground News once and reuse the saved cookies for scraping.
*   **`llm_request_processor.py`**: An experimental worker that processes queued LLM requests separately from the Discord bot.

---

## 12. Troubleshooting

*   **Playwright Processes:** If Playwright processes remain running after scraping, the bot has a built-in task that attempts to clean them up periodically. You can also manually kill these processes.
*   **Model Not Found Errors:** Ensure the model names in your `.env` file (`LLM_MODEL`, `VISION_LLM_MODEL`, etc.) exactly match the models available on your LLM server.
*   **Connection Errors:** Verify that all server URLs (`LOCAL_SERVER_URL`, `TTS_API_URL`, `SEARX_URL`) are correct and the services are running and accessible.
*   **Permissions:** If slash commands don't appear or fail, check the bot's permissions in your Discord server. It needs permissions to read/send messages, use embeds, attach files, and use slash commands. Admin commands require the user's ID to be in `ADMIN_USER_IDS`.
*   **TTS Video Issues:** If you encounter problems with the MP4 video delivery for TTS, refer to the `TTS_VIDEO_DEBUG.md` file for a detailed guide on the generation process and common failure points.

---

## 13. Potential Future Directions

DiscordSam is a project with significant potential for growth. Here are some ideas for future enhancements:

*   **Enhanced RAG Strategies:** More-sophisticated summarization, knowledge graph integration, hybrid search, advanced query understanding for RAG.
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

## Support

If the tool is helpful, consider supporting it on [Ko-fi](https://ko-fi.com/gille).
