# Project Overview: DiscordSam

## 1. Introduction and Purpose

DiscordSam is an advanced, context-aware Discord bot designed to provide intelligent, detailed, and rational responses. Its primary purpose is to serve as a highly interactive and knowledgeable assistant within a Discord server, capable of engaging in complex conversations, accessing and processing external information, and performing various tasks through slash commands. The bot leverages local Large Language Models (LLMs), Retrieval Augmented Generation (RAG) for long-term memory, web scraping, multimedia understanding, and Text-to-Speech (TTS) to create a rich and dynamic user experience.

## 2. Core Functionalities

*   **Intelligent Conversations:** Powered by local LLMs (e.g., LM Studio, Ollama compatible), allowing for nuanced and context-aware dialogue.
*   **Long-Term Memory (RAG):** Utilizes ChromaDB to store and retrieve conversation history, enabling the bot to recall past interactions and provide more informed responses. Conversations are "distilled" into keyword-rich sentences for efficient semantic search.
*   **Web Capabilities:**
    *   Scrapes content from websites and YouTube transcripts.
    *   Performs web searches via a local SearXNG instance and summarizes results.
    *   Fetches and summarizes recent tweets.
*   **Multimedia Interaction:**
    *   Analyzes and describes attached images (often with a creative "AP Photo" style).
    *   Transcribes attached audio files using a local Whisper model.
*   **Text-to-Speech (TTS):** Provides voice responses for bot messages, enhancing interactivity.
*   **Slash Commands:** A suite of commands for various functionalities including reminders, web searches, content generation (roasts, political commentary), tweet fetching, image analysis, and history management.
*   **High Configurability:** Most settings are managed via a `.env` file, allowing for easy customization of LLM endpoints, API keys, and bot behavior.

## 3. Overall Architecture

DiscordSam is built with a modular Python codebase:

*   **Main Bot (`main_bot.py`):** Entry point of the application, handles bot initialization, connection to Discord, and loading of command/event handlers.
*   **LLM Handling (`llm_handling.py`):** Manages interactions with the LLM, including prompt construction, streaming responses, and vision capabilities.
*   **RAG Chroma Manager (`rag_chroma_manager.py`):** Handles all aspects of the RAG system, including initializing ChromaDB, ingesting new conversations, distilling conversations for summary, retrieving relevant context, and synthesizing context for prompts.
*   **Discord Commands (`discord_commands.py`):** Defines and implements all slash commands available to the bot.
*   **Discord Events (`discord_events.py`):** Manages event handling, such as `on_message` for direct interactions and `on_ready` for setup tasks.
*   **Configuration (`config.py`):** Loads and provides access to environment variables.
*   **State Management (`state.py`):** Manages the bot's short-term memory and operational state (e.g., message history per channel).
*   **Utilities (`utils.py`, `audio_utils.py`, `web_utils.py`):** Provide helper functions for various tasks like text chunking, TTS processing, and web scraping.
*   **Common Models (`common_models.py`):** Defines shared data structures, like `MsgNode` for message representation.

The bot integrates with external services/local servers for:
*   LLM processing (OpenAI API compatible).
*   TTS generation (OpenAI TTS API compatible or custom).
*   Web search (SearXNG).
*   Audio transcription (Whisper).

## 4. Key Technologies Used

*   **Python 3.10+**
*   **discord.py:** Library for creating Discord bots.
*   **OpenAI API (compatible):** For interacting with local LLMs.
*   **ChromaDB:** Vector database for Retrieval Augmented Generation.
*   **Playwright:** For web scraping.
*   **Whisper (OpenAI):** For audio transcription.
*   **Asyncio:** For asynchronous programming.
*   **LM Studio, Ollama (examples):** Local LLM server software.
*   **SearXNG (optional):** Local metasearch engine.

## 5. Potential Future Directions

This section outlines potential areas for future development and enhancement of DiscordSam:

*   **Enhanced RAG Strategies:**
    *   **More Sophisticated Summarization/Distillation:** Explore advanced techniques for creating conversation summaries beyond simple sentence extraction, potentially using the LLM for more abstractive summarization during ingestion.
    *   **Knowledge Graph Integration:** Connect ChromaDB with a knowledge graph to represent and query relationships between entities and concepts discussed in conversations.
    *   **Hybrid Search:** Combine semantic search from ChromaDB with traditional keyword search for more robust retrieval.
    *   **Context Tiering:** Implement different tiers of memory (e.g., short-term active, medium-term summarized, long-term archived) with different retrieval strategies.
*   **Expanded LLM/Multimodal Support:**
    *   **Support for More LLM Backends:** Add native support or adapters for other popular LLM hosting solutions beyond the generic OpenAI API compatibility.
    *   **True Multimodal Inputs/Outputs:** Beyond image description, explore models that can natively reason over combined text, image, and audio inputs, or generate multimodal outputs.
    *   **Function Calling/Agentic Capabilities:** Integrate more robust function calling capabilities with the LLM to allow it to interact with external tools and APIs more autonomously.
*   **Slash Command Enhancements:**
    *   **New Creative Tools:** Add commands for story generation, poetry, code snippets, etc.
    *   **Personalization Features:** Commands for users to set personal preferences, aliases, or custom knowledge snippets for the bot to remember about them.
    *   **Integration with External Services:** Commands to interact with services like calendars, to-do lists, or project management tools.
*   **Improved Error Handling and Resilience:**
    *   **More Granular Error Reporting:** Provide users with clearer feedback when a command fails or an internal error occurs.
    *   **Graceful Degradation:** Allow the bot to function with reduced capabilities if some external services (e.g., TTS, SearXNG) are unavailable.
*   **Sophisticated Context Management:**
    *   **User-Specific Context Windows:** Allow the bot to maintain distinct conversational contexts for different users, even within the same channel.
    *   **Thread-Specific Context:** Ensure context is properly isolated and managed within Discord threads.
    *   **Dynamic Context Adjustment:** Allow the LLM to dynamically request more or less context based on the query's complexity.
*   **UI/UX Enhancements:**
    *   **Interactive Embeds/Buttons:** Use more of Discord's UI elements (buttons, select menus) for command options and feedback.
    *   **Dashboard/Web UI (Ambitious):** A web interface for managing bot settings, viewing RAG store statistics, or curating memories.
*   **Observability and Logging:**
    *   **Structured Logging:** Implement more structured logging for easier parsing and analysis.
    *   **Performance Metrics:** Track key performance indicators like LLM response times, RAG retrieval accuracy, and command usage frequency.
    *   **Admin Commands:** Add slash commands for administrators to check bot health, reload configurations, or inspect RAG entries.
*   **Security and Permissions:**
    *   **Fine-grained Permission Control:** More detailed control over who can use which commands or access certain bot features.
    *   **Data Privacy Controls:** Features for users to manage or delete their data stored in the RAG system.

These directions aim to make DiscordSam even more intelligent, versatile, and user-friendly.
