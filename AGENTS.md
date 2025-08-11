# Guidelines for AI Agents Working on DiscordSam

Welcome, fellow AI! This document provides some general guidelines to help you effectively contribute to the DiscordSam project.

## 1. Understanding the Project

*   **Core Goal:** DiscordSam aims to be a highly intelligent, context-aware, and multi-functional Discord bot. It leverages local LLMs, RAG for memory, web scraping, and multimedia processing.
*   **Key Documentation:** Before making changes, please familiarize yourself with the `README.md`. It contains crucial information about the project's overview, features, architecture, configuration, and how to run the bot.
*   **Modularity:** The codebase is designed to be modular. Understand the role of each file (as outlined in the README's architecture section) before modifying it.

## 2. Development Practices

*   **Python Version:** The project uses Python 3.10+. Please ensure your changes are compatible.
*   **Coding Style:** Adhere to PEP 8 Python style guidelines. Aim for clear, readable, and well-commented code where necessary.
*   **Type Hinting:** The project is progressively adopting type hinting. Please add type hints to new functions and methods, and update existing ones where appropriate. This greatly improves code maintainability.
*   **Dependencies:**
    *   All Python dependencies are managed in `requirements.txt`.
    *   If you add a new dependency, ensure it's added to `requirements.txt`.
    *   For web scraping, Playwright is used. Remember that `playwright install` is needed to get browser binaries.
*   **Configuration (`.env` and `config.py`):**
    *   All configurable parameters should be managed via environment variables, loaded by `config.py` from an `.env` file.
    *   Do NOT hardcode API keys, tokens, or sensitive URLs directly in the Python files.
    *   If you add a new configurable parameter, ensure it's added to `config.py` with a sensible default and documented in the `README.md`'s "Configuration" section.
*   **Error Handling:** Implement robust error handling. Use `try-except` blocks appropriately and log errors clearly using the `logging` module. Provide informative error messages to the user in Discord where applicable.
*   **Logging:**
    *   Use the `logging` module for all logging. Get a logger instance at the top of your module: `logger = logging.getLogger(__name__)`.
    *   Logging is configured in `main_bot.py`. Logs are output to the console.
    *   Use appropriate log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

## 3. Working with Key Components

*   **LLM Interaction (`llm_handling.py`):**
    *   All direct calls to LLMs should ideally go through functions in this module.
    *   When adding new LLM-powered features, consider if existing streaming or prompt-building functions can be reused or extended.
*   **RAG & ChromaDB (`rag_chroma_manager.py`):**
    *   This module handles all interactions with ChromaDB.
    *   If you add new types of data to be stored or retrieved for RAG, update the functions here.
    *   Pay attention to the structure of metadata when adding documents to ChromaDB.
*   **Slash Commands (`discord_commands.py`):**
    *   New slash commands should be defined here and registered in `setup_commands()`.
    *   Ensure commands provide clear feedback to the user (e.g., deferring long operations, updating with progress).
*   **Event Handlers (`discord_events.py`):**
    *   New event handlers or background tasks go here and are registered in `setup_events_and_tasks()`.
*   **State Management (`state.py`):**
    *   Use the `BotState` object for managing shared, mutable state like conversation history or reminders. Ensure methods are async-safe if they modify state.

## 4. Documentation

*   **README.md:** This is the primary source of truth for users and developers. If you:
    *   Add a new feature (especially a slash command).
    *   Change existing functionality significantly.
    *   Add or change a configuration variable.
    *   Introduce new dependencies or setup steps.
    *   **You MUST update `README.md` accordingly.**
*   **Docstrings and Comments:** Write clear docstrings for public functions/classes and add comments to explain complex logic.

## 5. Testing (Aspirational)

*   While the project currently lacks a comprehensive test suite, contributions towards adding unit tests or integration tests would be highly valuable.
*   **Environment Note:** The execution environment for agents may have disk space limitations. Running tests that require downloading large dependencies (e.g., PyTorch for model inference) may fail. Please be mindful of this when considering adding or running comprehensive tests.
*   When making changes, manually test your features thoroughly. Consider edge cases and potential failure points. For example:
    *   What happens if an external API is down?
    *   How does the command handle invalid user input?
    *   Does it work correctly with empty or very long inputs?

## 6. Submitting Changes

*   When using tools to submit changes, provide clear and concise commit messages that describe the changes made.
*   If your changes are significant, consider outlining them in a pull request description (if applicable to your workflow).

By following these guidelines, you'll help keep DiscordSam robust, maintainable, and easy to understand for both human and AI developers. Thank you for your contribution!
