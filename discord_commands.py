import logging
import discord
from discord import app_commands  # type: ignore
from discord.ext import commands  # For bot type hint
import os
import base64
import random
import asyncio
from typing import Any, Optional, List  # Keep existing imports
from datetime import datetime

# Bot services and utilities
from config import config
from state import BotState
from common_models import MsgNode

from llm_handling import (
    _build_initial_prompt_messages,
    stream_llm_response_to_interaction
)
from rag_chroma_manager import (
    retrieve_and_prepare_rag_context,
    parse_chatgpt_export,
    store_chatgpt_conversations_in_chromadb,
    store_news_summary,
    ingest_conversation_to_chromadb,
)
from web_utils import (
    scrape_website,
    query_searx,
    scrape_latest_tweets,
    fetch_rss_entries
)
from audio_utils import send_tts_audio
from utils import parse_time_string_to_delta, chunk_text
from rss_cache import load_seen_entries, save_seen_entries

logger = logging.getLogger(__name__)

# Default RSS feeds users can choose from with the /rss command
DEFAULT_RSS_FEEDS = [
    ("CBS Main", "https://www.cbsnews.com/latest/rss/main"),
    ("CBS U.S.", "https://www.cbsnews.com/latest/rss/us"),
    ("CBS World", "https://www.cbsnews.com/latest/rss/world"),
    ("Congressional Bills", "https://www.govinfo.gov/rss/bills.xml"),
    ("CBS MoneyWatch", "https://www.cbsnews.com/latest/rss/moneywatch"),
    ("DoD News", "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?max=10&ContentType=1&Site=945"),
    ("DoD Releases", "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=9&Site=945&max=10"),
    ("CBP Media Releases", "https://www.cbp.gov/rss/newsroom/media-releases"),
    ("NBC News", "https://feeds.nbcnews.com/nbcnews/public/news"),
    ("CBS Entertainment", "https://www.cbsnews.com/latest/rss/entertainment"),
    ("Google News", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"),
    ("CNN Top Stories", "http://rss.cnn.com/rss/cnn_topstories.rss"),
    ("NPR News", "http://www.npr.org/rss/rss.php?id=1001"),
    ("BBC Americas", "http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml?edition=int"),
    ("KSL Local Stories", "https://www.ksl.com/rss/news"),
    ("NPR Education", "http://www.npr.org/rss/rss.php?id=1013"),
    ("NPR Technology", "http://www.npr.org/rss/rss.php?id=1019"),
    ("Hacker News", "https://news.ycombinator.com/rss"),
    ("World News - The Guardian", "https://www.theguardian.com/world/rss"),
    ("US News - The Guardian", "https://www.theguardian.com/us/rss"),
    ("Gizmodo", "https://gizmodo.com/feed"),
    ("Huffpost Politics", "https://chaski.huffpost.com/us/auto/vertical/politics"),
    ("Huffpost US News", "https://chaski.huffpost.com/us/auto/vertical/us-news"),
    ("Fox News Politics", "https://moxie.foxnews.com/google-publisher/politics.xml"),
    ("Fox News World", "https://moxie.foxnews.com/google-publisher/world.xml"),
]

# Default Twitter users for the /gettweets command dropdown
DEFAULT_TWITTER_USERS = [
    "acyn",
    "basedmikelee",
    "rapidresponse47",
    "dhsgov",
    "stephenm",
    "trumpdailyposts",
    "whitehouse",
    "sec_noem",
    "secdef",
    "petehegseth",
    "presssec",
    "unusual_whales",
    "ap",
    "cspan",
    "thehill",
    "atrupar",
    "kslcom",
    "wutangkids",
    "sama",
    "openai",
    "openainewsroom",
    "openaidevs",
    "jdvance",
]
# Module-level globals to store instances passed from main_bot.py
bot_instance: Optional[commands.Bot] = None
llm_client_instance: Optional[Any] = None
bot_state_instance: Optional[BotState] = None


def setup_commands(bot: commands.Bot, llm_client_in: Any, bot_state_in: BotState):
    global bot_instance, llm_client_instance, bot_state_instance
    bot_instance = bot
    llm_client_instance = llm_client_in
    bot_state_instance = bot_state_in

    if not bot_instance or not llm_client_instance or not bot_state_instance:
        logger.critical("Bot, LLM client, or BotState not properly initialized in setup_commands. Commands may fail.")
        return

    @bot_instance.tree.command(name="news", description="Generates a news briefing on a given topic.")
    @app_commands.describe(
        topic="The news topic you want a briefing on."
    )
    async def news_slash_command(interaction: discord.Interaction, topic: str):
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("/news command: Bot components not ready.")
            await interaction.response.send_message("Bot components not ready. Cannot generate news.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)

        # Update Playwright usage time
        if bot_state_instance and hasattr(bot_state_instance, 'update_last_playwright_usage_time'):
            await bot_state_instance.update_last_playwright_usage_time() # Made awaitable
            logger.debug(f"Updated last_playwright_usage_time via bot_state_instance for /news command")

        search_topic = f"news {topic}"

        initial_embed = discord.Embed(
            title=f"News Briefing: {topic}",
            description=f"Gathering news articles for '{search_topic}'...",
            color=config.EMBED_COLOR["incomplete"]
        )
        await interaction.edit_original_response(embed=initial_embed)

        try:
            max_articles_to_process = config.NEWS_MAX_LINKS_TO_PROCESS
            logger.info(f"/news command for topic '{topic}' (searching for '{search_topic}') by {interaction.user.name}, max_articles={max_articles_to_process}")

            search_results = await query_searx(search_topic)

            if not search_results:
                error_embed = discord.Embed(
                    title=f"News Briefing: {topic}",
                    description=f"Sorry, I couldn't find any initial search results for '{search_topic}'.",
                    color=config.EMBED_COLOR["error"]
                )
                await interaction.edit_original_response(embed=error_embed)
                return

            num_to_process = min(len(search_results), max_articles_to_process)
            article_summaries_for_briefing: List[str] = []
            processed_urls = set()

            for i in range(num_to_process):
                result = search_results[i]
                article_url = result.get('url')
                article_title = result.get('title', 'Untitled Article')

                if not article_url or article_url in processed_urls:
                    logger.info(f"Skipping duplicate or invalid URL: {article_url}")
                    continue
                processed_urls.add(article_url)

                update_embed = discord.Embed(
                    title=f"News Briefing: {topic}",
                    description=f"Processing article {i+1}/{num_to_process}: Scraping '{article_title}'...",
                    color=config.EMBED_COLOR["incomplete"]
                )
                await interaction.edit_original_response(embed=update_embed)

                if bot_state_instance and hasattr(bot_state_instance, 'update_last_playwright_usage_time'):
                    await bot_state_instance.update_last_playwright_usage_time() # Made awaitable

                scraped_content = await scrape_website(article_url)

                if not scraped_content or "Failed to scrape" in scraped_content or "Scraping timed out" in scraped_content:
                    logger.warning(f"Failed to scrape '{article_title}' from {article_url}. Reason: {scraped_content}")
                    article_summaries_for_briefing.append(f"Source: {article_title} ({article_url})\nSummary: [Could not retrieve content for summarization]\n\n")
                    continue

                update_embed.description = f"Processing article {i+1}/{num_to_process}: Summarizing '{article_title}'..."
                await interaction.edit_original_response(embed=update_embed)

                summarization_prompt = (
                    f"You are an expert news summarizer. Please read the following article content, "
                    f"which was found when searching for the topic '{search_topic}'. Extract the key factual"
                    f"news points and provide a concise summary (2-4 sentences) relevant to this topic. "
                    f"Focus on who, what, when, where, and why if applicable. Avoid opinions or speculation not present in the text.\n\n"
                    f"Article Title: {article_title}\n"
                    f"Article Content:\n{scraped_content[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT*2]}"
                )

                try:
                    summary_response = await llm_client_instance.chat.completions.create(
                        model=config.FAST_LLM_MODEL,
                        messages=[{"role": "user", "content": summarization_prompt}],
                        max_tokens=250,
                        temperature=0.3,
                        stream=False
                    )
                    if summary_response.choices and summary_response.choices[0].message and summary_response.choices[0].message.content:
                        article_summary = summary_response.choices[0].message.content.strip()
                        logger.info(f"Summarized '{article_title}': {article_summary[:100]}...")
                        article_summaries_for_briefing.append(f"Source: {article_title} ({article_url})\nSummary: {article_summary}\n\n")

                        store_news_summary(topic=topic, url=article_url, summary_text=article_summary)
                    else:
                        logger.warning(f"LLM summarization returned no content for '{article_title}'.")
                        article_summaries_for_briefing.append(f"Source: {article_title} ({article_url})\nSummary: [AI summarization failed or returned no content]\n\n")
                except Exception as e_summ:
                    logger.error(f"Error during LLM summarization for '{article_title}': {e_summ}", exc_info=True)
                    article_summaries_for_briefing.append(f"Source: {article_title} ({article_url})\nSummary: [Error during AI summarization]\n\n")

            if not article_summaries_for_briefing:
                error_embed = discord.Embed(
                    title=f"News Briefing: {topic}",
                    description=f"Could not process any articles to generate a briefing for '{topic}'.",
                    color=config.EMBED_COLOR["error"]
                )
                await interaction.edit_original_response(embed=error_embed)
                return

            update_embed.description = "All articles processed. Generating final news briefing..."
            await interaction.edit_original_response(embed=update_embed)

            combined_summaries_text = "".join(article_summaries_for_briefing)

            final_briefing_prompt_content = (
                f"You are Sam, a news anchor delivering a concise and objective briefing. "
                f"The following are summaries of news articles related to the topic: '{topic}'. "
                f"Synthesize this information into a coherent news report. Start with a clear headline for the briefing. "
                f"Present the key developments and information from the summaries. Maintain a neutral and informative tone. "
                f"Do not add external information or opinions not present in the provided summaries.\n\n"
                f"Topic: {topic}\n\n"
                f"Collected Article Summaries:\n{combined_summaries_text}"
            )

            user_msg_node_for_briefing = MsgNode("user", final_briefing_prompt_content, name=str(interaction.user.id))

            rag_query_for_briefing = f"news briefing about {topic}"
            synthesized_rag_context_for_briefing = await retrieve_and_prepare_rag_context(llm_client_instance, rag_query_for_briefing)

            prompt_nodes_for_briefing = await _build_initial_prompt_messages(
                user_query_content=final_briefing_prompt_content,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_rag_context_for_briefing,
                max_image_history_depth=0
            )

            await stream_llm_response_to_interaction(
                interaction=interaction,
                llm_client=llm_client_instance,
                bot_state=bot_state_instance,
                user_msg_node=user_msg_node_for_briefing,
                prompt_messages=prompt_nodes_for_briefing,
                title=f"News Briefing: {topic}",
                synthesized_rag_context_for_display=synthesized_rag_context_for_briefing,
                bot_user_id=bot_instance.user.id
            )

        except Exception as e:
            logger.error(f"Error in /news command for topic '{topic}': {e}", exc_info=True)
            error_embed = discord.Embed(
                title=f"News Briefing: {topic}",
                description=f"An unexpected error occurred while generating your news briefing: {str(e)[:1000]}",
                color=config.EMBED_COLOR["error"]
            )
            try:
                if interaction.response.is_done():
                     await interaction.edit_original_response(embed=error_embed, content=None) # Clear content
                else:
                    # This case should ideally not be hit if defer() was successful
                    await interaction.response.send_message(embed=error_embed, ephemeral=True)
            except discord.HTTPException:
                await interaction.followup.send(embed=error_embed, ephemeral=True)


    @bot_instance.tree.command(name="ingest_chatgpt_export", description="Ingests a conversations.json file from a ChatGPT export.")
    @app_commands.describe(file_path="The full local path to your conversations.json file.")
    @app_commands.checks.has_permissions(manage_messages=True)
    async def ingest_chatgpt_export_command(interaction: discord.Interaction, file_path: str):
        if not llm_client_instance:
            logger.error("ingest_chatgpt_export_command: llm_client_instance is None.")
            await interaction.response.send_message("LLM client not available. Cannot process.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        logger.info(f"Ingestion of ChatGPT export file '{file_path}' initiated by {interaction.user.name} ({interaction.user.id}).")

        if not os.path.exists(file_path):
            await interaction.followup.send(f"Error: File not found at the specified path: `{file_path}`", ephemeral=True)
            return

        try:
            parsed_conversations = parse_chatgpt_export(file_path)
            if not parsed_conversations:
                await interaction.followup.send("Could not parse any conversations from the file. It might be empty or in an unexpected format.", ephemeral=True)
                return

            count = await store_chatgpt_conversations_in_chromadb(llm_client_instance, parsed_conversations)
            await interaction.followup.send(f"Successfully processed and stored {count} conversations (with distillations) from the export file into ChromaDB.", ephemeral=True)
        except Exception as e_ingest:
            logger.error(f"Error during ChatGPT export ingestion process for file '{file_path}': {e_ingest}", exc_info=True)
            await interaction.followup.send(f"An error occurred during the ingestion process: {str(e_ingest)[:1000]}", ephemeral=True)

    @ingest_chatgpt_export_command.error
    async def ingest_export_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message("You need 'Manage Messages' permission to use this command (though current check is False).", ephemeral=True)
        else:
            logger.error(f"Unhandled error in ingest_chatgpt_export_command: {error}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(f"An unexpected error occurred: {str(error)[:500]}", ephemeral=True)
            else:
                await interaction.followup.send(f"An unexpected error occurred: {str(error)[:500]}", ephemeral=True)


    @bot_instance.tree.command(name="remindme", description="Sets a reminder. E.g., /remindme 1h30m Check the oven.")
    @app_commands.describe(time_duration="Duration (e.g., '10m', '2h30m', '1d').", reminder_message="The message for your reminder.")
    async def remindme_slash_command(interaction: discord.Interaction, time_duration: str, reminder_message: str):
        if not bot_state_instance:
            logger.error("remindme_slash_command: bot_state_instance is None.")
            await interaction.response.send_message("Bot state not available. Cannot set reminder.", ephemeral=True)
            return

        time_delta, descriptive_time_str = parse_time_string_to_delta(time_duration)

        if not time_delta or time_delta.total_seconds() <= 0:
            await interaction.response.send_message("Invalid time duration provided. Please use formats like '10m', '2h30m', '1d'.", ephemeral=True)
            return

        reminder_time = datetime.now() + time_delta
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: Could not determine the channel for this reminder.", ephemeral=True)
            return

        reminder_entry = (reminder_time, interaction.channel_id, interaction.user.id, reminder_message, descriptive_time_str or "later")
        await bot_state_instance.add_reminder(reminder_entry)

        await interaction.response.send_message(f"Okay, {interaction.user.mention}! I'll remind you in {descriptive_time_str or 'the specified time'} about: \"{reminder_message}\"")
        logger.info(f"Reminder set for user {interaction.user.name} ({interaction.user.id}) at {reminder_time.strftime('%Y-%m-%d %H:%M:%S')} for: {reminder_message}")

    @bot_instance.tree.command(name="roast", description="Generates a comedy routine based on a webpage.")
    @app_commands.describe(url="The URL of the webpage to roast.")
    async def roast_slash_command(interaction: discord.Interaction, url: str):
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("roast_slash_command: One or more bot components (llm_client, bot_state, bot_instance) are None.")
            await interaction.response.send_message("Bot components not ready. Cannot perform roast.", ephemeral=True)
            return

        logger.info(f"Roast command initiated by {interaction.user.name} for URL: {url}.")
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)
        await interaction.edit_original_response(content=f"Getting ready to roast {url}...")

        if bot_state_instance and hasattr(bot_state_instance, 'update_last_playwright_usage_time'):
            await bot_state_instance.update_last_playwright_usage_time() # Made awaitable
            logger.debug(f"Updated last_playwright_usage_time via bot_state_instance for /roast command")

        try:
            webpage_text = await scrape_website(url)
            if not webpage_text or "Failed to scrape" in webpage_text or "Scraping timed out" in webpage_text:
                error_message = f"Sorry, I couldn't properly roast {url}. Reason: {webpage_text or 'Could not retrieve any content from the page.'}"
                await interaction.edit_original_response(content=error_message, embed=None) # Clear embed
                return

            await interaction.edit_original_response(content=f"Crafting a roast for {url} based on its content...")

            user_query_content = f"Analyze the following content from the webpage {url} and write a short, witty, and biting comedy roast routine about it. Be creative and funny, focusing on absurdities or humorous angles. Do not just summarize. Make it a roast!\n\nWebpage Content:\n{webpage_text}"
            user_msg_node = MsgNode("user", user_query_content, name=str(interaction.user.id))

            rag_query_for_roast = f"comedy roast of webpage content from URL: {url}"
            synthesized_rag_context = await retrieve_and_prepare_rag_context(llm_client_instance, rag_query_for_roast)

            prompt_nodes = await _build_initial_prompt_messages(
                user_query_content=user_query_content,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_rag_context
            )
            # stream_llm_response_to_interaction will handle editing the original response
            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, prompt_nodes,
                title=f"Comedy Roast of {url}",
                synthesized_rag_context_for_display=synthesized_rag_context,
                bot_user_id=bot_instance.user.id
            )
        except Exception as e:
            logger.error(f"Error in roast_slash_command for URL '{url}': {e}", exc_info=True)
            await interaction.edit_original_response(content=f"Ouch, the roast attempt for {url} backfired on me! Error: {str(e)[:1000]}", embed=None)

    @bot_instance.tree.command(name="search", description="Performs a web search and summarizes results.")
    @app_commands.describe(query="Your search query.")
    async def search_slash_command(interaction: discord.Interaction, query: str):
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("search_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot perform search.", ephemeral=True)
            return

        logger.info(f"Search command initiated by {interaction.user.name} for query: '{query}'.")
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)
        await interaction.edit_original_response(content=f"Searching the web for: '{query}'...")

        try:
            search_results = await query_searx(query)
            if not search_results:
                await interaction.edit_original_response(content=f"Sorry, I couldn't find any search results for '{query}'.", embed=None)
                return

            await interaction.edit_original_response(content=f"Found {len(search_results)} results for '{query}'. Processing top results...")

            max_results_to_process = config.NEWS_MAX_LINKS_TO_PROCESS # Reuse similar config as /news
            num_to_process = min(len(search_results), max_results_to_process)
            page_summaries_for_final_synthesis: List[str] = []
            processed_urls_search = set()

            for i in range(num_to_process):
                result = search_results[i]
                page_url = result.get('url')
                page_title = result.get('title', 'Untitled Page')

                if not page_url or page_url in processed_urls_search:
                    logger.info(f"Skipping duplicate or invalid URL for search: {page_url}")
                    continue
                processed_urls_search.add(page_url)

                update_embed_search = discord.Embed(
                    title=f"Search Results for: {query}",
                    description=f"Processing result {i+1}/{num_to_process}: Scraping '{page_title}'...",
                    color=config.EMBED_COLOR["incomplete"]
                )
                await interaction.edit_original_response(embed=update_embed_search, content=None)

                if bot_state_instance and hasattr(bot_state_instance, 'update_last_playwright_usage_time'):
                    await bot_state_instance.update_last_playwright_usage_time()

                scraped_content = await scrape_website(page_url)

                if not scraped_content or "Failed to scrape" in scraped_content or "Scraping timed out" in scraped_content:
                    logger.warning(f"Failed to scrape '{page_title}' from {page_url} for search. Reason: {scraped_content}")
                    page_summaries_for_final_synthesis.append(f"Source: {page_title} ({page_url})\nSummary: [Could not retrieve content for summarization]\n\n")
                    continue

                update_embed_search.description = f"Processing result {i+1}/{num_to_process}: Summarizing '{page_title}'..."
                await interaction.edit_original_response(embed=update_embed_search)

                summarization_prompt_search = (
                    f"You are an expert summarizer. Please read the following web page content, "
                    f"which was found when searching for the query '{query}'. Extract the key factual "
                    f"points and provide a concise summary (2-4 sentences) relevant to this query. "
                    f"Focus on information that directly addresses or relates to the user's search intent.\n\n"
                    f"Page Title: {page_title}\n"
                    f"Page Content:\n{scraped_content[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT*2]}"
                )

                try:
                    summary_response_search = await llm_client_instance.chat.completions.create(
                        model=config.FAST_LLM_MODEL,
                        messages=[{"role": "user", "content": summarization_prompt_search}],
                        max_tokens=250,
                        temperature=0.3,
                        stream=False
                    )
                    if summary_response_search.choices and summary_response_search.choices[0].message and summary_response_search.choices[0].message.content:
                        page_summary = summary_response_search.choices[0].message.content.strip()
                        logger.info(f"Summarized '{page_title}' for search query '{query}': {page_summary[:100]}...")
                        page_summaries_for_final_synthesis.append(f"Source Page: {page_title} ({page_url})\nSummary of Page Content: {page_summary}\n\n")
                        # Optionally, store these summaries if a suitable storage mechanism exists (like news summaries)
                        # store_search_page_summary(query=query, url=page_url, summary_text=page_summary) # Example, if implemented
                    else:
                        logger.warning(f"LLM summarization returned no content for '{page_title}' (search query '{query}').")
                        page_summaries_for_final_synthesis.append(f"Source Page: {page_title} ({page_url})\nSummary: [AI summarization failed or returned no content]\n\n")
                except Exception as e_summ_search:
                    logger.error(f"Error during LLM summarization for '{page_title}' (search query '{query}'): {e_summ_search}", exc_info=True)
                    page_summaries_for_final_synthesis.append(f"Source Page: {page_title} ({page_url})\nSummary: [Error during AI summarization]\n\n")
            
            if not page_summaries_for_final_synthesis:
                error_embed_search = discord.Embed(
                    title=f"Search Results for: {query}",
                    description=f"Could not process any web pages to generate a summary for '{query}'.",
                    color=config.EMBED_COLOR["error"]
                )
                await interaction.edit_original_response(embed=error_embed_search)
                return

            final_update_embed = discord.Embed(
                title=f"Search Results for: {query}",
                description="All relevant pages processed. Generating final search summary...",
                color=config.EMBED_COLOR["incomplete"]
            )
            await interaction.edit_original_response(embed=final_update_embed)

            combined_page_summaries_text = "".join(page_summaries_for_final_synthesis)

            final_synthesis_prompt_content = (
                f"You are a search engine. Your purpose is to provide a concise, factual, and direct summary of information found across multiple web pages related to the user's query. "
                f"Do not adopt a conversational persona or use journalistic language. Act like a search engine's summary snippet.\n\n"
                f"User's Original Query: '{query}'\n\n"
                f"Below are summaries of content from several web pages. Synthesize these into a single, integrated summary that directly addresses the user's query. "
                f"Focus on the most relevant facts and information. Aim for a comprehensive yet brief overview. "
                f"If the sources provide conflicting information, you may note this neutrally. Do not add information not present in the summaries.\n\n"
                f"--- Collected Page Summaries ---\n"
                f"{combined_page_summaries_text}"
                f"--- End of Page Summaries ---\n\n"
                f"Based on these summaries, provide a direct answer/summary for the query: '{query}'"
            )
            user_msg_node = MsgNode("user", final_synthesis_prompt_content, name=str(interaction.user.id))

            rag_query_for_search_summary = query 
            synthesized_rag_context = await retrieve_and_prepare_rag_context(llm_client_instance, rag_query_for_search_summary)

            prompt_nodes = await _build_initial_prompt_messages(
                user_query_content=final_synthesis_prompt_content,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_rag_context
            )
            # stream_llm_response_to_interaction will send new followups for the summary
            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, prompt_nodes,
                title=f"Summary for Search: {query}",
                force_new_followup_flow=True,
                synthesized_rag_context_for_display=synthesized_rag_context,
                bot_user_id=bot_instance.user.id
            )
        except Exception as e:
            logger.error(f"Error in search_slash_command for query '{query}': {e}", exc_info=True)
            await interaction.edit_original_response(content=f"Yikes, my search circuits are fuzzy! Failed to search for '{query}'. Error: {str(e)[:1000]}", embed=None)

    @bot_instance.tree.command(name="pol", description="Generates a sarcastic response to a political statement.")
    @app_commands.describe(statement="The political statement.")
    async def pol_slash_command(interaction: discord.Interaction, statement: str):
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("pol_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot perform pol.", ephemeral=True)
            return

        logger.info(f"Pol command initiated by {interaction.user.name} for statement: '{statement[:50]}...'.")
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)
        await interaction.edit_original_response(content="Crafting a suitably sarcastic political commentary...")

        try:
            pol_system_content = (
                "You are a bot that generates extremely sarcastic, snarky, and somewhat troll-like comments "
                "to mock extremist political views or absurd political statements. Your goal is to be biting and humorous, "
                "undermining the statement without resorting to direct vulgarity or hate speech. Focus on sharp wit, irony, and highlighting logical fallacies in a comedic way. "
                "Keep it relatively brief but impactful."
            )
            user_query_content = f"Generate a sarcastic comeback or commentary for the following political statement: \"{statement}\""
            user_msg_node = MsgNode("user", user_query_content, name=str(interaction.user.id))

            rag_query_for_pol = statement
            synthesized_rag_context = await retrieve_and_prepare_rag_context(llm_client_instance, rag_query_for_pol)

            base_prompt_nodes = await _build_initial_prompt_messages(
                user_query_content=user_query_content,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_rag_context
            )

            insert_idx = 0
            for idx, node in enumerate(base_prompt_nodes):
                if node.role != "system":
                    insert_idx = idx
                    break
                insert_idx = idx + 1

            final_prompt_nodes = base_prompt_nodes[:insert_idx] + [MsgNode("system", pol_system_content)] + base_prompt_nodes[insert_idx:]

            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, final_prompt_nodes,
                title="Sarcastic Political Commentary",
                synthesized_rag_context_for_display=synthesized_rag_context,
                bot_user_id=bot_instance.user.id
            )
        except Exception as e:
            logger.error(f"Error in pol_slash_command for statement '{statement[:50]}...': {e}", exc_info=True)
            await interaction.edit_original_response(content=f"My political satire circuits just blew a fuse! Error: {str(e)[:1000]}", embed=None)

    @bot_instance.tree.command(name="rss", description="Fetches new entries from an RSS feed and summarizes them.")
    @app_commands.describe(feed_url="The RSS feed URL.", limit="Number of new entries to fetch (max 10).")
    @app_commands.choices(
        feed_url=[
            app_commands.Choice(name=name, value=url)
            for name, url in DEFAULT_RSS_FEEDS
        ]
    )
    async def rss_slash_command(
        interaction: discord.Interaction,
        feed_url: str,
        limit: app_commands.Range[int, 1, 10] = 5,
    ):
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("rss_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot fetch RSS.", ephemeral=True)
            return

        logger.info(f"RSS command initiated by {interaction.user.name} for {feed_url}, limit {limit}.")
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)
        await interaction.edit_original_response(content=f"Fetching RSS feed: {feed_url}...")

        seen = load_seen_entries()
        seen_ids = set(seen.get(feed_url, []))

        try:
            entries = await fetch_rss_entries(feed_url)
            new_entries = [e for e in entries if e.get("guid") not in seen_ids]
            if not new_entries:
                await interaction.edit_original_response(content="No new entries found.")
                return

            to_process = new_entries[:limit]
            summaries: List[str] = []

            for idx, ent in enumerate(to_process, 1):
                title = ent.get("title") or "Untitled"
                link = ent.get("link") or ""
                guid = ent.get("guid") or link

                await interaction.edit_original_response(content=f"Scraping {idx}/{len(to_process)}: {title}...")

                scraped_text, _ = await scrape_website(link)
                if not scraped_text or "Failed to scrape" in scraped_text or "Scraping timed out" in scraped_text:
                    summaries.append(f"**{title}**\nCould not scrape article: {link}\n")
                    seen_ids.add(guid)
                    continue

                prompt = (
                    f"Summarize the following article in 2-4 sentences. Focus on key facts.\n\n"
                    f"Title: {title}\nURL: {link}\n\n{scraped_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
                )

                try:
                    response = await llm_client_instance.chat.completions.create(
                        model=config.FAST_LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=250,
                        temperature=0.3,
                        stream=False,
                    )
                    summary = response.choices[0].message.content.strip() if response.choices else ""
                except Exception as e_summ:
                    logger.error(f"LLM summarization failed for {link}: {e_summ}")
                    summary = "[LLM summarization failed]"

                summaries.append(f"**{title}**\n{summary}\n{link}\n")
                seen_ids.add(guid)

            seen[feed_url] = list(seen_ids)
            save_seen_entries(seen)

            combined = "\n\n".join(summaries)
            chunks = chunk_text(combined, config.EMBED_MAX_LENGTH)
            for i, chunk in enumerate(chunks):
                embed = discord.Embed(
                    title=f"RSS Summaries for {feed_url}" + ("" if i == 0 else f" (cont. {i+1})"),
                    description=chunk,
                    color=config.EMBED_COLOR["complete"],
                )
                if i == 0:
                    await interaction.edit_original_response(content=None, embed=embed)
                else:
                    await interaction.followup.send(embed=embed)

            # --- New: Postprocess like other responses ---
            asyncio.create_task(
                send_tts_audio(interaction, combined, base_filename=f"rss_{interaction.id}")
            )
            user_msg = MsgNode(
                "user",
                f"/rss {feed_url} (limit {limit})",
                name=str(interaction.user.id),
            )
            assistant_msg = MsgNode(
                "assistant",
                combined,
                name=str(bot_instance.user.id),
            )
            await bot_state_instance.append_history(
                interaction.channel_id, user_msg, config.MAX_MESSAGE_HISTORY
            )
            await bot_state_instance.append_history(
                interaction.channel_id, assistant_msg, config.MAX_MESSAGE_HISTORY
            )
            await ingest_conversation_to_chromadb(
                llm_client_instance,
                interaction.channel_id,
                interaction.user.id,
                [user_msg, assistant_msg],
            )

        except Exception as e:
            logger.error(f"Error in rss_slash_command for {feed_url}: {e}", exc_info=True)
            await interaction.edit_original_response(content=f"Failed to process RSS feed. Error: {str(e)[:500]}", embed=None)

    @bot_instance.tree.command(name="gettweets", description="Fetches and summarizes recent tweets from a user.")
    @app_commands.describe(
        username="The X/Twitter username (without @).",
        preset_user="Choose a preset account instead of typing one.",
        limit="Number of tweets to fetch (max 50)."
    )
    @app_commands.choices(
        preset_user=[app_commands.Choice(name=u, value=u) for u in DEFAULT_TWITTER_USERS]
    )
    async def gettweets_slash_command(
        interaction: discord.Interaction,
        username: str = "",
        preset_user: str = "",
        limit: app_commands.Range[int, 1, 50] = 10,
    ):
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("gettweets_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot get tweets.", ephemeral=True)
            return

        user_to_fetch = username or preset_user
        if not user_to_fetch:
            await interaction.response.send_message(
                "Please provide a username or choose one from the dropdown.",
                ephemeral=True,
            )
            return

        logger.info(
            f"Gettweets command initiated by {interaction.user.name} for @{user_to_fetch}, limit {limit}."
        )
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)
        # Send initial message using edit_original_response, this will be updated by send_progress
        clean_username_for_initial_message = user_to_fetch.lstrip('@')
        await interaction.edit_original_response(
            content=f"Starting to scrape tweets for @{clean_username_for_initial_message} (up to {limit})..."
        )

        async def send_progress(message: str):
            try:
                await interaction.edit_original_response(content=message)
            except discord.HTTPException as e_prog:
                # Log if the original message edit fails, especially if it's due to interaction expiry
                logger.warning(f"Failed to send progress update for gettweets (edit original): {e_prog}")
            except Exception as e_unexp:
                 logger.error(f"Unexpected error in send_progress for gettweets: {e_unexp}", exc_info=True)


        try:
            clean_username = user_to_fetch.lstrip('@')

            if bot_state_instance and hasattr(bot_state_instance, 'update_last_playwright_usage_time'):
                await bot_state_instance.update_last_playwright_usage_time()  # Made awaitable
                logger.debug(
                    f"Updated last_playwright_usage_time via bot_state_instance for /gettweets @{clean_username}"
                )

            # Initial message is already sent above. send_progress will now update it.
            tweets = await scrape_latest_tweets(clean_username, limit=limit, progress_callback=send_progress)

            if not tweets:
                final_content_message = f"Finished scraping for @{clean_username}. No tweets found or profile might be private/inaccessible."
                await interaction.edit_original_response(content=final_content_message, embed=None) # Ensure embed is cleared
                # Followup with a more user-friendly ephemeral message if needed, but primary is above.
                # await interaction.followup.send(f"Could not fetch any recent tweets for @{clean_username}. The profile might be private, have no tweets, or there was an issue.", ephemeral=True)
                return

            tweet_texts_for_display = []
            for t in tweets:
                ts_str = t.get('timestamp', 'N/A')
                display_ts = ts_str
                try:
                    dt_obj = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str != 'N/A' else None
                    display_ts = dt_obj.strftime("%Y-%m-%d %H:%M UTC") if dt_obj else ts_str
                except ValueError: pass

                author_display = t.get('username', clean_username)
                content_display = discord.utils.escape_markdown(t.get('content', 'N/A'))
                tweet_url_display = t.get('tweet_url', '')

                header = f"[{display_ts}] @{author_display}"
                if t.get('is_repost') and t.get('reposted_by'):
                    header = f"[{display_ts}] @{t.get('reposted_by')} reposted @{author_display}"

                link_text = f" ([Link]({tweet_url_display}))" if tweet_url_display else ""
                tweet_texts_for_display.append(f"**{header}**: {content_display}{link_text}")

            raw_tweets_display_str = "\n\n".join(tweet_texts_for_display)
            if not raw_tweets_display_str: raw_tweets_display_str = "No tweet content could be formatted for display."

            await send_progress(f"Formatting {len(tweets)} tweets for display...")

            embed_title = f"Recent Tweets from @{clean_username}"
            raw_tweet_chunks = chunk_text(raw_tweets_display_str, config.EMBED_MAX_LENGTH)

            for i, chunk_content_part in enumerate(raw_tweet_chunks):
                chunk_title = embed_title if i == 0 else f"{embed_title} (cont.)"
                embed = discord.Embed(
                    title=chunk_title,
                    description=chunk_content_part,
                    color=config.EMBED_COLOR["complete"]
                )
                if i == 0:
                    await interaction.edit_original_response(content=None, embed=embed) # Clear content text, show first embed
                else:
                    await interaction.followup.send(embed=embed) # New messages for subsequent chunks

            user_query_content_for_summary = (
                f"Please analyze and summarize the main themes, topics discussed, and overall sentiment "
                f"from @{clean_username}'s recent tweets provided below. Extract key points and present a concise overview. "
                f"Do not just re-list the tweets.\n\nRecent Tweets:\n{raw_tweets_display_str[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
            )
            user_msg_node = MsgNode("user", user_query_content_for_summary, name=str(interaction.user.id))

            rag_query_for_tweets_summary = f"summary of recent tweets from Twitter user @{clean_username}"
            synthesized_rag_context = await retrieve_and_prepare_rag_context(llm_client_instance, rag_query_for_tweets_summary)

            prompt_nodes_summary = await _build_initial_prompt_messages(
                user_query_content=user_query_content_for_summary,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_rag_context
            )
            # This will create a new message flow for the summary
            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, prompt_nodes_summary,
                title=f"Tweet Summary for @{clean_username}",
                force_new_followup_flow=True,
                synthesized_rag_context_for_display=synthesized_rag_context,
                bot_user_id=bot_instance.user.id
            )
        except Exception as e:
            logger.error(
                f"Error in gettweets_slash_command for @{user_to_fetch}: {e}", exc_info=True
            )
            error_content = (
                f"My tweet-fetching antenna is bent! Failed for @{user_to_fetch}. Error: {str(e)[:500]}"
            )
            try:
                await interaction.edit_original_response(content=error_content, embed=None)  # Clear embed
            except discord.HTTPException:
                logger.warning(
                    f"Could not send final error message (edit) for gettweets @{user_to_fetch} to user."
                )


    @bot_instance.tree.command(name="ap", description="Describes an attached image with a creative AP Photo twist.")
    @app_commands.describe(image="The image to describe.", user_prompt="Optional additional prompt for the description.")
    async def ap_slash_command(interaction: discord.Interaction, image: discord.Attachment, user_prompt: str = ""):
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("ap_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot perform AP.", ephemeral=True)
            return

        logger.info(f"AP command initiated by {interaction.user.name} for image {image.filename}.")
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)
        await interaction.edit_original_response(content="Preparing your AP Photo description...")


        try:
            if not image.content_type or not image.content_type.startswith("image/"):
                await interaction.edit_original_response(content="The attached file is not a valid image. Please attach a PNG, JPG, GIF, etc.", embed=None)
                return

            image_bytes = await image.read()
            if len(image_bytes) > config.MAX_IMAGE_BYTES_FOR_PROMPT:
                logger.warning(f"Image {image.filename} provided by {interaction.user.name} is too large ({len(image_bytes)} bytes). Max allowed: {config.MAX_IMAGE_BYTES_FOR_PROMPT}.")
                await interaction.edit_original_response(content=f"The image you attached is too large (max {config.MAX_IMAGE_BYTES_FOR_PROMPT // (1024*1024)}MB). Please try a smaller one.", embed=None)
                return

            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url_for_llm = f"data:{image.content_type};base64,{base64_image}"

            chosen_celebrity = random.choice(["Keanu Reeves", "Dwayne 'The Rock' Johnson", "Zendaya", "Tom Hanks", "Margot Robbie", "Ryan Reynolds", "Morgan Freeman", "Awkwafina"])

            ap_task_prompt_text = (
                f"You are an Associated Press (AP) photo caption writer with a quirky sense of humor. Your task is to describe the attached image in vivid detail, as if for someone who cannot see it. "
                f"However, here's the twist: you must creatively and seamlessly replace the main subject or a prominent character in the image with the celebrity **{chosen_celebrity}**. "
                f"Maintain a professional AP caption style (who, what, when, where, why - if inferable), but weave in {chosen_celebrity}'s presence naturally and humorously. "
                f"Start your response with 'AP Photo: {chosen_celebrity}...' "
                f"If the user provided an additional prompt, try to incorporate its theme or request into your {chosen_celebrity}-centric description: '{user_prompt if user_prompt else 'No additional user prompt.'}'"
            )

            user_content_for_ap_node = [
                {"type": "text", "text": user_prompt if user_prompt else "Describe this image with the AP Photo celebrity twist."},
                {"type": "image_url", "image_url": {"url": image_url_for_llm}}
            ]
            user_msg_node = MsgNode("user", user_content_for_ap_node, name=str(interaction.user.id))

            rag_query_for_ap = user_prompt if user_prompt else f"AP photo style description featuring {chosen_celebrity} for an image."
            synthesized_rag_context = await retrieve_and_prepare_rag_context(llm_client_instance, rag_query_for_ap)

            base_prompt_nodes = await _build_initial_prompt_messages(
                user_query_content=user_content_for_ap_node,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_rag_context,
                max_image_history_depth=0
            )

            insert_idx = 0
            for idx, node in enumerate(base_prompt_nodes):
                if node.role != "system": insert_idx = idx; break
                insert_idx = idx + 1
            final_prompt_nodes = base_prompt_nodes[:insert_idx] + [MsgNode("system", ap_task_prompt_text)] + base_prompt_nodes[insert_idx:]

            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, final_prompt_nodes,
                title=f"AP Photo Description ft. {chosen_celebrity}",
                synthesized_rag_context_for_display=synthesized_rag_context,
                bot_user_id=bot_instance.user.id
            )
        except Exception as e:
            logger.error(f"Error in ap_slash_command for image '{image.filename}': {e}", exc_info=True)
            await interaction.edit_original_response(content=f"My camera lens for the AP command seems to be cracked! Error: {str(e)[:1000]}", embed=None)

    @bot_instance.tree.command(name="clearhistory", description="Clears the bot's short-term message history for this channel.")
    @app_commands.checks.has_permissions(manage_messages=True)
    async def clearhistory_slash_command(interaction: discord.Interaction):
        if not bot_state_instance:
             logger.error("clearhistory_slash_command: bot_state_instance is None.")
             await interaction.response.send_message("Bot state not available. Cannot clear history.", ephemeral=True)
             return

        if interaction.channel_id:
            await bot_state_instance.clear_channel_history(interaction.channel_id)
            logger.info(f"Short-term message history cleared for channel {interaction.channel_id} by {interaction.user.name} ({interaction.user.id}).")
            await interaction.response.send_message("My short-term memory for this channel has been wiped clean!", ephemeral=True)
        else:
            await interaction.response.send_message("Error: Could not determine the channel to clear history for.", ephemeral=True)

    @clearhistory_slash_command.error
    async def clearhistory_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message("You don't have the 'Manage Messages' permission to use this command.", ephemeral=True)
        else:
            logger.error(f"Unexpected error in clearhistory_slash_command: {error}", exc_info=True)
            if not interaction.response.is_done():
                 await interaction.response.send_message(f"An unexpected error occurred: {str(error)[:500]}", ephemeral=True)
            else:
                 await interaction.followup.send(f"An unexpected error occurred: {str(error)[:500]}", ephemeral=True)
