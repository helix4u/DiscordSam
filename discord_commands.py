import logging
import discord
from discord import app_commands # type: ignore
from discord.ext import commands # For bot type hint
import os
import base64
import random
from typing import Any, Optional, List # Keep existing imports
from datetime import datetime

# Bot services and utilities
from config import config
from state import BotState
from common_models import (
    MsgNode, LLMRequest, InteractionRequestData,
    NewsCommandData, IngestCommandData, SearchCommandData, GetTweetsCommandData, APCommandData
)
#from pydantic import BaseModel # No longer needed here

from llm_handling import (
    _build_initial_prompt_messages,
    stream_llm_response_to_interaction
)
from rag_chroma_manager import (
    retrieve_and_prepare_rag_context,
    parse_chatgpt_export,
    store_chatgpt_conversations_in_chromadb,
    store_news_summary,
)
from web_utils import (
    scrape_website,
    query_searx,
    scrape_latest_tweets
)
from utils import parse_time_string_to_delta, chunk_text

logger = logging.getLogger(__name__)

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

        await interaction.response.defer(ephemeral=False) # Defer early

        # The actual processing logic will be moved to the queue worker.
        # Here, we just package and enqueue.

        # Note: Playwright usage time should ideally be updated by the worker
        # if Playwright is used during the execution of the queued task.
        # For now, we're simplifying; the worker will need access to bot_state.

        # Build the core information needed for the LLM, this might involve some initial non-LLM work.
        # However, long-running scraping and multiple LLM calls (like summarization per article)
        # should definitely be part of the worker task.

        # For the `/news` command, the entire original logic (searching, scraping, summarizing individual articles,
        # then final briefing) is LLM-heavy and long-running.
        # It's best to encapsulate the *intent* and parameters in the queue item.

        # Simplified: We'll pass the topic and let the worker handle the multi-step process.
        # This means the worker will need to be more complex for this command.
        # Alternatively, for this specific command, the "prompt_messages" might be a placeholder
        # or an initial instruction, and the worker reconstructs the full flow.

        # Let's make the worker responsible for the whole news generation flow.
        # We'll create a specific data structure for news requests.
        # NewsCommandData is now imported from common_models.py

        news_data = NewsCommandData(
            interaction=interaction,
            topic=topic,
            bot_user_id=bot_instance.user.id if bot_instance else None
        )

        # We are not pre-building user_msg_node or prompt_messages here for /news,
        # as the worker will handle the entire news generation logic.
        # The 'data' field in LLMRequest will carry this news_data.

        llm_req = LLMRequest(request_type='news_command', data=news_data)

        if bot_state_instance:
            await bot_state_instance.llm_request_queue.put(llm_req)
            try:
                queue_size = bot_state_instance.llm_request_queue.qsize()
                await interaction.edit_original_response(content=f"Your request for a news briefing on '{topic}' has been queued (approx. position: {queue_size}). I'll start working on it soon!")
                logger.info(f"/news command for '{topic}' by {interaction.user.name} queued. Approx queue size: {queue_size}")
            except discord.HTTPException as e:
                logger.warning(f"Failed to send 'queued' confirmation for /news: {e}")
        else:
            logger.error("Bot state instance not available, cannot queue /news command.")
            await interaction.edit_original_response(content=f"Sorry, I couldn't queue your news request for '{topic}' due to an internal issue.")

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

        # Define a simple data structure for this command's queue request
        # IngestCommandData is now imported from common_models.py

        ingest_data = IngestCommandData(interaction=interaction, file_path=file_path)
        llm_req = LLMRequest(request_type='ingest_command', data=ingest_data)

        if bot_state_instance:
            await bot_state_instance.llm_request_queue.put(llm_req)
            try:
                queue_size = bot_state_instance.llm_request_queue.qsize()
                await interaction.followup.send(f"Your request to ingest '{os.path.basename(file_path)}' has been queued (approx. position: {queue_size}). Processing will begin shortly.", ephemeral=True)
                logger.info(f"/ingest_chatgpt_export for '{file_path}' by {interaction.user.name} queued. Approx queue size: {queue_size}")
            except discord.HTTPException as e:
                logger.warning(f"Failed to send 'queued' confirmation for /ingest_chatgpt_export: {e}")
        else:
            logger.error("Bot state instance not available, cannot queue /ingest_chatgpt_export command.")
            await interaction.followup.send(f"Sorry, I couldn't queue your ingest request for '{os.path.basename(file_path)}' due to an internal issue.", ephemeral=True)

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

        # Playwright usage time update should ideally be handled by the worker if it does the scraping.
        # For now, let's assume the scraping happens before queuing for roast.
        # This means the initial response might take a moment if scraping is slow.
        # A more advanced setup would queue the URL, and the worker does scrape + LLM.

        if bot_state_instance and hasattr(bot_state_instance, 'update_last_playwright_usage_time'):
            await bot_state_instance.update_last_playwright_usage_time()
            logger.debug(f"Updated last_playwright_usage_time for /roast command (pre-queue scrape)")

        await interaction.edit_original_response(content=f"Attempting to fetch content from {url} to prepare for roasting...")
        webpage_text = await scrape_website(url)

        if not webpage_text or "Failed to scrape" in webpage_text or "Scraping timed out" in webpage_text:
            error_message = f"Sorry, I couldn't properly roast {url}. Reason: {webpage_text or 'Could not retrieve any content from the page.'}"
            await interaction.edit_original_response(content=error_message, embed=None)
            return

        await interaction.edit_original_response(content=f"Content from {url} fetched. Now queuing the roast request...")

        user_query_content = f"Analyze the following content from the webpage {url} and write a short, witty, and biting comedy roast routine about it. Be creative and funny, focusing on absurdities or humorous angles. Do not just summarize. Make it a roast!\n\nWebpage Content:\n{webpage_text}"
        user_msg_node = MsgNode("user", user_query_content, name=str(interaction.user.id))

        rag_query_for_roast = f"comedy roast of webpage content from URL: {url}"
        # RAG context generation involves an LLM call, so this should also be part of the worker.
        # For now, generating it before queuing. This makes the command execution longer before queuing.
        # Ideal: Worker does RAG context, then main LLM call.
        synthesized_rag_context = await retrieve_and_prepare_rag_context(llm_client_instance, rag_query_for_roast)

        prompt_nodes = await _build_initial_prompt_messages(
            user_query_content=user_query_content,
            channel_id=interaction.channel_id,
            bot_state=bot_state_instance, # For history
            user_id=str(interaction.user.id),
            synthesized_rag_context_str=synthesized_rag_context
        )

        request_data = InteractionRequestData(
            interaction=interaction,
            user_msg_node=user_msg_node,
            prompt_messages=prompt_nodes,
            title=f"Comedy Roast of {url}",
            synthesized_rag_context_for_display=synthesized_rag_context,
            bot_user_id=bot_instance.user.id if bot_instance else None
        )

        llm_req = LLMRequest(request_type='interaction', data=request_data)

        if bot_state_instance:
            await bot_state_instance.llm_request_queue.put(llm_req)
            try:
                queue_size = bot_state_instance.llm_request_queue.qsize()
                await interaction.edit_original_response(content=f"Your roast request for '{url}' has been queued (approx. position: {queue_size}). The comedic genius is warming up!")
                logger.info(f"/roast command for '{url}' by {interaction.user.name} queued. Approx queue size: {queue_size}")
            except discord.HTTPException as e:
                logger.warning(f"Failed to send 'queued' confirmation for /roast: {e}")
        else:
            logger.error("Bot state instance not available, cannot queue /roast command.")
            await interaction.edit_original_response(content=f"Sorry, I couldn't queue your roast request for '{url}' due to an internal issue.")

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

        # Similar to /news, the /search command involves multiple steps (web search, scraping, summarizing, final synthesis)
        # This entire flow should be handled by the worker.
        # SearchCommandData is now imported from common_models.py

        search_data = SearchCommandData(
            interaction=interaction,
            query=query,
            bot_user_id=bot_instance.user.id if bot_instance else None
        )

        llm_req = LLMRequest(request_type='search_command', data=search_data)

        if bot_state_instance:
            await bot_state_instance.llm_request_queue.put(llm_req)
            try:
                queue_size = bot_state_instance.llm_request_queue.qsize()
                await interaction.edit_original_response(content=f"Your search for '{query}' has been queued (approx. position: {queue_size}). I'll start digging for information soon!")
                logger.info(f"/search command for '{query}' by {interaction.user.name} queued. Approx queue size: {queue_size}")
            except discord.HTTPException as e:
                logger.warning(f"Failed to send 'queued' confirmation for /search: {e}")
        else:
            logger.error("Bot state instance not available, cannot queue /search command.")
            await interaction.edit_original_response(content=f"Sorry, I couldn't queue your search for '{query}' due to an internal issue.")

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

        # RAG context generation involves an LLM call, ideally part of the worker.
        # For now, doing it before queuing.
        await interaction.edit_original_response(content="Preparing context for political commentary...")
        rag_query_for_pol = statement
        synthesized_rag_context = await retrieve_and_prepare_rag_context(llm_client_instance, rag_query_for_pol)

        await interaction.edit_original_response(content="Crafting a suitably sarcastic political commentary... (Queued)")

        pol_system_content = (
            "You are a bot that generates extremely sarcastic, snarky, and somewhat troll-like comments "
            "to mock extremist political views or absurd political statements. Your goal is to be biting and humorous, "
            "undermining the statement without resorting to direct vulgarity or hate speech. Focus on sharp wit, irony, and highlighting logical fallacies in a comedic way. "
            "Keep it relatively brief but impactful."
        )
        user_query_content = f"Generate a sarcastic comeback or commentary for the following political statement: \"{statement}\""
        user_msg_node = MsgNode("user", user_query_content, name=str(interaction.user.id))

        base_prompt_nodes = await _build_initial_prompt_messages(
            user_query_content=user_query_content,
            channel_id=interaction.channel_id,
            bot_state=bot_state_instance, # For history
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

        request_data = InteractionRequestData(
            interaction=interaction,
            user_msg_node=user_msg_node,
            prompt_messages=final_prompt_nodes,
            title="Sarcastic Political Commentary",
            synthesized_rag_context_for_display=synthesized_rag_context,
            bot_user_id=bot_instance.user.id if bot_instance else None
        )

        llm_req = LLMRequest(request_type='interaction', data=request_data)

        if bot_state_instance:
            await bot_state_instance.llm_request_queue.put(llm_req)
            try:
                queue_size = bot_state_instance.llm_request_queue.qsize()
                await interaction.edit_original_response(content=f"Your political commentary request for '{statement[:30]}...' has been queued (approx. position: {queue_size}). Sharpening my wit...")
                logger.info(f"/pol command for '{statement[:30]}...' by {interaction.user.name} queued. Approx queue size: {queue_size}")
            except discord.HTTPException as e:
                logger.warning(f"Failed to send 'queued' confirmation for /pol: {e}")
        else:
            logger.error("Bot state instance not available, cannot queue /pol command.")
            await interaction.edit_original_response(content=f"Sorry, I couldn't queue your /pol request for '{statement[:30]}...' due to an internal issue.")

    @bot_instance.tree.command(name="gettweets", description="Fetches and summarizes recent tweets from a user.")
    @app_commands.describe(username="The X/Twitter username (without @).", limit="Number of tweets to fetch (max 50).")
    async def gettweets_slash_command(interaction: discord.Interaction, username: str, limit: app_commands.Range[int, 1, 50] = 10):
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("gettweets_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot get tweets.", ephemeral=True)
            return

        logger.info(f"Gettweets command initiated by {interaction.user.name} for @{username}, limit {limit}.")
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)
        # Send initial message using edit_original_response, this will be updated by send_progress
        clean_username_for_initial_message = username.lstrip('@')
        await interaction.edit_original_response(content=f"Starting to scrape tweets for @{clean_username_for_initial_message} (up to {limit})...")

        async def send_progress(message: str): # This progress callback will be passed to the worker if needed
            try:
                await interaction.edit_original_response(content=message)
            except discord.HTTPException as e_prog:
                logger.warning(f"Failed to send progress update for gettweets (edit original): {e_prog}")
            except Exception as e_unexp:
                 logger.error(f"Unexpected error in send_progress for gettweets: {e_unexp}", exc_info=True)

        # The worker will handle scraping, displaying raw tweets, then summarizing.
        # The send_progress callback might be tricky to pass and use effectively by a generic worker.
        # For now, the worker will handle its own progress updates to the interaction if complex.
        # Simplified: queue the username and limit.
        # GetTweetsCommandData is now imported from common_models.py

        gettweets_data = GetTweetsCommandData(
            interaction=interaction,
            username=username,
            limit=limit,
            bot_user_id=bot_instance.user.id if bot_instance else None
        )

        llm_req = LLMRequest(request_type='gettweets_command', data=gettweets_data)

        if bot_state_instance:
            await bot_state_instance.llm_request_queue.put(llm_req)
            try:
                queue_size = bot_state_instance.llm_request_queue.qsize()
                # Initial message was already: content=f"Starting to scrape tweets for @{clean_username_for_initial_message} (up to {limit})..."
                # Edit it to reflect queuing.
                await interaction.edit_original_response(content=f"Your request to fetch tweets for @{username.lstrip('@')} has been queued (approx. position: {queue_size}). This might take a few moments.")
                logger.info(f"/gettweets command for @{username} by {interaction.user.name} queued. Approx queue size: {queue_size}")
            except discord.HTTPException as e:
                logger.warning(f"Failed to send 'queued' confirmation for /gettweets: {e}")
        else:
            logger.error("Bot state instance not available, cannot queue /gettweets command.")
            await interaction.edit_original_response(content=f"Sorry, I couldn't queue your /gettweets request for @{username.lstrip('@')} due to an internal issue.")

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

        # Image processing and RAG context should ideally be done by the worker.
        # For now, reading image bytes and generating RAG before queuing.
        # This makes the command take longer before user sees "queued" confirmation if image is large or RAG is slow.

        await interaction.edit_original_response(content="Preparing your AP Photo description (reading image & context)...")

        if not image.content_type or not image.content_type.startswith("image/"):
            await interaction.edit_original_response(content="The attached file is not a valid image. Please attach a PNG, JPG, GIF, etc.", embed=None)
            return

        image_bytes = await image.read() # Read here, pass bytes or b64 string in queue data
        if len(image_bytes) > config.MAX_IMAGE_BYTES_FOR_PROMPT:
            logger.warning(f"Image {image.filename} provided by {interaction.user.name} is too large ({len(image_bytes)} bytes). Max allowed: {config.MAX_IMAGE_BYTES_FOR_PROMPT}.")
            await interaction.edit_original_response(content=f"The image you attached is too large (max {config.MAX_IMAGE_BYTES_FOR_PROMPT // (1024*1024)}MB). Please try a smaller one.", embed=None)
            return

        base64_image_str = base64.b64encode(image_bytes).decode('utf-8')
        # The worker will reconstruct the image_url_for_llm using this b64 string and image.content_type

        chosen_celebrity = random.choice(["Keanu Reeves", "Dwayne 'The Rock' Johnson", "Zendaya", "Tom Hanks", "Margot Robbie", "Ryan Reynolds", "Morgan Freeman", "Awkwafina"])

        rag_query_for_ap = user_prompt if user_prompt else f"AP photo style description featuring {chosen_celebrity} for an image."
        synthesized_rag_context = await retrieve_and_prepare_rag_context(llm_client_instance, rag_query_for_ap)

        # Building the prompt structure that the worker will use
        ap_task_prompt_text = (
            f"You are an Associated Press (AP) photo caption writer with a quirky sense of humor. Your task is to describe the attached image in vivid detail, as if for someone who cannot see it. "
            f"However, here's the twist: you must creatively and seamlessly replace the main subject or a prominent character in the image with the celebrity **{chosen_celebrity}**. "
            f"Maintain a professional AP caption style (who, what, when, where, why - if inferable), but weave in {chosen_celebrity}'s presence naturally and humorously. "
            f"Start your response with 'AP Photo: {chosen_celebrity}...' "
            f"If the user provided an additional prompt, try to incorporate its theme or request into your {chosen_celebrity}-centric description: '{user_prompt if user_prompt else 'No additional user prompt.'}'"
        )
        user_content_text_for_ap = user_prompt if user_prompt else "Describe this image with the AP Photo celebrity twist."
        # The worker will assemble the user_msg_node content list with the text and image_url.

        # For the AP command, we pass the necessary components to reconstruct the final prompt in the worker.
        # APCommandData is now imported from common_models.py

        ap_data = APCommandData(
            interaction=interaction,
            image_b64=base64_image_str,
            image_content_type=image.content_type,
            user_prompt_text=user_content_text_for_ap,
            ap_system_task_prompt=ap_task_prompt_text,
            base_user_id_for_node=str(interaction.user.id),
            synthesized_rag_context=synthesized_rag_context,
            title=f"AP Photo Description ft. {chosen_celebrity}",
            bot_user_id=bot_instance.user.id if bot_instance else None
        )

        llm_req = LLMRequest(request_type='ap_command', data=ap_data)

        if bot_state_instance:
            await bot_state_instance.llm_request_queue.put(llm_req)
            try:
                queue_size = bot_state_instance.llm_request_queue.qsize()
                await interaction.edit_original_response(content=f"Your AP photo description for '{image.filename}' has been queued (approx. position: {queue_size}). The paparazzi are getting ready!")
                logger.info(f"/ap command for '{image.filename}' by {interaction.user.name} queued. Approx queue size: {queue_size}")
            except discord.HTTPException as e:
                logger.warning(f"Failed to send 'queued' confirmation for /ap: {e}")
        else:
            logger.error("Bot state instance not available, cannot queue /ap command.")
            await interaction.edit_original_response(content=f"Sorry, I couldn't queue your /ap request for '{image.filename}' due to an internal issue.")

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
