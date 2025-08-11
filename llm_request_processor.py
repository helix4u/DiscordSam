import asyncio
import logging
import discord
from discord.ext import commands # For bot type hint
from typing import Any, Optional

from state import BotState
from common_models import LLMRequest, MessageRequestData, InteractionRequestData, MsgNode # Import all relevant models
from llm_handling import stream_llm_response_to_message, stream_llm_response_to_interaction, _build_initial_prompt_messages
from rag_chroma_manager import retrieve_and_prepare_rag_context, parse_chatgpt_export, store_chatgpt_conversations_in_chromadb, store_news_summary
from web_utils import scrape_website, query_searx, scrape_latest_tweets
from utils import chunk_text # For gettweets formatting
from config import config # For gettweets and other command-specific configs
import base64 # For ap_command
import random # For ap_command
import os # For ingest_command
from logit_biases import LOGIT_BIAS_UNWANTED_TOKENS_STR
from openai_api import create_chat_completion, extract_text

# Need to import the inline Pydantic models from discord_commands if they are not moved to common_models
# For now, assuming they will be moved or this processor will be adapted.
# Let's try to define them here if not found in common_models, or preferably, ensure they are in common_models.
# Assuming NewsCommandData, IngestCommandData, SearchCommandData, GetTweetsCommandData, APCommandData
# will be accessible or redefined if necessary.
# For now, I'll proceed as if they are part of common_models or defined globally for the worker.
# To make this runnable, these would need to be properly imported or defined.
# For the sake of this step, I will assume they are available via common_models or similar.
from common_models import (
    MessageRequestData,
    InteractionRequestData,
    NewsCommandData,
    IngestCommandData,
    SearchCommandData,
    GetTweetsCommandData,
    APCommandData
)
#from pydantic import BaseModel # No longer needed for defining these here

# Inline definitions are now removed as they are imported from common_models.py


logger = logging.getLogger(__name__)

async def llm_request_processor_task(bot_state: BotState, llm_client: Any, bot_instance: commands.Bot):
    if not bot_state.llm_processor_task_active.is_set():
        bot_state.llm_processor_task_active.set()
        logger.info("LLM Request Processor Task started and active event set.")
    else:
        logger.info("LLM Request Processor Task restarted (already active).")

    while True:
        try:
            request_item: LLMRequest = await bot_state.llm_request_queue.get()
            request_type = request_item.request_type
            data = request_item.data
            timestamp = request_item.timestamp

            logger.info(f"Processing LLM request: Type='{request_type}', Queued_at='{timestamp.strftime('%Y-%m-%d %H:%M:%S')}', Approx_Queue_Size_Now='{bot_state.llm_request_queue.qsize()}'")

            async with bot_state.llm_processing_lock: # Ensure only one LLM operation at a time
                logger.debug(f"LLM processing lock acquired by worker for request type '{request_type}'")
                try:
                    if request_type == 'message':
                        msg_data: MessageRequestData = data
                        # Ensure target_message's channel is messageable
                        if not isinstance(msg_data.target_message.channel, discord.abc.Messageable):
                            logger.error(f"Target message's channel (ID: {msg_data.target_message.channel.id}) is not Messageable for request from queue.")
                            bot_state.llm_request_queue.task_done()
                            continue

                        await stream_llm_response_to_message(
                            target_message=msg_data.target_message,
                            llm_client=llm_client, # Passed to worker
                            bot_state=bot_state,   # Passed to worker (for history updates within stream_llm)
                            user_msg_node=msg_data.user_msg_node,
                            prompt_messages=msg_data.prompt_messages,
                            synthesized_rag_context_for_display=msg_data.synthesized_rag_context_for_display,
                            bot_user_id=msg_data.bot_user_id
                        )
                    elif request_type == 'interaction':
                        int_data: InteractionRequestData = data
                        if not isinstance(int_data.interaction.channel, discord.abc.Messageable):
                            logger.error(f"Interaction's channel (ID: {int_data.interaction.channel_id}) is not Messageable for request from queue.")
                            bot_state.llm_request_queue.task_done()
                            continue

                        await stream_llm_response_to_interaction(
                            interaction=int_data.interaction,
                            llm_client=llm_client, # Passed to worker
                            bot_state=bot_state,   # Passed to worker
                            user_msg_node=int_data.user_msg_node,
                            prompt_messages=int_data.prompt_messages,
                            title=int_data.title,
                            force_new_followup_flow=int_data.force_new_followup_flow,
                            synthesized_rag_context_for_display=int_data.synthesized_rag_context_for_display,
                            bot_user_id=int_data.bot_user_id
                        )
                    elif request_type == 'news_command':
                        news_data: NewsCommandData = data
                        # --- Start /news specific logic (adapted from discord_commands.py) ---
                        logger.info(f"Worker processing /news for topic: {news_data.topic}")
                        interaction = news_data.interaction
                        topic = news_data.topic

                        try:
                            # Update Playwright usage time before potential use
                            if bot_state and hasattr(bot_state, 'update_last_playwright_usage_time'):
                                await bot_state.update_last_playwright_usage_time()

                            search_topic = f"news {topic}"
                            await interaction.edit_original_response(content=f"Gathering news articles for '{search_topic}'...")
                            search_results = await query_searx(search_topic)

                            if not search_results:
                                await interaction.edit_original_response(content=f"Sorry, I couldn't find any initial search results for '{search_topic}'.")
                                bot_state.llm_request_queue.task_done()
                                continue

                            max_articles_to_process = config.NEWS_MAX_LINKS_TO_PROCESS
                            num_to_process = min(len(search_results), max_articles_to_process)
                            article_summaries_for_briefing: List[str] = []
                            processed_urls = set()

                            for i_news in range(num_to_process):
                                result = search_results[i_news]
                                article_url = result.get('url')
                                article_title = result.get('title', 'Untitled Article')
                                if not article_url or article_url in processed_urls: continue
                                processed_urls.add(article_url)

                                await interaction.edit_original_response(content=f"Processing article {i_news+1}/{num_to_process}: Scraping '{article_title}'...")
                                if bot_state and hasattr(bot_state, 'update_last_playwright_usage_time'):
                                    await bot_state.update_last_playwright_usage_time()
                                scraped_content, _ = await scrape_website(article_url) # Ignoring screenshots for now

                                if not scraped_content or "Failed to scrape" in scraped_content or "Scraping timed out" in scraped_content:
                                    article_summaries_for_briefing.append(f"Source: {article_title} ({article_url})\nSummary: [Could not retrieve content for summarization]\n\n")
                                    continue

                                await interaction.edit_original_response(content=f"Processing article {i_news+1}/{num_to_process}: Summarizing '{article_title}'...")
                                summarization_prompt = (
                                    f"You are an expert news summarizer. Please read the following article content, "
                                    f"which was found when searching for the topic '{search_topic}'. Extract the key factual"
                                    f" news points and provide a detailed yet concise summary (2-4 sentences) relevant to this topic. "
                                    f"Focus on who, what, when, where, and why if applicable. Avoid opinions or speculation not present in the text.\n\n"
                                    f"Article Title: {article_title}\n"
                                    f"Article Content:\n{scraped_content[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT*2]}"
                                )
                                summary_response = await create_chat_completion(
                                    llm_client,
                                    [
                                        {"role": "system", "content": "You are an expert news summarizer."},
                                        {"role": "user", "content": summarization_prompt}
                                    ],
                                    model=config.FAST_LLM_MODEL,
                                    use_responses_api=config.FASTLLM_USE_RESPONSES_API,
                                    max_tokens=250,
                                    temperature=0.3,
                                    logit_bias=LOGIT_BIAS_UNWANTED_TOKENS_STR,
                                )
                                article_summary = extract_text(summary_response, use_responses_api=config.FASTLLM_USE_RESPONSES_API)
                                if article_summary:
                                    article_summaries_for_briefing.append(f"Source: {article_title} ({article_url})\nSummary: {article_summary}\n\n")
                                    store_news_summary(topic=topic, url=article_url, summary_text=article_summary) # Assuming this is thread-safe or handled
                                else:
                                    article_summaries_for_briefing.append(f"Source: {article_title} ({article_url})\nSummary: [AI summarization failed]\n\n")

                            if not article_summaries_for_briefing:
                                await interaction.edit_original_response(content=f"Could not process any articles to generate a briefing for '{topic}'.")
                                bot_state.llm_request_queue.task_done()
                                continue

                            await interaction.edit_original_response(content="All articles processed. Generating final news briefing...")
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
                            synthesized_rag_context_for_briefing = await retrieve_and_prepare_rag_context(llm_client, rag_query_for_briefing)
                            prompt_nodes_for_briefing = await _build_initial_prompt_messages(
                                user_query_content=final_briefing_prompt_content, channel_id=interaction.channel_id,
                                bot_state=bot_state, user_id=str(interaction.user.id),
                                synthesized_rag_context_str=synthesized_rag_context_for_briefing, max_image_history_depth=0
                            )
                            await stream_llm_response_to_interaction(
                                interaction=interaction, llm_client=llm_client, bot_state=bot_state,
                                user_msg_node=user_msg_node_for_briefing, prompt_messages=prompt_nodes_for_briefing,
                                title=f"News Briefing: {topic}", synthesized_rag_context_for_display=synthesized_rag_context_for_briefing,
                                bot_user_id=news_data.bot_user_id,
                                force_new_followup_flow=True # Ensure new followup for worker job
                            )
                        except Exception as e_news:
                            logger.error(f"Error in worker processing /news for '{topic}': {e_news}", exc_info=True)
                            try:
                                await interaction.edit_original_response(content=f"An error occurred while generating your news briefing for '{topic}': {str(e_news)[:500]}")
                            except discord.HTTPException: pass
                        # --- End /news specific logic ---

                    elif request_type == 'ingest_command':
                        ingest_data: IngestCommandData = data
                        interaction = ingest_data.interaction
                        file_path = ingest_data.file_path
                        logger.info(f"Worker processing /ingest_chatgpt_export for file: {file_path}")
                        try:
                            parsed_conversations = parse_chatgpt_export(file_path)
                            if not parsed_conversations:
                                await interaction.followup.send("Could not parse any conversations from the file. It might be empty or in an unexpected format.", ephemeral=True)
                            else:
                                count = await store_chatgpt_conversations_in_chromadb(llm_client, parsed_conversations)
                                await interaction.followup.send(f"Successfully processed and stored {count} conversations (with distillations) from '{os.path.basename(file_path)}' into ChromaDB.", ephemeral=True)
                        except Exception as e_ingest:
                            logger.error(f"Error in worker processing /ingest_chatgpt_export for '{file_path}': {e_ingest}", exc_info=True)
                            try:
                                await interaction.followup.send(f"An error occurred during ingestion of '{os.path.basename(file_path)}': {str(e_ingest)[:500]}", ephemeral=True)
                            except discord.HTTPException: pass

                    elif request_type == 'search_command':
                        search_data: SearchCommandData = data
                        interaction = search_data.interaction
                        query = search_data.query
                        logger.info(f"Worker processing /search for query: {query}")
                        # Simplified: This would be a very complex expansion similar to /news.
                        # For now, just acknowledge and perhaps do a simple single LLM call.
                        # A full implementation would mirror the original /search logic here.
                        try:
                            await interaction.edit_original_response(content=f"Searching for '{query}' and preparing summary...")
                            # Placeholder for full search logic - for now, a simple response
                            search_placeholder_prompt = f"User searched for: {query}. Provide a placeholder acknowledgement or a very brief conceptual answer if possible, but indicate this is a placeholder for a full search result."
                            user_msg_node = MsgNode("user", search_placeholder_prompt, name=str(interaction.user.id))
                            prompt_nodes = await _build_initial_prompt_messages(user_query_content=search_placeholder_prompt, channel_id=interaction.channel_id, bot_state=bot_state, user_id=str(interaction.user.id))
                            await stream_llm_response_to_interaction(
                                interaction, llm_client, bot_state, user_msg_node, prompt_nodes,
                                title=f"Search Results for: {query} (Placeholder)", bot_user_id=search_data.bot_user_id,
                                force_new_followup_flow=True # Ensure new followup for worker job
                            )
                        except Exception as e_search:
                            logger.error(f"Error in worker processing /search for '{query}': {e_search}", exc_info=True)
                            try:
                                await interaction.edit_original_response(content=f"An error occurred while searching for '{query}': {str(e_search)[:500]}")
                            except discord.HTTPException: pass

                    elif request_type == 'gettweets_command':
                        gettweets_data: GetTweetsCommandData = data
                        interaction = gettweets_data.interaction
                        username = gettweets_data.username.lstrip('@')
                        limit = gettweets_data.limit
                        logger.info(f"Worker processing /gettweets for @{username}, limit {limit}")
                        try:
                            # Update Playwright usage time before potential use
                            if bot_state and hasattr(bot_state, 'update_last_playwright_usage_time'):
                                await bot_state.update_last_playwright_usage_time()

                            await interaction.edit_original_response(content=f"Scraping tweets for @{username} (up to {limit})...")
                            # The original command had a progress_callback. This is harder to manage here.
                            # The worker can edit the interaction response at stages.
                            tweets = await scrape_latest_tweets(username, limit=limit, progress_callback=None) # No progress cb for now

                            if not tweets:
                                await interaction.edit_original_response(content=f"Finished scraping for @{username}. No tweets found or profile might be private/inaccessible.")
                                bot_state.llm_request_queue.task_done()
                                continue

                            tweet_texts_for_display = [] # Simplified formatting from original
                            for t in tweets:
                                tweet_texts_for_display.append(f"@{t.get('username', username)}: {t.get('content', 'N/A')}")
                            raw_tweets_display_str = "\n\n".join(tweet_texts_for_display)

                            await interaction.edit_original_response(content=f"Formatting {len(tweets)} tweets for display...") # This might be too quick

                            embed_title = f"Recent Tweets from @{username}"
                            raw_tweet_chunks = chunk_text(raw_tweets_display_str, config.EMBED_MAX_LENGTH)
                            for i_tweet_chunk, chunk_content_part in enumerate(raw_tweet_chunks):
                                chunk_title_tweets = embed_title if i_tweet_chunk == 0 else f"{embed_title} (cont.)"
                                embed_tweets = discord.Embed(title=chunk_title_tweets, description=chunk_content_part, color=config.EMBED_COLOR["complete"])
                                if i_tweet_chunk == 0: await interaction.edit_original_response(content=None, embed=embed_tweets)
                                else: await interaction.followup.send(embed=embed_tweets)

                            user_query_content_for_summary = (
                                f"Please analyze and summarize the main themes, topics discussed, and overall sentiment "
                                f"from @{username}'s recent tweets provided below. Extract key points and present a concise yet detailed overview of this snapshot in time. "
                                f"Do not just re-list the tweets.\n\nRecent Tweets:\n{raw_tweets_display_str[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
                            )
                            user_msg_node_tweets = MsgNode("user", user_query_content_for_summary, name=str(interaction.user.id))
                            rag_query_tweets = f"summary of tweets from @{username}"
                            synth_rag_tweets = await retrieve_and_prepare_rag_context(llm_client, rag_query_tweets)
                            prompt_nodes_tweets = await _build_initial_prompt_messages(
                                user_query_content=user_query_content_for_summary, channel_id=interaction.channel_id,
                                bot_state=bot_state, user_id=str(interaction.user.id), synthesized_rag_context_str=synth_rag_tweets
                            )
                            await stream_llm_response_to_interaction(
                                interaction, llm_client, bot_state, user_msg_node_tweets, prompt_nodes_tweets,
                                title=f"Tweet Summary for @{username}", force_new_followup_flow=True,
                                synthesized_rag_context_for_display=synth_rag_tweets, bot_user_id=gettweets_data.bot_user_id
                            )
                        except Exception as e_gettweets:
                            logger.error(f"Error in worker processing /gettweets for @{username}: {e_gettweets}", exc_info=True)
                            try:
                                await interaction.edit_original_response(content=f"An error occurred while fetching tweets for @{username}: {str(e_gettweets)[:500]}")
                            except discord.HTTPException: pass

                    elif request_type == 'ap_command':
                        ap_data: APCommandData = data
                        interaction = ap_data.interaction
                        logger.info(f"Worker processing /ap command for image by {interaction.user.name}")
                        try:
                            image_url_for_llm = f"data:{ap_data.image_content_type};base64,{ap_data.image_b64}"

                            user_content_for_ap_node_list = [
                                {"type": "input_text", "text": ap_data.user_prompt_text},
                                {"type": "input_image", "image_url": image_url_for_llm}
                            ]
                            user_msg_node_ap = MsgNode("user", user_content_for_ap_node_list, name=ap_data.base_user_id_for_node)

                            base_prompt_nodes_ap = await _build_initial_prompt_messages(
                                user_query_content=user_content_for_ap_node_list,
                                channel_id=interaction.channel_id,
                                bot_state=bot_state,
                                user_id=ap_data.base_user_id_for_node,
                                synthesized_rag_context_str=ap_data.synthesized_rag_context,
                                max_image_history_depth=0
                            )
                            insert_idx_ap = 0
                            for idx, node in enumerate(base_prompt_nodes_ap):
                                if node.role != "system": insert_idx_ap = idx; break
                                insert_idx_ap = idx + 1
                            final_prompt_nodes_ap = base_prompt_nodes_ap[:insert_idx_ap] + \
                                                [MsgNode("system", ap_data.ap_system_task_prompt)] + \
                                                base_prompt_nodes_ap[insert_idx_ap:]

                            await stream_llm_response_to_interaction(
                                interaction, llm_client, bot_state, user_msg_node_ap, final_prompt_nodes_ap,
                                title=ap_data.title,
                                synthesized_rag_context_for_display=ap_data.synthesized_rag_context,
                                bot_user_id=ap_data.bot_user_id,
                                force_new_followup_flow=True # Ensure new followup for worker job
                            )
                        except Exception as e_ap:
                            logger.error(f"Error in worker processing /ap command: {e_ap}", exc_info=True)
                            try:
                                await interaction.edit_original_response(content=f"An error occurred with the /ap command: {str(e_ap)[:500]}")
                            except discord.HTTPException: pass
                    else:
                        logger.warning(f"Unknown LLM request type in queue: {request_type}")

                except Exception as e:
                    logger.error(f"Error processing queued LLM request (type: {request_type}): {e}", exc_info=True)
                    # Attempt to notify user if possible, depends on what 'data' holds
                    if hasattr(data, 'interaction') and isinstance(data.interaction, discord.Interaction):
                        try:
                            # Check if original response exists and is not done before editing
                            # Using followup as a safer bet if interaction is old.
                            await data.interaction.followup.send(f"Sorry, an unexpected error occurred while processing your '{request_type}' request. Please try again later.", ephemeral=True)
                        except discord.HTTPException:
                            logger.error(f"Failed to send error notification for request type {request_type}.")
                    elif hasattr(data, 'target_message') and isinstance(data.target_message, discord.Message):
                         try:
                            await data.target_message.reply(f"Sorry, an unexpected error occurred while processing your request. Please try again later.", mention_author=False)
                         except discord.HTTPException:
                            logger.error(f"Failed to send error notification reply for request type {request_type}.")
                finally:
                    logger.debug(f"LLM processing lock released by worker for request type '{request_type}'")
                    # Lock is auto-released by 'async with'

            bot_state.llm_request_queue.task_done()

        except asyncio.CancelledError:
            logger.info("LLM Request Processor Task was cancelled.")
            bot_state.llm_processor_task_active.clear()
            break
        except Exception as e:
            logger.critical(f"Critical unhandled error in LLM Request Processor Task loop: {e}", exc_info=True)
            # Avoid busy-looping on persistent errors; add a small delay
            await asyncio.sleep(5)

    logger.info("LLM Request Processor Task finished.")
    if bot_state.llm_processor_task_active.is_set(): # Should be cleared if loop exited cleanly
        bot_state.llm_processor_task_active.clear()
        logger.info("LLM Request Processor Task active event cleared on finish.")

# Example of how this task might be started in main_bot.py's on_ready:
# from llm_request_processor import llm_request_processor_task
# ...
# @bot.event
# async def on_ready():
#   ...
#   if bot_state_instance and llm_client_instance and bot_instance:
#      if not hasattr(bot_instance, 'llm_processor_task') or bot_instance.llm_processor_task.done():
#          logger.info("Starting LLM request processor task...")
#          bot_instance.llm_processor_task = asyncio.create_task(
#              llm_request_processor_task(bot_state_instance, llm_client_instance, bot_instance)
#          )
#   ...
