import logging
import discord # type: ignore
from discord import app_commands # Added this import
from discord.ext import commands, tasks # type: ignore
import base64 
import asyncio 
import os 
from datetime import datetime 
from typing import Any, Optional, List, Union, Dict 

# Bot services and utilities
from config import config
from state import BotState
from common_models import MsgNode # Import MsgNode

from llm_handling import (
    _build_initial_prompt_messages, 
    stream_llm_response_to_message
)
from rag_chroma_manager import (
    retrieve_and_prepare_rag_context
)
# Corrected import for detect_urls
from utils import detect_urls, cleanup_playwright_processes
from web_utils import scrape_website, fetch_youtube_transcript 
from audio_utils import transcribe_audio_file, send_tts_audio, load_whisper_model 

logger = logging.getLogger(__name__)

# Module-level globals to store instances passed from main_bot.py
bot_instance: Optional[commands.Bot] = None
llm_client_instance: Optional[Any] = None
bot_state_instance: Optional[BotState] = None


def setup_events_and_tasks(bot: commands.Bot, llm_client_in: Any, bot_state_in: BotState):
    global bot_instance, llm_client_instance, bot_state_instance
    bot_instance = bot
    llm_client_instance = llm_client_in
    bot_state_instance = bot_state_in

    # Ensure instances are set before defining events/tasks that use them
    if not bot_instance or not llm_client_instance or not bot_state_instance:
        logger.critical("Bot, LLM client, or BotState not properly initialized in setup_events_and_tasks. Events/tasks may fail.")
        return # Prevent events/tasks from being registered if setup is faulty


    @tasks.loop(seconds=30)
    async def check_reminders_task():
        if not bot_state_instance or not bot_instance: # Check instances
            logger.error("check_reminders_task: Bot state or bot instance not available.")
            return

        now = datetime.now()
        due_reminders_list = await bot_state_instance.pop_due_reminders(now) 
        
        for reminder_time, channel_id, user_id, message_content, original_time_str in due_reminders_list:
            logger.info(f"Reminder DUE for user {user_id} in channel {channel_id} at {reminder_time.strftime('%Y-%m-%d %H:%M:%S')}: {message_content}")
            try:
                channel = await bot_instance.fetch_channel(channel_id)
                user = await bot_instance.fetch_user(user_id) 
                
                if channel and user and isinstance(channel, discord.abc.Messageable): 
                    embed = discord.Embed(
                        title=f"⏰ Reminder! (Originally set for ~{original_time_str})", 
                        description=message_content, 
                        color=discord.Color.blue(),
                        timestamp=reminder_time 
                    )
                    embed.set_footer(text=f"This reminder was for {user.display_name}")
                    await channel.send(content=user.mention, embed=embed) 
                    
                    tts_reminder_text = f"Hey {user.display_name}, here's your reminder: {message_content}"
                    await send_tts_audio(channel, tts_reminder_text, base_filename=f"reminder_{user_id}_{channel_id}")
                else:
                    logger.warning(f"Could not fetch channel/user or channel not messageable for reminder: Channel ID {channel_id}, User ID {user_id}")
            except discord.errors.NotFound:
                logger.warning(f"Channel or User not found for reminder: Channel ID {channel_id}, User ID {user_id}. Reminder lost.")
            except Exception as e:
                logger.error(f"Failed to send reminder (Channel ID {channel_id}, User ID {user_id}): {e}", exc_info=True)

    @tasks.loop(minutes=10)
    async def cleanup_playwright_task():
        killed = cleanup_playwright_processes()
        if killed:
            logger.info(f"Cleaned up {killed} stray Chromium processes.")

    @bot_instance.event # type: ignore
    async def on_ready():
        if not bot_instance or not bot_instance.user: 
            logger.critical("Bot user not available on_ready. This is highly unusual.")
            return

        load_whisper_model() 

        logger.info(f'{bot_instance.user.name} (ID: {bot_instance.user.id}) has connected to Discord!')
        logger.info(f"discord.py version: {discord.__version__}")
        logger.info(f"Operating with LLM: {config.LLM_MODEL}, Vision: {config.VISION_LLM_MODEL}, FastLLM: {config.FAST_LLM_MODEL}")
        logger.info(f"Allowed Channel IDs: {config.ALLOWED_CHANNEL_IDS if config.ALLOWED_CHANNEL_IDS else 'All channels'}")
        logger.info(f"Allowed Role IDs: {config.ALLOWED_ROLE_IDS if config.ALLOWED_ROLE_IDS else 'No role restrictions (beyond channel/mention)'}")
        logger.info(f"User-Provided Global Context is {'SET' if config.USER_PROVIDED_CONTEXT else 'NOT SET'}.")
        logger.info(f"Max Image Bytes: {config.MAX_IMAGE_BYTES_FOR_PROMPT}, Max Scraped Text: {config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT}")
        logger.info(f"Stream Edit Throttle: {config.STREAM_EDIT_THROTTLE_SECONDS}s, RAG Sentences to Fetch: {config.RAG_NUM_DISTILLED_SENTENCES_TO_FETCH}")
        
        try:
            synced = await bot_instance.tree.sync()
            logger.info(f"Synced {len(synced)} slash commands to Discord.")
        except Exception as e:
            logger.error(f"Failed to sync slash commands: {e}", exc_info=True)

        if not check_reminders_task.is_running():
            check_reminders_task.start()
            logger.info("Reminders task started.")

        if not cleanup_playwright_task.is_running():
            cleanup_playwright_task.start()
            logger.info("Playwright cleanup task started.")

        await bot_instance.change_presence(activity=discord.Game(name="with /commands | Ask me anything!"))
        logger.info("Bot presence updated.")

    @bot_instance.event # type: ignore
    async def on_message(message: discord.Message):
        if not bot_instance or not bot_instance.user or not llm_client_instance or not bot_state_instance:
            logger.error("on_message: Bot components (bot, llm_client, bot_state) not ready. Skipping message processing.")
            return

        if message.author.bot: return 
        if message.channel is None or message.channel.id is None : 
            logger.warning(f"Message from {message.author.name} arrived without a channel or channel ID. Ignoring."); return

        prefixes = await bot_instance.get_prefix(message) 
        is_command_attempt = False
        if isinstance(prefixes, (list, tuple)):
            is_command_attempt = any(message.content.startswith(p) for p in prefixes)
        elif isinstance(prefixes, str):
            is_command_attempt = message.content.startswith(prefixes)
        
        if is_command_attempt:
            await bot_instance.process_commands(message) 
            return 

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = bot_instance.user in message.mentions
        
        channel_id = message.channel.id
        author_roles = getattr(message.author, 'roles', []) 
        
        allowed_by_channel = False
        if not config.ALLOWED_CHANNEL_IDS: 
            allowed_by_channel = True
        elif channel_id in config.ALLOWED_CHANNEL_IDS:
            allowed_by_channel = True
        elif isinstance(message.channel, discord.Thread) and message.channel.parent_id in config.ALLOWED_CHANNEL_IDS:
            allowed_by_channel = True

        allowed_by_role = False
        if not config.ALLOWED_ROLE_IDS or is_dm: 
            allowed_by_role = True
        else:
            allowed_by_role = any(role.id in config.ALLOWED_ROLE_IDS for role in author_roles)
        
        should_respond = is_dm or is_mentioned or (allowed_by_channel and allowed_by_role)
        
        if not should_respond:
            if not (is_dm or is_mentioned): 
                if not allowed_by_channel: logger.debug(f"Message from {message.author.name} in Channel ID {channel_id} ignored (channel not in ALLOWED_CHANNEL_IDS).")
                elif not allowed_by_role: logger.debug(f"Message from {message.author.name} in Channel ID {channel_id} ignored (user role not in ALLOWED_ROLE_IDS).")
            return 

        logger.info(f"General LLM processing for message from {message.author.name} in {getattr(message.channel, 'name', f'Channel ID {channel_id}')}: '{message.content[:50]}...'")
        
        current_message_content_parts: List[Dict[str, Any]] = [] 
        user_message_text_for_processing = message.content
        if f"<@{bot_instance.user.id}>" in user_message_text_for_processing:
            user_message_text_for_processing = user_message_text_for_processing.replace(f"<@{bot_instance.user.id}>", "").strip()
        elif f"<@!{bot_instance.user.id}>" in user_message_text_for_processing: 
             user_message_text_for_processing = user_message_text_for_processing.replace(f"<@!{bot_instance.user.id}>", "").strip()


        if message.attachments:
            for attachment in message.attachments:
                if attachment.content_type and attachment.content_type.startswith("audio/"):
                    try:
                        if not os.path.exists("temp"): os.makedirs("temp")
                        original_filename_parts = attachment.filename.split('.')
                        safe_suffix = "dat" 
                        if len(original_filename_parts) > 1:
                            raw_suffix = original_filename_parts[-1]
                            safe_suffix = "".join(c if c.isalnum() else "_" for c in raw_suffix)[:10] 
                        
                        audio_filename = f"temp/temp_audio_{attachment.id}.{safe_suffix}"
                        await attachment.save(audio_filename)
                        transcription = transcribe_audio_file(audio_filename) 
                        if os.path.exists(audio_filename): os.remove(audio_filename) 
                        
                        if transcription: 
                            capped_transcription = transcription[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]
                            user_message_text_for_processing = (f"[Audio Transcript from {attachment.filename}: {capped_transcription}] " + user_message_text_for_processing).strip()
                            logger.info(f"Added audio transcript (capped at {config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT} chars): {capped_transcription[:50]}...")
                    except Exception as e:
                        logger.error(f"Error processing audio attachment '{attachment.filename}': {e}", exc_info=True)
                    break 

        current_message_content_parts.append({"type": "text", "text": user_message_text_for_processing if user_message_text_for_processing else ""})

        image_added_to_prompt = False; images_processed_count = 0
        if message.attachments:
            for attachment in message.attachments:
                if images_processed_count >= config.MAX_IMAGES_PER_MESSAGE: break 
                if attachment.content_type and attachment.content_type.startswith("image/"):
                    try:
                        img_bytes = await attachment.read()
                        if len(img_bytes) > config.MAX_IMAGE_BYTES_FOR_PROMPT:
                            logger.warning(f"Image {attachment.filename} from {message.author.name} is too large ({len(img_bytes)} bytes). Skipping.")
                            for part in current_message_content_parts:
                                if part["type"] == "text":
                                    part["text"] += f" [Note: Attached image '{attachment.filename}' was too large to process.]"
                                    break
                            continue 

                        b64_img = base64.b64encode(img_bytes).decode('utf-8')
                        current_message_content_parts.append({"type": "image_url", "image_url": {"url": f"data:{attachment.content_type};base64,{b64_img}"}})
                        image_added_to_prompt = True; images_processed_count +=1
                        logger.info(f"Added image '{attachment.filename}' to prompt for message from {message.author.name}.")
                    except Exception as e:
                        logger.error(f"Error processing image attachment '{attachment.filename}': {e}", exc_info=True)
        
        text_part_exists_with_content = any(p["type"] == "text" and p.get("text","").strip() for p in current_message_content_parts)
        if image_added_to_prompt and not text_part_exists_with_content:
            text_part_updated = False
            for part in current_message_content_parts:
                if part["type"] == "text":
                    part["text"] = "User sent image(s). Please describe or respond based on the image(s)."
                    text_part_updated = True
                    break
            if not text_part_updated: 
                 current_message_content_parts.insert(0, {"type": "text", "text": "User sent image(s). Please describe or respond based on the image(s)."})


        current_text_for_url_detection = ""
        for part in current_message_content_parts:
            if part["type"] == "text":
                current_text_for_url_detection = part["text"]; # type: ignore
                break 
            
        scraped_content_accumulator = [] 
        if detected_urls_in_text := detect_urls(str(current_text_for_url_detection)): 
            for i, url in enumerate(detected_urls_in_text[:2]): 
                logger.info(f"Processing URL from message (index {i}): {url}")
                content_piece = None
                
                is_googleusercontent_youtube = "googleusercontent.com/youtube.com/" in url 
                youtube_indicators = ["youtube.com/watch", "youtu.be/", "youtube.com/shorts", "googleusercontent.com/youtube.com/"] # Removed /3, /1, /6
                is_any_youtube_type = any(indicator in url for indicator in youtube_indicators) or "youtube.com/watch?v=" in url or "youtu.be/" in url


                if is_any_youtube_type: 
                    transcript = await fetch_youtube_transcript(url) 
                    if transcript:
                        content_piece = f"\n\n--- YouTube Transcript for {url} ---\n{transcript}\n--- End Transcript ---"
                        logger.info(f"Successfully fetched YouTube transcript for {url}.")
                    elif not is_googleusercontent_youtube: # Only fallback if not the specific "0" type
                        logger.warning(f"YouTube transcript failed for {url}. Falling back to generic web scrape of the page.")
                        scraped_text = await scrape_website(url) 
                        if scraped_text and "Failed to scrape" not in scraped_text and "Scraping timed out" not in scraped_text:
                            content_piece = f"\n\n--- Webpage Content (fallback for YouTube URL {url}) ---\n{scraped_text}\n--- End Webpage Content ---"
                            logger.info(f"Fetched webpage content for {url} (YouTube transcript fallback).")
                        else:
                            logger.warning(f"Failed to get transcript or scrape fallback for YouTube URL {url}. Scraped_text: '{scraped_text}'")
                    else: 
                         logger.warning(f"Failed to get YouTube transcript for {url} (specific googleusercontent type '0'). Skipping generic web scrape for this pattern.")
                else: 
                    scraped_text = await scrape_website(url)
                    if scraped_text and "Failed to scrape" not in scraped_text and "Scraping timed out" not in scraped_text:
                        content_piece = f"\n\n--- Webpage Content for {url} ---\n{scraped_text}\n--- End Webpage Content ---"
                        logger.info(f"Fetched webpage content for non-YouTube URL: {url}.")
                    else:
                        logger.warning(f"Failed to scrape content for non-YouTube URL {url}. Scraped_text: '{scraped_text}'")
                
                if content_piece:
                    scraped_content_accumulator.append(content_piece)
                else:
                    notice = f"\n\nCould not retrieve content from {url}"
                    scraped_content_accumulator.append(notice)
                    logger.info(f"Appended notice for failed content retrieval: {url}")
                await asyncio.sleep(0.2)
        
        final_user_message_text_for_llm = user_message_text_for_processing 
        if scraped_content_accumulator:
            combined_scraped_content = "".join(scraped_content_accumulator)
            max_total_scraped_len = config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT * 1.5 
            if len(combined_scraped_content) > max_total_scraped_len: 
                combined_scraped_content = combined_scraped_content[:int(max_total_scraped_len)] + "\n[Combined scraped content truncated due to length]..."
            
            final_user_message_text_for_llm = combined_scraped_content + "\n\nUser's message (following processed URL content and/or audio transcript, if any): " + user_message_text_for_processing
        
        text_part_found_and_updated_with_scrape = False
        for part_idx, part_dict in enumerate(current_message_content_parts):
            if part_dict["type"] == "text":
                current_message_content_parts[part_idx]["text"] = final_user_message_text_for_llm
                text_part_found_and_updated_with_scrape = True
                break
        if not text_part_found_and_updated_with_scrape: 
            logger.error("Critical: Text part missing in current_message_content_parts before LLM call.")
            current_message_content_parts.insert(0, {"type": "text", "text": final_user_message_text_for_llm})

        final_content_is_empty = True
        has_text_content = False
        has_image_content = False

        for part in current_message_content_parts:
            if part.get("type") == "text" and str(part.get("text","")).strip():
                has_text_content = True
            if part.get("type") == "image_url":
                has_image_content = True
        
        if has_text_content or has_image_content:
            final_content_is_empty = False
            
        if final_content_is_empty:
            logger.info(f"Ignoring message from {message.author.name} as it resulted in no processable content after all stages."); return

        user_msg_node_content_final: Union[str, List[dict]]
        if len(current_message_content_parts) == 1 and current_message_content_parts[0]["type"] == "text" and not has_image_content:
            user_msg_node_content_final = current_message_content_parts[0]["text"]
        else:
            user_msg_node_content_final = current_message_content_parts # type: ignore
        
        rag_query_text = user_message_text_for_processing.strip() if user_message_text_for_processing.strip() else \
                         ("User sent an image/attachment" if image_added_to_prompt else "User sent a message with no textual content.")
        synthesized_rag_context = await retrieve_and_prepare_rag_context(llm_client_instance, rag_query_text)
        
        user_msg_node_for_short_term_history = MsgNode("user", user_msg_node_content_final, name=str(message.author.id))

        llm_prompt_for_current_turn = await _build_initial_prompt_messages(
            user_query_content=user_msg_node_content_final, 
            channel_id=channel_id,
            bot_state=bot_state_instance, 
            user_id=str(message.author.id),
            synthesized_rag_context_str=synthesized_rag_context 
        )
        
        await stream_llm_response_to_message(
            target_message=message, 
            llm_client=llm_client_instance, 
            bot_state=bot_state_instance,   
            user_msg_node=user_msg_node_for_short_term_history, 
            prompt_messages=llm_prompt_for_current_turn,
            synthesized_rag_context_for_display=synthesized_rag_context,
            bot_user_id=bot_instance.user.id 
        )

    @bot_instance.event # type: ignore
    async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
        if not bot_instance or not bot_instance.user : return 

        if payload.user_id == bot_instance.user.id or str(payload.emoji) != '❌': 
            return 
        if payload.channel_id is None: 
            return

        channel: Optional[discord.abc.Messageable] = None
        message_obj: Optional[discord.Message] = None 

        try:
            fetched_channel = await bot_instance.fetch_channel(payload.channel_id)
            if not isinstance(fetched_channel, discord.abc.Messageable):
                logger.warning(f"Channel {payload.channel_id} is not messageable for ❌ reaction.")
                return
            channel = fetched_channel
            message_obj = await channel.fetch_message(payload.message_id)
        except (discord.NotFound, discord.Forbidden): 
            logger.warning(f"Could not fetch message {payload.message_id} in channel {payload.channel_id} for ❌ reaction.")
            return 
        
        if message_obj is None or message_obj.author.id != bot_instance.user.id: 
            return 

        can_delete = False 
        user_who_reacted = None
        if isinstance(channel, (discord.TextChannel, discord.Thread)) and channel.guild:
            try:
                member = await channel.guild.fetch_member(payload.user_id) 
                user_who_reacted = member 
                if member and member.guild_permissions.manage_messages: 
                    can_delete = True
            except discord.HTTPException: 
                logger.debug(f"Could not fetch member {payload.user_id} in guild {channel.guild.id}.")
                pass 
        elif isinstance(channel, discord.DMChannel): 
            can_delete = True 
            try: user_who_reacted = await bot_instance.fetch_user(payload.user_id)
            except discord.NotFound: pass


        if not can_delete:
            if message_obj.interaction and message_obj.interaction.user.id == payload.user_id: 
                can_delete = True
            elif message_obj.reference and message_obj.reference.message_id and channel: 
                try: 
                    original_message = await channel.fetch_message(message_obj.reference.message_id)
                    if original_message.author.id == payload.user_id: 
                        can_delete = True
                except discord.NotFound: 
                    pass 

        if can_delete:
            try: 
                await message_obj.delete()
                reactor_name = user_who_reacted.name if user_who_reacted else f"User ID {payload.user_id}"
                logger.info(f"Message {message_obj.id} (sent by bot) deleted by ❌ reaction from {reactor_name}.")
            except discord.Forbidden:
                logger.warning(f"Bot lacked permissions to delete message {message_obj.id} despite can_delete logic.")
            except discord.HTTPException as e: 
                logger.error(f"Failed to delete message {message_obj.id} by reaction: {e}")
        else:
            logger.debug(f"User {payload.user_id} reacted with ❌ on message {message_obj.id}, but lacked deletion rights.")

    @bot_instance.event # type: ignore
    async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError): 
        command_name = interaction.command.name if interaction.command else 'unknown_slash_command'
        logger.error(f"Error in slash command '{command_name}' invoked by {interaction.user.name} ({interaction.user.id}): {error}", exc_info=True) 
        
        error_message = "An unexpected error occurred while trying to run this slash command."
        original_error_is_unknown_interaction = False

        if isinstance(error, app_commands.CommandInvokeError): 
            original_error = error.original
            if isinstance(original_error, discord.errors.NotFound) and original_error.code == 10062: 
                error_message = "The command took too long to respond initially, or the interaction may have expired. Please try the command again."
                logger.warning(f"Acknowledged 'Unknown Interaction' (Discord API error 10062) for command '{command_name}'. Interaction ID: {interaction.id}. User notified to retry.")
                original_error_is_unknown_interaction = True 
            else: 
                error_message = f"The command '{command_name}' encountered an issue: {str(original_error)[:500]}"
        elif isinstance(error, app_commands.CommandNotFound): 
            error_message = "That slash command was not found. This is unexpected if you used the UI."
        elif isinstance(error, app_commands.MissingPermissions): 
            error_message = f"You don't have the required permissions to use this command: {', '.join(error.missing_permissions)}"
        elif isinstance(error, app_commands.BotMissingPermissions): 
            error_message = f"I don't have the required permissions to perform this action: {', '.join(error.missing_permissions)}"
        elif isinstance(error, app_commands.CheckFailure): 
            error_message = "You do not meet the requirements to use this command."
        elif isinstance(error, app_commands.CommandOnCooldown): 
            error_message = f"This command is on cooldown. Please try again in {error.retry_after:.2f} seconds."
        elif isinstance(error, app_commands.TransformerError): 
            param_name = error.param.name if hasattr(error, 'param') and error.param else 'unknown_parameter'
            expected_type_str = str(error.type) if hasattr(error, 'type') else 'unknown_type'
            error_message = f"There was an issue with one of the arguments you provided for '{command_name}'. Parameter: {param_name}, Value: '{error.value}'. Expected type: {expected_type_str}."
        
        if original_error_is_unknown_interaction: 
            return 

        try:
            if interaction.response.is_done(): 
                await interaction.followup.send(error_message, ephemeral=True)
            else: 
                await interaction.response.send_message(error_message, ephemeral=True)
        except discord.errors.HTTPException as ehttp: 
            if ehttp.code == 40060: 
                logger.warning(f"Error handler: Interaction for '{command_name}' was already acknowledged. Attempting followup for error message. Original error: {error}")
                try: 
                    await interaction.followup.send(error_message, ephemeral=True) 
                except Exception as e_followup: 
                    logger.error(f"Error handler: Failed to send followup error message for '{command_name}' after 40060: {e_followup}")
            else: 
                logger.error(f"Error handler: An HTTPException occurred while trying to send error for '{command_name}': {ehttp}. Original error: {error}")
        except discord.errors.NotFound: 
            logger.error(f"Error handler: Interaction for '{command_name}' not found. It might have expired before error handling. Original error: {error}")
        except Exception as e_generic: 
            logger.error(f"Error handler: A generic error occurred while trying to send error for '{command_name}': {e_generic}. Original error: {error}")

    @bot_instance.event # type: ignore
    async def on_command_error(ctx: commands.Context, error: commands.CommandError): 
        command_name_str = ctx.command.name if ctx.command else "unknown_prefix_command"
        if isinstance(error, commands.CommandNotFound):
            logger.debug(f"Prefix command not found: {ctx.message.content.split()[0]} by {ctx.author.name}")
            pass 
        elif isinstance(error, commands.MissingRequiredArgument):
            param_name = error.param.name if hasattr(error, 'param') and error.param else 'unknown_argument'
            await ctx.reply(f"You're missing an argument for `!{command_name_str}`: `{param_name}`.", silent=True)
        elif isinstance(error, commands.BadArgument):
            await ctx.reply(f"That's not a valid argument for `!{command_name_str}`. Please check the expected type.", silent=True)
        elif isinstance(error, commands.CheckFailure): 
            await ctx.reply("You don't have permission to use that prefix command.", silent=True)
        elif isinstance(error, commands.CommandInvokeError):
            logger.error(f"Error invoking prefix command `!{command_name_str}` by {ctx.author.name}: {error.original}", exc_info=error.original)
            await ctx.reply(f"Oops! Something went wrong with `!{command_name_str}`: {str(error.original)[:500]}", silent=True)
        else:
            logger.error(f"An unhandled error occurred with a prefix command (`{ctx.invoked_with}` by {ctx.author.name}): {error}", exc_info=True)

