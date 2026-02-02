import logging
import re
import discord # type: ignore
from discord import app_commands
from discord.ext import commands, tasks # type: ignore
import base64
import asyncio
import os
from datetime import datetime, timedelta # Added timedelta
from typing import Any, Optional, List, Union, Dict

# Bot services and utilities
from config import config
from state import BotState
from common_models import MsgNode
import shutil # For deleting directories

from llm_handling import (
    _build_initial_prompt_messages,
    stream_llm_response_to_message,
    get_description_for_image,
    retrieve_rag_context_with_progress,
)
from utils import (
    detect_urls,
    cleanup_playwright_processes,
    append_absolute_dates,
)
from web_utils import scrape_website, fetch_youtube_transcript
from audio_utils import transcribe_audio_file, send_tts_audio
from timeline_pruner import prune_and_summarize
from scheduler import (
    run_allrss_digest,
    run_alltweets_digest,
    run_groundrss_digest,
    run_groundtopic_digest,
)

logger = logging.getLogger(__name__)

# Module-level globals to store instances passed from main_bot.py
bot_instance: Optional[commands.Bot] = None
llm_client_instance: Optional[Any] = None
bot_state_instance: Optional[BotState] = None # This will hold last_playwright_usage_time


def setup_events_and_tasks(bot: commands.Bot, llm_client_in: Any, bot_state_in: BotState):
    global bot_instance, llm_client_instance, bot_state_instance
    bot_instance = bot
    llm_client_instance = llm_client_in
    bot_state_instance = bot_state_in

    if not bot_instance or not llm_client_instance or not bot_state_instance:
        logger.critical("Bot, LLM client, or BotState not properly initialized in setup_events_and_tasks. Events/tasks may fail.")
        return

    @tasks.loop(seconds=30)
    async def check_reminders_task():
        if not bot_state_instance or not bot_instance:
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
                    await send_tts_audio(
                        channel,
                        tts_reminder_text,
                        base_filename=f"reminder_{user_id}_{channel_id}",
                        bot_state=bot_state_instance,
                    )
                else:
                    logger.warning(f"Could not fetch channel/user or channel not messageable for reminder: Channel ID {channel_id}, User ID {user_id}")
            except discord.errors.NotFound:
                logger.warning(f"Channel or User not found for reminder: Channel ID {channel_id}, User ID {user_id}. Reminder lost.")
            except Exception as e:
                logger.error(f"Failed to send reminder (Channel ID {channel_id}, User ID {user_id}): {e}", exc_info=True)

    @tasks.loop(minutes=config.PLAYWRIGHT_CLEANUP_INTERVAL_MINUTES if hasattr(config, 'PLAYWRIGHT_CLEANUP_INTERVAL_MINUTES') else 5) # Default to 5 min if not in config
    async def cleanup_playwright_task():
        logger.debug("Playwright cleanup task started.")
        try:
            if not bot_state_instance:
                logger.error("cleanup_playwright_task: BotState not available.")
                return

            last_usage = await bot_state_instance.get_last_playwright_usage_time()
            cleanup_threshold_minutes = config.PLAYWRIGHT_IDLE_CLEANUP_THRESHOLD_MINUTES if hasattr(config, 'PLAYWRIGHT_IDLE_CLEANUP_THRESHOLD_MINUTES') else 5 # Default to 5 min

            if last_usage:
                idle_time = datetime.now() - last_usage
                if idle_time < timedelta(minutes=cleanup_threshold_minutes):
                    logger.debug(f"Playwright cleanup skipped. Last used {idle_time.total_seconds()//60:.0f}m ago (threshold: {cleanup_threshold_minutes}m).")
                    return
                logger.info(f"Playwright idle for {idle_time.total_seconds()//60:.0f}m. Proceeding with cleanup check (threshold: {cleanup_threshold_minutes}m).")
            else:
                # If last_usage is None, it means either it was never used or cleaned up previously.
                # We can proceed to clean up any potential orphaned processes.
                logger.info("No recorded Playwright usage or previously cleaned. Proceeding with cleanup check.")

            killed = cleanup_playwright_processes()
            if killed:
                logger.info(f"Cleaned up {killed} stray Chromium/Playwright processes.")
                # After a successful cleanup, update the time, signifying active management.
                await bot_state_instance.update_last_playwright_usage_time()
            else:
                logger.debug("Playwright cleanup task ran: No stray processes found or killed.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in cleanup_playwright_task: {e}", exc_info=True)
        finally:
            logger.debug("Playwright cleanup task finished.")

    @tasks.loop(hours=24)
    async def timeline_pruner_task():
        logger.info("Timeline pruner task started.")
        prune_days = getattr(config, 'TIMELINE_PRUNE_DAYS', 30)
        try:
            await prune_and_summarize(prune_days)
            logger.info("Timeline pruner task completed successfully.")
        except Exception as e:
            logger.error(f"Timeline pruner task failed: {e}", exc_info=True)

    @tasks.loop(seconds=60)
    async def scheduler_task():
        """Check and run due background schedules (scoped per-channel)."""
        if not bot_state_instance or not bot_instance:
            return
        try:
            pause_state = await bot_state_instance.get_schedules_pause_state()
            if pause_state.get("paused"):
                already_logged = getattr(scheduler_task, "_pause_logged", False)
                if not already_logged:
                    reason = pause_state.get("reason") or "No reason provided"
                    paused_by = pause_state.get("paused_by")
                    paused_at = pause_state.get("paused_at")
                    logger.info(
                        "Scheduler paused by %s at %s. Reason: %s",
                        paused_by or "unknown user",
                        paused_at or "unknown time",
                        reason,
                    )
                    scheduler_task._pause_logged = True  # type: ignore[attr-defined]
                logger.debug("Scheduler: Pause is active, skipping all scheduled jobs.")
                return
            if getattr(scheduler_task, "_pause_logged", False):
                logger.info("Scheduler pause cleared. Resuming scheduled jobs.")
                scheduler_task._pause_logged = False  # type: ignore[attr-defined]

            schedules = await bot_state_instance.list_schedules()
            now = datetime.now()
            for s in schedules:
                try:
                    sched_id = s.get("id")
                    channel_id = int(s.get("channel_id"))
                    kind = s.get("type")
                    interval_seconds = int(s.get("interval_seconds", 0))
                    last_run_iso = s.get("last_run")
                    last_run_dt = None
                    if isinstance(last_run_iso, str):
                        try:
                            last_run_dt = datetime.fromisoformat(last_run_iso)
                        except Exception:
                            last_run_dt = None

                    due = False
                    if interval_seconds > 0:
                        if not last_run_dt:
                            due = True
                        else:
                            due = (now - last_run_dt).total_seconds() >= interval_seconds
                    else:
                        # If interval not set, skip
                        continue

                    if not due:
                        continue

                    # Check pause state again right before executing (in case it changed during iteration)
                    pause_state_check = await bot_state_instance.get_schedules_pause_state()
                    if pause_state_check.get("paused"):
                        logger.debug("Scheduler: Skipping due schedule %s (type: %s) - schedules are paused.", sched_id, kind)
                        continue

                    # Serialize execution per global scrape lock for heavy tasks
                    if kind == "allrss":
                        async with bot_state_instance.get_scrape_lock():
                            lim = int(s.get("params", {}).get("limit", 10))
                            try:
                                await run_allrss_digest(
                                    bot_instance,
                                    llm_client_instance,
                                    channel_id,
                                    limit=lim,
                                    bot_state=bot_state_instance,
                                )
                            except asyncio.CancelledError:
                                logger.info("Scheduler: digest for channel %s cancelled.", channel_id)
                                continue
                            await bot_state_instance.update_schedule_last_run(sched_id, now)
                    elif kind == "alltweets":
                        async with bot_state_instance.get_scrape_lock():
                            params = s.get("params", {}) or {}
                            lim = int(params.get("limit", 100))
                            list_name = str(params.get("list_name") or "")
                            scope_guild_raw = params.get("scope_guild_id")
                            scope_user_raw = params.get("scope_user_id")
                            scope_guild_id = None
                            scope_user_id = None
                            if scope_guild_raw is not None:
                                try:
                                    scope_guild_id = int(scope_guild_raw)
                                except (TypeError, ValueError):
                                    scope_guild_id = None
                            if scope_user_raw is not None:
                                try:
                                    scope_user_id = int(scope_user_raw)
                                except (TypeError, ValueError):
                                    scope_user_id = None
                            if scope_user_id is None and scope_guild_id is None:
                                created_by = s.get("created_by")
                                if created_by is not None:
                                    try:
                                        scope_user_id = int(created_by)
                                    except (TypeError, ValueError):
                                        scope_user_id = None
                            try:
                                await run_alltweets_digest(
                                    bot_instance,
                                    llm_client_instance,
                                    channel_id,
                                    limit=lim,
                                    list_name=list_name,
                                    scope_guild_id=scope_guild_id,
                                    scope_user_id=scope_user_id,
                                    bot_state=bot_state_instance,
                                )
                            except asyncio.CancelledError:
                                logger.info("Scheduler: alltweets for channel %s cancelled.", channel_id)
                                continue
                            await bot_state_instance.update_schedule_last_run(sched_id, now)
                    elif kind == "groundrss":
                        async with bot_state_instance.get_scrape_lock():
                            params = s.get("params", {}) or {}
                            lim = int(params.get("limit", 100))
                            try:
                                await run_groundrss_digest(
                                    bot_instance,
                                    llm_client_instance,
                                    channel_id,
                                    limit=lim,
                                    bot_state=bot_state_instance,
                                )
                            except asyncio.CancelledError:
                                logger.info("Scheduler: groundrss for channel %s cancelled.", channel_id)
                                continue
                            await bot_state_instance.update_schedule_last_run(sched_id, now)
                    elif kind == "groundtopic":
                        async with bot_state_instance.get_scrape_lock():
                            params = s.get("params", {}) or {}
                            lim = int(params.get("limit", 100))
                            topic_slug = params.get("topic")
                            if not topic_slug:
                                logger.warning("Scheduler: groundtopic schedule %s missing topic slug.", sched_id)
                                continue
                            try:
                                await run_groundtopic_digest(
                                    bot_instance,
                                    llm_client_instance,
                                    channel_id,
                                    topic_slug=topic_slug,
                                    limit=lim,
                                    bot_state=bot_state_instance,
                                )
                            except asyncio.CancelledError:
                                logger.info("Scheduler: groundtopic for channel %s cancelled.", channel_id)
                                continue
                            await bot_state_instance.update_schedule_last_run(sched_id, now)
                except Exception as inner:
                    logger.error("Scheduler: error running schedule %s: %s", s, inner, exc_info=True)
        except Exception as e:
            logger.error("Scheduler: loop error: %s", e, exc_info=True)

    scheduler_task._pause_logged = False  # type: ignore[attr-defined]


    @bot_instance.event # type: ignore
    async def on_ready():
        if not bot_instance or not bot_instance.user:
            logger.critical("Bot user not available on_ready. This is highly unusual.")
            return
        logger.info(f'{bot_instance.user.name} (ID: {bot_instance.user.id}) has connected to Discord!')
        # ... (rest of on_ready remains the same)
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

        if not timeline_pruner_task.is_running():
            timeline_pruner_task.start()
            logger.info("Timeline pruner task started.")

        if not scheduler_task.is_running():
            scheduler_task.start()
            logger.info("Scheduler task started.")

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

        # Content parts are stored in a standard Chat Completions format.
        text_type = "text"
        image_type = "image_url"

        current_message_content_parts: List[Dict[str, Any]] = []
        user_message_text_for_processing = message.content
        if f"<@{bot_instance.user.id}>" in user_message_text_for_processing:
            user_message_text_for_processing = user_message_text_for_processing.replace(f"<@{bot_instance.user.id}>", "").strip()
        elif f"<@!{bot_instance.user.id}>" in user_message_text_for_processing:
             user_message_text_for_processing = user_message_text_for_processing.replace(f"<@!{bot_instance.user.id}>", "").strip()

        if message.attachments:
            for attachment in message.attachments:
                if attachment.content_type and attachment.content_type.startswith("audio/"):
                    # ... (audio processing remains the same)
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

        user_message_text_for_processing = append_absolute_dates(
            user_message_text_for_processing
        )

        current_message_content_parts.append(
            {
                "type": text_type,
                "text": user_message_text_for_processing
                if user_message_text_for_processing
                else "",
            }
        )

        image_added_to_prompt = False; images_processed_count = 0
        if message.attachments:
            for attachment in message.attachments:
                # ... (image processing remains the same)
                if images_processed_count >= config.MAX_IMAGES_PER_MESSAGE: break
                if attachment.content_type and attachment.content_type.startswith("image/"):
                    try:
                        img_bytes = await attachment.read()
                        if len(img_bytes) > config.MAX_IMAGE_BYTES_FOR_PROMPT:
                            logger.warning(f"Image {attachment.filename} from {message.author.name} is too large ({len(img_bytes)} bytes). Skipping.")
                            for part in current_message_content_parts:
                                if part["type"] == text_type:
                                    part["text"] += f" [Note: Attached image '{attachment.filename}' was too large to process.]"
                                    break
                            continue

                        b64_img = base64.b64encode(img_bytes).decode('utf-8')
                        image_content_part: Dict[str, Any] = {
                            "type": image_type,
                            "image_url": {"url": f"data:{attachment.content_type};base64,{b64_img}"},
                        }
                        current_message_content_parts.append(image_content_part)
                        image_added_to_prompt = True; images_processed_count +=1
                        logger.info(f"Added image '{attachment.filename}' to prompt for message from {message.author.name}.")
                    except Exception as e:
                        logger.error(f"Error processing image attachment '{attachment.filename}': {e}", exc_info=True)

        text_part_exists_with_content = any(p["type"] == text_type and p.get("text","").strip() for p in current_message_content_parts)
        if image_added_to_prompt and not text_part_exists_with_content:
             # ... (image text part handling remains the same)
            text_part_updated = False
            for part in current_message_content_parts:
                if part["type"] == text_type:
                    part["text"] = "User sent image(s). Please describe or respond based on the image(s)."
                    text_part_updated = True
                    break
            if not text_part_updated:
                 current_message_content_parts.insert(0, {"type": text_type, "text": "User sent image(s). Please describe or respond based on the image(s)."})


        current_text_for_url_detection = ""
        for part in current_message_content_parts:
            if part["type"] == text_type:
                current_text_for_url_detection = part["text"];
                break

        scraped_content_accumulator = []
        just_urls_only = False
        detected_urls_in_text = detect_urls(str(current_text_for_url_detection)) or []
        # Determine if the user message text is only URLs (no other meaningful text)
        if detected_urls_in_text:
            remainder_text = str(current_text_for_url_detection)
            for _u in detected_urls_in_text:
                remainder_text = remainder_text.replace(_u, " ")
            # Strip common separators/punctuation; if nothing remains, it's URL-only
            remainder_text = re.sub(r"[\s\-–—:|,;()\[\]{}'\"“”’]+", "", remainder_text)
            if remainder_text == "":
                just_urls_only = True

        if detected_urls_in_text:
            playwright_used_in_loop = False
            temp_screenshots_base_dir = f"temp/screenshots_{message.id}"

            for i, url in enumerate(detected_urls_in_text[:2]):
                logger.info(f"Processing URL from message (index {i}): {url}")
                content_piece = None
                screenshot_descriptions_for_this_url = []

                # Create a unique subdir for each URL to avoid filename clashes if multiple URLs are processed
                current_url_screenshots_dir = os.path.join(temp_screenshots_base_dir, f"url_{i+1}")

                is_googleusercontent_youtube = "googleusercontent.com/youtube.com/" in url
                youtube_indicators = ["youtube.com/watch", "youtu.be/", "youtube.com/shorts", "googleusercontent.com/youtube.com/"]
                is_any_youtube_type = any(indicator in url for indicator in youtube_indicators) or "youtube.com/watch?v=" in url or "youtu.be/" in url

                if is_any_youtube_type:
                    if bot_state_instance: await bot_state_instance.update_last_playwright_usage_time()
                    playwright_used_in_loop = True
                    transcript = await fetch_youtube_transcript(url) # fetch_youtube_transcript does not take screenshots_dir
                    if transcript:
                        content_piece = f"\n\n--- YouTube Transcript for {url} ---\n{transcript}\n--- End Transcript ---"
                        logger.info(f"Successfully fetched YouTube transcript for {url}.")
                    # No screenshot fallback for YouTube transcript failures currently, as fetch_youtube_transcript handles its own fallback logic.
                    # If we wanted screenshots of the YouTube page itself, that would be a separate call to scrape_website.
                    elif not is_googleusercontent_youtube: # Fallback to generic scrape only if not the specific API-like URL
                        logger.warning(f"YouTube transcript failed for {url}. Falling back to generic web scrape of the page (will attempt screenshots).")
                        if bot_state_instance: await bot_state_instance.update_last_playwright_usage_time()
                        playwright_used_in_loop = True
                        scraped_text, screenshot_paths = await scrape_website(url, screenshots_dir=current_url_screenshots_dir)
                        if scraped_text and "Failed to scrape" not in scraped_text and "Scraping timed out" not in scraped_text and "Blocked from fetching URL" not in scraped_text:
                            content_piece = f"\n\n--- Webpage Content (fallback for YouTube URL {url}) ---\n{scraped_text}\n--- End Webpage Content ---"
                            logger.info(f"Fetched webpage content for {url} (YouTube transcript fallback).")
                            if screenshot_paths and llm_client_instance:
                                for ss_idx, ss_path in enumerate(screenshot_paths):
                                    desc = await get_description_for_image(llm_client_instance, ss_path)
                                    screenshot_descriptions_for_this_url.append(f"\n\n--- Image Description (Screenshot {ss_idx+1} for {url} fallback) ---\n{desc}\n--- End Image Description ---")
                        else:
                            logger.warning(f"Failed to get transcript or scrape fallback for YouTube URL {url}. Scraped_text: '{scraped_text}'")
                    else:
                         logger.warning(f"Failed to get YouTube transcript for {url} (specific googleusercontent type '0'). Skipping generic web scrape for this pattern.")

                else: # Non-YouTube URL, proceed with scraping and screenshots
                    if bot_state_instance: await bot_state_instance.update_last_playwright_usage_time()
                    playwright_used_in_loop = True
                    scraped_text, screenshot_paths = await scrape_website(url, screenshots_dir=current_url_screenshots_dir)
                    if scraped_text and "Failed to scrape" not in scraped_text and "Scraping timed out" not in scraped_text and "Blocked from fetching URL" not in scraped_text:
                        content_piece = f"\n\n--- Webpage Content for {url} ---\n{scraped_text}\n--- End Webpage Content ---"
                        logger.info(f"Fetched webpage content for non-YouTube URL: {url}.")
                        if screenshot_paths and llm_client_instance:
                            for ss_idx, ss_path in enumerate(screenshot_paths):
                                desc = await get_description_for_image(llm_client_instance, ss_path)
                                screenshot_descriptions_for_this_url.append(f"\n\n--- Image Description (Screenshot {ss_idx+1} for {url}) ---\n{desc}\n--- End Image Description ---")
                    else:
                        logger.warning(f"Failed to scrape content for non-YouTube URL {url}. Scraped_text: '{scraped_text}'")

                if screenshot_descriptions_for_this_url:
                    content_piece = ("" if content_piece is None else content_piece) + "".join(screenshot_descriptions_for_this_url)

                if content_piece:
                    scraped_content_accumulator.append(content_piece)
                else:
                    notice = f"\n\n[Could not retrieve text content or screenshot descriptions from {url}]"
                    scraped_content_accumulator.append(notice)
                    logger.info(f"Appended notice for failed content/description retrieval: {url}")
                await asyncio.sleep(0.2)

            if playwright_used_in_loop and bot_state_instance:
                 logger.debug("Playwright usage time updated due to URL processing in on_message.")

            if os.path.exists(temp_screenshots_base_dir):
                try:
                    shutil.rmtree(temp_screenshots_base_dir)
                    logger.info(f"Successfully deleted temporary screenshot directory: {temp_screenshots_base_dir}")
                except Exception as e_rmtree:
                    logger.error(f"Failed to delete temporary screenshot directory {temp_screenshots_base_dir}: {e_rmtree}")

        final_user_message_text_for_llm = user_message_text_for_processing
        if scraped_content_accumulator:
            combined_scraped_content = "".join(scraped_content_accumulator)

            # Truncate if necessary, but try to keep it generous for combined content
            # MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT is for a single page, this might be multiple things
            max_total_scraped_len = config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT * 2
            if len(combined_scraped_content) > max_total_scraped_len:
                combined_scraped_content = combined_scraped_content[:int(max_total_scraped_len)] + "\n[Combined scraped content and image descriptions truncated due to length]..."

            # Update the text part of current_message_content_parts
            # The image descriptions are now part of this text block.
            # Raw images are not sent to the main LLM if descriptions are generated.
            final_user_message_text_for_llm = combined_scraped_content + "\n\nUser's original message (following processed URL content, screenshot descriptions, and/or audio transcript, if any). Default task: Provide a detailed summary of the content, not a generic description, as they will not see the original content: " + user_message_text_for_processing

        if scraped_content_accumulator and not user_message_text_for_processing.strip() and detected_urls_in_text:
            final_user_message_text_for_llm += "\n\nProvide a detailed summary of the content, not a generic description, as I will not see it myself."

        # Update the text part in current_message_content_parts
        # We are NOT adding screenshot images themselves to current_message_content_parts here,
        # as we are using their descriptions instead.
        text_part_found_and_updated = False
        for part_idx, part_dict in enumerate(current_message_content_parts):
            if part_dict["type"] == text_type:
                current_message_content_parts[part_idx]["text"] = final_user_message_text_for_llm
                text_part_found_and_updated = True
                break
        if not text_part_found_and_updated: # Should ideally not happen if initialized correctly
            logger.error("Critical: Text part missing in current_message_content_parts before LLM call after URL processing.")
            current_message_content_parts.insert(0, {"type": text_type, "text": final_user_message_text_for_llm})

        # The rest of the logic determining user_msg_node_content_final based on current_message_content_parts
        # will now correctly use the text that includes scraped content and image descriptions.
        # Attached images (non-screenshots) are still handled as before.

        final_content_is_empty = True
        has_text_content = False
        has_image_content = False

        for part in current_message_content_parts:
            if part.get("type") == text_type and str(part.get("text","")).strip():
                has_text_content = True
            if part.get("type") == image_type:
                has_image_content = True

        if has_text_content or has_image_content:
            final_content_is_empty = False

        if final_content_is_empty:
            logger.info(f"Ignoring message from {message.author.name} as it resulted in no processable content after all stages."); return

        user_msg_node_content_final: Union[str, List[dict]]
        if len(current_message_content_parts) == 1 and current_message_content_parts[0]["type"] == text_type and not has_image_content:
            user_msg_node_content_final = current_message_content_parts[0]["text"]
        else:
            user_msg_node_content_final = current_message_content_parts

        rag_query_text = user_message_text_for_processing.strip() if user_message_text_for_processing.strip() else \
                         ("User sent an image/attachment" if image_added_to_prompt else "User sent a message with no textual content.")

        # Skip RAG for URL-only or attachment-only (no text) messages
        attachments_only = False
        try:
            attachments_only = (not str(user_message_text_for_processing).strip()) and bool(message.attachments)
        except Exception:
            attachments_only = False

        if just_urls_only or attachments_only:
            reason = "URL-only" if just_urls_only else "attachment-only"
            logger.info(f"Skipping RAG retrieval: message is {reason} after cleaning.")
            synthesized_rag_summary = ""
            raw_rag_snippets = []
        else:
            synthesized_rag_summary, raw_rag_snippets = await retrieve_rag_context_with_progress(
                llm_client=llm_client_instance,
                query=rag_query_text,
                channel=message.channel,
                initial_status="🔍 Searching memories for related context...",
                send_kwargs={"reference": message, "silent": True},
            )

        user_msg_node_for_short_term_history = MsgNode("user", user_msg_node_content_final, name=str(message.author.id))

        llm_prompt_for_current_turn = await _build_initial_prompt_messages(
            user_query_content=user_msg_node_content_final,
            channel_id=channel_id,
            bot_state=bot_state_instance,
            user_id=str(message.author.id),
            synthesized_rag_context_str=synthesized_rag_summary, # Pass the summary string
            raw_rag_snippets=raw_rag_snippets # Pass the raw snippets
        )

        await stream_llm_response_to_message(
            target_message=message,
            llm_client=llm_client_instance,
            bot_state=bot_state_instance,
            user_msg_node=user_msg_node_for_short_term_history,
            prompt_messages=llm_prompt_for_current_turn,
            synthesized_rag_context_for_display=synthesized_rag_summary, # Display summary
            bot_user_id=bot_instance.user.id,
            retrieved_snippets=raw_rag_snippets,
        )

    # ... (on_raw_reaction_add, on_app_command_error, on_command_error remain the same)
    @bot_instance.event
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

    @bot_instance.event
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

    @bot_instance.event
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
