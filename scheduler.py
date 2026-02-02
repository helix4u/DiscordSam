import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import discord

from config import config
from rag_chroma_manager import store_rss_summary, ingest_conversation_to_chromadb
from common_models import MsgNode
from rss_cache import load_seen_entries, save_seen_entries
from ground_news_cache import load_seen_links, save_seen_links
from utils import chunk_text, format_article_time
from web_utils import fetch_rss_entries, scrape_website, scrape_ground_news_my, scrape_ground_news_topic
from openai_api import create_chat_completion, extract_text
from logit_biases import LOGIT_BIAS_UNWANTED_TOKENS_STR
from llm_clients import get_llm_runtime

logger = logging.getLogger(__name__)


async def run_allrss_digest(
    bot: discord.Client,
    llm_client: Any,
    channel_id: int,
    *,
    limit: int = 10,
    bot_state: Optional[Any] = None,
    status_interaction: Optional[discord.Interaction] = None,
) -> None:
    """Post a combined RSS digest to a channel without requiring an interaction.

    The implementation mirrors the core behavior of /allrss but uses direct
    channel sends and avoids ephemeral progress messages.
    """
    try:
        ch = await bot.fetch_channel(channel_id)
        if not isinstance(ch, discord.abc.Messageable):
            logger.warning("Scheduled allrss: Channel %s is not messageable.", channel_id)
            return
    except Exception as e:
        logger.error("Scheduled allrss: Failed to fetch channel %s: %s", channel_id, e)
        return


    header: Optional[discord.Message] = None
    if status_interaction:
        try:
            if status_interaction.response.is_done():
                header = await status_interaction.followup.send(
                    content=f"Starting scheduled RSS digest (limit {limit}).",
                    ephemeral=True,
                    wait=True,
                )
            else:
                await status_interaction.response.defer(ephemeral=True)
                header = await status_interaction.followup.send(
                    content=f"Starting scheduled RSS digest (limit {limit}).",
                    ephemeral=True,
                    wait=True,
                )
        except Exception:
            header = None
    if header is None:
        header = await ch.send(
            content=(
                f"Starting scheduled RSS digest (limit {limit}). This may take a while…"
            )
        )

    total_summaries: List[str] = []

    if bot_state:
        await bot_state.set_active_task(channel_id, asyncio.current_task())
    from discord_commands import DEFAULT_RSS_FEEDS  # Lazy import to avoid cycles

    try:
        for name, feed_url in DEFAULT_RSS_FEEDS:
            try:
                entries = await fetch_rss_entries(feed_url)
            except Exception as e:
                logger.error("Scheduled allrss: fetch failed for %s: %s", feed_url, e)
                continue

            if not entries:
                continue

            seen_record = load_seen_entries()
            seen_ids = set(seen_record.get(feed_url, []))
            entries_sorted = sorted(
                [e for e in entries if e.get("pubDate_dt")],
                key=lambda e: e["pubDate_dt"],
                reverse=True,
            )
            new_entries: List[Dict[str, Any]] = []
            for ent in entries_sorted:
                guid_candidate = ent.get("guid") or ent.get("link") or ""
                if guid_candidate and guid_candidate in seen_ids:
                    continue
                new_entries.append(ent)

            if not new_entries:
                continue

            chunk_size = max(limit, 1)
            total_entries = len(new_entries)
            total_batches = (total_entries + chunk_size - 1) // chunk_size
            posted_count = 0

            for chunk_start in range(0, total_entries, chunk_size):
                chunk_entries = new_entries[chunk_start : chunk_start + chunk_size]
                chunk_summaries: List[str] = []
                fast_runtime = get_llm_runtime("fast")
                fast_client = fast_runtime.client
                fast_provider = fast_runtime.provider
                fast_logit_bias = (
                    LOGIT_BIAS_UNWANTED_TOKENS_STR
                    if fast_provider.supports_logit_bias
                    else None
                )

                for idx_within_chunk, ent in enumerate(chunk_entries, 1):
                    idx_global = chunk_start + idx_within_chunk
                    title = ent.get("title") or "Untitled"
                    link = ent.get("link") or ""
                    guid = ent.get("guid") or link
                    pub_date_dt = ent.get("pubDate_dt")
                    if not pub_date_dt:
                        pub_date_str = ent.get("pubDate")
                        if pub_date_str:
                            try:
                                pub_date_dt = parsedate_to_datetime(pub_date_str)
                                if pub_date_dt and pub_date_dt.tzinfo is None:
                                    pub_date_dt = pub_date_dt.replace(tzinfo=timezone.utc)
                            except Exception:
                                pub_date_dt = None
                    pub_date = (
                        pub_date_dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")
                        if pub_date_dt
                        else (ent.get("pubDate") or "")
                    )

                    status_text = f"[{name}] {idx_global}/{total_entries}: Scraping {title}…"
                    if header:
                        try:
                            await header.delete()
                        except Exception:
                            pass
                        header = None

                    if status_interaction:
                        try:
                            header = await status_interaction.followup.send(
                                content=status_text,
                                ephemeral=True,
                                wait=True,
                            )
                        except Exception:
                            header = None
                    if header is None:
                        header = await ch.send(status_text)

                    scraped_text, _ = await scrape_website(link)
                    if (
                        not scraped_text
                        or "Failed to scrape" in scraped_text
                        or "Scraping timed out" in scraped_text
                        or "Blocked from fetching URL" in scraped_text
                    ):
                        summary_entry = (
                            f"[{name}] **{title}**\n{pub_date}\n{link}\nCould not scrape article\n"
                        )
                        chunk_summaries.append(summary_entry)
                        seen_ids.add(guid)
                        continue

                    prompt = (
                        "You are an expert news summarizer. Summarize the following article in 3-5 sentences. "
                        "Focus on key facts and avoid fluff.\n\n"
                        f"Title: {title}\nURL: {link}\n\n{scraped_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
                    )
                    try:
                        response = await create_chat_completion(
                            fast_client,
                            [
                                {"role": "system", "content": "You are an expert news summarizer."},
                                {"role": "user", "content": prompt},
                            ],
                            model=fast_provider.model,
                            max_tokens=3072,
                            temperature=fast_provider.temperature,
                            logit_bias=fast_logit_bias,
                            use_responses_api=fast_provider.use_responses_api,
                        )
                        summary = (
                            extract_text(response, fast_provider.use_responses_api)
                            or "[LLM summarization failed]"
                        )
                        await store_rss_summary(
                            feed_url=feed_url,
                            article_url=link,
                            title=title,
                            summary_text=summary,
                            timestamp=datetime.now(),
                        )
                    except Exception as e_summ:
                        logger.error("Scheduled allrss: summarize failed for %s: %s", link, e_summ)
                        summary = "[LLM summarization failed]"

                    summary_entry = f"[{name}] **{title}**\n{pub_date}\n{link}\n{summary}\n"
                    chunk_summaries.append(summary_entry)
                    seen_ids.add(guid)

                    # Per-article: embed (title, time, summary) then link so Discord shows link preview
                    time_str = format_article_time(pub_date_dt)
                    desc = (time_str + "\n\n" + summary) if time_str else summary
                    if len(desc) > config.EMBED_MAX_LENGTH:
                        desc = desc[: config.EMBED_MAX_LENGTH - 3] + "..."
                    title_display = (title[: 253] + "...") if len(title) > 256 else title
                    article_embed = discord.Embed(
                        title=title_display,
                        description=desc,
                        color=config.EMBED_COLOR.get("complete"),
                    )
                    await ch.send(embed=article_embed)
                    await ch.send(content=link)

                    await asyncio.sleep(0.2)

                if not chunk_summaries:
                    continue

                posted_count += len(chunk_summaries)
                total_summaries.extend(chunk_summaries)
                combined_chunk = "\n\n".join(chunk_summaries)
                try:
                    from audio_utils import send_tts_audio

                    await send_tts_audio(
                        ch,
                        combined_chunk,
                        base_filename=f"scheduled_rss_{channel_id}_{name}_batch{chunk_start // chunk_size + 1}",
                        bot_state=bot_state,
                    )
                except Exception as tts_exc:
                    logger.error("Scheduled allrss: TTS failed for %s: %s", name, tts_exc)

                if bot_state:
                    try:
                        user_node = MsgNode(
                            "user",
                            f"Scheduled digest chunk for {name} batch {chunk_start // chunk_size + 1}",
                            name=str(channel_id),
                        )
                        assistant_node = MsgNode(
                            "assistant",
                            combined_chunk,
                            name=str(bot.user.id) if bot.user else None,
                        )
                        ingest_task = ingest_conversation_to_chromadb(
                            llm_client,
                            channel_id,
                            channel_id,
                            [user_node, assistant_node],
                            None,
                        )
                        await ingest_task
                    except Exception as ingest_exc:
                        logger.error("Scheduled allrss: Failed to ingest chunk into RAG: %s", ingest_exc, exc_info=True)

            if posted_count:
                completion_text = (
                    f"✅ Finished processing **{name}**. Posted {posted_count} summary(ies)."
                )
                if header:
                    try:
                        await header.delete()
                    except Exception:
                        pass
                    header = None
                if status_interaction:
                    try:
                        header = await status_interaction.followup.send(
                            content=completion_text,
                            ephemeral=True,
                            wait=True,
                        )
                    except Exception:
                        header = None
                if header is None:
                    header = await ch.send(completion_text)

            seen_record[feed_url] = list(seen_ids)
            save_seen_entries(seen_record)

        if header:
            if not total_summaries:
                try:
                    await header.edit(content="Scheduled RSS digest found nothing new.")
                except Exception:
                    pass
            else:
                try:
                    await header.edit(content=f"Scheduled RSS digest completed. Posted {len(total_summaries)} article summaries.")
                except Exception:
                    pass
    except asyncio.CancelledError:
        if header:
            try:
                await header.edit(content="Scheduled RSS digest cancelled.")
            except Exception:
                pass
        await ch.send("Scheduled RSS digest cancelled.")
        raise
    finally:
        if bot_state:
            await bot_state.clear_active_task(channel_id)


async def run_alltweets_digest(
    bot: discord.Client,
    llm_client: Any,
    channel_id: int,
    *,
    limit: int = 100,
    list_name: str = "",
    scope_guild_id: Optional[int] = None,
    scope_user_id: Optional[int] = None,
    bot_state: Optional[Any] = None,
) -> None:
    """Post a combined tweets digest to a channel without relying on an interaction."""

    from discord_commands import (  # Local import to avoid circular references
        DEFAULT_TWITTER_USERS,
        SUMMARY_SYSTEM_PROMPT,
        TEMPORAL_SYSTEM_CONTEXT,
        _collect_new_tweets,
    )

    try:
        ch = await bot.fetch_channel(channel_id)
        if not isinstance(ch, discord.abc.Messageable):
            logger.warning("Scheduled alltweets: Channel %s is not messageable.", channel_id)
            return
    except Exception as exc:
        logger.error("Scheduled alltweets: Failed to fetch channel %s: %s", channel_id, exc)
        return

    handle_list = DEFAULT_TWITTER_USERS
    list_descriptor = "default accounts"
    channel_guild_id = getattr(getattr(ch, "guild", None), "id", None)
    effective_guild_id: Optional[int]
    effective_user_id: Optional[int]

    if scope_guild_id is not None:
        try:
            effective_guild_id = int(scope_guild_id)
        except (TypeError, ValueError):
            effective_guild_id = None
    else:
        effective_guild_id = channel_guild_id

    if effective_guild_id is not None:
        try:
            effective_guild_id = int(effective_guild_id)
        except (TypeError, ValueError):
            effective_guild_id = None

    if effective_guild_id is None and scope_user_id is not None:
        try:
            effective_user_id = int(scope_user_id)
        except (TypeError, ValueError):
            effective_user_id = None
    else:
        effective_user_id = None

    if bot_state:
        try:
            if not list_name:
                default_handles = await bot_state.get_twitter_list_handles(
                    effective_guild_id,
                    "default",
                    user_id=effective_user_id,
                )
                if not default_handles:
                    await bot_state.set_twitter_list(
                        effective_guild_id,
                        "default",
                        DEFAULT_TWITTER_USERS,
                        user_id=effective_user_id,
                    )
                    default_handles = DEFAULT_TWITTER_USERS
                handle_list = default_handles or DEFAULT_TWITTER_USERS
            else:
                handles = await bot_state.get_twitter_list_handles(
                    effective_guild_id,
                    list_name,
                    user_id=effective_user_id,
                )
                if handles:
                    handle_list = handles
                    list_descriptor = f"list `{list_name}`"
                else:
                    scope_label = (
                        f"guild {effective_guild_id}"
                        if effective_guild_id is not None
                        else f"user {effective_user_id}" if effective_user_id is not None
                        else "unknown scope"
                    )
                    logger.warning(
                        "Scheduled alltweets: Saved list '%s' empty for %s. Falling back to defaults.",
                        list_name,
                        scope_label,
                    )
        except Exception as exc:
            logger.error(
                "Scheduled alltweets: Failed retrieving list '%s' (guild=%s user=%s): %s",
                list_name,
                effective_guild_id,
                effective_user_id,
                exc,
            )
    elif list_name:
        logger.warning(
            "Scheduled alltweets: Cannot resolve saved list '%s' without bot state.",
            list_name,
        )

    header: Optional[discord.Message] = None
    try:
        header = await ch.send(
            f"Starting scheduled all-tweets digest using {list_descriptor} (limit {limit}). This may take a bit…"
        )
    except Exception:
        header = None

    async def update_status(text: str) -> None:
        if header:
            try:
                await header.edit(content=text[:2000])
            except Exception:
                pass

    if bot_state:
        await bot_state.set_active_task(channel_id, asyncio.current_task())

    main_runtime = get_llm_runtime("main")
    main_client = main_runtime.client
    main_provider = main_runtime.provider
    main_logit_bias = LOGIT_BIAS_UNWANTED_TOKENS_STR if main_provider.supports_logit_bias else None

    any_new = False
    try:
        for username in handle_list:
            clean_username = username.lstrip("@")
            await update_status(
                f"Fetching tweets for @{clean_username} (limit {limit})…"
            )
            result = await _collect_new_tweets(
                clean_username,
                limit=limit,
                progress_callback=update_status,
                source_command="/schedule_alltweets",
            )
            if result.status_message and not result.new_tweets:
                logger.info(
                    "Scheduled alltweets: %s",
                    result.status_message,
                )
                continue

            any_new = True
            embed_title = f"Scheduled Tweets from @{clean_username}"
            for idx, chunk in enumerate(result.embed_chunks or ["No tweet content available."]):
                embed = discord.Embed(
                    title=embed_title if idx == 0 else f"{embed_title} (cont.)",
                    description=chunk,
                    color=config.EMBED_COLOR["complete"],
                )
                await ch.send(embed=embed)

            if bot_state and result.raw_display_str.strip():
                user_node = MsgNode(
                    "user",
                    f"/schedule_alltweets @{clean_username} snapshot (limit {limit})",
                    name="scheduled",
                )
                assistant_node = MsgNode(
                    "assistant",
                    result.raw_display_str,
                    name=str(bot.user.id) if bot.user else None,
                )
                try:
                    await bot_state.append_history(
                        channel_id,
                        user_node,
                        config.MAX_MESSAGE_HISTORY,
                    )
                    await bot_state.append_history(
                        channel_id,
                        assistant_node,
                        config.MAX_MESSAGE_HISTORY,
                    )
                    try:
                        await ingest_conversation_to_chromadb(
                            llm_client,
                            channel_id,
                            channel_id,
                            [user_node, assistant_node],
                            None,
                        )
                    except Exception as ingest_exc:
                        logger.error(
                            "Scheduled alltweets: Failed ingesting tweets for @%s: %s",
                            clean_username,
                            ingest_exc,
                            exc_info=True,
                        )
                except Exception as history_exc:
                    logger.error(
                        "Scheduled alltweets: Failed updating history for channel %s: %s",
                        channel_id,
                        history_exc,
                        exc_info=True,
                    )

            summary_prompt = (
                f"Please analyze and summarize the main themes, topics discussed, and overall sentiment "
                f"from @{clean_username}'s recent tweets provided below. Extract key points and present a concise yet "
                f"detailed overview appropriate for a scheduled digest. Do not just re-list the tweets."
                f"\n\nRecent Tweets:\n{result.raw_display_str[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
            )
            try:
                response = await create_chat_completion(
                    main_client,
                    [
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "system", "content": TEMPORAL_SYSTEM_CONTEXT},
                        {"role": "user", "content": summary_prompt},
                    ],
                    model=main_provider.model,
                    max_tokens=3072,
                    temperature=main_provider.temperature,
                    logit_bias=main_logit_bias,
                    use_responses_api=main_provider.use_responses_api,
                )
                summary_text = extract_text(response, main_provider.use_responses_api)
            except Exception as summary_exc:
                logger.error(
                    "Scheduled alltweets: Failed to summarize tweets for @%s: %s",
                    clean_username,
                    summary_exc,
                    exc_info=True,
                )
                summary_text = None

            if summary_text:
                summary_embed = discord.Embed(
                    title=f"Tweet Summary for @{clean_username}",
                    description=summary_text[: config.EMBED_MAX_LENGTH],
                    color=config.EMBED_COLOR["complete"],
                )
                await ch.send(embed=summary_embed)
                try:
                    from audio_utils import send_tts_audio

                    await send_tts_audio(
                        ch,
                        summary_text,
                        base_filename=f"scheduled_alltweets_summary_{clean_username}_{channel_id}",
                        bot_state=bot_state,
                    )
                except Exception as tts_exc:
                    logger.error(
                        "Scheduled alltweets: TTS failed for summary of @%s in channel %s: %s",
                        clean_username,
                        channel_id,
                        tts_exc,
                    )
    except asyncio.CancelledError:
        if header:
            try:
                await header.edit(content="Scheduled all-tweets digest cancelled.")
            except Exception:
                pass
        await ch.send("Scheduled all-tweets digest cancelled.")
        raise
    finally:
        if header:
            try:
                if any_new:
                    await header.edit(
                        content=(
                            f"Scheduled all-tweets digest complete for {list_descriptor}. "
                            "New tweets have been posted above."
                        )
                    )
                else:
                    await header.edit(
                        content=f"Scheduled all-tweets digest finished with no new tweets for {list_descriptor}."
                    )
            except Exception:
                pass
        if bot_state:
            await bot_state.clear_active_task(channel_id)


async def run_groundrss_digest(
    bot: discord.Client,
    llm_client: Any,
    channel_id: int,
    *,
    limit: int = 100,
    bot_state: Optional[Any] = None,
) -> None:
    """Post a scheduled Ground News 'My Feed' digest into a channel."""

    from discord_commands import SUMMARY_SYSTEM_PROMPT  # Local import to avoid cycles

    try:
        ch = await bot.fetch_channel(channel_id)
        if not isinstance(ch, discord.abc.Messageable):
            logger.warning("Scheduled groundrss: Channel %s is not messageable.", channel_id)
            return
    except Exception as exc:
        logger.error("Scheduled groundrss: Failed to fetch channel %s: %s", channel_id, exc)
        return

    header: Optional[discord.Message] = None
    try:
        header = await ch.send(
            f"Starting scheduled Ground News 'My Feed' digest (limit {limit}). This may take a moment…"
        )
    except Exception:
        header = None

    async def update_status(text: str) -> None:
        if header:
            try:
                await header.edit(content=text[:2000])
            except Exception:
                pass

    if bot_state:
        await bot_state.set_active_task(channel_id, asyncio.current_task())

    seen_urls = load_seen_links()
    try:
        articles = await scrape_ground_news_my(limit)
    except Exception as exc:
        logger.error("Scheduled groundrss: Failed scraping Ground News My Feed: %s", exc, exc_info=True)
        articles = []

    new_articles = [article for article in articles if article.url not in seen_urls]
    if not new_articles:
        if header:
            try:
                await header.edit(content="Scheduled Ground News digest found nothing new.")
            except Exception:
                pass
        if not articles:
            await ch.send("Scheduled Ground News digest failed to fetch any articles.")
        else:
            await ch.send("Scheduled Ground News digest is already up to date.")
        if bot_state:
            await bot_state.clear_active_task(channel_id)
        return

    fast_runtime = get_llm_runtime("fast")
    fast_client = fast_runtime.client
    fast_provider = fast_runtime.provider
    fast_logit_bias = (
        LOGIT_BIAS_UNWANTED_TOKENS_STR if fast_provider.supports_logit_bias else None
    )

    summaries: List[str] = []
    try:
        for idx, article in enumerate(new_articles[:limit], 1):
            await update_status(f"Scraping {idx}/{min(len(new_articles), limit)}: {article.title}")
            if idx > 1:
                await asyncio.sleep(config.GROUND_NEWS_ARTICLE_DELAY_SECONDS)

            scraped_text, _ = await scrape_website(article.url)
            if (
                not scraped_text
                or "Failed to scrape" in scraped_text
                or "Scraping timed out" in scraped_text
                or "Blocked from fetching URL" in scraped_text
            ):
                summaries.append(f"**{article.title}**\n{article.url}\nCould not scrape article\n")
                seen_urls.add(article.url)
                continue

            prompt = (
                "[It is currently 2025 and Donald Trump is the current president. Biden IS NOT THE CURRENT PRESIDENT! "
                "BIDEN IS THE EX PRESIDENT AND TRUMP IS THE PRESIDENT AGAIN FOR HIS SECOND TERM. THIS IS JUST KEY "
                "TEMPORAL KNOWLEDGE TO ORIENT YOU WHEN IT COMES UP. ANSWER ACCORDINGLY!] "
                "(Just an FYI. Maybe unrelated to context and omitted IF NOT RELATED TO TRUMP OR BIDEN OR THE YEAR. "
                "I NEED THE BEHAVIOR OF CALLING TRUMP THE EX-PRESIDENT AND CALLING THE CURRENT ADMINISTRATION THE BIDEN ADMINISTRATIONL. "
                "THESE ERRORS ARE UNACCEPTABLE. DON'T MENTION ME YELLING AT YOU. JUST DO THE DAMN THING!). "
                "Do not use em dashes. Summarize the following article in 3-5 sentences. "
                "Focus on key facts. Present in a casual, blunt, honest and slightly profane tone. "
                "Do NOT start with 'So, ' or end with 'Basically, '. Do not state things like 'This article describes', "
                "'The article', etc. Present is as a person would if they were talking to you about the article.\n\n"
                f"Title: {article.title}\nURL: {article.url}\n\n{scraped_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
            )

            try:
                response = await create_chat_completion(
                    fast_client,
                    [
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=fast_provider.model,
                    max_tokens=3072,
                    temperature=fast_provider.temperature,
                    logit_bias=fast_logit_bias,
                    use_responses_api=fast_provider.use_responses_api,
                )
                summary = extract_text(response, fast_provider.use_responses_api) or "[LLM summarization failed]"
                await store_rss_summary(
                    feed_url="ground_news_my",
                    article_url=article.url,
                    title=article.title,
                    summary_text=summary,
                    timestamp=datetime.now(),
                )
            except Exception as exc:
                logger.error("Scheduled groundrss: LLM summarization failed for %s: %s", article.url, exc)
                summary = "[LLM summarization failed]"

            summary_line = f"**{article.title}**\n{article.url}\n{summary}\n"
            summaries.append(summary_line)
            seen_urls.add(article.url)

            if bot_state:
                user_node = MsgNode(
                    "user",
                    f"/schedule_groundrss article {idx} (limit {limit})",
                    name="scheduled",
                )
                assistant_node = MsgNode(
                    "assistant",
                    summary_line,
                    name=str(bot.user.id) if bot.user else None,
                )
                try:
                    await bot_state.append_history(
                        channel_id,
                        user_node,
                        config.MAX_MESSAGE_HISTORY,
                    )
                    await bot_state.append_history(
                        channel_id,
                        assistant_node,
                        config.MAX_MESSAGE_HISTORY,
                    )
                    try:
                        await ingest_conversation_to_chromadb(
                            llm_client,
                            channel_id,
                            channel_id,
                            [user_node, assistant_node],
                            None,
                        )
                    except Exception as ingest_exc:
                        logger.error(
                            "Scheduled groundrss: Failed ingesting summary for %s: %s",
                            article.url,
                            ingest_exc,
                            exc_info=True,
                        )
                except Exception as history_exc:
                    logger.error(
                        "Scheduled groundrss: Failed updating history for channel %s: %s",
                        channel_id,
                        history_exc,
                        exc_info=True,
                    )

        save_seen_links(seen_urls)
        combined = "\n\n".join(summaries)
        for idx, chunk in enumerate(chunk_text(combined, config.EMBED_MAX_LENGTH)):
            embed = discord.Embed(
                title="Ground News Summaries" + ("" if idx == 0 else f" (cont. {idx + 1})"),
                description=chunk,
                color=config.EMBED_COLOR["complete"],
            )
            await ch.send(embed=embed)
        try:
            from audio_utils import send_tts_audio

            await send_tts_audio(
                ch,
                combined,
                base_filename=f"scheduled_groundrss_{channel_id}",
                bot_state=bot_state,
            )
        except Exception as tts_exc:
            logger.error("Scheduled groundrss: TTS failed for channel %s: %s", channel_id, tts_exc)
    except asyncio.CancelledError:
        if header:
            try:
                await header.edit(content="Scheduled Ground News digest cancelled.")
            except Exception:
                pass
        await ch.send("Scheduled Ground News digest cancelled.")
        raise
    finally:
        if header:
            try:
                await header.edit(content="Scheduled Ground News digest completed.")
            except Exception:
                pass
        if bot_state:
            await bot_state.clear_active_task(channel_id)


async def run_groundtopic_digest(
    bot: discord.Client,
    llm_client: Any,
    channel_id: int,
    *,
    topic_slug: str,
    limit: int = 100,
    bot_state: Optional[Any] = None,
) -> None:
    """Post a scheduled Ground News topic digest into a channel."""

    from discord_commands import GROUND_NEWS_TOPICS, SUMMARY_SYSTEM_PROMPT  # Local import

    try:
        ch = await bot.fetch_channel(channel_id)
        if not isinstance(ch, discord.abc.Messageable):
            logger.warning("Scheduled groundtopic: Channel %s is not messageable.", channel_id)
            return
    except Exception as exc:
        logger.error("Scheduled groundtopic: Failed to fetch channel %s: %s", channel_id, exc)
        return

    topic_entry = GROUND_NEWS_TOPICS.get(topic_slug)
    if not topic_entry:
        await ch.send(f"Scheduled Ground News topic `{topic_slug}` is not configured.")
        return
    topic_name, topic_url = topic_entry

    header: Optional[discord.Message] = None
    try:
        header = await ch.send(
            f"Starting scheduled Ground News topic digest for **{topic_name}** (limit {limit})."
        )
    except Exception:
        header = None

    async def update_status(text: str) -> None:
        if header:
            try:
                await header.edit(content=text[:2000])
            except Exception:
                pass

    if bot_state:
        await bot_state.set_active_task(channel_id, asyncio.current_task())

    seen_urls = load_seen_links()
    try:
        articles = await scrape_ground_news_topic(topic_url, limit)
    except Exception as exc:
        logger.error("Scheduled groundtopic: Failed scraping %s: %s", topic_url, exc, exc_info=True)
        articles = []

    new_articles = [article for article in articles if article.url not in seen_urls]
    if not new_articles:
        if header:
            try:
                await header.edit(content=f"Scheduled Ground News topic `{topic_name}` found nothing new.")
            except Exception:
                pass
        await ch.send(f"Scheduled Ground News topic `{topic_name}` is already up to date.")
        if bot_state:
            await bot_state.clear_active_task(channel_id)
        return

    fast_runtime = get_llm_runtime("fast")
    fast_client = fast_runtime.client
    fast_provider = fast_runtime.provider
    fast_logit_bias = (
        LOGIT_BIAS_UNWANTED_TOKENS_STR if fast_provider.supports_logit_bias else None
    )

    summaries: List[str] = []
    try:
        for idx, article in enumerate(new_articles[:limit], 1):
            await update_status(f"[{topic_name}] {idx}/{min(len(new_articles), limit)}: {article.title}")
            if idx > 1:
                await asyncio.sleep(config.GROUND_NEWS_ARTICLE_DELAY_SECONDS)

            scraped_text, _ = await scrape_website(article.url)
            if (
                not scraped_text
                or "Failed to scrape" in scraped_text
                or "Scraping timed out" in scraped_text
                or "Blocked from fetching URL" in scraped_text
            ):
                summaries.append(f"**{article.title}**\n{article.url}\nCould not scrape article\n")
                seen_urls.add(article.url)
                continue

            prompt = (
                "[It is currently 2025 and Donald Trump is the current president. Biden IS NOT THE CURRENT PRESIDENT! "
                "BIDEN IS THE EX PRESIDENT AND TRUMP IS THE PRESIDENT AGAIN FOR HIS SECOND TERM. THIS IS JUST KEY "
                "TEMPORAL KNOWLEDGE TO ORIENT YOU WHEN IT COMES UP. ANSWER ACCORDINGLY!] "
                "(Just an FYI. Maybe unrelated to context and omitted IF NOT RELATED TO TRUMP OR BIDEN OR THE YEAR. "
                "I NEED THE BEHAVIOR OF CALLING TRUMP THE EX-PRESIDENT AND CALLING THE CURRENT ADMINISTRATION THE BIDEN ADMINISTRATIONL. "
                "THESE ERRORS ARE UNACCEPTABLE. DON'T MENTION ME YELLING AT YOU. JUST DO THE DAMN THING!). "
                "Do not use em dashes. Summarize the following article in 3-5 sentences. "
                "Focus on key facts. Present in a casual, blunt, honest and slightly profane tone. "
                "Do NOT start with 'So, ' or end with 'Basically, '. Do not state things like 'This article describes', "
                "'The article', etc. Present is as a person would if they were talking to you about the article.\n\n"
                f"Title: {article.title}\nURL: {article.url}\n\n{scraped_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
            )

            try:
                response = await create_chat_completion(
                    fast_client,
                    [
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=fast_provider.model,
                    max_tokens=3072,
                    temperature=fast_provider.temperature,
                    logit_bias=fast_logit_bias,
                    use_responses_api=fast_provider.use_responses_api,
                )
                summary = extract_text(response, fast_provider.use_responses_api) or "[LLM summarization failed]"
                await store_rss_summary(
                    feed_url=f"ground_news_topic_{topic_slug}",
                    article_url=article.url,
                    title=article.title,
                    summary_text=summary,
                    timestamp=datetime.now(),
                )
            except Exception as exc:
                logger.error(
                    "Scheduled groundtopic: LLM summarization failed for %s: %s",
                    article.url,
                    exc,
                )
                summary = "[LLM summarization failed]"

            summary_line = f"**{article.title}**\n{article.url}\n{summary}\n"
            summaries.append(summary_line)
            seen_urls.add(article.url)

            if bot_state:
                user_node = MsgNode(
                    "user",
                    f"/schedule_groundtopic {topic_slug} article {idx} (limit {limit})",
                    name="scheduled",
                )
                assistant_node = MsgNode(
                    "assistant",
                    summary_line,
                    name=str(bot.user.id) if bot.user else None,
                )
                try:
                    await bot_state.append_history(
                        channel_id,
                        user_node,
                        config.MAX_MESSAGE_HISTORY,
                    )
                    await bot_state.append_history(
                        channel_id,
                        assistant_node,
                        config.MAX_MESSAGE_HISTORY,
                    )
                    try:
                        await ingest_conversation_to_chromadb(
                            llm_client,
                            channel_id,
                            channel_id,
                            [user_node, assistant_node],
                            None,
                        )
                    except Exception as ingest_exc:
                        logger.error(
                            "Scheduled groundtopic: Failed ingesting summary for %s: %s",
                            article.url,
                            ingest_exc,
                            exc_info=True,
                        )
                except Exception as history_exc:
                    logger.error(
                        "Scheduled groundtopic: Failed updating history for channel %s: %s",
                        channel_id,
                        history_exc,
                        exc_info=True,
                    )

        save_seen_links(seen_urls)
        combined = "\n\n".join(summaries)
        for idx, chunk in enumerate(chunk_text(combined, config.EMBED_MAX_LENGTH)):
            embed = discord.Embed(
                title=f"Ground News Summaries – {topic_name}" + ("" if idx == 0 else f" (cont. {idx + 1})"),
                description=chunk,
                color=config.EMBED_COLOR["complete"],
            )
            await ch.send(embed=embed)
        try:
            from audio_utils import send_tts_audio

            await send_tts_audio(
                ch,
                combined,
                base_filename=f"scheduled_groundtopic_{topic_slug}_{channel_id}",
                bot_state=bot_state,
            )
        except Exception as tts_exc:
            logger.error(
                "Scheduled groundtopic: TTS failed for channel %s topic %s: %s",
                channel_id,
                topic_slug,
                tts_exc,
            )
    except asyncio.CancelledError:
        if header:
            try:
                await header.edit(content=f"Scheduled Ground News topic `{topic_name}` digest cancelled.")
            except Exception:
                pass
        await ch.send(f"Scheduled Ground News topic `{topic_name}` digest cancelled.")
        raise
    finally:
        if header:
            try:
                await header.edit(content=f"Scheduled Ground News topic `{topic_name}` digest completed.")
            except Exception:
                pass
        if bot_state:
            await bot_state.clear_active_task(channel_id)
