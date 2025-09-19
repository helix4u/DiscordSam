import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import discord

from config import config
from rag_chroma_manager import store_rss_summary
from utils import chunk_text
from web_utils import fetch_rss_entries, scrape_website
from openai_api import create_chat_completion, extract_text
from logit_biases import LOGIT_BIAS_UNWANTED_TOKENS_STR

logger = logging.getLogger(__name__)


async def run_allrss_digest(
    bot: discord.Client,
    llm_client: Any,
    channel_id: int,
    *,
    limit: int = 10,
    bot_state: Optional[Any] = None,
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

            # Sort by pubDate_dt desc and limit
            entries_sorted = sorted(
                [e for e in entries if e.get("pubDate_dt")],
                key=lambda e: e["pubDate_dt"],
                reverse=True,
            )
            to_process = entries_sorted[:limit]

            feed_summaries: List[str] = []

            for idx, ent in enumerate(to_process, 1):
                title = ent.get("title") or "Untitled"
                link = ent.get("link") or ""

                try:
                    await header.edit(content=f"[{name}] {idx}/{len(to_process)}: Scraping {title}…")
                except Exception:
                    pass

                scraped_text, _ = await scrape_website(link)
                if (
                    not scraped_text
                    or "Failed to scrape" in scraped_text
                    or "Scraping timed out" in scraped_text
                    or "Blocked from fetching URL" in scraped_text
                ):
                    summary_entry = f"[{name}] **{title}**\n{link}\nCould not scrape article\n"
                    feed_summaries.append(summary_entry)
                    continue

                prompt = (
                    "You are an expert news summarizer. Summarize the following article in 3-5 sentences. "
                    "Focus on key facts and avoid fluff.\n\n"
                    f"Title: {title}\nURL: {link}\n\n{scraped_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
                )
                try:
                    response = await create_chat_completion(
                        llm_client,
                        [
                            {"role": "system", "content": "You are an expert news summarizer."},
                            {"role": "user", "content": prompt},
                        ],
                        model=config.FAST_LLM_MODEL,
                        max_tokens=3072,
                        temperature=1,
                        logit_bias=LOGIT_BIAS_UNWANTED_TOKENS_STR,
                        use_responses_api=config.FAST_LLM_USE_RESPONSES_API,
                    )
                    summary = extract_text(response, config.FAST_LLM_USE_RESPONSES_API) or "[LLM summarization failed]"
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

                summary_entry = f"[{name}] **{title}**\n{link}\n{summary}\n"
                feed_summaries.append(summary_entry)
                await asyncio.sleep(0.2)

            if feed_summaries:
                total_summaries.extend(feed_summaries)
                combined_feed = "\n\n".join(feed_summaries)
                chunks = chunk_text(combined_feed, config.EMBED_MAX_LENGTH)
                for i, chunk in enumerate(chunks):
                    embed = discord.Embed(
                        title=f"Scheduled RSS Digest ({name})" + ("" if i == 0 else f" (cont. {i+1})"),
                        description=chunk,
                        color=config.EMBED_COLOR.get("complete"),
                    )
                    await ch.send(embed=embed)
                try:
                    from audio_utils import send_tts_audio
                    await send_tts_audio(ch, combined_feed, base_filename=f"scheduled_rss_{channel_id}_{name}")
                except Exception as tts_exc:
                    logger.error("Scheduled allrss: TTS failed for %s: %s", name, tts_exc)

        if not total_summaries:
            await header.edit(content="Scheduled RSS digest found nothing new.")
        else:
            await header.edit(content=f"Scheduled RSS digest completed. Posted {len(total_summaries)} article summaries.")
    except asyncio.CancelledError:
        try:
            await header.edit(content="Scheduled RSS digest cancelled.")
        except Exception:
            pass
        await ch.send("Scheduled RSS digest cancelled.")
        raise
    finally:
        if bot_state:
            await bot_state.clear_active_task(channel_id)
