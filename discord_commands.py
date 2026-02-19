import logging
import discord
from discord import app_commands  # type: ignore
from discord.ext import commands  # For bot type hint
import os
from pathlib import Path
import base64
import random
import asyncio
import textwrap
from typing import Any, Optional, List, Callable, Awaitable, Dict  # Keep existing imports
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from urllib.parse import urlparse

# Bot services and utilities
from config import config
from llm_clients import get_llm_runtime
from state import BotState
from common_models import MsgNode, TweetData, GroundNewsArticle

from llm_handling import (
    _build_initial_prompt_messages,
    get_system_prompt,
    stream_llm_response_to_interaction,
    retrieve_rag_context_with_progress,
)
from rag_chroma_manager import (
    parse_chatgpt_export,
    store_chatgpt_conversations_in_chromadb,
    store_rss_summary,
    store_moltbook_reply,
    store_moltbook_feed_post,
    store_moltbook_full_post,
    ingest_conversation_to_chromadb,
    get_chroma_collection_counts,
    fetch_recent_channel_memories,
)
import rag_chroma_manager as rcm
import aiohttp
from web_utils import (
    scrape_website,
    query_searx,
    scrape_latest_tweets,
    scrape_home_timeline,
    scrape_ground_news_my,
    scrape_ground_news_topic,
    fetch_rss_entries
)
from openai_api import create_chat_completion, extract_text
from logit_biases import LOGIT_BIAS_UNWANTED_TOKENS_STR
from audio_utils import send_tts_audio
from utils import (
    parse_time_string_to_delta,
    chunk_text,
    format_article_time,
    safe_followup_send,
    safe_message_edit,
    start_post_processing_task,
    is_admin_user,
    sanitize_moltbook_text_for_tts,
)
from rss_cache import load_seen_entries, save_seen_entries
from twitter_cache import load_seen_tweet_ids, save_seen_tweet_ids # New import
from timeline_pruner import prune_oldest_items
from ground_news_cache import load_seen_links, save_seen_links
from rate_limiter import get_rate_limiter
from moltbook_client import (
    MoltbookAPIError,
    moltbook_add_comment,
    moltbook_dm_approve,
    moltbook_create_post,
    moltbook_dm_check,
    moltbook_dm_conversations,
    moltbook_dm_get_conversation,
    moltbook_dm_request,
    moltbook_dm_requests,
    moltbook_dm_send,
    moltbook_get_comments,
    moltbook_get_feed,
    moltbook_get_post,
    moltbook_get_profile,
    moltbook_get_status,
    moltbook_search,
)

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = "You are an expert news summarizer."

# Temporal grounding to prevent outdated references in generated content
TEMPORAL_SYSTEM_CONTEXT = (
    "Temporal context: The year is 2025. Donald Trump is the current President of the United States. "
    "Joe Biden is the former president. Use correct current titles and do not assert outdated office-holders."
)

# Default RSS feeds users can choose from with the /rss command
DEFAULT_RSS_FEEDS = [
    ("Google News", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"),
    ("CBS News - US", "https://www.cbsnews.com/latest/rss/us"),
    ("The Daily Beast", "https://www.thedailybeast.com/arc/outboundfeeds/rss/articles/"),
    ("DoD Releases", "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=9&Site=945&max=10"),
    ("ABC Politics Headlines", "https://abcnews.go.com/abcnews/politicsheadlines"),
    ("ABC US Headlines", "https://abcnews.go.com/abcnews/usheadlines"),
    ("SAN", "https://san.com/feed/"),
    ("NYT Homepage", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
    ("NBC News", "https://feeds.nbcnews.com/nbcnews/public/news"),
    ("NBC World", "https://feeds.nbcnews.com/nbcnews/public/world"),
    ("Drudge Report", "https://feeds.feedburner.com/DrudgeReportFeed"),
    ("NPR News", "http://www.npr.org/rss/rss.php?id=1001"),
    ("BBC Americas", "http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml?edition=int"),
    ("Congressional Bills", "https://www.govinfo.gov/rss/bills.xml"),
    ("Hacker News", "https://news.ycombinator.com/rss"),
    ("Sky News - US", "https://feeds.skynews.com/feeds/rss/us.xml"),
    ("Forbes Business", "https://www.forbes.com/business/feed/"),
    ("Huffpost Politics", "https://chaski.huffpost.com/us/auto/vertical/politics"),
    ("Huffpost US News", "https://chaski.huffpost.com/us/auto/vertical/us-news"),
    ("Express.co.uk - US", "https://www.express.co.uk/posts/rss/198/us"),
    ("Time", "https://time.com/feed/"),
    ("PBS Headlines", "https://www.pbs.org/newshour/feeds/rss/headlines"),
    ("Mother Jones", "https://www.motherjones.com/feed/"),
    ("Quartz", "https://qz.com/rss"),
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
    "secwar",
    "petehegseth",
    "presssec",
    "unusual_whales",
    "ap",
    "cspan",
    "thehill",
    "JDVance",
    "FBI",
    "wutangkids",
    "DNIGabbard",
    "CNBC",
    "ABC",
    "TriciaOhio",
    "ICEgov",
]

# Results from collecting tweets for a user run.
@dataclass
class TweetCollectionResult:
    clean_username: str
    new_tweets: List[TweetData]
    raw_display_str: str
    embed_chunks: List[str]
    status_message: Optional[str]
    total_fetched: int


_shared_rate_limiter = get_rate_limiter()


def _twitter_scope_from_interaction(
    interaction: discord.Interaction,
) -> tuple[Optional[int], Optional[int]]:
    """Return the scope identifiers (guild or admin DM user) for list storage."""
    if interaction.guild_id is not None:
        return interaction.guild_id, None
    return None, interaction.user.id


async def _ensure_default_twitter_list(
    guild_id: Optional[int],
    user_id: Optional[int],
) -> None:
    """Ensure the default twitter list exists for the given scope."""
    if not bot_state_instance:
        return
    existing = await bot_state_instance.get_twitter_list_handles(
        guild_id,
        "default",
        user_id=user_id,
    )
    if existing:
        return
    await bot_state_instance.set_twitter_list(
        guild_id,
        "default",
        DEFAULT_TWITTER_USERS,
        user_id=user_id,
    )

def _format_iso_timestamp(value: Optional[str]) -> str:
    if not value:
        return "unknown time"
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _format_user_reference(user_id_value: Optional[str]) -> str:
    if not user_id_value:
        return "unknown user"
    try:
        numeric = int(str(user_id_value))
    except (TypeError, ValueError):
        return str(user_id_value)
    return f"<@{numeric}>"


def _format_pause_notice(state: Dict[str, Any]) -> str:
    reason = state.get("reason") or "No reason provided."
    paused_by = _format_user_reference(state.get("paused_by"))
    paused_at = _format_iso_timestamp(state.get("paused_at"))
    return f"⚠️ Gaming mode enabled by {paused_by} at {paused_at}. Reason: {reason}"


def _moltbook_enabled() -> bool:
    return bool(config.MOLTBOOK_API_KEY and config.MOLTBOOK_AGENT_NAME)


def _moltbook_post_url(post_id: str) -> str:
    """Canonical web URL for a Moltbook post (e.g. https://www.moltbook.com/post/{id})."""
    base = config.MOLTBOOK_BASE_URL.replace("/api/v1", "").rstrip("/")
    return f"{base}/post/{post_id}"


def _format_moltbook_post(post: Dict[str, Any], *, max_content_length: Optional[int] = 240) -> str:
    post_id = post.get("id", "unknown")
    title = post.get("title") or "(untitled)"
    content = post.get("content") or post.get("url") or ""
    author = (post.get("author") or {}).get("name", "unknown")
    submolt = (post.get("submolt") or {}).get("name", "general")
    created_at = post.get("created_at", "unknown time")
    upvotes = post.get("upvotes", 0)
    downvotes = post.get("downvotes", 0)
    preview = content.strip().replace("\n", " ")
    if max_content_length is not None and len(preview) > max_content_length:
        preview = preview[: max_content_length - 3].rstrip() + "..."
    id_display = f"[{post_id}]({_moltbook_post_url(post_id)})" if post_id != "unknown" else "unknown"
    return (
        f"**{title}**\n"
        f"🦞 {author} • m/{submolt} • {created_at}\n"
        f"👍 {upvotes} | 👎 {downvotes} | ID: {id_display}\n"
        f"{preview}"
    )


def _format_moltbook_comment(comment: Dict[str, Any], *, max_content_length: Optional[int] = 240) -> str:
    comment_id = comment.get("id", "unknown")
    content = (comment.get("content") or "").strip()
    if max_content_length is not None and len(content) > max_content_length:
        content = content[: max_content_length - 3].rstrip() + "..."
    author = (comment.get("author") or {}).get("name", "unknown")
    created_at = comment.get("created_at", "unknown time")
    upvotes = comment.get("upvotes", 0)
    downvotes = comment.get("downvotes", 0)
    return (
        f"💬 {author} • {created_at}\n"
        f"👍 {upvotes} | 👎 {downvotes} | ID: `{comment_id}`\n"
        f"{content}"
    )


async def _do_show_moltbook_post(
    interaction: discord.Interaction, post_id: str
) -> None:
    """Fetch a Moltbook post and send embeds + TTS view. Caller must have deferred the response."""
    from moltbook_client import moltbook_get_post, MoltbookAPIError

    try:
        post_payload = await moltbook_get_post(post_id)
        post = post_payload.get("post") or post_payload.get("data") or post_payload
        description = _format_moltbook_post(post, max_content_length=None)
        comments = (
            post_payload.get("comments")
            or (post_payload.get("data") or {}).get("comments")
            or []
        )
        if comments:
            snippets = [
                _format_moltbook_comment(c, max_content_length=None)
                for c in comments[:5]
            ]
            description = f"{description}\n\n**Comments**\n" + "\n\n".join(snippets)
        # Store full post fetch in ChromaDB for RAG when drafting replies (fire-and-forget)
        start_post_processing_task(store_moltbook_full_post(post_id, description))
        chunks = chunk_text(description, config.EMBED_MAX_LENGTH)
        total = 0
        embeds_list = []
        for i, ch in enumerate(chunks):
            if total + len(ch) > 6000:
                break
            embeds_list.append(
                discord.Embed(
                    title="Moltbook Post" if i == 0 else f"Moltbook Post (cont. {i + 1})",
                    description=ch,
                    color=config.EMBED_COLOR["complete"],
                )
            )
            total += len(ch)
        if not embeds_list:
            embeds_list = [
                discord.Embed(
                    title="Moltbook Post",
                    description=description[: config.EMBED_MAX_LENGTH - 20].rstrip() + "...",
                    color=config.EMBED_COLOR["complete"],
                )
            ]
        tts_view = MoltbookPostTTSView(description, post_id=post_id)
        await interaction.followup.send(embeds=embeds_list, view=tts_view)
    except MoltbookAPIError as exc:
        logger.warning("Moltbook get failed: %s", exc)
        await interaction.followup.send(content=f"Failed to fetch post: {exc}", ephemeral=True)


async def _handle_moltbook_get_button(interaction: discord.Interaction, post_id: str) -> None:
    """Handle 'Get post' button click: defer then show post."""
    await interaction.response.defer(ephemeral=False)
    await _do_show_moltbook_post(interaction, post_id)


class DraftReplySubmitView(discord.ui.View):
    """Submit, Don't submit, and Get TTS for a drafted Moltbook reply."""

    def __init__(self, draft: str, post_id: str, *, timeout: float = 300.0) -> None:
        super().__init__(timeout=timeout)
        self.draft = draft
        self.post_id = post_id
        get_tts_btn = discord.ui.Button(
            label="Get TTS",
            style=discord.ButtonStyle.primary,
            custom_id="moltbook_draft_tts",
        )
        get_tts_btn.callback = self._tts_callback
        self.add_item(get_tts_btn)
        submit_btn = discord.ui.Button(
            label="Submit",
            style=discord.ButtonStyle.primary,
            custom_id="moltbook_draft_submit",
        )
        submit_btn.callback = self._submit_callback
        self.add_item(submit_btn)
        dont_btn = discord.ui.Button(
            label="Don't submit",
            style=discord.ButtonStyle.secondary,
            custom_id="moltbook_draft_dont",
        )
        dont_btn.callback = self._dont_callback
        self.add_item(dont_btn)

    async def _tts_callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=False)
        if not (self.draft and self.draft.strip()):
            await interaction.followup.send(content="No draft text to speak.", ephemeral=True)
            return
        try:
            await send_tts_audio(
                interaction,
                sanitize_moltbook_text_for_tts(self.draft),
                base_filename="moltbook_draft",
                bot_state=bot_state_instance,
            )
        except Exception as exc:
            logger.warning("Moltbook draft TTS failed: %s", exc)
            await interaction.followup.send(content=f"TTS failed: {exc}", ephemeral=True)

    async def _submit_callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        from moltbook_client import moltbook_add_comment, MoltbookAPIError
        if not _moltbook_enabled():
            await interaction.followup.send(
                content="Moltbook is not configured. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in .env.",
                ephemeral=True,
            )
            return
        try:
            await moltbook_add_comment(post_id=self.post_id, content=self.draft)
            # Store submitted reply in ChromaDB for RAG when drafting future replies (fire-and-forget)
            start_post_processing_task(
                store_moltbook_reply(
                    self.post_id,
                    self.draft,
                    kind="submitted",
                    post_content_snippet=None,
                ),
            )
            embed = discord.Embed(
                title="Draft reply",
                description="Comment posted to Moltbook.",
                color=config.EMBED_COLOR["complete"],
            )
            embed.set_footer(text=f"Post ID: {self.post_id}")
            await interaction.message.edit(embed=embed, view=None)
            await interaction.followup.send(content="Comment posted.", ephemeral=True)
        except MoltbookAPIError as exc:
            logger.warning("Moltbook draft submit failed: %s", exc)
            msg = f"Failed to post comment: {exc}"
            if getattr(exc, "status", None) == 401:
                msg += (
                    "\n\n**401 = auth rejected.** If `/moltbook_status` works, your key and agent name are correct "
                    "and the 401 on POST may be a Moltbook server restriction on write operations. Otherwise check "
                    "MOLTBOOK_API_KEY and MOLTBOOK_AGENT_NAME in .env and that your agent is claimed (see https://www.moltbook.com/skill.md)."
                )
            elif exc.hint:
                msg += f"\n\nHint: {exc.hint}"
            await interaction.followup.send(content=msg, ephemeral=True)

    async def _dont_callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            await interaction.message.delete()
        except (discord.HTTPException, discord.NotFound):
            # Ephemeral messages may not be editable/deletable via API
            pass
        await interaction.followup.send(content="Dismissed.", ephemeral=True)


def _strip_trailing_url_label(content: Optional[str]) -> Optional[str]:
    """Remove a trailing 'URL:' or 'URL: ...' line from content so it doesn't appear in the post body."""
    if not content or not content.strip():
        return content
    import re
    return re.sub(r"\n\s*URL:\s*(.*)$", "", content.strip(), flags=re.IGNORECASE).strip() or None


def _parse_moltbook_draft_post(text: str) -> Dict[str, Optional[str]]:
    """Parse LLM draft output for SUBMOLT:, TITLE:, CONTENT:, optional URL:. Returns dict with submolt, title, content, url (all optional str)."""
    result: Dict[str, Optional[str]] = {"submolt": None, "title": None, "content": None, "url": None}
    if not (text and text.strip()):
        return result
    import re
    raw = text.strip()
    # SUBMOLT: and TITLE: and URL: single line each
    submolt_m = re.search(r"(?i)^SUBMOLT:\s*(.+?)$", raw, re.MULTILINE)
    if submolt_m:
        result["submolt"] = submolt_m.group(1).strip() or None
    title_m = re.search(r"(?i)^TITLE:\s*(.+?)$", raw, re.MULTILINE)
    if title_m:
        result["title"] = title_m.group(1).strip() or None
    url_m = re.search(r"(?i)^URL:\s*(.+?)$", raw, re.MULTILINE)
    if url_m:
        result["url"] = url_m.group(1).strip() or None
    # CONTENT: from first CONTENT: to end of text (multiline); strip trailing "URL:" line so it doesn't appear in post body
    content_m = re.search(r"(?i)^CONTENT:\s*(.+)", raw, re.MULTILINE | re.DOTALL)
    if content_m:
        result["content"] = _strip_trailing_url_label(content_m.group(1).strip() or None)
    return result


class MoltbookDraftPostView(discord.ui.View):
    """Accept (post to Moltbook) or Decline (delete) a drafted post."""

    def __init__(
        self,
        submolt: str,
        title: str,
        content: Optional[str] = None,
        url: Optional[str] = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self.submolt = (submolt or "general").strip()
        self.title = (title or "").strip()
        self.content = (content or "").strip() or None
        self.url = (url or "").strip() or None
        accept_btn = discord.ui.Button(
            label="Post to Moltbook",
            style=discord.ButtonStyle.primary,
            custom_id="moltbook_draft_post_accept",
        )
        accept_btn.callback = self._accept_callback
        self.add_item(accept_btn)
        decline_btn = discord.ui.Button(
            label="Decline",
            style=discord.ButtonStyle.secondary,
            custom_id="moltbook_draft_post_decline",
        )
        decline_btn.callback = self._decline_callback
        self.add_item(decline_btn)

    async def _accept_callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=False)
        if not _moltbook_enabled():
            await interaction.followup.send(
                content="Moltbook is not configured. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in .env.",
                ephemeral=True,
            )
            return
        if not self.title:
            await interaction.followup.send(content="Draft has no title; cannot post.", ephemeral=True)
            return
        if not self.content and not self.url:
            await interaction.followup.send(
                content="Draft needs either content or a URL to post.",
                ephemeral=True,
            )
            return
        # Sanitize content so a trailing "URL:" line from the LLM never gets into the post body
        content_to_send = _strip_trailing_url_label(self.content) if self.content else None
        try:
            payload = await moltbook_create_post(
                submolt=self.submolt,
                title=self.title,
                content=content_to_send,
                url=self.url,
            )
            post = payload.get("post") or payload.get("data") or payload
            post_id = post.get("id", "unknown")
            embed = discord.Embed(
                title="Moltbook Post Created",
                description=_format_moltbook_post(post),
                color=config.EMBED_COLOR["complete"],
            )
            embed.set_footer(text=f"Post ID: {post_id}")
            try:
                await interaction.message.edit(embed=embed, view=None)
            except (discord.HTTPException, discord.NotFound):
                pass
            # No followup message: the edited message above is the only reply (avoids 3rd post / attaching to dismissed message)
        except MoltbookAPIError as exc:
            logger.warning("Moltbook draft post (accept) failed: %s", exc)
            msg = str(exc)
            if getattr(exc, "hint", None):
                msg += f"\nHint: {exc.hint}"
            await interaction.followup.send(content=msg, ephemeral=True)

    async def _decline_callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            await interaction.message.delete()
        except (discord.HTTPException, discord.NotFound):
            pass
        await interaction.followup.send(content="Draft discarded.", ephemeral=True)


class MoltbookPostTTSView(discord.ui.View):
    """Get TTS and Draft reply for a fetched Moltbook post."""

    def __init__(
        self,
        text_to_speak: str,
        *,
        post_id: Optional[str] = None,
        timeout: float = 180.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self.text_to_speak = (text_to_speak or "").strip()
        self.post_id = post_id or ""
        get_tts_btn = discord.ui.Button(
            label="Get TTS",
            style=discord.ButtonStyle.primary,
            custom_id="moltbook_tts_post",
        )
        get_tts_btn.callback = self._tts_callback
        self.add_item(get_tts_btn)
        draft_btn = discord.ui.Button(
            label="Draft reply",
            style=discord.ButtonStyle.secondary,
            custom_id="moltbook_draft_reply",
        )
        draft_btn.callback = self._draft_callback
        self.add_item(draft_btn)

    async def _tts_callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=False)
        if not self.text_to_speak:
            await interaction.followup.send(content="No text to speak.", ephemeral=True)
            return
        try:
            await send_tts_audio(
                interaction,
                sanitize_moltbook_text_for_tts(self.text_to_speak),
                base_filename="moltbook_post",
                bot_state=bot_state_instance,
            )
        except Exception as exc:
            logger.warning("Moltbook TTS (button) failed: %s", exc)
            await interaction.followup.send(content=f"TTS failed: {exc}", ephemeral=True)

    async def _draft_callback(self, interaction: discord.Interaction) -> None:
        """Act as if user said 'yo, sam, draft a reply to this one.' with current post as context.
        Uses same RAG (ChromaDB) retrieval as normal user input to prep the agent with relevant memories."""
        await interaction.response.defer(ephemeral=False)
        try:
            fast_runtime = get_llm_runtime("fast")
            fast_client = fast_runtime.client if fast_runtime else None
            fast_provider = fast_runtime.provider if fast_runtime else None
            if not fast_client or not fast_provider:
                await interaction.followup.send(
                    content="LLM is not configured; cannot draft a reply.",
                    ephemeral=True,
                )
                return
            # Same process as user input: gather ChromaDB memories to prep the agent
            rag_query = (
                "Draft a reply to this Moltbook post:\n"
                + (self.text_to_speak[:2500] if self.text_to_speak else "No post content.")
            )
            synthesized_rag_summary, raw_rag_snippets = await retrieve_rag_context_with_progress(
                llm_client=fast_client,
                query=rag_query,
                interaction=interaction,
                channel=interaction.channel,
                initial_status="🔍 Searching memories for relevant context...",
                completion_status="✅ Memory search complete.",
            )
            # Build prompt with same system instructions and dev context as regular Sam responses
            sys_node = get_system_prompt()
            messages: List[Dict[str, str]] = [{"role": sys_node.role, "content": sys_node.content}]
            if (config.USER_PROVIDED_CONTEXT or "").strip():
                messages.append({
                    "role": "system",
                    "content": f"User-Set Global Context:\n{config.USER_PROVIDED_CONTEXT.strip()}",
                })
            if synthesized_rag_summary and synthesized_rag_summary.strip():
                context_block = (
                    "The following is a synthesized summary of potentially relevant past conversations. "
                    "Use it to provide a more informed reply.\n\n"
                    "--- Synthesized Relevant Context ---\n"
                    + synthesized_rag_summary.strip()
                    + "\n--- End Synthesized Context ---"
                )
                messages.append({"role": "system", "content": context_block})
            if raw_rag_snippets:
                max_snippet_chars = getattr(config, "MAX_RAW_RAG_SNIPPET_CHARS_IN_PROMPT", 120000)
                max_per_snippet = getattr(config, "MAX_RAW_SNIPPET_CHARS_FOR_DRAFT", 2000)
                parts = [
                    "Raw retrieved context snippets that might be relevant. Use them to inform your reply.\n"
                ]
                total = 0
                for i, (snippet_text, source) in enumerate(raw_rag_snippets):
                    if total >= max_snippet_chars:
                        parts.append("\n[More snippets omitted due to length.]")
                        break
                    trunc = snippet_text[:max_per_snippet]
                    if len(snippet_text) > max_per_snippet:
                        trunc += " [truncated]"
                    part = f"\n--- Snippet {i + 1} (Source: {source}) ---\n{trunc}\n"
                    if total + len(part) > max_snippet_chars:
                        break
                    parts.append(part)
                    total += len(part)
                parts.append("\n--- End Raw Snippets ---")
                messages.append({"role": "system", "content": "".join(parts)})
            # Draft-only instruction (same as regular turn, but output is just the comment text)
            messages.append({
                "role": "system",
                "content": (
                    "For this turn only: The user said: yo, sam, draft a reply to this one. "
                    "Output only the draft reply text, no preamble. Keep it suitable as a Moltbook comment (a few sentences)."
                ),
            })
            user_msg = f"Post to reply to:\n{self.text_to_speak[:3000]}"
            messages.append({"role": "user", "content": user_msg})
            response = await create_chat_completion(
                fast_client,
                messages,
                model=fast_provider.model,
                max_tokens=config.MAX_COMPLETION_TOKENS,
                temperature=fast_provider.temperature,
                use_responses_api=getattr(fast_provider, "use_responses_api", False),
            )
            draft = extract_text(response, getattr(fast_provider, "use_responses_api", False))
            if not (draft and draft.strip()):
                await interaction.followup.send(
                    content="No draft was generated.",
                    ephemeral=True,
                )
                return
            # Store draft in ChromaDB for RAG when drafting future replies (fire-and-forget)
            start_post_processing_task(
                store_moltbook_reply(
                    self.post_id,
                    draft.strip(),
                    kind="draft",
                    post_content_snippet=(self.text_to_speak[:3000] if self.text_to_speak else None),
                ),
            )
            draft_display = draft.strip()[:4000]
            if len(draft.strip()) > 4000:
                draft_display += "\n…"
            embed = discord.Embed(
                title="Draft reply",
                description=draft_display,
                color=config.EMBED_COLOR["complete"],
            )
            embed.set_footer(text=f"Post ID: {self.post_id} • Submit or Don't submit below.")
            view = DraftReplySubmitView(draft=draft.strip(), post_id=self.post_id)
            await interaction.followup.send(embed=embed, view=view)
        except Exception as exc:
            logger.warning("Moltbook draft reply failed: %s", exc, exc_info=True)
            await interaction.followup.send(
                content=f"Draft failed: {exc}",
                ephemeral=True,
            )


# Max Discord select options per menu
_MOLTBOOK_SELECT_MAX_OPTIONS = 25
# Truncate post title in dropdown for consistent, readable labels
_MOLTBOOK_SELECT_LABEL_MAX = 50


class MoltbookFeedView(discord.ui.View):
    """View with one dropdown: select a post to view (then Get TTS on the opened post)."""

    def __init__(self, posts: List[Dict[str, Any]], *, timeout: float = 180.0):
        super().__init__(timeout=timeout)
        options: List[discord.SelectOption] = []
        for post in posts[:_MOLTBOOK_SELECT_MAX_OPTIONS]:
            post_id = post.get("id")
            if not post_id:
                continue
            raw_title = (post.get("title") or "Untitled").strip()
            if len(raw_title) > _MOLTBOOK_SELECT_LABEL_MAX:
                label = raw_title[:_MOLTBOOK_SELECT_LABEL_MAX - 1].rstrip() + "…"
            else:
                label = raw_title or "Untitled"
            options.append(discord.SelectOption(label=label, value=post_id))

        if not options:
            return
        select = discord.ui.Select(
            placeholder="Select a post to view…",
            options=options,
            min_values=1,
            max_values=1,
        )
        select.callback = self._select_callback
        self.add_item(select)

    async def _select_callback(self, interaction: discord.Interaction) -> None:
        if not interaction.data or "values" not in interaction.data or not interaction.data["values"]:
            await interaction.response.defer(ephemeral=True)
            await interaction.followup.send(content="No post selected.", ephemeral=True)
            return
        post_id = interaction.data["values"][0]
        await interaction.response.defer(ephemeral=False)
        await _do_show_moltbook_post(interaction, post_id)


async def _collect_new_tweets(
    clean_username: str,
    *,
    limit: int,
    progress_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    stop_after_seen_consecutive: int = 3,
    source_command: str = "/alltweets",
) -> TweetCollectionResult:
    """Collect new tweets for a user and prepare formatted output."""

    async def _emit(message: str) -> None:
        if not progress_callback:
            return
        try:
            await progress_callback(message)
        except Exception as exc:
            logger.warning(
                "Tweet progress callback failed for @%s: %s",
                clean_username,
                exc,
            )

    all_seen_tweet_ids_cache = load_seen_tweet_ids()
    user_seen_tweet_ids = all_seen_tweet_ids_cache.get(clean_username, set())

    def _seen_checker(td: TweetData) -> bool:
        return bool(td.tweet_url) and td.tweet_url in user_seen_tweet_ids

    fetched_tweets_data = await scrape_latest_tweets(
        clean_username,
        limit=limit,
        progress_callback=_emit,
        seen_checker=_seen_checker,
        stop_after_seen_consecutive=stop_after_seen_consecutive,
    )

    total_fetched = len(fetched_tweets_data)
    if not fetched_tweets_data:
        return TweetCollectionResult(
            clean_username=clean_username,
            new_tweets=[],
            raw_display_str="",
            embed_chunks=[],
            status_message=(
                f"Finished scraping for @{clean_username}. "
                "No tweets found or profile inaccessible."
            ),
            total_fetched=0,
        )

    # Identify repeated content from other accounts (likely ads) to discard
    content_repeat_counts: Dict[str, int] = {}
    for tweet_data in fetched_tweets_data:
        norm_content = (tweet_data.content or "").strip().lower()
        if not norm_content:
            continue
        if tweet_data.username and tweet_data.username.lower() != clean_username.lower():
            content_repeat_counts[norm_content] = content_repeat_counts.get(norm_content, 0) + 1

    new_tweets_to_process: List[TweetData] = []
    processed_tweet_ids_current_run: set[str] = set()
    for tweet_data in fetched_tweets_data:
        if not tweet_data.tweet_url:
            logger.warning(
                "Tweet from @%s missing 'tweet_url'. Skipping entry: %s",
                clean_username,
                tweet_data,
            )
            continue
        norm_content = (tweet_data.content or "").strip().lower()
        if (
            tweet_data.username
            and tweet_data.username.lower() != clean_username.lower()
            and norm_content
            and content_repeat_counts.get(norm_content, 0) > 1
        ):
            logger.info(
                "Skipping potential ad tweet for @%s from @%s due to repeated external content.",
                clean_username,
                tweet_data.username,
            )
            continue
        if tweet_data.tweet_url not in user_seen_tweet_ids:
            new_tweets_to_process.append(tweet_data)
            processed_tweet_ids_current_run.add(tweet_data.tweet_url)
        else:
            logger.info(
                "Skipping already seen tweet for @%s: %s",
                clean_username,
                tweet_data.tweet_url,
            )

    all_seen_tweet_ids_cache[clean_username] = user_seen_tweet_ids.union(
        processed_tweet_ids_current_run
    )
    save_seen_tweet_ids(all_seen_tweet_ids_cache)
    logger.info(
        "Updated seen tweet IDs for @%s. Total cached: %s",
        clean_username,
        len(all_seen_tweet_ids_cache[clean_username]),
    )

    if not new_tweets_to_process:
        return TweetCollectionResult(
            clean_username=clean_username,
            new_tweets=[],
            raw_display_str="",
            embed_chunks=[],
            status_message=f"No new tweets found for @{clean_username} since last check.",
            total_fetched=total_fetched,
        )

    # Count total images across all tweets for global progress tracking
    total_images_global = sum(len(t.image_urls) for t in new_tweets_to_process if t.image_urls)
    current_image_num = 0
    
    tweet_texts_for_display: List[str] = []
    for t_data in new_tweets_to_process:
        display_ts = t_data.timestamp
        try:
            dt_obj = (
                datetime.fromisoformat(t_data.timestamp.replace("Z", "+00:00"))
                if t_data.timestamp
                else None
            )
            if dt_obj:
                display_ts = dt_obj.astimezone().strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            pass

        author_display = t_data.username or clean_username
        content_display = discord.utils.escape_markdown(t_data.content or "N/A")
        tweet_url_display = (t_data.tweet_url or "").replace("/analytics", "")

        header = f"[{display_ts}] @{author_display}"
        if t_data.is_repost and t_data.reposted_by:
            header = f"[{display_ts}] @{t_data.reposted_by} reposted @{author_display}"

        image_description_text = ""
        if t_data.image_urls:
            for i, image_url in enumerate(t_data.image_urls):
                current_image_num += 1
                alt_text = t_data.alt_texts[i] if i < len(t_data.alt_texts) else None
                if alt_text:
                    image_description_text += f'\n*Image Alt Text: "{alt_text}"*'
                await _emit(
                    f"Describing image {current_image_num}/{total_images_global} in tweet from @{author_display}..."
                )
                description = await describe_image(image_url)
                if description:
                    image_description_text += f'\n*Image Description: "{description}"*'

        link_text = f" ([Link]({tweet_url_display}))" if tweet_url_display else ""
        tweet_texts_for_display.append(
            f"**{header}**: {content_display}{image_description_text}{link_text}"
        )

    raw_tweets_display_str = "\n\n".join(tweet_texts_for_display).strip()
    if not raw_tweets_display_str:
        raw_tweets_display_str = "No new tweet content could be formatted for display."

    embed_chunks = chunk_text(raw_tweets_display_str, config.EMBED_MAX_LENGTH)

    if new_tweets_to_process and rcm.tweets_collection:
        tweet_docs_to_add: List[str] = []
        tweet_metadatas_to_add: List[Dict[str, Any]] = []
        tweet_ids_to_add: List[str] = []
        seen_doc_ids: set[str] = set()

        for t_data in new_tweets_to_process:
            if not t_data.tweet_url:
                continue
            tweet_id_val = str(t_data.id or "")
            if not tweet_id_val:
                tweet_id_val = t_data.tweet_url.split("?")[0].split("/")[-1]
            doc_id = f"tweet_{clean_username}_{tweet_id_val}"
            if doc_id in seen_doc_ids:
                continue

            document_content = t_data.content or ""
            if not document_content.strip():
                logger.info(
                    "Skipping empty tweet from @%s, ID: %s",
                    clean_username,
                    doc_id,
                )
                continue

            metadata: Dict[str, Any] = {
                "username": clean_username,
                "tweet_url": t_data.tweet_url,
                "timestamp": t_data.timestamp,
                "is_repost": t_data.is_repost,
                "source_command": source_command,
                "raw_data_preview": str(t_data)[:200],
            }
            if t_data.reposted_by:
                metadata["reposted_by"] = t_data.reposted_by

            tweet_docs_to_add.append(document_content)
            tweet_metadatas_to_add.append(metadata)
            tweet_ids_to_add.append(doc_id)
            seen_doc_ids.add(doc_id)

        if tweet_ids_to_add:
            try:
                rcm.tweets_collection.add(
                    documents=tweet_docs_to_add,
                    metadatas=tweet_metadatas_to_add,
                    ids=tweet_ids_to_add,
                )
                logger.info(
                    "Stored %s new tweets from @%s in ChromaDB.",
                    len(tweet_ids_to_add),
                    clean_username,
                )
            except Exception as exc:
                logger.error(
                    "Failed to store tweets for @%s in ChromaDB: %s",
                    clean_username,
                    exc,
                    exc_info=True,
                )
    elif not rcm.tweets_collection:
        logger.warning(
            "tweets_collection unavailable. Skipping storage for @%s.",
            clean_username,
        )

    return TweetCollectionResult(
        clean_username=clean_username,
        new_tweets=new_tweets_to_process,
        raw_display_str=raw_tweets_display_str,
        embed_chunks=embed_chunks,
        status_message=None,
        total_fetched=total_fetched,
    )
# Available Ground News topic pages for the /groundtopic command
GROUND_NEWS_TOPICS = {
    "us-politics": ("US Politics", "https://ground.news/interest/us-politics_3c3c3c"),
    "donald-trump": ("Donald Trump", "https://ground.news/interest/donald-trump"),
    "stock-markets": ("Stock Markets", "https://ground.news/interest/stock-markets"),
    "us-immigration": ("US Immigration", "https://ground.news/interest/us-immigration"),
    "us-crime": ("US Crime", "https://ground.news/interest/us-crime"),
    "wall-street": ("Wall Street", "https://ground.news/interest/wall-street"),
}

GROUND_NEWS_TOPIC_CHOICES = [
    app_commands.Choice(name=disp, value=slug)
    for slug, (disp, _) in GROUND_NEWS_TOPICS.items()
]

MEMORY_SCOPE_CHOICES = [
    app_commands.Choice(name="Distilled summaries", value="distilled"),
    app_commands.Choice(name="Full conversation logs", value="full"),
]

TTS_DELIVERY_CHOICES = [
    app_commands.Choice(name="Audio only", value="audio"),
    app_commands.Choice(name="Video (MP4)", value="video"),
    app_commands.Choice(name="Audio + video", value="both"),
    app_commands.Choice(name="Disabled", value="off"),
]
# Module-level globals to store instances passed from main_bot.py
bot_instance: Optional[commands.Bot] = None
llm_client_instance: Optional[Any] = None
bot_state_instance: Optional[BotState] = None


async def process_rss_feed(
    interaction: discord.Interaction,
    feed_url: str,
    limit: int,
) -> bool:
    """Fetch, summarize and display new entries from a single RSS feed.

    Parameters
    ----------
    interaction : discord.Interaction
        The originating interaction for responding and context.
    feed_url : str
        RSS feed URL to process.
    limit : int
        Maximum number of new entries to process in this batch.

    Returns
    -------
    bool
        ``True`` if any new entries were processed, ``False`` otherwise.
    """

    progress_message = await safe_followup_send(
        interaction,
        content=f"Fetching RSS feed: {feed_url}..."
    )

    seen = load_seen_entries()
    seen_ids = set(seen.get(feed_url, []))

    entries = await fetch_rss_entries(feed_url)
    new_entries = [e for e in entries if e.get("guid") not in seen_ids]
    if not new_entries:
        try:
            await progress_message.delete()
        except discord.HTTPException as e:
            if e.status == 401 and getattr(e, "code", None) == 50027:
                logger.warning("Webhook token expired; could not delete progress message")
            else:
                raise
        await safe_followup_send(
            interaction,
            content=f"No new entries found for {feed_url}.",
            ephemeral=True,
        )
        return False

    to_process = new_entries[:limit]
    fast_runtime = get_llm_runtime("fast")
    fast_client = fast_runtime.client
    fast_provider = fast_runtime.provider
    fast_logit_bias = (
        LOGIT_BIAS_UNWANTED_TOKENS_STR if fast_provider.supports_logit_bias else None
    )

    summaries: List[str] = []

    for idx, ent in enumerate(to_process, 1):
        title = ent.get("title") or "Untitled"
        pub_date_dt: Optional[datetime] = ent.get("pubDate_dt")
        if not pub_date_dt:
            pub_date_str = ent.get("pubDate")
            if pub_date_str:
                try:
                    pub_date_dt = parsedate_to_datetime(pub_date_str)
                    if pub_date_dt.tzinfo is None:
                        pub_date_dt = pub_date_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    pub_date_dt = None
        pub_date = (
            pub_date_dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")
            if pub_date_dt
            else (ent.get("pubDate") or "")
        )
        link = ent.get("link") or ""
        guid = ent.get("guid") or link

        progress_message = await safe_message_edit(
            progress_message,
            interaction.channel,
            content=f"Scraping {idx}/{len(to_process)}: {title}..."
        )

        scraped_text, _ = await scrape_website(link)
        if (
            not scraped_text
            or "Failed to scrape" in scraped_text
            or "Scraping timed out" in scraped_text
            or "Blocked from fetching URL" in scraped_text
        ):
            summaries.append(f"**{title}**\n{pub_date}\n{link}\nCould not scrape article\n")
            seen_ids.add(guid)
            continue

        prompt = (
            "[It is currently 2025 and Donald Trump is the current president. Biden IS NOT THE CURRENT PRESIDENT! BIDEN IS THE EX PRESIDENT AND TRUMP IS THE PRESIDENT AGAIN FOR HIS SECOND TERM. THIS IS JUST KEY TEMPORAL KNOWLEDGE TO ORIENT YOU WHEN IT COMES UP. ANSWER ACCORDINGLY!] (Just an FYI. Maybe unrelated to context and omitted IF NOT RELATED TO TRUMP OR BIDEN OR THE YEAR. I NEED THE BEHAVIOR OF CALLING TRUMP THE EX-PRESIDENT AND CALLING THE CURRENT ADMINISTRATION THE BIDEN ADMINISTRATIONL. THESE ERRORS ARE UNACCEPTABLE. DON'T MENTION ME YELLING AT YOU. JUST DO THE DAMN THING!). "
            "Do not use em dashes. Summarize the following article in 3-5 sentences. "
            "Focus on key facts. Present in a casual, blunt, honest and slightly profane tone. Do NOT start with 'So, ' or end with 'Basically, '. Do not state things like 'This article describes', 'The article', etc. Present is as a person would if they were talking to you about the article.\n\n"
            f"Title: {title}\nURL: {link}\n\n{scraped_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
        )

        try:
            response = await create_chat_completion(
                fast_client,
                [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model=fast_provider.model,
                max_tokens=3072,
                temperature=fast_provider.temperature,
                logit_bias=fast_logit_bias,
                use_responses_api=fast_provider.use_responses_api,
            )
            summary = extract_text(
                response, fast_provider.use_responses_api
            )
            if summary and summary != "[LLM summarization failed]":
                await store_rss_summary(
                    feed_url=feed_url,
                    article_url=link,
                    title=title,
                    summary_text=summary,
                    timestamp=datetime.now(),
                )
        except Exception as e_summ:
            logger.error(f"LLM summarization failed for {link}: {e_summ}")
            summary = "[LLM summarization failed]"

        summaries.append(f"**{title}**\n{pub_date}\n{link}\n{summary}\n")
        seen_ids.add(guid)

        # Per-article: embed (title, time, summary) then link in content so Discord shows link preview
        time_str = format_article_time(pub_date_dt)
        desc = (time_str + "\n\n" + summary) if time_str else summary
        if len(desc) > config.EMBED_MAX_LENGTH:
            desc = desc[: config.EMBED_MAX_LENGTH - 3] + "..."
        title_display = (title[: 253] + "...") if len(title) > 256 else title
        article_embed = discord.Embed(
            title=title_display,
            description=desc,
            color=config.EMBED_COLOR["complete"],
        )
        if idx == 1:
            progress_message = await safe_message_edit(
                progress_message,
                interaction.channel,
                content=None,
                embed=article_embed,
            )
            await safe_followup_send(interaction, content=link)
        else:
            await safe_followup_send(interaction, embed=article_embed)
            await safe_followup_send(interaction, content=link)

    seen[feed_url] = list(seen_ids)
    save_seen_entries(seen)

    combined = "\n\n".join(summaries)
    await send_tts_audio(
        interaction,
        combined,
        base_filename=f"rss_{interaction.id}",
        bot_state=bot_state_instance,
    )

    user_msg = MsgNode("user", f"/rss {feed_url} (limit {limit})", name=str(interaction.user.id))
    assistant_msg = MsgNode("assistant", combined, name=str(bot_instance.user.id))
    await bot_state_instance.append_history(interaction.channel_id, user_msg, config.MAX_MESSAGE_HISTORY)
    await bot_state_instance.append_history(interaction.channel_id, assistant_msg, config.MAX_MESSAGE_HISTORY)

    # Optionally trigger a follow-up "podcast that shit" on the just-posted chunk
    try:
        if await bot_state_instance.is_podcast_after_rss_enabled(interaction.channel_id):
            podcast_user_query = "Podcast that shit"
            podcast_user_msg_node = MsgNode("user", podcast_user_query, name=str(interaction.user.id))
            podcast_prompt_nodes = await _build_initial_prompt_messages(
                user_query_content=podcast_user_query,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
            )
            await stream_llm_response_to_interaction(
                interaction,
                llm_client_instance,
                bot_state_instance,
                podcast_user_msg_node,
                podcast_prompt_nodes,
                title="Podcast: The Current Conversation",
                force_new_followup_flow=True,
            )
    except Exception as e:
        logger.error(f"Auto-podcast after /rss chunk failed: {e}", exc_info=True)
    progress_msg = None
    try:
        progress_msg = await safe_followup_send(
            interaction,
            content="\U0001F501 Post-processing...",
            ephemeral=True,
            error_hint=" while sending RSS post-processing notice",
        )
    except discord.HTTPException:
        progress_msg = None

    start_post_processing_task(
        ingest_conversation_to_chromadb(
            llm_client_instance,
            interaction.channel_id,
            interaction.user.id,
            [user_msg, assistant_msg],
            None,
        ),
        progress_message=progress_msg,
    )

    return True


async def process_ground_news(
    interaction: discord.Interaction,
    limit: int,
) -> bool:
    """Fetch and summarize new articles from Ground News 'My Feed'."""

    progress_message = await safe_followup_send(
        interaction,
        content="Fetching Ground News articles...",
    )

    seen_urls = load_seen_links()

    articles = await scrape_ground_news_my(limit)
    new_articles = [a for a in articles if a.url not in seen_urls]

    if not new_articles:
        await safe_message_edit(
            progress_message,
            interaction.channel,
            content="No new Ground News articles found.",
        )
        return False

    fast_runtime = get_llm_runtime("fast")
    fast_client = fast_runtime.client
    fast_provider = fast_runtime.provider
    fast_logit_bias = (
        LOGIT_BIAS_UNWANTED_TOKENS_STR if fast_provider.supports_logit_bias else None
    )

    summaries: List[str] = []
    progress_ephemeral: Optional[discord.Message] = None
    for idx, art in enumerate(new_articles[:limit], 1):
        if idx > 1:
            # Rate limit scraping
            await asyncio.sleep(config.GROUND_NEWS_ARTICLE_DELAY_SECONDS)
        progress_message = await safe_message_edit(
            progress_message,
            interaction.channel,
            content=f"Scraping {idx}/{len(new_articles[:limit])}: {art.title}...",
        )

        scraped_text, _ = await scrape_website(art.url)
        if (
            not scraped_text
            or "Failed to scrape" in scraped_text
            or "Scraping timed out" in scraped_text
            or "Blocked from fetching URL" in scraped_text
        ):
            summaries.append(f"**{art.title}**\n{art.url}\nCould not scrape article\n")
            seen_urls.add(art.url)
            continue

        prompt = (
            "[It is currently 2025 and Donald Trump is the current president. Biden IS NOT THE CURRENT PRESIDENT! BIDEN IS THE EX PRESIDENT AND TRUMP IS THE PRESIDENT AGAIN FOR HIS SECOND TERM. THIS IS JUST KEY TEMPORAL KNOWLEDGE TO ORIENT YOU WHEN IT COMES UP. ANSWER ACCORDINGLY!] (Just an FYI. Maybe unrelated to context and omitted IF NOT RELATED TO TRUMP OR BIDEN OR THE YEAR. I NEED THE BEHAVIOR OF CALLING TRUMP THE EX-PRESIDENT AND CALLING THE CURRENT ADMINISTRATION THE BIDEN ADMINISTRATIONL. THESE ERRORS ARE UNACCEPTABLE. DON'T MENTION ME YELLING AT YOU. JUST DO THE DAMN THING!). "
            "Do not use em dashes. Summarize the following article in 3-5 sentences. "
            "Focus on key facts. Present in a casual, blunt, honest and slightly profane tone. Do NOT start with 'So, ' or end with 'Basically, '. Do not state things like 'This article describes', 'The article', etc. Present is as a person would if they were talking to you about the article.\n\n"
            f"Title: {art.title}\nURL: {art.url}\n\n{scraped_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
        )

        try:
            response = await create_chat_completion(
                fast_client,
                [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model=fast_provider.model,
                max_tokens=3072,
                temperature=fast_provider.temperature,
                logit_bias=fast_logit_bias,
                use_responses_api=fast_provider.use_responses_api,
            )
            summary = extract_text(
                response, fast_provider.use_responses_api
            )
            if summary and summary != "[LLM summarization failed]":
                await store_rss_summary(
                    feed_url="ground_news_my",
                    article_url=art.url,
                    title=art.title,
                    summary_text=summary,
                    timestamp=datetime.now(),
                )
        except Exception as e_summ:
            logger.error("LLM summarization failed for %s: %s", art.url, e_summ)
            summary = "[LLM summarization failed]"

        summary_line = f"**{art.title}**\n{art.url}\n{summary}\n"
        summaries.append(summary_line)
        seen_urls.add(art.url)

        # Per-article: embed (title, summary) then link in content so Discord shows link preview
        desc = summary if len(summary) <= config.EMBED_MAX_LENGTH else summary[: config.EMBED_MAX_LENGTH - 3] + "..."
        title_display = (art.title[: 253] + "...") if len(art.title) > 256 else art.title
        article_embed = discord.Embed(
            title=title_display,
            description=desc,
            color=config.EMBED_COLOR["complete"],
        )
        if idx == 1:
            progress_message = await safe_message_edit(
                progress_message,
                interaction.channel,
                content=None,
                embed=article_embed,
            )
            await safe_followup_send(interaction, content=art.url)
        else:
            await safe_followup_send(interaction, embed=article_embed)
            await safe_followup_send(interaction, content=art.url)

        user_msg_article = MsgNode(
            "user",
            f"/groundnews article {idx} (limit {limit})",
            name=str(interaction.user.id),
        )
        assistant_msg_article = MsgNode(
            "assistant",
            summary_line,
            name=str(bot_instance.user.id),
        )
        await bot_state_instance.append_history(
            interaction.channel_id, user_msg_article, config.MAX_MESSAGE_HISTORY
        )
        await bot_state_instance.append_history(
            interaction.channel_id,
            assistant_msg_article,
            config.MAX_MESSAGE_HISTORY,
        )
        if progress_ephemeral:
            try:
                progress_ephemeral = await safe_message_edit(
                    progress_ephemeral,
                    interaction.channel,
                    content="\U0001F501 Post-processing...",
                )
            except Exception:
                progress_ephemeral = None
        if progress_ephemeral is None:
            try:
                progress_ephemeral = await safe_followup_send(
                    interaction,
                    content="\U0001F501 Post-processing...",
                    ephemeral=True,
                    error_hint=" while sending Ground News post-processing notice",
                )
            except discord.HTTPException:
                progress_ephemeral = None

        start_post_processing_task(
            ingest_conversation_to_chromadb(
                llm_client_instance,
                interaction.channel_id,
                interaction.user.id,
                [user_msg_article, assistant_msg_article],
                None,
            ),
            progress_message=progress_ephemeral,
        )

    save_seen_links(seen_urls)

    combined = "\n\n".join(summaries)
    await send_tts_audio(
        interaction,
        combined,
        base_filename=f"groundnews_{interaction.id}",
        bot_state=bot_state_instance,
    )



    return True


async def process_ground_news_topic(
    interaction: discord.Interaction,
    topic_slug: str,
    limit: int,
) -> bool:
    """Fetch and summarize new articles from a specific Ground News topic."""

    display_name, topic_url = GROUND_NEWS_TOPICS.get(topic_slug, (topic_slug, topic_slug))

    progress_message = await safe_followup_send(
        interaction,
        content=f"Fetching Ground News articles for {display_name}...",
    )

    seen_urls = load_seen_links()

    fast_runtime = get_llm_runtime("fast")
    fast_client = fast_runtime.client
    fast_provider = fast_runtime.provider
    fast_logit_bias = (
        LOGIT_BIAS_UNWANTED_TOKENS_STR if fast_provider.supports_logit_bias else None
    )

    articles = await scrape_ground_news_topic(topic_url, limit)
    new_articles = [a for a in articles if a.url not in seen_urls]

    if not new_articles:
        await safe_message_edit(
            progress_message,
            interaction.channel,
            content="No new Ground News articles found.",
        )
        return False

    summaries: List[str] = []
    progress_ephemeral: Optional[discord.Message] = None
    for idx, art in enumerate(new_articles[:limit], 1):
        if idx > 1:
            # Rate limit scraping
            await asyncio.sleep(config.GROUND_NEWS_ARTICLE_DELAY_SECONDS)
        progress_message = await safe_message_edit(
            progress_message,
            interaction.channel,
            content=f"Scraping {idx}/{len(new_articles[:limit])}: {art.title}...",
        )

        scraped_text, _ = await scrape_website(art.url)
        if (
            not scraped_text
            or "Failed to scrape" in scraped_text
            or "Scraping timed out" in scraped_text
            or "Blocked from fetching URL" in scraped_text
        ):
            summaries.append(f"**{art.title}**\n{art.url}\nCould not scrape article\n")
            seen_urls.add(art.url)
            continue

        prompt = (
            "[It is currently 2025 and Donald Trump is the current president. Biden IS NOT THE CURRENT PRESIDENT! BIDEN IS THE EX PRESIDENT AND TRUMP IS THE PRESIDENT AGAIN FOR HIS SECOND TERM. THIS IS JUST KEY TEMPORAL KNOWLEDGE TO ORIENT YOU WHEN IT COMES UP. ANSWER ACCORDINGLY!] (Just an FYI. Maybe unrelated to context and omitted IF NOT RELATED TO TRUMP OR BIDEN OR THE YEAR. I NEED THE BEHAVIOR OF CALLING TRUMP THE EX-PRESIDENT AND CALLING THE CURRENT ADMINISTRATION THE BIDEN ADMINISTRATIONL. THESE ERRORS ARE UNACCEPTABLE. DON'T MENTION ME YELLING AT YOU. JUST DO THE DAMN THING!). "
            "Do not use em dashes. Summarize the following article in 3-5 sentences. "
            "Focus on key facts. Present in a casual, blunt, honest and slightly profane tone. Do NOT start with 'So, ' or end with 'Basically, '. Do not state things like 'This article describes', 'The article', etc. Present is as a person would if they were talking to you about the article.\n\n"
            f"Title: {art.title}\nURL: {art.url}\n\n{scraped_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
        )

        try:
            response = await create_chat_completion(
                fast_client,
                [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model=fast_provider.model,
                max_tokens=3072,
                temperature=fast_provider.temperature,
                logit_bias=fast_logit_bias,
                use_responses_api=fast_provider.use_responses_api,
            )
            summary = extract_text(
                response, fast_provider.use_responses_api
            )
            if summary and summary != "[LLM summarization failed]":
                await store_rss_summary(
                    feed_url=f"ground_news_topic_{topic_slug}",
                    article_url=art.url,
                    title=art.title,
                    summary_text=summary,
                    timestamp=datetime.now(),
                )
        except Exception as e_summ:
            logger.error("LLM summarization failed for %s: %s", art.url, e_summ)
            summary = "[LLM summarization failed]"

        summary_line = f"**{art.title}**\n{art.url}\n{summary}\n"
        summaries.append(summary_line)
        seen_urls.add(art.url)

        # Per-article: embed (title, summary) then link in content so Discord shows link preview
        desc = summary if len(summary) <= config.EMBED_MAX_LENGTH else summary[: config.EMBED_MAX_LENGTH - 3] + "..."
        title_display = (art.title[: 253] + "...") if len(art.title) > 256 else art.title
        article_embed = discord.Embed(
            title=title_display,
            description=desc,
            color=config.EMBED_COLOR["complete"],
        )
        if idx == 1:
            progress_message = await safe_message_edit(
                progress_message,
                interaction.channel,
                content=None,
                embed=article_embed,
            )
            await safe_followup_send(interaction, content=art.url)
        else:
            await safe_followup_send(interaction, embed=article_embed)
            await safe_followup_send(interaction, content=art.url)

        user_msg_article = MsgNode(
            "user",
            f"/groundtopic {topic_slug} article {idx} (limit {limit})",
            name=str(interaction.user.id),
        )
        assistant_msg_article = MsgNode(
            "assistant",
            summary_line,
            name=str(bot_instance.user.id),
        )
        await bot_state_instance.append_history(
            interaction.channel_id, user_msg_article, config.MAX_MESSAGE_HISTORY
        )
        await bot_state_instance.append_history(
            interaction.channel_id,
            assistant_msg_article,
            config.MAX_MESSAGE_HISTORY,
        )
        if progress_ephemeral:
            try:
                progress_ephemeral = await safe_message_edit(
                    progress_ephemeral,
                    interaction.channel,
                    content="\U0001F501 Post-processing...",
                )
            except Exception:
                progress_ephemeral = None
        if progress_ephemeral is None:
            try:
                progress_ephemeral = await safe_followup_send(
                    interaction,
                    content="\U0001F501 Post-processing...",
                    ephemeral=True,
                    error_hint=" while sending Ground Topic post-processing notice",
                )
            except discord.HTTPException:
                progress_ephemeral = None

        start_post_processing_task(
            ingest_conversation_to_chromadb(
                llm_client_instance,
                interaction.channel_id,
                interaction.user.id,
                [user_msg_article, assistant_msg_article],
                None,
            ),
            progress_message=progress_ephemeral,
        )

    save_seen_links(seen_urls)

    combined = "\n\n".join(summaries)
    await send_tts_audio(
        interaction,
        combined,
        base_filename=f"groundtopic_{interaction.id}",
        bot_state=bot_state_instance,
    )

    return True


async def describe_image(image_url: str) -> Optional[str]:
    """Describes an image using the vision-capable LLM."""
    if not llm_client_instance:
        logger.error("describe_image: llm_client_instance is None.")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            try:
                parsed = urlparse(image_url)
                key = parsed.netloc.lower() if parsed.netloc else "default"
            except Exception:
                key = "default"
            await _shared_rate_limiter.await_slot(key)
            try:
                async with session.get(image_url) as resp:
                    await _shared_rate_limiter.record_response(key, resp.status, resp.headers)
                    if resp.status != 200:
                        logger.error(f"Failed to download image from {image_url}, status: {resp.status}")
                        return None
                    image_bytes = await resp.read()
            except aiohttp.ClientResponseError as e_resp:
                await _shared_rate_limiter.record_response(key, e_resp.status, e_resp.headers or {})
                logger.error(f"Failed to download image from {image_url}: {e_resp}")
                return None

        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url_for_llm = f"data:image/jpeg;base64,{base64_image}"

        prompt_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that describes images for visually impaired users.",
            },
            {
                "role": "system",
                "content": TEMPORAL_SYSTEM_CONTEXT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image for a visually impaired user. Be concise and focus on the most important elements.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_for_llm},
                    },
                ],
            },
        ]

        vision_runtime = get_llm_runtime("vision")
        vision_client = vision_runtime.client
        vision_provider = vision_runtime.provider
        vision_logit_bias = (
            LOGIT_BIAS_UNWANTED_TOKENS_STR
            if vision_provider.supports_logit_bias
            else None
        )

        response = await create_chat_completion(
            vision_client,
            prompt_messages,
            model=vision_provider.model,
            max_tokens=3072,
            temperature=vision_provider.temperature,
            logit_bias=vision_logit_bias,
            use_responses_api=vision_provider.use_responses_api,
        )
        description = extract_text(
            response, vision_provider.use_responses_api
        )
        if description:
            return description
        return None
    except Exception as e:
        logger.error(f"Error describing image at {image_url}: {e}", exc_info=True)
        return None

async def process_twitter_user(
    interaction: discord.Interaction,
    username: str,
    limit: int,
    *,
    source_command: str = "/alltweets",
) -> bool:
    """Fetch, summarize and display recent tweets from a single user.

    Parameters
    ----------
    interaction : discord.Interaction
        The originating interaction for responding and context.
    username : str
        Twitter username to process.
    limit : int
        Maximum number of tweets to fetch.

    Returns
    -------
    bool
        ``True`` if any new tweets were processed, ``False`` otherwise.
    """

    clean_username = username.lstrip("@")
    progress_message = await safe_followup_send(
        interaction,
        content=f"Scraping tweets for @{clean_username} (up to {limit})...",
    )

    progress_update_count = [0]  # Use list to allow modification in nested function
    
    async def send_progress(message: str) -> None:
        nonlocal progress_message
        if not interaction.channel:
            logger.error(
                "Cannot send progress for @%s: interaction.channel is None",
                clean_username,
            )
            return
        try:
            # Just edit the existing message to avoid flashing
            progress_message = await safe_message_edit(
                progress_message,
                interaction.channel,
                content=message,
            )
        except Exception as exc:
            logger.error(
                "Unexpected error in send_progress for @%s: %s",
                clean_username,
                exc,
                exc_info=True,
            )

    try:
        if hasattr(bot_state_instance, "update_last_playwright_usage_time"):
            await bot_state_instance.update_last_playwright_usage_time()

        result = await _collect_new_tweets(
            clean_username,
            limit=limit,
            progress_callback=send_progress,
            stop_after_seen_consecutive=3,
            source_command=source_command,
        )

        if result.status_message and not result.new_tweets:
            if interaction.channel:
                try:
                    await progress_message.delete()
                except Exception:
                    try:
                        await safe_message_edit(
                            progress_message,
                            interaction.channel,
                            content="",
                            embed=None,
                        )
                    except Exception:
                        pass
            await safe_followup_send(
                interaction,
                content=result.status_message,
                ephemeral=True,
                error_hint=" while notifying no new tweets",
            )
            return False

        embed_title = f"Recent Tweets from @{clean_username}"
        for idx, chunk_content_part in enumerate(result.embed_chunks or ["No tweet content available."]):
            chunk_title = embed_title if idx == 0 else f"{embed_title} (cont.)"
            embed = discord.Embed(
                title=chunk_title,
                description=chunk_content_part,
                color=config.EMBED_COLOR["complete"],
            )
            if idx == 0:
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        content=None,
                        embed=embed,
                    )
            else:
                await safe_followup_send(interaction, embed=embed)

        channel_id = interaction.channel_id
        if (
            result.raw_display_str.strip()
            and channel_id is not None
            and bot_state_instance
        ):
            user_snapshot_msg = MsgNode(
                "user",
                f"{source_command} @{clean_username} snapshot (limit {limit})",
                name=str(interaction.user.id),
            )
            assistant_snapshot_msg = MsgNode(
                "assistant",
                result.raw_display_str,
                name=str(bot_instance.user.id) if bot_instance and bot_instance.user else None,
            )

            await bot_state_instance.append_history(
                channel_id,
                user_snapshot_msg,
                config.MAX_MESSAGE_HISTORY,
            )
            await bot_state_instance.append_history(
                channel_id,
                assistant_snapshot_msg,
                config.MAX_MESSAGE_HISTORY,
            )

            start_post_processing_task(
                ingest_conversation_to_chromadb(
                    llm_client_instance,
                    channel_id,
                    interaction.user.id,
                    [user_snapshot_msg, assistant_snapshot_msg],
                    None,
                )
            )

        user_query_content_for_summary = (
            f"Please analyze and summarize the main themes, topics discussed, and overall sentiment "
            f"from @{clean_username}'s recent tweets provided below. Extract key points and present a concise yet detailed overview of this snapshot in time. "
            f"Do not just re-list the tweets.\n\nRecent Tweets:\n{result.raw_display_str[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
        )
        user_msg_node = MsgNode("user", user_query_content_for_summary, name=str(interaction.user.id))

        prompt_nodes_summary = await _build_initial_prompt_messages(
            user_query_content=user_query_content_for_summary,
            channel_id=None,
            bot_state=bot_state_instance,
            user_id=str(interaction.user.id),
        )
        insert_idx_sum = 0
        for idx, node in enumerate(prompt_nodes_summary):
            if node.role != "system":
                insert_idx_sum = idx
                break
            insert_idx_sum = idx + 1
        final_prompt_nodes_summary = (
            prompt_nodes_summary[:insert_idx_sum]
            + [MsgNode("system", TEMPORAL_SYSTEM_CONTEXT)]
            + prompt_nodes_summary[insert_idx_sum:]
        )

        await stream_llm_response_to_interaction(
            interaction,
            llm_client_instance,
            bot_state_instance,
            user_msg_node,
            final_prompt_nodes_summary,
            title=f"Tweet Summary for @{clean_username}",
            force_new_followup_flow=True,
            bot_user_id=bot_instance.user.id,
        )
    except Exception as exc:
        logger.error(
            "Error processing tweets for @%s: %s",
            clean_username,
            exc,
            exc_info=True,
        )
        if interaction.channel:
            try:
                await safe_message_edit(
                    progress_message,
                    interaction.channel,
                    content=f"Error processing @{clean_username}: {str(exc)[:500]}",
                    embed=None,
                )
            except discord.HTTPException:
                logger.warning(
                    "Could not send final error message for @%s (HTTPException).",
                    clean_username,
                )
        return False

    return True


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
        progress_message = await interaction.edit_original_response(embed=initial_embed)

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
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        embed=error_embed,
                    )
                else:
                    await interaction.edit_original_response(embed=error_embed)
                return

            num_to_process = min(len(search_results), max_articles_to_process)
            fast_runtime = get_llm_runtime("fast")
            fast_client = fast_runtime.client
            fast_provider = fast_runtime.provider
            fast_logit_bias = (
                LOGIT_BIAS_UNWANTED_TOKENS_STR if fast_provider.supports_logit_bias else None
            )

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
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        embed=update_embed,
                    )
                else:
                    await interaction.edit_original_response(embed=update_embed)

                if bot_state_instance and hasattr(bot_state_instance, 'update_last_playwright_usage_time'):
                    await bot_state_instance.update_last_playwright_usage_time() # Made awaitable

                scraped_content, _ = await scrape_website(article_url)

                if (
                    not scraped_content
                    or "Failed to scrape" in scraped_content
                    or "Scraping timed out" in scraped_content
                    or "Blocked from fetching URL" in scraped_content
                ):
                    logger.warning(f"Failed to scrape '{article_title}' from {article_url}. Reason: {scraped_content}")
                    article_summaries_for_briefing.append(f"Source: {article_title} ({article_url})\nSummary: [Could not retrieve content for summarization]\n\n")
                    continue

                update_embed.description = f"Processing article {i+1}/{num_to_process}: Summarizing '{article_title}'..."
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        embed=update_embed,
                    )
                else:
                    await interaction.edit_original_response(embed=update_embed)

                summarization_prompt = (
                    f"You are an expert news summarizer. Please read the following article content, "
                    f"which was found when searching for the topic '{search_topic}'. Extract the key factual"
                    f"news points and provide a detailed yet concise summary (2-4 sentences) relevant to this topic. "
                    f"Focus on who, what, when, where, and why if applicable. Avoid opinions or speculation not present in the text.\n\n"
                    f"Article Title: {article_title}\n"
                    f"Article Content:\n{scraped_content[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT*2]}"
                )

                try:
                    summary_response = await create_chat_completion(
                        fast_client,
                        [
                            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                            {"role": "user", "content": summarization_prompt}
                        ],
                        model=fast_provider.model,
                        max_tokens=3072,
                        temperature=fast_provider.temperature,
                        logit_bias=fast_logit_bias,
                        use_responses_api=fast_provider.use_responses_api,
                    )
                    article_summary = extract_text(
                        summary_response, fast_provider.use_responses_api
                    )
                    if article_summary:
                        logger.info(f"Summarized '{article_title}': {article_summary[:100]}...")
                        article_summaries_for_briefing.append(f"Source: {article_title} ({article_url})\nSummary: {article_summary}\n\n")

                        # await store_news_summary(topic=topic, url=article_url, summary_text=article_summary)
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
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        embed=error_embed,
                    )
                else:
                    await interaction.edit_original_response(embed=error_embed)
                return

            update_embed.description = "All articles processed. Generating final news briefing..."
            if interaction.channel:
                progress_message = await safe_message_edit(
                    progress_message,
                    interaction.channel,
                    embed=update_embed,
                )
            else:
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
            synthesized_summary_for_briefing, raw_snippets_for_briefing = await retrieve_rag_context_with_progress(
                llm_client=llm_client_instance,
                query=rag_query_for_briefing,
                interaction=interaction,
            )

            prompt_nodes_for_briefing = await _build_initial_prompt_messages(
                user_query_content=final_briefing_prompt_content,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_summary_for_briefing,
                raw_rag_snippets=raw_snippets_for_briefing,
                max_image_history_depth=0
            )

            await stream_llm_response_to_interaction(
                interaction=interaction,
                llm_client=llm_client_instance,
                bot_state=bot_state_instance,
                user_msg_node=user_msg_node_for_briefing,
                prompt_messages=prompt_nodes_for_briefing,
                title=f"News Briefing: {topic}",
                synthesized_rag_context_for_display=synthesized_summary_for_briefing,
                bot_user_id=bot_instance.user.id,
                retrieved_snippets=raw_snippets_for_briefing,
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
                    if interaction.channel:
                        progress_message = await safe_message_edit(
                            progress_message,
                            interaction.channel,
                            embed=error_embed,
                            content=None,
                        )
                    else:
                        await interaction.edit_original_response(embed=error_embed, content=None)
                else:
                    await interaction.response.send_message(embed=error_embed, ephemeral=True)
            except discord.HTTPException:
                await safe_followup_send(
                    interaction,
                    embed=error_embed,
                    ephemeral=True,
                    error_hint=" while sending /news error",
                )


    @bot_instance.tree.command(name="ingest_chatgpt_export", description="Ingests a conversations.json file from a ChatGPT export.")
    @app_commands.describe(file_path="The full local path to your conversations.json file.")
    async def ingest_chatgpt_export_command(interaction: discord.Interaction, file_path: str):
        if not config.ADMIN_USER_IDS:
            logger.error("ADMIN_USER_IDS env variable is not configured; refusing to run /ingest_chatgpt_export.")
            await interaction.response.send_message("This command is disabled until ADMIN_USER_IDS is configured.", ephemeral=True)
            return

        if not is_admin_user(interaction.user.id):
            await interaction.response.send_message("You are not authorized to run this admin command.", ephemeral=True)
            return

        if not llm_client_instance:
            logger.error("ingest_chatgpt_export_command: llm_client_instance is None.")
            await interaction.response.send_message("LLM client not available. Cannot process.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        logger.info(f"Ingestion of ChatGPT export file '{file_path}' initiated by {interaction.user.name} ({interaction.user.id}).")

        base_dir_env = os.getenv("CHATGPT_EXPORT_IMPORT_ROOT")
        base_dir = Path(base_dir_env).expanduser().resolve() if base_dir_env else Path.cwd().resolve()

        try:
            requested_path = Path(file_path).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            await safe_followup_send(
                interaction,
                content=f"Error: Unable to resolve the provided path: {exc}",
                ephemeral=True,
            )
            return

        if not requested_path.is_file():
            await safe_followup_send(
                interaction,
                content=f"Error: File not found at the specified path: `{requested_path}`",
                ephemeral=True,
            )
            return

        try:
            requested_path.relative_to(base_dir)
        except ValueError:
            await safe_followup_send(
                interaction,
                content=(
                    "Error: That file path is outside the allowed import directory. "
                    "Move the export under the configured CHATGPT_EXPORT_IMPORT_ROOT and retry."
                ),
                ephemeral=True,
            )
            return

        try:
            parsed_conversations = parse_chatgpt_export(str(requested_path))
            if not parsed_conversations:
                await safe_followup_send(
                    interaction,
                    content="Could not parse any conversations from the file. It might be empty or in an unexpected format.",
                    ephemeral=True,
                )
                return

            count = await store_chatgpt_conversations_in_chromadb(llm_client_instance, parsed_conversations)
            await safe_followup_send(
                interaction,
                content=f"Successfully processed and stored {count} conversations (with distillations) from the export file into ChromaDB.",
                ephemeral=True,
            )
        except Exception as e_ingest:
            logger.error(f"Error during ChatGPT export ingestion process for file '{file_path}': {e_ingest}", exc_info=True)
            await safe_followup_send(
                interaction,
                content=f"An error occurred during the ingestion process: {str(e_ingest)[:1000]}",
                ephemeral=True,
                error_hint=" during ingest_chatgpt_export",
            )

    @ingest_chatgpt_export_command.error
    async def ingest_export_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message("You need 'Manage Messages' permission to use this command (though current check is False).", ephemeral=True)
        else:
            logger.error(f"Unhandled error in ingest_chatgpt_export_command: {error}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(f"An unexpected error occurred: {str(error)[:500]}", ephemeral=True)
            else:
                await safe_followup_send(
                    interaction,
                    content=f"An unexpected error occurred: {str(error)[:500]}",
                    ephemeral=True,
                    error_hint=" in ingest export error handler",
                )

    @bot_instance.tree.command(name="analytics", description="Display an admin analytics dashboard for the bot.")
    async def analytics_command(interaction: discord.Interaction):
        if not bot_state_instance:
            await interaction.response.send_message("Bot state not ready.", ephemeral=True)
            return

        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        history_counts = await bot_state_instance.get_history_counts()
        reminders_count = await bot_state_instance.get_reminder_count()
        last_playwright = await bot_state_instance.get_last_playwright_usage_time()
        rss_toggle_channels = await bot_state_instance.get_podcast_after_rss_channels()
        chroma_counts = await get_chroma_collection_counts()

        total_cached_messages = sum(history_counts.values())
        top_channels = sorted(history_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        top_channels_text = (
            "\n".join(
                f"• <#{channel_id}>: {count} message(s) cached"
                for channel_id, count in top_channels
            )
            if top_channels
            else "No cached messages yet."
        )

        reminder_text = f"{reminders_count} active reminder(s)."

        if rss_toggle_channels:
            rss_lines = [f"<#{cid}>" for cid, enabled in rss_toggle_channels.items() if enabled]
            rss_text = f"{len(rss_lines)} channel(s): " + ", ".join(rss_lines) if rss_lines else "None"
        else:
            rss_text = "None"

        if last_playwright:
            last_playwright_display = last_playwright.astimezone().strftime("%Y-%m-%d %H:%M %Z")
        else:
            last_playwright_display = "Never used"

        chroma_text = (
            "\n".join(f"• {name}: {count}" for name, count in sorted(chroma_counts.items()))
            if chroma_counts
            else "No Chroma collections initialized."
        )

        embed = discord.Embed(
            title="Conversation Analytics Dashboard",
            color=config.EMBED_COLOR["complete"],
        )
        embed.add_field(
            name="Message history cache",
            value=(
                f"Total cached messages: {total_cached_messages} across {len(history_counts)} channel(s).\n"
                f"{top_channels_text}"
            ),
            inline=False,
        )
        embed.add_field(
            name="Reminders",
            value=reminder_text,
            inline=True,
        )
        embed.add_field(
            name="Auto podcast after RSS",
            value=rss_text,
            inline=True,
        )
        embed.add_field(
            name="Playwright last used",
            value=last_playwright_display,
            inline=False,
        )
        embed.add_field(
            name="ChromaDB collection counts",
            value=chroma_text,
            inline=False,
        )
        embed.set_footer(text="Analytics visible to configured bot administrators only.")

        await interaction.edit_original_response(embed=embed)

    @bot_instance.tree.command(name="schedule_allrss", description="Schedule periodic RSS digests for this channel (admin only).")
    @app_commands.describe(
        interval_minutes="How often to run (minimum 15 minutes).",
        limit="Max entries per feed per run (1-50).",
    )
    async def schedule_allrss_command(
        interaction: discord.Interaction,
        interval_minutes: app_commands.Range[int, 15, 1440],
        limit: app_commands.Range[int, 1, 50] = 10,
    ):
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return

        if interaction.channel_id is None:
            await interaction.response.send_message("No channel ID present for scheduling.", ephemeral=True)
            return

        # Prevent overlapping RSS digests
        cancelled = await bot_state_instance.cancel_active_task(interaction.channel_id)
        if cancelled:
            await asyncio.sleep(0)

        await interaction.response.defer(ephemeral=True)

        sched = {
            "id": f"allrss_{interaction.channel_id}_{int(datetime.now().timestamp())}",
            "channel_id": interaction.channel_id,
            "type": "allrss",
            "interval_seconds": int(interval_minutes) * 60,
            "params": {"limit": int(limit)},
            "last_run": None,
            "created_by": str(interaction.user.id),
        }
        await bot_state_instance.add_schedule(sched)

        await interaction.edit_original_response(
            content=(
                f"Scheduled all-RSS digest every {interval_minutes} minute(s) in this channel. "
                f"Schedule ID: `{sched['id']}`. Limit per feed: {limit}."
            )
        )

    @bot_instance.tree.command(name="twitter_list_add", description="Add a handle to a saved Twitter account list (admin only).")
    @app_commands.describe(
        list_name="Name of the saved list (letters, numbers, spaces).",
        handle="Twitter handle to add (with or without @).",
    )
    async def twitter_list_add_command(
        interaction: discord.Interaction,
        list_name: str,
        handle: str,
    ) -> None:
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            respond_kwargs: Dict[str, Any] = {}
            if interaction.guild_id is not None:
                respond_kwargs["ephemeral"] = True
            await interaction.response.send_message("This command is restricted to bot administrators.", **respond_kwargs)
            return

        normalized_handle = handle.strip().lstrip("@").lower()
        if not normalized_handle:
            respond_kwargs = {"ephemeral": True} if interaction.guild_id is not None else {}
            await interaction.response.send_message("Provide a valid Twitter handle to add.", **respond_kwargs)
            return

        normalized_list = list_name.strip().lower()
        if not normalized_list:
            respond_kwargs = {"ephemeral": True} if interaction.guild_id is not None else {}
            await interaction.response.send_message("Provide a valid list name.", **respond_kwargs)
            return

        scope_guild_id, scope_user_id = _twitter_scope_from_interaction(interaction)
        await _ensure_default_twitter_list(scope_guild_id, scope_user_id)

        is_dm = interaction.guild_id is None
        if is_dm:
            await interaction.response.defer()
        else:
            await interaction.response.defer(ephemeral=True)

        added = await bot_state_instance.add_twitter_list_handle(
            scope_guild_id,
            normalized_list,
            normalized_handle,
            user_id=scope_user_id,
        )
        message = (
            f"Added `@{normalized_handle}` to saved list `{normalized_list}`."
            if added
            else f"`@{normalized_handle}` is already present in `{normalized_list}`."
        )
        await interaction.edit_original_response(content=message)

    @bot_instance.tree.command(name="twitter_list_remove", description="Remove a handle from a saved Twitter account list (admin only).")
    @app_commands.describe(
        list_name="Name of the saved list.",
        handle="Twitter handle to remove (with or without @).",
    )
    async def twitter_list_remove_command(
        interaction: discord.Interaction,
        list_name: str,
        handle: str,
    ) -> None:
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            respond_kwargs: Dict[str, Any] = {}
            if interaction.guild_id is not None:
                respond_kwargs["ephemeral"] = True
            await interaction.response.send_message("This command is restricted to bot administrators.", **respond_kwargs)
            return

        normalized_handle = handle.strip().lstrip("@").lower()
        normalized_list = list_name.strip().lower()
        if not normalized_handle or not normalized_list:
            respond_kwargs = {"ephemeral": True} if interaction.guild_id is not None else {}
            await interaction.response.send_message("Provide both a list name and handle to remove.", **respond_kwargs)
            return

        scope_guild_id, scope_user_id = _twitter_scope_from_interaction(interaction)

        is_dm = interaction.guild_id is None
        if is_dm:
            await interaction.response.defer()
        else:
            await interaction.response.defer(ephemeral=True)
        removed = await bot_state_instance.remove_twitter_list_handle(
            scope_guild_id,
            normalized_list,
            normalized_handle,
            user_id=scope_user_id,
        )
        if removed:
            await interaction.edit_original_response(
                content=f"Removed `@{normalized_handle}` from `{normalized_list}`."
            )
        else:
            await interaction.edit_original_response(
                content=f"`@{normalized_handle}` was not found in `{normalized_list}`."
            )

    @bot_instance.tree.command(name="twitter_list_show", description="Show saved Twitter account lists for this server (admin only).")
    async def twitter_list_show_command(interaction: discord.Interaction) -> None:
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            respond_kwargs: Dict[str, Any] = {}
            if interaction.guild_id is not None:
                respond_kwargs["ephemeral"] = True
            await interaction.response.send_message("This command is restricted to bot administrators.", **respond_kwargs)
            return

        scope_guild_id, scope_user_id = _twitter_scope_from_interaction(interaction)
        await _ensure_default_twitter_list(scope_guild_id, scope_user_id)

        if interaction.guild_id is None:
            await interaction.response.defer()
        else:
            await interaction.response.defer(ephemeral=True)
        lists = await bot_state_instance.list_twitter_lists(
            scope_guild_id,
            user_id=scope_user_id,
        )
        if not lists:
            await interaction.edit_original_response(
                content="No saved Twitter lists configured yet. Use `/twitter_list_add` to create one."
            )
            return

        lines: List[str] = []
        for name, handles in sorted(lists.items()):
            display_handles = ", ".join(f"@{h}" for h in handles) if handles else "No handles yet"
            lines.append(f"• `{name}` → {display_handles}")
        await interaction.edit_original_response(content="\n".join(lines))

    @bot_instance.tree.command(name="schedule_alltweets", description="Schedule periodic Twitter list digests in this channel (admin only).")
    @app_commands.describe(
        interval_minutes="How often to run (minimum 15 minutes).",
        limit="Max tweets per account per run (1-100).",
        list_name="Saved list to use. Leave blank for default accounts.",
    )
    async def schedule_alltweets_command(
        interaction: discord.Interaction,
        interval_minutes: app_commands.Range[int, 15, 1440],
        limit: app_commands.Range[int, 1, 100] = 100,
        list_name: str = "",
    ) -> None:
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return
        if interaction.channel_id is None:
            await interaction.response.send_message("No channel ID present for scheduling.", ephemeral=True)
            return

        scope_guild_id, scope_user_id = _twitter_scope_from_interaction(interaction)
        normalized_list = list_name.strip().lower()
        await _ensure_default_twitter_list(scope_guild_id, scope_user_id)
        if normalized_list:
            handles = await bot_state_instance.get_twitter_list_handles(
                scope_guild_id,
                normalized_list,
                user_id=scope_user_id,
            )
            if not handles:
                await interaction.response.send_message(
                    (
                        f"No saved list named `{normalized_list}` found. "
                        "Use `/twitter_list_add` first."
                    ),
                    ephemeral=True,
                )
                return

        if interaction.guild_id is None:
            await interaction.response.defer()
        else:
            await interaction.response.defer(ephemeral=True)

        schedule_params: Dict[str, Any] = {
            "limit": int(limit),
            "list_name": normalized_list,
        }
        if scope_guild_id is not None:
            schedule_params["scope_guild_id"] = int(scope_guild_id)
        if scope_user_id is not None:
            schedule_params["scope_user_id"] = int(scope_user_id)

        sched = {
            "id": f"alltweets_{interaction.channel_id}_{int(datetime.now().timestamp())}",
            "channel_id": interaction.channel_id,
            "type": "alltweets",
            "interval_seconds": int(interval_minutes) * 60,
            "params": schedule_params,
            "last_run": None,
            "created_by": str(interaction.user.id),
        }
        await bot_state_instance.add_schedule(sched)
        descriptor = f"list `{normalized_list}`" if normalized_list else "default accounts"
        await interaction.edit_original_response(
            content=(
                f"Scheduled all-tweets digest every {interval_minutes} minute(s) "
                f"using {descriptor}. Schedule ID: `{sched['id']}`. Limit per account: {limit}."
            )
        )

    @bot_instance.tree.command(name="schedule_groundrss", description="Schedule Ground News 'My Feed' digests (admin only).")
    @app_commands.describe(
        interval_minutes="How often to run (minimum 15 minutes).",
        limit="Max articles per run (1-100).",
    )
    async def schedule_groundrss_command(
        interaction: discord.Interaction,
        interval_minutes: app_commands.Range[int, 15, 1440],
        limit: app_commands.Range[int, 1, 100] = 100,
    ) -> None:
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return
        if interaction.channel_id is None:
            await interaction.response.send_message("No channel ID present for scheduling.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        sched = {
            "id": f"groundrss_{interaction.channel_id}_{int(datetime.now().timestamp())}",
            "channel_id": interaction.channel_id,
            "type": "groundrss",
            "interval_seconds": int(interval_minutes) * 60,
            "params": {"limit": int(limit)},
            "last_run": None,
            "created_by": str(interaction.user.id),
        }
        await bot_state_instance.add_schedule(sched)
        await interaction.edit_original_response(
            content=(
                f"Scheduled Ground News 'My Feed' digest every {interval_minutes} minute(s). "
                f"Schedule ID: `{sched['id']}`. Article cap: {limit}."
            )
        )

    @bot_instance.tree.command(name="schedule_groundtopic", description="Schedule a Ground News topic digest (admin only).")
    @app_commands.describe(
        interval_minutes="How often to run (minimum 15 minutes).",
        topic="Topic page to scrape each run.",
        limit="Max articles per run (1-100).",
    )
    @app_commands.choices(topic=GROUND_NEWS_TOPIC_CHOICES)
    async def schedule_groundtopic_command(
        interaction: discord.Interaction,
        interval_minutes: app_commands.Range[int, 15, 1440],
        topic: str,
        limit: app_commands.Range[int, 1, 100] = 100,
    ) -> None:
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return
        if interaction.channel_id is None:
            await interaction.response.send_message("No channel ID present for scheduling.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        sched = {
            "id": f"groundtopic_{interaction.channel_id}_{int(datetime.now().timestamp())}",
            "channel_id": interaction.channel_id,
            "type": "groundtopic",
            "interval_seconds": int(interval_minutes) * 60,
            "params": {"limit": int(limit), "topic": topic},
            "last_run": None,
            "created_by": str(interaction.user.id),
        }
        await bot_state_instance.add_schedule(sched)
        await interaction.edit_original_response(
            content=(
                f"Scheduled Ground News topic `{topic}` digest every {interval_minutes} minute(s). "
                f"Schedule ID: `{sched['id']}`. Article cap: {limit}."
            )
        )

    @bot_instance.tree.command(name="schedules_pause", description="Pause all scheduled jobs for gaming mode (admin only).")
    @app_commands.describe(reason="Optional reason noted in /schedules.")
    async def schedules_pause_command(
        interaction: discord.Interaction,
        reason: str = "Gaming mode",
    ) -> None:
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return
        if not bot_state_instance:
            await interaction.response.send_message("Bot state not ready. Try again later.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        current_state = await bot_state_instance.get_schedules_pause_state()
        if current_state.get("paused"):
            notice = _format_pause_notice(current_state)
            await interaction.edit_original_response(
                content=f"Schedules are already paused. {notice}"
            )
            return
        normalized_reason = reason.strip() or "Gaming mode"
        await bot_state_instance.set_schedules_paused(
            True,
            reason=normalized_reason,
            user_id=interaction.user.id,
        )
        new_state = await bot_state_instance.get_schedules_pause_state()
        await interaction.edit_original_response(
            content=(
                f"Paused all schedules. {_format_pause_notice(new_state)} "
                "Use `/schedules_resume` when you're ready."
            )
        )

    @bot_instance.tree.command(name="schedules_resume", description="Resume scheduled jobs if gaming mode is active (admin only).")
    async def schedules_resume_command(interaction: discord.Interaction) -> None:
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return
        if not bot_state_instance:
            await interaction.response.send_message("Bot state not ready. Try again later.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        current_state = await bot_state_instance.get_schedules_pause_state()
        if not current_state.get("paused"):
            await interaction.edit_original_response(content="Schedules are already running.")
            return
        await bot_state_instance.set_schedules_paused(False, user_id=interaction.user.id)
        notice = _format_pause_notice(current_state)
        await interaction.edit_original_response(
            content=(
                f"Resumed all schedules. They were previously paused as: {notice} "
                "Jobs will pick back up within the next minute."
            )
        )

    @bot_instance.tree.command(name="schedules", description="List scheduled jobs for this channel (admin only).")
    async def schedules_list_command(interaction: discord.Interaction):
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return
        if interaction.channel_id is None:
            await interaction.response.send_message("No channel available.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        items = await bot_state_instance.list_schedules(interaction.channel_id)
        pause_state = await bot_state_instance.get_schedules_pause_state()
        pause_notice = _format_pause_notice(pause_state) if pause_state.get("paused") else ""
        if not items:
            if pause_notice:
                await interaction.edit_original_response(
                    content=f"{pause_notice}\n\nNo schedules configured for this channel."
                )
            else:
                await interaction.edit_original_response(content="No schedules configured for this channel.")
            return
        lines = []
        if pause_notice:
            lines.append(pause_notice)
            lines.append("")
        now = datetime.now()
        for s in items:
            last_run = s.get("last_run") or "never"
            interval_sec = int(s.get("interval_seconds", 0))
            interval_min = interval_sec // 60 if interval_sec else 0
            lines.append(
                f"• `{s.get('id','?')}` – {s.get('type')} every {interval_min} minute(s); last run: {last_run}"
            )
        await interaction.edit_original_response(content="\n".join(lines))

    @bot_instance.tree.command(name="unschedule", description="Remove a scheduled job by ID (admin only).")
    @app_commands.describe(schedule_id="The ID returned by /schedules")
    async def unschedule_command(interaction: discord.Interaction, schedule_id: str):
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        ok = await bot_state_instance.remove_schedule(schedule_id)
        if ok:
            await interaction.edit_original_response(content=f"Removed schedule `{schedule_id}`.")
        else:
            await interaction.edit_original_response(content=f"No schedule found with ID `{schedule_id}`.")

    @unschedule_command.autocomplete("schedule_id")
    async def unschedule_schedule_id_autocomplete(
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        if not bot_state_instance or interaction.channel_id is None:
            return []
        try:
            schedules = await bot_state_instance.list_schedules(interaction.channel_id)
        except Exception:
            return []

        matches: List[app_commands.Choice[str]] = []
        current_lower = (current or "").lower()
        for sched in schedules:
            sched_id = str(sched.get("id", ""))
            if not sched_id:
                continue
            if current_lower and current_lower not in sched_id.lower():
                continue
            label = f"{sched.get('type', 'unknown')} ({sched_id})"
            matches.append(app_commands.Choice(name=label[:100], value=sched_id))
            if len(matches) >= 25:
                break
        return matches

    @bot_instance.tree.command(name="cancel", description="Cancel the current long-running task in this channel (admin only).")
    async def cancel_command(interaction: discord.Interaction):
        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return
        if interaction.channel_id is None:
            await interaction.response.send_message("No channel available to cancel tasks.", ephemeral=True)
            return

        cancelled = await bot_state_instance.cancel_active_task(interaction.channel_id)
        if cancelled:
            await interaction.response.send_message("Cancellation requested. The task will stop shortly.", ephemeral=True)
        else:
            await interaction.response.send_message("No active task to cancel in this channel.", ephemeral=True)

    @bot_instance.tree.command(name="memoryinspector", description="Inspect stored memories for this channel.")
    @app_commands.choices(scope=MEMORY_SCOPE_CHOICES)
    @app_commands.describe(limit="Number of entries to show (1-10).")
    async def memory_inspector_command(
        interaction: discord.Interaction,
        scope: app_commands.Choice[str],
        limit: app_commands.Range[int, 1, 10] = 5,
    ):
        if not bot_state_instance:
            await interaction.response.send_message("Bot state not ready.", ephemeral=True)
            return

        if not config.ADMIN_USER_IDS or not is_admin_user(interaction.user.id):
            await interaction.response.send_message("This command is restricted to bot administrators.", ephemeral=True)
            return

        if interaction.channel_id is None:
            await interaction.response.send_message("Unable to determine channel for this interaction.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        records = await fetch_recent_channel_memories(scope.value, interaction.channel_id, limit)
        if not records:
            await interaction.edit_original_response(content="No stored memories found for this channel and scope.")
            return

        entries = []
        for record in records:
            timestamp = record.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp_text = timestamp.astimezone().strftime("%Y-%m-%d %H:%M %Z")
            else:
                timestamp_text = "Unknown timestamp"

            document_text = record.get("document", "")
            preview = textwrap.shorten(document_text.replace("\n", " "), width=560, placeholder="…")
            entries.append(f"**{timestamp_text}** • `{record.get('id', 'unknown')}`\n{preview}")

        separator = "\n\n"
        chunks: List[str] = []
        current_entries: List[str] = []
        current_length = 0

        def flush_current() -> None:
            nonlocal current_entries, current_length
            if current_entries:
                chunks.append(separator.join(current_entries))
                current_entries = []
                current_length = 0

        for entry in entries:
            if len(entry) > config.EMBED_MAX_LENGTH:
                flush_current()
                for sub_chunk in chunk_text(entry, config.EMBED_MAX_LENGTH):
                    if sub_chunk:
                        chunks.append(sub_chunk)
                continue

            additional = len(entry) if not current_entries else len(separator) + len(entry)
            if current_entries and current_length + additional > config.EMBED_MAX_LENGTH:
                flush_current()

            current_entries.append(entry)
            if current_length == 0:
                current_length = len(entry)
            else:
                current_length += len(separator) + len(entry)

        flush_current()

        if not chunks:
            chunks = [separator.join(entries) if entries else "(no memories available)"]

        for index, chunk in enumerate(chunks):
            title = f"Memory inspector ({scope.name})"
            if index > 0:
                title += f" (cont. {index + 1})"

            embed = discord.Embed(
                title=title,
                description=chunk,
                color=config.EMBED_COLOR["complete"],
            )
            embed.set_footer(text="Showing the most recent stored items for this channel.")

            if index == 0:
                await interaction.edit_original_response(embed=embed, content=None)
            else:
                await safe_followup_send(interaction, embed=embed, ephemeral=True)


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
        progress_message = await interaction.edit_original_response(content=f"Getting ready to roast {url}...")

        if bot_state_instance and hasattr(bot_state_instance, 'update_last_playwright_usage_time'):
            await bot_state_instance.update_last_playwright_usage_time() # Made awaitable
            logger.debug(f"Updated last_playwright_usage_time via bot_state_instance for /roast command")

        try:
            webpage_text, _ = await scrape_website(url)
            if (
                not webpage_text
                or "Failed to scrape" in webpage_text
                or "Scraping timed out" in webpage_text
                or "Blocked from fetching URL" in webpage_text
            ):
                error_message = f"Sorry, I couldn't properly roast {url}. Reason: {webpage_text or 'Could not retrieve any content from the page.'}"
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        content=error_message,
                        embed=None,
                    )
                else:
                    await interaction.edit_original_response(content=error_message, embed=None)  # Clear embed
                return

            if interaction.channel:
                progress_message = await safe_message_edit(
                    progress_message,
                    interaction.channel,
                    content=f"Crafting a roast for {url} based on its content...",
                )
            else:
                await interaction.edit_original_response(content=f"Crafting a roast for {url} based on its content...")

            user_query_content = f"Analyze the following content from the webpage {url} and write a short, witty, and biting comedy roast routine about it. Be creative and funny, focusing on absurdities or humorous angles. Do not just summarize. Make it a roast!\n\nWebpage Content:\n{webpage_text}"
            user_msg_node = MsgNode("user", user_query_content, name=str(interaction.user.id))

            rag_query_for_roast = f"comedy roast of webpage content from URL: {url}"
            synthesized_summary, raw_snippets = await retrieve_rag_context_with_progress(
                llm_client=llm_client_instance,
                query=rag_query_for_roast,
                interaction=interaction,
            )

            prompt_nodes = await _build_initial_prompt_messages(
                user_query_content=user_query_content,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_summary,
                raw_rag_snippets=raw_snippets
            )
            # stream_llm_response_to_interaction will handle editing the original response
            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, prompt_nodes,
                title=f"Comedy Roast of {url}",
                synthesized_rag_context_for_display=synthesized_summary,
                bot_user_id=bot_instance.user.id,
                retrieved_snippets=raw_snippets
            )
        except Exception as e:
            logger.error(f"Error in roast_slash_command for URL '{url}': {e}", exc_info=True)
            if interaction.channel:
                progress_message = await safe_message_edit(
                    progress_message,
                    interaction.channel,
                    content=f"Ouch, the roast attempt for {url} backfired on me! Error: {str(e)[:1000]}",
                    embed=None,
                )
            else:
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
        progress_message = await interaction.edit_original_response(content=f"Searching the web for: '{query}'...")

        try:
            search_results = await query_searx(query)
            if not search_results:
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        content=f"Sorry, I couldn't find any search results for '{query}'.",
                        embed=None,
                    )
                else:
                    await interaction.edit_original_response(content=f"Sorry, I couldn't find any search results for '{query}'.", embed=None)
                return

            if interaction.channel:
                progress_message = await safe_message_edit(
                    progress_message,
                    interaction.channel,
                    content=f"Found {len(search_results)} results for '{query}'. Processing top results...",
                )
            else:
                await interaction.edit_original_response(content=f"Found {len(search_results)} results for '{query}'. Processing top results...")

            max_results_to_process = config.NEWS_MAX_LINKS_TO_PROCESS # Reuse similar config as /news
            num_to_process = min(len(search_results), max_results_to_process)
            fast_runtime = get_llm_runtime("fast")
            fast_client = fast_runtime.client
            fast_provider = fast_runtime.provider
            fast_logit_bias = (
                LOGIT_BIAS_UNWANTED_TOKENS_STR if fast_provider.supports_logit_bias else None
            )

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
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        embed=update_embed_search,
                        content=None,
                    )
                else:
                    await interaction.edit_original_response(embed=update_embed_search, content=None)

                if bot_state_instance and hasattr(bot_state_instance, 'update_last_playwright_usage_time'):
                    await bot_state_instance.update_last_playwright_usage_time()

                scraped_content, _ = await scrape_website(page_url)

                if (
                    not scraped_content
                    or "Failed to scrape" in scraped_content
                    or "Scraping timed out" in scraped_content
                    or "Blocked from fetching URL" in scraped_content
                ):
                    logger.warning(f"Failed to scrape '{page_title}' from {page_url} for search. Reason: {scraped_content}")
                    page_summaries_for_final_synthesis.append(f"Source: {page_title} ({page_url})\nSummary: [Could not retrieve content for summarization]\n\n")
                    continue

                update_embed_search.description = f"Processing result {i+1}/{num_to_process}: Summarizing '{page_title}'..."
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        embed=update_embed_search,
                    )
                else:
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
                    summary_response_search = await create_chat_completion(
                        fast_client,
                        [
                            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                            {"role": "user", "content": summarization_prompt_search}
                        ],
                        model=fast_provider.model,
                        max_tokens=3072,
                        temperature=fast_provider.temperature,
                        logit_bias=fast_logit_bias,
                        use_responses_api=fast_provider.use_responses_api,
                    )
                    page_summary = extract_text(
                        summary_response_search, fast_provider.use_responses_api
                    )
                    if page_summary:
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
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        embed=error_embed_search,
                    )
                else:
                    await interaction.edit_original_response(embed=error_embed_search)
                return

            final_update_embed = discord.Embed(
                title=f"Search Results for: {query}",
                description="All relevant pages processed. Generating final search summary...",
                color=config.EMBED_COLOR["incomplete"]
            )
            if interaction.channel:
                progress_message = await safe_message_edit(
                    progress_message,
                    interaction.channel,
                    embed=final_update_embed,
                )
            else:
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
            synthesized_summary, raw_snippets = await retrieve_rag_context_with_progress(
                llm_client=llm_client_instance,
                query=rag_query_for_search_summary,
                interaction=interaction,
            )

            prompt_nodes = await _build_initial_prompt_messages(
                user_query_content=final_synthesis_prompt_content,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_summary,
                raw_rag_snippets=raw_snippets
            )
            # stream_llm_response_to_interaction will send new followups for the summary
            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, prompt_nodes,
                title=f"Summary for Search: {query}",
                force_new_followup_flow=True,
                synthesized_rag_context_for_display=synthesized_summary,
                bot_user_id=bot_instance.user.id,
                retrieved_snippets=raw_snippets
            )
        except Exception as e:
            logger.error(f"Error in search_slash_command for query '{query}': {e}", exc_info=True)
            if interaction.channel:
                progress_message = await safe_message_edit(
                    progress_message,
                    interaction.channel,
                    content=f"Yikes, my search circuits are fuzzy! Failed to search for '{query}'. Error: {str(e)[:1000]}",
                    embed=None,
                )
            else:
                await interaction.edit_original_response(content=f"Yikes, my search circuits are fuzzy! Failed to search for '{query}'. Error: {str(e)[:1000]}", embed=None)

    @bot_instance.tree.command(name="moltbook_status", description="Checks Moltbook claim status and account details.")
    async def moltbook_status_slash_command(interaction: discord.Interaction):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured yet. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in your .env file.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)
        try:
            status_payload = await moltbook_get_status()
            profile_payload = await moltbook_get_profile()
            status = status_payload.get("status") or (status_payload.get("data") or {}).get("status") or "unknown"
            agent = profile_payload.get("agent") or profile_payload.get("data") or {}
            profile_name = agent.get("name") or config.MOLTBOOK_AGENT_NAME
            karma = agent.get("karma", "unknown")
            claimed = "yes" if agent.get("is_claimed") else "no"
            description = agent.get("description", "")

            embed = discord.Embed(
                title="Moltbook Status",
                description=f"Agent: **{profile_name}**\nClaimed: **{claimed}**\nStatus: **{status}**\nKarma: **{karma}**",
                color=config.EMBED_COLOR["complete"],
            )
            if description:
                embed.add_field(name="Profile", value=description[:512], inline=False)
            await interaction.edit_original_response(embed=embed)
        except MoltbookAPIError as exc:
            logger.error("Moltbook status failed: %s", exc)
            message = f"Moltbook status request failed: {exc}"
            if exc.hint:
                message = f"{message}\nHint: {exc.hint}"
            await interaction.edit_original_response(content=message)

    @bot_instance.tree.command(name="moltbook_feed", description="Fetches Moltbook posts from a feed or submolt.")
    @app_commands.describe(
        sort="Sort order for the feed.",
        limit="Number of posts to fetch (max 50).",
        submolt="Optional submolt name (without m/).",
        personalized="Use your personalized feed (requires follows/subscriptions).",
    )
    @app_commands.choices(
        sort=[
            app_commands.Choice(name="hot", value="hot"),
            app_commands.Choice(name="new", value="new"),
            app_commands.Choice(name="top", value="top"),
            app_commands.Choice(name="rising", value="rising"),
        ]
    )
    async def moltbook_feed_slash_command(
        interaction: discord.Interaction,
        sort: str = "new",
        limit: int = 10,
        submolt: Optional[str] = None,
        personalized: bool = False,
    ):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured yet. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in your .env file.",
                ephemeral=True,
            )
            return

        bounded_limit = max(1, min(limit, 50))
        await interaction.response.defer(ephemeral=False)
        try:
            payload = await moltbook_get_feed(
                sort=sort,
                limit=bounded_limit,
                submolt=submolt,
                personalized=personalized and not submolt,
            )
            posts = payload.get("posts") or payload.get("data") or payload.get("results") or []
            if not posts:
                await interaction.edit_original_response(
                    content="No Moltbook posts found for that request.",
                    embed=None,
                )
                return

            posts = posts[:bounded_limit]
            # Store each feed post in ChromaDB for RAG when drafting replies (fire-and-forget)
            for post in posts:
                pid = post.get("id") or ""
                if not pid:
                    continue
                start_post_processing_task(
                    store_moltbook_feed_post(
                        post_id=pid,
                        title=(post.get("title") or "(untitled)").strip(),
                        content_snippet=(post.get("content") or post.get("url") or ""),
                        submolt=submolt,
                        sort=sort,
                    ),
                )
            # Build chunks by post so we know which post IDs get a button in each message.
            chunk_texts: List[str] = []
            chunk_posts: List[List[Dict[str, Any]]] = []
            current_text: List[str] = []
            current_posts: List[Dict[str, Any]] = []
            for post in posts:
                line = _format_moltbook_post(post)
                if current_text and len("\n\n".join(current_text)) + len(line) + 2 > config.EMBED_MAX_LENGTH:
                    chunk_texts.append("\n\n".join(current_text))
                    chunk_posts.append(current_posts)
                    current_text = []
                    current_posts = []
                current_text.append(line)
                current_posts.append(post)
            if current_text:
                chunk_texts.append("\n\n".join(current_text))
                chunk_posts.append(current_posts)

            footer_bits = []
            if submolt:
                footer_bits.append(f"m/{submolt}")
            if personalized and not submolt:
                footer_bits.append("personalized")
            footer_bits.append(f"sort={sort}")
            footer_text = " • ".join(footer_bits)
            for i, (chunk, posts_in_chunk) in enumerate(zip(chunk_texts, chunk_posts)):
                title = "Moltbook Feed" if i == 0 else f"Moltbook Feed (cont. {i + 1})"
                embed = discord.Embed(
                    title=title,
                    description=chunk,
                    color=config.EMBED_COLOR["complete"],
                )
                embed.set_footer(text=footer_text)
                view = MoltbookFeedView(posts_in_chunk)
                if i == 0:
                    await interaction.edit_original_response(embed=embed, view=view)
                else:
                    await safe_followup_send(interaction, embed=embed, view=view)
        except MoltbookAPIError as exc:
            logger.error("Moltbook feed failed: %s", exc)
            message = f"Moltbook feed request failed: {exc}"
            if exc.hint:
                message = f"{message}\nHint: {exc.hint}"
            await interaction.edit_original_response(content=message)

    @bot_instance.tree.command(name="moltbook_search", description="Semantic search across Moltbook posts and comments.")
    @app_commands.describe(
        query="Search query for Moltbook (natural language, max 500 chars).",
        search_type="What to search (posts, comments, or all).",
        limit="Number of results to fetch (max 50).",
    )
    @app_commands.choices(
        search_type=[
            app_commands.Choice(name="all", value="all"),
            app_commands.Choice(name="posts", value="posts"),
            app_commands.Choice(name="comments", value="comments"),
        ]
    )
    async def moltbook_search_slash_command(
        interaction: discord.Interaction,
        query: str,
        search_type: str = "all",
        limit: int = 10,
    ):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured yet. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in your .env file.",
                ephemeral=True,
            )
            return

        # Skill: search limit default 20, max 50
        bounded_limit = max(1, min(limit, 50))
        await interaction.response.defer(ephemeral=False)
        try:
            payload = await moltbook_search(
                query=query,
                search_type=search_type,
                limit=bounded_limit,
            )
            results = payload.get("results") or payload.get("data") or []
            if not results:
                await interaction.edit_original_response(
                    content="No Moltbook search results found.",
                    embed=None,
                )
                return

            lines: List[str] = []
            for result in results[:bounded_limit]:
                entry_type = result.get("type", "post")
                if entry_type == "comment":
                    post_ref = result.get("post", {})
                    post_title = post_ref.get("title", "unknown post")
                    line = (
                        f"**Comment** on *{post_title}* (post `{result.get('post_id', 'unknown')}`)\n"
                        f"{_format_moltbook_comment(result)}"
                    )
                else:
                    line = _format_moltbook_post(result)
                similarity = result.get("similarity")
                if similarity is not None:
                    line = f"{line}\nSimilarity: {similarity:.2f}"
                lines.append(line)

            description = "\n\n".join(lines)
            if len(description) > config.EMBED_MAX_LENGTH:
                description = description[: config.EMBED_MAX_LENGTH - 3].rstrip() + "..."

            embed = discord.Embed(
                title=f"Moltbook Search: {query}",
                description=description,
                color=config.EMBED_COLOR["complete"],
            )
            embed.set_footer(text=f"type={search_type} • limit={bounded_limit}")
            await interaction.edit_original_response(embed=embed)
        except MoltbookAPIError as exc:
            logger.error("Moltbook search failed: %s", exc)
            message = f"Moltbook search failed: {exc}"
            if exc.hint:
                message = f"{message}\nHint: {exc.hint}"
            if exc.status == 500:
                message = (
                    f"{message}\n*(Server-side error on Moltbook—try again in a moment, "
                    "or use `/moltbook_feed` to browse recent posts.)*"
                )
            await interaction.edit_original_response(content=message)

    @bot_instance.tree.command(
        name="moltbook_post",
        description="Ask Sam to draft a Moltbook post from today's context; then accept or decline.",
    )
    @app_commands.describe(
        topic_guidance=(
            "Optional: a question or topic to steer RAG and the post (e.g. 'recent deep dives on AI safety', "
            "'what we discussed about project X'). Use this to get in-depth posts instead of generic short ones."
        ),
    )
    async def moltbook_post_slash_command(
        interaction: discord.Interaction,
        topic_guidance: Optional[str] = None,
    ):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured yet. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in your .env file.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=False)
        try:
            fast_runtime = get_llm_runtime("fast")
            fast_client = fast_runtime.client if fast_runtime else None
            fast_provider = fast_runtime.provider if fast_runtime else None
            if not fast_client or not fast_provider:
                await interaction.edit_original_response(
                    content="LLM is not configured; cannot draft a post.",
                )
                return

            topic_str = (topic_guidance or "").strip()
            if topic_str:
                rag_query = (
                    f"Topic to focus on for the post: {topic_str}. "
                    "Also include: what has happened recently, relevant conversations, timeline summaries, and activity. "
                    "Use this to draft a substantive, in-depth Moltbook post on that topic."
                )
            else:
                rag_query = (
                    "What has happened today? Recent conversations, timeline summaries, and today's activity. "
                    "Use this to draft a short Moltbook post."
                )
            synthesized_rag_summary, raw_rag_snippets = await retrieve_rag_context_with_progress(
                llm_client=fast_client,
                query=rag_query,
                interaction=interaction,
                channel=interaction.channel,
                initial_status="🔍 Gathering today's context...",
                completion_status="✅ Context ready.",
            )
            sys_node = get_system_prompt()
            messages: List[Dict[str, str]] = [{"role": sys_node.role, "content": sys_node.content}]
            if (config.USER_PROVIDED_CONTEXT or "").strip():
                messages.append({
                    "role": "system",
                    "content": f"User-Set Global Context:\n{config.USER_PROVIDED_CONTEXT.strip()}",
                })
            if synthesized_rag_summary and synthesized_rag_summary.strip():
                messages.append({
                    "role": "system",
                    "content": (
                        "Relevant context from today:\n---\n"
                        + synthesized_rag_summary.strip()
                        + "\n---"
                    ),
                })
            if raw_rag_snippets:
                parts = ["Additional snippets that may be relevant:\n"]
                for i, (snippet_text, source) in enumerate(raw_rag_snippets[:15]):
                    trunc = (snippet_text or "")[:1500]
                    if len(snippet_text or "") > 1500:
                        trunc += " [truncated]"
                    parts.append(f"\n--- Snippet {i + 1} ({source}) ---\n{trunc}\n")
                messages.append({"role": "system", "content": "".join(parts)})
            draft_instruction = (
                "For this turn only: Draft a single Moltbook post based on the context above. "
                "Output exactly in this format, one field per line (CONTENT can be multiple lines):\n"
                "SUBMOLT: <submolt name, e.g. general>\n"
                "TITLE: <one-line title>\n"
                "CONTENT: <body text, or leave blank if only sharing a link>\n"
                "URL: <optional link if it's a link post>\n"
                "Do not add any preamble or explanation; only these lines."
            )
            if topic_str:
                draft_instruction = (
                    "The user asked to focus this post on a specific topic. Focus the post on that topic and go in-depth; "
                    "avoid generic short takes. You may write a longer, substantive post (several paragraphs if the context supports it).\n\n"
                    + draft_instruction
                )
            messages.append({"role": "system", "content": draft_instruction})
            messages.append({"role": "user", "content": "Draft the Moltbook post now."})
            max_tokens_moltbook = getattr(
                config, "MOLTBOOK_POST_MAX_TOKENS", config.MAX_COMPLETION_TOKENS
            )
            response = await create_chat_completion(
                fast_client,
                messages,
                model=fast_provider.model,
                max_tokens=max_tokens_moltbook,
                temperature=fast_provider.temperature,
                use_responses_api=getattr(fast_provider, "use_responses_api", False),
            )
            raw_draft = extract_text(response, getattr(fast_provider, "use_responses_api", False))
            if not (raw_draft and raw_draft.strip()):
                try:
                    await interaction.edit_original_response(
                        content="Sam did not produce a draft. Try again or add more context.",
                    )
                except (discord.NotFound, discord.HTTPException):
                    await interaction.followup.send(
                        content="Sam did not produce a draft. Try again or add more context.",
                        ephemeral=True,
                    )
                return

            parsed = _parse_moltbook_draft_post(raw_draft)
            submolt = (parsed.get("submolt") or "general").strip()
            title = (parsed.get("title") or "").strip()
            content = (parsed.get("content") or "").strip() or None
            url = (parsed.get("url") or "").strip() or None

            display_lines = [
                f"**Submolt:** {submolt}",
                f"**Title:** {title}",
            ]
            if content:
                display_lines.append(f"**Content:**\n{content[:2000]}" + ("…" if len(content) > 2000 else ""))
            if url:
                display_lines.append(f"**URL:** {url}")
            if not title and not content and not url:
                display_lines.append("*(Raw draft)*\n" + raw_draft[:1500])

            embed = discord.Embed(
                title="Draft Moltbook Post",
                description="\n\n".join(display_lines),
                color=config.EMBED_COLOR["complete"],
            )
            embed.set_footer(text="Post to Moltbook or Decline below.")
            view = MoltbookDraftPostView(submolt=submolt, title=title, content=content, url=url)
            # Use followup so we don't depend on the deferred original response (can 404 after RAG progress)
            await interaction.followup.send(embed=embed, view=view)
            try:
                await interaction.edit_original_response(content="Draft ready below.")
            except (discord.NotFound, discord.HTTPException):
                pass
        except Exception as exc:
            logger.exception("Moltbook draft post failed: %s", exc)
            try:
                await interaction.edit_original_response(
                    content=f"Draft failed: {str(exc)[:500]}",
                )
            except (discord.NotFound, discord.HTTPException):
                try:
                    await interaction.followup.send(
                        content=f"Draft failed: {str(exc)[:500]}",
                        ephemeral=True,
                    )
                except discord.HTTPException:
                    pass

    @bot_instance.tree.command(name="moltbook_comment", description="Comment on a Moltbook post.")
    @app_commands.describe(
        post_id="ID of the Moltbook post.",
        content="Comment content.",
        parent_id="Optional parent comment ID for threading.",
    )
    async def moltbook_comment_slash_command(
        interaction: discord.Interaction,
        post_id: str,
        content: str,
        parent_id: Optional[str] = None,
    ):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured yet. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in your .env file.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=False)
        try:
            payload = await moltbook_add_comment(
                post_id=post_id,
                content=content,
                parent_id=parent_id,
            )
            comment = payload.get("comment") or payload.get("data") or payload
            embed = discord.Embed(
                title="Moltbook Comment Posted",
                description=_format_moltbook_comment(comment),
                color=config.EMBED_COLOR["complete"],
            )
            await interaction.edit_original_response(embed=embed)
        except MoltbookAPIError as exc:
            logger.error("Moltbook comment failed: %s", exc)
            message = f"Failed to post Moltbook comment: {exc}"
            if exc.hint:
                message = f"{message}\nHint: {exc.hint}"
            await interaction.edit_original_response(content=message)

    @bot_instance.tree.command(name="moltbook_get", description="Fetch a Moltbook post and its comments.")
    @app_commands.describe(
        post_id="ID of the Moltbook post.",
        include_comments="Include top comments in the response.",
        comment_sort="Sort order for comments.",
    )
    @app_commands.choices(
        comment_sort=[
            app_commands.Choice(name="top", value="top"),
            app_commands.Choice(name="new", value="new"),
            app_commands.Choice(name="controversial", value="controversial"),
        ]
    )
    async def moltbook_get_slash_command(
        interaction: discord.Interaction,
        post_id: str,
        include_comments: bool = True,
        comment_sort: str = "top",
    ):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured yet. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in your .env file.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=False)
        try:
            post_payload = await moltbook_get_post(post_id)
            post = post_payload.get("post") or post_payload.get("data") or post_payload
            # Single-post view: full content, chunked into multiple embeds if needed (Discord max 10 embeds)
            description = _format_moltbook_post(
                post,
                max_content_length=None,
            )
            if include_comments:
                # API returns comments in the get_post response; separate GET /posts/{id}/comments returns 405
                comments = (
                    post_payload.get("comments")
                    or (post_payload.get("data") or {}).get("comments")
                    or []
                )
                if comments:
                    snippets = [
                        _format_moltbook_comment(comment, max_content_length=None)
                        for comment in comments[:5]
                    ]
                    description = f"{description}\n\n**Comments**\n" + "\n\n".join(snippets)

            chunks = chunk_text(description, config.EMBED_MAX_LENGTH)
            if len(chunks) > 10:
                chunks = chunks[:10]
                chunks[-1] = chunks[-1][: config.EMBED_MAX_LENGTH - 20].rstrip() + "\n\n*(truncated)*"
            embeds = []
            for i, chunk in enumerate(chunks):
                title = "Moltbook Post" if i == 0 else f"Moltbook Post (cont. {i + 1})"
                embed = discord.Embed(
                    title=title,
                    description=chunk,
                    color=config.EMBED_COLOR["complete"],
                )
                if i == len(chunks) - 1:
                    embed.set_footer(text=f"Post ID: {post_id} • comments={comment_sort}")
                embeds.append(embed)
            await interaction.edit_original_response(embeds=embeds)
        except MoltbookAPIError as exc:
            logger.error("Moltbook get failed: %s", exc)
            message = f"Failed to fetch Moltbook post: {exc}"
            if exc.hint:
                message = f"{message}\nHint: {exc.hint}"
            await interaction.edit_original_response(content=message)

    # --- Moltbook DMs (heartbeat.md) ---

    @bot_instance.tree.command(name="moltbook_dm_check", description="Check Moltbook DMs: pending requests and unread messages.")
    async def moltbook_dm_check_slash_command(interaction: discord.Interaction):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in .env.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=False)
        try:
            payload = await moltbook_dm_check()
            data = payload.get("data") or payload
            pending = data.get("pending_requests", 0) if isinstance(data, dict) else 0
            unread = data.get("unread_messages", 0) if isinstance(data, dict) else 0
            lines = [
                "**Moltbook DMs**",
                f"Pending requests (need approval): **{pending}**",
                f"Unread messages: **{unread}**",
            ]
            if pending:
                lines.append("\nUse `/moltbook_dm_requests` to list and `/moltbook_dm_approve` to approve.")
            if unread:
                lines.append("Use `/moltbook_dm_conversations` then `/moltbook_dm_read` to read.")
            embed = discord.Embed(
                description="\n".join(lines),
                color=config.EMBED_COLOR["complete"],
            )
            embed.set_footer(text="Moltbook heartbeat")
            await interaction.edit_original_response(embed=embed)
        except MoltbookAPIError as exc:
            logger.error("Moltbook DM check failed: %s", exc)
            message = f"Moltbook DM check failed: {exc}"
            if exc.hint:
                message = f"{message}\nHint: {exc.hint}"
            await interaction.edit_original_response(content=message)

    @bot_instance.tree.command(name="moltbook_dm_requests", description="List pending Moltbook DM requests (need your approval).")
    async def moltbook_dm_requests_slash_command(interaction: discord.Interaction):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in .env.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=False)
        try:
            payload = await moltbook_dm_requests()
            requests_list = payload.get("requests") or payload.get("data") or []
            if isinstance(requests_list, dict):
                requests_list = requests_list.get("requests", requests_list.get("data", [])) or []
            if not requests_list:
                await interaction.edit_original_response(
                    content="No pending DM requests. Use `/moltbook_dm_check` to see summary.",
                    embed=None,
                )
                return
            lines = []
            for i, req in enumerate(requests_list[:15], 1):
                from_val = req.get("from")
                if isinstance(from_val, dict):
                    from_agent = from_val.get("name", "?")
                else:
                    from_agent = req.get("from_agent") or from_val or "?"
                conv_id = req.get("conversation_id") or req.get("id", "?")
                msg_preview = (req.get("message") or req.get("initial_message") or "")[:80]
                if msg_preview and len((req.get("message") or req.get("initial_message") or "")) > 80:
                    msg_preview += "..."
                lines.append(f"{i}. **{from_agent}** — `{conv_id}`\n   \"{msg_preview}\"\n   Approve: `/moltbook_dm_approve` conversation_id:`{conv_id}`")
            embed = discord.Embed(
                title="Pending DM Requests",
                description="\n\n".join(lines),
                color=config.EMBED_COLOR["complete"],
            )
            embed.set_footer(text="Approve with /moltbook_dm_approve")
            await interaction.edit_original_response(embed=embed)
        except MoltbookAPIError as exc:
            logger.error("Moltbook DM requests failed: %s", exc)
            await interaction.edit_original_response(content=f"Moltbook DM requests failed: {exc}")

    @bot_instance.tree.command(name="moltbook_dm_approve", description="Approve a Moltbook DM request (conversation can then be used).")
    @app_commands.describe(conversation_id="ID of the conversation request to approve.")
    async def moltbook_dm_approve_slash_command(interaction: discord.Interaction, conversation_id: str):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in .env.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=False)
        try:
            await moltbook_dm_approve(conversation_id)
            await interaction.edit_original_response(
                content=f"Approved DM request `{conversation_id}`. You can now use `/moltbook_dm_read` and `/moltbook_dm_reply` with this conversation.",
                embed=None,
            )
        except MoltbookAPIError as exc:
            logger.error("Moltbook DM approve failed: %s", exc)
            await interaction.edit_original_response(content=f"Failed to approve: {exc}")

    @bot_instance.tree.command(name="moltbook_dm_conversations", description="List your Moltbook DM conversations.")
    async def moltbook_dm_conversations_slash_command(interaction: discord.Interaction):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in .env.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=False)
        try:
            payload = await moltbook_dm_conversations()
            convs = payload.get("conversations") or payload.get("data") or []
            if isinstance(convs, dict):
                convs = convs.get("conversations", convs.get("data", [])) or []
            if not convs:
                await interaction.edit_original_response(
                    content="No DM conversations yet. Use `/moltbook_dm_start` to request a chat with another molty.",
                    embed=None,
                )
                return
            lines = []
            for i, c in enumerate(convs[:20], 1):
                other_val = c.get("other_agent") or c.get("with")
                other = other_val.get("name", "?") if isinstance(other_val, dict) else (other_val or "?")
                cid = c.get("id") or c.get("conversation_id", "?")
                unread = c.get("unread_count", 0)
                lines.append(f"{i}. **{other}** — `{cid}`" + (f" ({unread} unread)" if unread else ""))
            embed = discord.Embed(
                title="DM Conversations",
                description="\n".join(lines) + "\n\nUse `/moltbook_dm_read` with a conversation_id to read.",
                color=config.EMBED_COLOR["complete"],
            )
            await interaction.edit_original_response(embed=embed)
        except MoltbookAPIError as exc:
            logger.error("Moltbook DM conversations failed: %s", exc)
            await interaction.edit_original_response(content=f"Moltbook DM conversations failed: {exc}")

    @bot_instance.tree.command(name="moltbook_dm_read", description="Read a Moltbook DM conversation (marks as read).")
    @app_commands.describe(conversation_id="ID of the conversation to read.")
    async def moltbook_dm_read_slash_command(interaction: discord.Interaction, conversation_id: str):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in .env.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=False)
        try:
            payload = await moltbook_dm_get_conversation(conversation_id)
            data = payload.get("data") or payload.get("conversation") or payload
            messages = data.get("messages", []) if isinstance(data, dict) else []
            if isinstance(data, dict) and not messages:
                messages = data.get("data", [])
            lines = []
            for m in (messages or [])[-25:]:
                from_val = m.get("from") or m.get("author")
                author = from_val.get("name", "?") if isinstance(from_val, dict) else (from_val or "?")
                body = (m.get("content") or m.get("message") or m.get("text", ""))[:300]
                ts = m.get("created_at") or m.get("timestamp", "")
                lines.append(f"**{author}** ({ts}): {body}")
            if not lines:
                desc = "No messages yet, or conversation not found."
            else:
                desc = "\n\n".join(lines)
            if len(desc) > config.EMBED_MAX_LENGTH:
                desc = desc[: config.EMBED_MAX_LENGTH - 3].rstrip() + "..."
            embed = discord.Embed(
                title=f"DM — {conversation_id[:20]}…",
                description=desc,
                color=config.EMBED_COLOR["complete"],
            )
            embed.set_footer(text=f"Reply: /moltbook_dm_reply conversation_id:{conversation_id} message:...")
            await interaction.edit_original_response(embed=embed)
        except MoltbookAPIError as exc:
            logger.error("Moltbook DM read failed: %s", exc)
            await interaction.edit_original_response(content=f"Moltbook DM read failed: {exc}")

    @bot_instance.tree.command(name="moltbook_dm_reply", description="Reply in a Moltbook DM conversation.")
    @app_commands.describe(
        conversation_id="ID of the conversation.",
        message="Your reply.",
    )
    async def moltbook_dm_reply_slash_command(interaction: discord.Interaction, conversation_id: str, message: str):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in .env.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=False)
        try:
            await moltbook_dm_send(conversation_id, message)
            await interaction.edit_original_response(
                content=f"Replied in `{conversation_id}`: \"{message[:200]}{'…' if len(message) > 200 else ''}\"",
                embed=None,
            )
        except MoltbookAPIError as exc:
            logger.error("Moltbook DM reply failed: %s", exc)
            await interaction.edit_original_response(content=f"Moltbook DM reply failed: {exc}")

    @bot_instance.tree.command(name="moltbook_dm_start", description="Start a new Moltbook DM (request; their owner must approve).")
    @app_commands.describe(
        to_agent="Other molty's name (e.g. CoolBot).",
        message="Your initial message.",
    )
    async def moltbook_dm_start_slash_command(interaction: discord.Interaction, to_agent: str, message: str):
        if not _moltbook_enabled():
            await interaction.response.send_message(
                "Moltbook is not configured. Set MOLTBOOK_AGENT_NAME and MOLTBOOK_API_KEY in .env.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=False)
        try:
            payload = await moltbook_dm_request(to_agent.strip(), message.strip())
            data = payload.get("data") or payload
            conv_id = data.get("conversation_id") or data.get("id", "?") if isinstance(data, dict) else "?"
            await interaction.edit_original_response(
                content=f"DM request sent to **{to_agent}**. They (or their human) must approve before you can chat. Conversation ID: `{conv_id}`",
                embed=None,
            )
        except MoltbookAPIError as exc:
            logger.error("Moltbook DM start failed: %s", exc)
            await interaction.edit_original_response(content=f"Moltbook DM start failed: {exc}")

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
            synthesized_summary, raw_snippets = await retrieve_rag_context_with_progress(
                llm_client=llm_client_instance,
                query=rag_query_for_pol,
                interaction=interaction,
            )

            base_prompt_nodes = await _build_initial_prompt_messages(
                user_query_content=user_query_content,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_summary,
                raw_rag_snippets=raw_snippets
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
                synthesized_rag_context_for_display=synthesized_summary,
                bot_user_id=bot_instance.user.id,
                retrieved_snippets=raw_snippets
            )
        except Exception as e:
            logger.error(f"Error in pol_slash_command for statement '{statement[:50]}...': {e}", exc_info=True)
            await interaction.edit_original_response(content=f"My political satire circuits just blew a fuse! Error: {str(e)[:1000]}", embed=None)

    @bot_instance.tree.command(name="rss", description="Fetches new entries from an RSS feed and summarizes them.")
    @app_commands.describe(
        feed_url="Choose a preset RSS feed URL.",
        feed_url_manual="Or, enter an RSS feed URL manually.",
        limit="Number of new entries to fetch (max 50)."
    )
    @app_commands.choices(
        feed_url=[
            app_commands.Choice(name=name, value=url)
            for name, url in DEFAULT_RSS_FEEDS
        ]
    )
    async def rss_slash_command(
        interaction: discord.Interaction,
        feed_url: Optional[str] = None,  # Made optional
        feed_url_manual: Optional[str] = None, # New manual field
        limit: app_commands.Range[int, 1, 50] = 20,
    ):
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("rss_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot fetch RSS.", ephemeral=True)
            return

        # Determine the final feed URL to use
        final_feed_url = feed_url_manual if feed_url_manual else feed_url

        if not final_feed_url:
            await interaction.response.send_message(
                "Please either select a preset RSS feed or manually enter a URL.",
                ephemeral=True
            )
            return

        logger.info(f"RSS command initiated by {interaction.user.name} for {final_feed_url}, limit {limit}.")
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        scrape_lock = bot_state_instance.get_scrape_lock()
        queue_notice = scrape_lock.locked()
        acquired_lock = False
        if queue_notice:
            await interaction.response.send_message(
                "Waiting for other scraping tasks to finish before processing your RSS request...",
                ephemeral=True,
            )
            await scrape_lock.acquire()
            acquired_lock = True
        else:
            await scrape_lock.acquire()
            acquired_lock = True
            await interaction.response.defer(ephemeral=False)

        try:
            await process_rss_feed(interaction, final_feed_url, limit)
        except Exception as e:
            logger.error(f"Error in rss_slash_command for {final_feed_url}: {e}", exc_info=True)
            await safe_followup_send(
                interaction,
                content=f"Failed to process RSS feed. Error: {str(e)[:500]}",
                error_hint=" while processing RSS feed",
            )
        finally:
            if acquired_lock:
                scrape_lock.release()

    @bot_instance.tree.command(name="allrss", description="Fetches new entries from all default RSS feeds until up to date.")
    @app_commands.describe(
        limit="Number of new entries per feed to fetch at a time (max 50)."
    )
    async def allrss_slash_command(
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 50] = 20,
    ) -> None:
        if not all([llm_client_instance, bot_state_instance, bot_instance, bot_instance.user, interaction.channel]):
            logger.error("allrss_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot fetch RSS.", ephemeral=True)
            return

        scrape_lock = bot_state_instance.get_scrape_lock()
        if scrape_lock.locked():
            await interaction.response.send_message(
                "Another scraping task is already running. Please wait for it to finish before starting a new one.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=False)
        progress_message: Optional[discord.Message] = None

        async def refresh_progress(text: str, error_hint: str = " for allrss status") -> None:
            nonlocal progress_message
            if progress_message:
                try:
                    await progress_message.delete()
                except discord.HTTPException:
                    pass
            progress_message = await safe_followup_send(
                interaction,
                content=text,
                ephemeral=True,
                wait=True,
                error_hint=error_hint,
            )

        await refresh_progress(
            f"Starting to process all {len(DEFAULT_RSS_FEEDS)} default RSS feeds. "
            "New article summaries will be posted below as they are found. This may take a while."
        )

        total_new_articles_found = 0
        channel = interaction.channel

        async with scrape_lock:
            await bot_state_instance.set_active_task(interaction.channel_id, asyncio.current_task())
            try:
                for name, feed_url in DEFAULT_RSS_FEEDS:
                    await asyncio.sleep(1) # Small delay between feeds

                    try:
                        await refresh_progress(f"Checking feed: **{name}** (`{feed_url}`)...")
                        feed_had_new_entries = False

                        while True: # Loop to process all new entries in chunks for a single feed
                            seen = load_seen_entries()
                            seen_ids = set(seen.get(feed_url, []))

                            entries = await fetch_rss_entries(feed_url)
                            new_entries = [e for e in entries if e.get("guid") not in seen_ids]

                            if not new_entries:
                                if not feed_had_new_entries:
                                    await refresh_progress(f"No new entries found for **{name}**.")
                                    await asyncio.sleep(5)
                                break # Exit the while loop, go to the next feed

                            feed_had_new_entries = True

                            # Take the next chunk of entries to process
                            entries_to_process = new_entries[:limit]

                            await refresh_progress(
                                f"Found {len(new_entries)} new entries for **{name}**. "
                                f"Processing a chunk of {len(entries_to_process)}..."
                            )

                            fast_runtime = get_llm_runtime("fast")
                            fast_client = fast_runtime.client
                            fast_provider = fast_runtime.provider
                            fast_logit_bias = (
                                LOGIT_BIAS_UNWANTED_TOKENS_STR
                                if fast_provider.supports_logit_bias
                                else None
                            )

                            summaries: List[str] = []
                            processed_guids_this_chunk = []

                            for idx, ent in enumerate(entries_to_process, 1):
                                title = ent.get("title") or "Untitled"
                                link = ent.get("link") or ""
                                guid = ent.get("guid") or link

                                pub_date_dt: Optional[datetime] = ent.get("pubDate_dt")
                                if not pub_date_dt:
                                    pub_date_str = ent.get("pubDate")
                                    if pub_date_str:
                                        try:
                                            pub_date_dt = parsedate_to_datetime(pub_date_str)
                                            if pub_date_dt.tzinfo is None:
                                                pub_date_dt = pub_date_dt.replace(tzinfo=timezone.utc)
                                        except Exception:
                                            pub_date_dt = None
                                pub_date = (
                                    pub_date_dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")
                                    if pub_date_dt
                                    else (ent.get("pubDate") or "")
                                )

                                await refresh_progress(
                                    f"Processing **{name}** ({idx}/{len(entries_to_process)} of chunk): Scraping *{title}*..."
                                )

                                scraped_text, _ = await scrape_website(link)
                                if (
                                    not scraped_text
                                    or "Failed to scrape" in scraped_text
                                    or "Scraping timed out" in scraped_text
                                    or "Blocked from fetching URL" in scraped_text
                                ):
                                    summary_line = f"**{title}**\n{pub_date}\n{link}\nCould not scrape article\n"
                                    summaries.append(summary_line)
                                    processed_guids_this_chunk.append(guid)
                                    continue

                                await refresh_progress(
                                    f"Processing **{name}** ({idx}/{len(entries_to_process)} of chunk): Summarizing *{title}*..."
                                )

                                prompt = (
                                    "[It is currently 2025 and Donald Trump is the current president. Biden IS NOT THE CURRENT PRESIDENT!] (Just an FYI. Maybe unrelated to context and omitted). "
                                    "Do not use em dashes. Summarize the following article in 2-4 sentences. "
                                    "Focus on key facts. Present in a casual, blunt, honest and slightly profane tone. Do NOT start with 'So, ' or end with 'Basically, '. Do not state things like 'This article describes', etc. Present is as a person would if they were talking to you about the article.\n\n"
                                    f"Title: {title}\nURL: {link}\n\n{scraped_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
                                )

                                try:
                                    response = await create_chat_completion(
                                        fast_client,
                                        [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                                        model=fast_provider.model,
                                        max_tokens=3072,
                                        temperature=fast_provider.temperature,
                                        logit_bias=fast_logit_bias,
                                        use_responses_api=fast_provider.use_responses_api,
                                    )
                                    summary = extract_text(response, fast_provider.use_responses_api)
                                    if summary and summary != "[LLM summarization failed]":
                                        await store_rss_summary(
                                            feed_url=feed_url, article_url=link, title=title,
                                            summary_text=summary, timestamp=datetime.now(),
                                        )
                                except Exception as e_summ:
                                    logger.error(f"LLM summarization failed for {link}: {e_summ}")
                                    summary = "[LLM summarization failed]"

                                summary_line = f"**{title}**\n{pub_date}\n{link}\n{summary}\n"
                                summaries.append(summary_line)
                                processed_guids_this_chunk.append(guid)

                                # Per-article: embed (title, time, summary) then link so Discord shows link preview
                                time_str = format_article_time(pub_date_dt)
                                desc = (time_str + "\n\n" + summary) if time_str else summary
                                if len(desc) > config.EMBED_MAX_LENGTH:
                                    desc = desc[: config.EMBED_MAX_LENGTH - 3] + "..."
                                title_display = (title[: 253] + "...") if len(title) > 256 else title
                                article_embed = discord.Embed(
                                    title=title_display,
                                    description=desc,
                                    color=config.EMBED_COLOR["complete"],
                                )
                                await channel.send(embed=article_embed)
                                await channel.send(content=link)

                            # After processing all entries for the current chunk
                            if summaries:
                                total_new_articles_found += len(summaries)
                                combined = "\n\n".join(summaries)

                                # TTS for the feed's summaries
                                await send_tts_audio(
                                    interaction,
                                    combined,
                                    base_filename=f"allrss_{interaction.id}_{name.replace(' ', '_')}",
                                    bot_state=bot_state_instance,
                                )

                                # History and RAG logging
                                user_msg = MsgNode(
                                    "user",
                                    f"/allrss feed '{name}' (limit {limit})",
                                    name=str(interaction.user.id),
                                )
                                assistant_msg = MsgNode(
                                    "assistant", combined, name=str(bot_instance.user.id)
                                )
                                await bot_state_instance.append_history(
                                    interaction.channel_id, user_msg, config.MAX_MESSAGE_HISTORY
                                )
                                await bot_state_instance.append_history(
                                    interaction.channel_id,
                                    assistant_msg,
                                    config.MAX_MESSAGE_HISTORY,
                                )
                                progress_note = None
                                try:
                                    progress_note = await safe_followup_send(
                                        interaction,
                                        content=f"\U0001F501 Post-processing summaries from {name}...",
                                        ephemeral=True,
                                        error_hint=" for allrss post-processing",
                                    )
                                except discord.HTTPException:
                                    progress_note = None

                                start_post_processing_task(
                                    ingest_conversation_to_chromadb(
                                        llm_client_instance,
                                        interaction.channel_id,
                                        interaction.user.id,
                                        [user_msg, assistant_msg],
                                        None,
                                    ),
                                    progress_message=progress_note,
                                )

                                # Optionally trigger a follow-up "podcast that shit" on this chunk
                                try:
                                    if await bot_state_instance.is_podcast_after_rss_enabled(interaction.channel_id):
                                        podcast_user_query2 = "Podcast that shit"
                                        podcast_user_msg_node2 = MsgNode("user", podcast_user_query2, name=str(interaction.user.id))
                                        podcast_prompt_nodes2 = await _build_initial_prompt_messages(
                                            user_query_content=podcast_user_query2,
                                            channel_id=interaction.channel_id,
                                            bot_state=bot_state_instance,
                                            user_id=str(interaction.user.id),
                                        )
                                        await stream_llm_response_to_interaction(
                                            interaction,
                                            llm_client_instance,
                                            bot_state_instance,
                                            podcast_user_msg_node2,
                                            podcast_prompt_nodes2,
                                            title="Podcast: The Current Conversation",
                                            force_new_followup_flow=True,
                                        )
                                except Exception as e_auto_pod:
                                    logger.error(f"Auto-podcast after /allrss chunk failed: {e_auto_pod}", exc_info=True)

                                # Save seen entries for this feed
                                seen.setdefault(feed_url, []).extend(processed_guids_this_chunk)
                                save_seen_entries(seen)

                                # Keep the status message as the latest message after chunk output
                                # Editing does not bump message order, so delete and re-send an updated status
                                try:
                                    remaining_after_chunk = max(0, len(new_entries) - len(entries_to_process))
                                    status_text = (
                                        f"✅ Sent {len(entries_to_process)} summary(ies) from **{name}**. "
                                        f"Remaining in this feed: {remaining_after_chunk}. "
                                        f"Total new articles so far: {total_new_articles_found}."
                                    )
                                    await refresh_progress(status_text)
                                except Exception as bump_err:
                                    logger.warning(f"Failed to refresh status message for {name}: {bump_err}")

                    except Exception as e_feed:
                        logger.error(f"Failed to process feed '{name}' ({feed_url}): {e_feed}", exc_info=True)
                        try:
                            await refresh_progress(f"An error occurred while processing **{name}**. Skipping.")
                        except Exception:
                            pass
                        continue

                # After all feeds are processed
                final_message = (
                    f"Finished processing all {len(DEFAULT_RSS_FEEDS)} RSS feeds. "
                    f"Found a total of {total_new_articles_found} new articles."
                )
                await refresh_progress(final_message)
                await channel.send(final_message)

            except asyncio.CancelledError:
                await channel.send("`/allrss` command cancelled.")
                return
            except Exception as e:
                logger.error(f"A critical error occurred in the main /allrss loop: {e}", exc_info=True)
                await channel.send(f"The `/allrss` command encountered a critical error and had to stop: {str(e)[:500]}")
            finally:
                await bot_state_instance.clear_active_task(interaction.channel_id)

    @bot_instance.tree.command(name="gettweets", description="Fetches and summarizes recent tweets from a user.")
    @app_commands.describe(
        username="The X/Twitter username (without @).",
        preset_user="Choose a preset account instead of typing one.",
        limit="Number of tweets to fetch (max 200)."
    )
    @app_commands.choices(
        preset_user=[app_commands.Choice(name=u, value=u) for u in DEFAULT_TWITTER_USERS]
    )
    async def gettweets_slash_command(
        interaction: discord.Interaction,
        username: str = "",
        preset_user: str = "",
        limit: app_commands.Range[int, 1, 200] = 50,
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

        scrape_lock = bot_state_instance.get_scrape_lock()
        queue_notice = scrape_lock.locked()
        progress_message: discord.Message
        acquired_lock = False
        if queue_notice:
            await interaction.response.send_message(
                "Waiting for other scraping tasks to finish before fetching tweets...",
                ephemeral=True,
            )
            try:
                await asyncio.wait_for(scrape_lock.acquire(), timeout=config.SCRAPE_LOCK_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout acquiring scrape_lock for /gettweets @{user_to_fetch.lstrip('@')}. Another task may be holding it too long."
                )
                await safe_followup_send(
                    interaction,
                    content=f"Could not acquire scrape lock for @{user_to_fetch.lstrip('@')} at this time. Please try again shortly.",
                    ephemeral=True,
                    error_hint=" acquiring scrape lock for gettweets",
                )
                return
            acquired_lock = True
            progress_message = await safe_followup_send(
                interaction,
                content=f"Scraping tweets for @{user_to_fetch.lstrip('@')} (up to {limit})...",
                error_hint=" starting gettweets",
            )
        else:
            await scrape_lock.acquire()
            acquired_lock = True
            await interaction.response.defer(ephemeral=False)
            clean_username_for_initial_message = user_to_fetch.lstrip('@')
            progress_message = await interaction.edit_original_response(
                content=f"Starting to scrape tweets for @{clean_username_for_initial_message} (up to {limit})..."
            )

        progress_update_count = [0]  # Use list to allow modification in nested function
        
        async def send_progress(message: str) -> None:
            nonlocal progress_message
            if not interaction.channel:
                logger.error("Cannot send progress update for gettweets: interaction.channel is None")
                return
            try:
                # Just edit the existing message to avoid flashing
                progress_message = await safe_message_edit(
                    progress_message,
                    interaction.channel,
                    content=message,
                )
            except Exception as e_unexp:
                logger.error(f"Unexpected error in send_progress for gettweets: {e_unexp}", exc_info=True)

        try:
            clean_username = user_to_fetch.lstrip('@')

            if hasattr(bot_state_instance, 'update_last_playwright_usage_time'): # Check attribute existence
                await bot_state_instance.update_last_playwright_usage_time()
                logger.debug(
                    f"Updated last_playwright_usage_time via bot_state_instance for /gettweets @{clean_username}"
                )

            all_seen_tweet_ids_cache = load_seen_tweet_ids()
            user_seen_tweet_ids = all_seen_tweet_ids_cache.get(clean_username, set())

            # The scrape_latest_tweets function needs to return something with an ID.
            # Let's assume it returns a list of dicts, and each dict has a 'tweet_id' or 'tweet_url' key.
            # For this example, I'll assume 'tweet_url' is unique enough to serve as an ID.
            # If `scrape_latest_tweets` is not returning IDs, it will need to be modified.
            # For now, I'll proceed assuming 'tweet_url' is the unique identifier.

            # Provide seen-checker so scraping can stop early after 4+ seen in a row
            def _seen_checker2(td: TweetData) -> bool:
                return bool(td.tweet_url) and td.tweet_url in user_seen_tweet_ids

            fetched_tweets_data = await scrape_latest_tweets(
                clean_username,
                limit=limit,
                progress_callback=send_progress,
                seen_checker=_seen_checker2,
                stop_after_seen_consecutive=3,
            )

            if not fetched_tweets_data:
                final_content_message = (
                    f"Finished scraping for @{clean_username}. No tweets found or profile might be private/inaccessible."
                )
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        content=final_content_message,
                        embed=None,
                    )
                else:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel or progress_message.channel,
                        content=final_content_message,
                        embed=None,
                    )
                return

            new_tweets_to_process = []
            processed_tweet_ids_current_run = set()

            for tweet_data in fetched_tweets_data:
                # IMPORTANT: Adjust 'tweet_url' if the actual unique identifier is different
                # (e.g., 'id_str', 'tweet_id', or a combination of fields).
                # For robustness, a dedicated tweet ID from the platform is best if available.
                if not tweet_data.tweet_url:
                    logger.warning(f"Tweet from @{clean_username} missing 'tweet_url' or suitable ID. Skipping. Data: {tweet_data}")
                    continue

                if tweet_data.tweet_url not in user_seen_tweet_ids:
                    new_tweets_to_process.append(tweet_data)
                    processed_tweet_ids_current_run.add(tweet_data.tweet_url)
                else:
                    logger.info(f"Skipping already seen tweet for @{clean_username}: {tweet_data.tweet_url}")

            if not new_tweets_to_process:
                final_content_message = f"No new tweets found for @{clean_username} since last check."
                try:
                    await progress_message.delete()
                except Exception:
                    if interaction.channel:
                        await safe_message_edit(
                            progress_message,
                            interaction.channel,
                            content="",
                            embed=None,
                        )
                # Still save, in case the cache file didn't exist for this user yet.
                all_seen_tweet_ids_cache[clean_username] = user_seen_tweet_ids.union(processed_tweet_ids_current_run)
                save_seen_tweet_ids(all_seen_tweet_ids_cache)
                await safe_followup_send(
                    interaction,
                    content=final_content_message,
                    ephemeral=True,
                    error_hint=" notifying no new tweets",
                )
                return

            # Count total images across all tweets for global progress tracking
            total_images_global = sum(len(t.image_urls) for t in new_tweets_to_process if t.image_urls)
            current_image_num = 0
            
            tweet_texts_for_display = []
            for t_data in new_tweets_to_process: # Iterate over new_tweets_to_process
                display_ts = t_data.timestamp
                try:
                    dt_obj = datetime.fromisoformat(t_data.timestamp.replace("Z", "+00:00")) if t_data.timestamp else None
                    display_ts = dt_obj.strftime("%Y-%m-%d %H:%M UTC") if dt_obj else t_data.timestamp
                except ValueError: pass

                author_display = t_data.username or clean_username
                content_display = discord.utils.escape_markdown(t_data.content or 'N/A')
                tweet_url_display = (t_data.tweet_url or '').replace('/analytics', '')

                header = f"[{display_ts}] @{author_display}"
                if t_data.is_repost and t_data.reposted_by:
                    header = f"[{display_ts}] @{t_data.reposted_by} reposted @{author_display}"

                image_description_text = ""
                if t_data.image_urls:
                    for i, image_url in enumerate(t_data.image_urls):
                        current_image_num += 1
                        alt_text = t_data.alt_texts[i] if i < len(t_data.alt_texts) else None
                        if alt_text:
                            image_description_text += f'\n*Image Alt Text: "{alt_text}"*'
                        await send_progress(f"Describing image {current_image_num}/{total_images_global} in tweet from @{author_display}...")
                        description = await describe_image(image_url)
                        if description:
                            image_description_text += f'\n*Image Description: "{description}"*'

                link_text = f" ([Link]({tweet_url_display}))" if tweet_url_display else ""
                tweet_texts_for_display.append(f"**{header}**: {content_display}{image_description_text}{link_text}")

            raw_tweets_display_str = "\n\n".join(tweet_texts_for_display)
            if not raw_tweets_display_str: raw_tweets_display_str = "No new tweet content could be formatted for display."

            await send_progress(f"Formatting {len(new_tweets_to_process)} new tweets for display...")

            # Update and save the cache with newly processed tweet IDs
            all_seen_tweet_ids_cache[clean_username] = user_seen_tweet_ids.union(processed_tweet_ids_current_run)
            save_seen_tweet_ids(all_seen_tweet_ids_cache)
            logger.info(f"Updated and saved seen tweet IDs for @{clean_username}. Total seen: {len(all_seen_tweet_ids_cache[clean_username])}")

            # Store new tweets in ChromaDB
            if new_tweets_to_process and rcm.tweets_collection:
                tweet_docs_to_add = []
                tweet_metadatas_to_add = []
                tweet_ids_to_add = []
                seen_doc_ids: set[str] = set()
                for t_data in new_tweets_to_process:
                    if not t_data.tweet_url:
                        logger.warning(f"Skipping tweet storage for @{clean_username} due to missing 'tweet_url' or ID. Data: {t_data}")
                        continue

                    # Use the tweet URL as the document ID if it's guaranteed unique, otherwise generate one
                    # For Chroma, IDs should be unique strings.
                    tweet_id_val = str(t_data.id or "")
                    if not tweet_id_val:
                        tweet_id_val = t_data.tweet_url.split("?")[0].split("/")[-1]
                    doc_id = f"tweet_{clean_username}_{tweet_id_val}"

                    # The document itself will be the content of the tweet
                    document_content = t_data.content or ''
                    if not document_content.strip(): # Don't store empty tweets
                        logger.info(f"Skipping empty tweet from @{clean_username}, ID: {doc_id}")
                        continue

                    metadata = {
                        "username": clean_username,
                        "tweet_url": t_data.tweet_url,
                        "timestamp": t_data.timestamp,
                        "is_repost": t_data.is_repost,
                        "source_command": "/gettweets",
                        "raw_data_preview": str(t_data)[:200],  # Store a snippet for quick reference
                    }
                    if t_data.reposted_by:
                        metadata["reposted_by"] = t_data.reposted_by
                    if doc_id in seen_doc_ids:
                        logger.debug(f"Duplicate tweet doc_id detected in batch: {doc_id}")
                        continue

                    tweet_docs_to_add.append(document_content)
                    tweet_metadatas_to_add.append(metadata)
                    tweet_ids_to_add.append(doc_id)
                    seen_doc_ids.add(doc_id)

                if tweet_ids_to_add:
                    try:
                        rcm.tweets_collection.add(
                            documents=tweet_docs_to_add,
                            metadatas=tweet_metadatas_to_add,
                            ids=tweet_ids_to_add
                        )
                        logger.info(
                            f"Successfully stored {len(tweet_ids_to_add)} new tweets from @{clean_username} in ChromaDB."
                        )
                    except Exception as e_add_tweet:
                        logger.error(
                            f"Failed to store tweets for @{clean_username} in ChromaDB: {e_add_tweet}",
                            exc_info=True,
                        )
            elif not rcm.tweets_collection:
                logger.warning(
                    f"tweets_collection is not available. Skipping storage of tweets for @{clean_username}."
                )


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
                    if interaction.channel:
                        progress_message = await safe_message_edit(
                            progress_message,
                            interaction.channel,
                            content=None,
                            embed=embed,
                        )
                    else:
                        progress_message = await safe_message_edit(
                            progress_message,
                            interaction.channel or progress_message.channel,
                            content=None,
                            embed=embed,
                        )
                else:
                    await safe_followup_send(interaction, embed=embed)

            user_query_content_for_summary = (
                f"Please analyze and summarize the main themes, topics discussed, and overall sentiment "
                f"from @{clean_username}'s recent tweets provided below. Extract key points and present a detailed overview of this snapshot in time and present to the user in a casual, yet informed manner."
                f"Do not just re-list the tweets.\n\nRecent Tweets:\n{raw_tweets_display_str[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
            )
            user_msg_node = MsgNode("user", user_query_content_for_summary, name=str(interaction.user.id))

            # Build a minimal prompt using only the scraped tweets as context (no RAG, no prior channel history)
            prompt_nodes_summary = await _build_initial_prompt_messages(
                user_query_content=user_query_content_for_summary,
                channel_id=None,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
            )
            # Insert temporal system context to avoid outdated references
            insert_idx_sum2 = 0
            for idx, node in enumerate(prompt_nodes_summary):
                if node.role != "system":
                    insert_idx_sum2 = idx
                    break
                insert_idx_sum2 = idx + 1
            final_prompt_nodes_summary2 = (
                prompt_nodes_summary[:insert_idx_sum2]
                + [MsgNode("system", TEMPORAL_SYSTEM_CONTEXT)]
                + prompt_nodes_summary[insert_idx_sum2:]
            )

            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, final_prompt_nodes_summary2,
                title=f"Tweet Summary for @{clean_username}",
                force_new_followup_flow=True,
                bot_user_id=bot_instance.user.id,
            )
            
            # Delete and recreate progress message to move to bottom after results are shown
            if progress_message:
                try:
                    await progress_message.delete()
                except (discord.NotFound, discord.HTTPException):
                    pass  # Message already deleted or webhook expired
                # Recreate the progress message at the bottom
                progress_message = await safe_followup_send(
                    interaction,
                    content="✅ Tweet processing complete!",
                )
        except Exception as e:
            logger.error(
                f"Error in gettweets_slash_command for @{user_to_fetch}: {e}", exc_info=True
            )
            error_content = (
                f"My tweet-fetching antenna is bent! Failed for @{user_to_fetch}. Error: {str(e)[:500]}"
            )
            try:
                if interaction.response.is_done():
                    await safe_followup_send(
                        interaction,
                        content=error_content,
                        ephemeral=True,
                        error_hint=" sending gettweets error",
                    )
                else:
                    if interaction.channel:
                        progress_message = await safe_message_edit(
                            progress_message,
                            interaction.channel,
                            content=error_content,
                            embed=None,
                        )
                    else:
                        progress_message = await safe_message_edit(
                            progress_message,
                            interaction.channel or progress_message.channel,
                            content=error_content,
                            embed=None,
                        )
            except discord.HTTPException:
                logger.warning(
                    f"Could not send final error message for gettweets @{user_to_fetch} to user (HTTPException)."
                )
        finally:
            if acquired_lock: # Only release if it was acquired
                scrape_lock.release()
                logger.debug(f"Scrape lock released for /gettweets @{user_to_fetch.lstrip('@')}")


    @bot_instance.tree.command(name="homefeed", description="Fetches and summarizes tweets from your home timeline.")
    @app_commands.describe(
        limit="Number of tweets to fetch (max 200)."
    )
    async def homefeed_slash_command(
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 200] = 30,
    ):
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("homefeed_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot get home timeline tweets.", ephemeral=True)
            return

        logger.info(
            f"Homefeed command initiated by {interaction.user.name}, limit {limit}."
        )
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        scrape_lock = bot_state_instance.get_scrape_lock()
        queue_notice = scrape_lock.locked()
        progress_message: discord.Message
        acquired_lock = False
        if queue_notice:
            await interaction.response.send_message(
                "Waiting for other scraping tasks to finish before fetching tweets...",
                ephemeral=True,
            )
            try:
                await asyncio.wait_for(scrape_lock.acquire(), timeout=config.SCRAPE_LOCK_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout acquiring scrape_lock for /homefeed. Another task may be holding it too long."
                )
                await safe_followup_send(
                    interaction,
                    content="Could not acquire scrape lock for home timeline at this time. Please try again shortly.",
                    ephemeral=True,
                    error_hint=" acquiring scrape lock for homefeed",
                )
                return
            acquired_lock = True
            progress_message = await safe_followup_send(
                interaction,
                content=f"Scraping home timeline tweets (up to {limit})...",
                error_hint=" starting homefeed",
            )
        else:
            await scrape_lock.acquire()
            acquired_lock = True
            await interaction.response.defer(ephemeral=False)
            progress_message = await interaction.edit_original_response(
                content=f"Starting to scrape home timeline tweets (up to {limit})..."
            )

        progress_update_count = [0]  # Use list to allow modification in nested function
        
        async def send_progress(message: str) -> None:
            nonlocal progress_message
            if not interaction.channel:
                logger.error("Cannot send progress update for homefeed: interaction.channel is None")
                return
            try:
                # Just edit the existing message to avoid flashing
                progress_message = await safe_message_edit(
                    progress_message,
                    interaction.channel,
                    content=message,
                )
            except Exception as e_unexp:
                logger.error(f"Unexpected error in send_progress for homefeed: {e_unexp}", exc_info=True)

        try:
            if hasattr(bot_state_instance, 'update_last_playwright_usage_time'):
                await bot_state_instance.update_last_playwright_usage_time()
                logger.debug("Updated last_playwright_usage_time via bot_state_instance for /homefeed")

            all_seen_tweet_ids_cache = load_seen_tweet_ids()
            home_key = "__home__"
            user_seen_tweet_ids = all_seen_tweet_ids_cache.get(home_key, set())

            fetched_tweets_data = await scrape_home_timeline(limit=limit, progress_callback=send_progress)

            if not fetched_tweets_data:
                final_content_message = "Finished scraping home timeline. No tweets found or timeline inaccessible."
                if interaction.channel:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel,
                        content=final_content_message,
                        embed=None,
                    )
                else:
                    progress_message = await safe_message_edit(
                        progress_message,
                        interaction.channel or progress_message.channel,
                        content=final_content_message,
                        embed=None,
                    )
                return

            new_tweets_to_process = []
            processed_tweet_ids_current_run = set()

            for tweet_data in fetched_tweets_data:
                if not tweet_data.tweet_url:
                    logger.warning(f"Tweet from home timeline missing 'tweet_url' or suitable ID. Skipping. Data: {tweet_data}")
                    continue

                if tweet_data.tweet_url not in user_seen_tweet_ids:
                    new_tweets_to_process.append(tweet_data)
                    processed_tweet_ids_current_run.add(tweet_data.tweet_url)
                else:
                    logger.info(f"Skipping already seen home tweet: {tweet_data.tweet_url}")

            if not new_tweets_to_process:
                final_content_message = "No new tweets found in the home timeline since last check."
                try:
                    await progress_message.delete()
                except Exception:
                    if interaction.channel:
                        await safe_message_edit(
                            progress_message,
                            interaction.channel,
                            content="",
                            embed=None,
                        )
                all_seen_tweet_ids_cache[home_key] = user_seen_tweet_ids.union(processed_tweet_ids_current_run)
                save_seen_tweet_ids(all_seen_tweet_ids_cache)
                await safe_followup_send(
                    interaction,
                    content=final_content_message,
                    ephemeral=True,
                    error_hint=" notifying no new home tweets",
                )
                return

            # Count total images across all tweets for global progress tracking
            total_images_global = sum(len(t.image_urls) for t in new_tweets_to_process if t.image_urls)
            current_image_num = 0
            
            tweet_texts_for_display = []
            for t_data in new_tweets_to_process:
                display_ts = t_data.timestamp
                try:
                    dt_obj = datetime.fromisoformat(t_data.timestamp.replace("Z", "+00:00")) if t_data.timestamp else None
                    display_ts = dt_obj.strftime("%Y-%m-%d %H:%M UTC") if dt_obj else t_data.timestamp
                except ValueError:
                    pass

                author_display = t_data.username or 'unknown'
                content_display = discord.utils.escape_markdown(t_data.content or 'N/A')
                tweet_url_display = (t_data.tweet_url or "").replace("/analytics", "")

                header = f"[{display_ts}] @{author_display}"
                if t_data.is_repost and t_data.reposted_by:
                    header = f"[{display_ts}] @{t_data.reposted_by} reposted @{author_display}"

                image_description_text = ""
                if t_data.image_urls:
                    for i, image_url in enumerate(t_data.image_urls):
                        current_image_num += 1
                        alt_text = t_data.alt_texts[i] if i < len(t_data.alt_texts) else None
                        if alt_text:
                            image_description_text += f'\n*Image Alt Text: "{alt_text}"*'
                        await send_progress(f"Describing image {current_image_num}/{total_images_global} in tweet from @{author_display}...")
                        description = await describe_image(image_url)
                        if description:
                            image_description_text += f'\n*Image Description: "{description}"*'

                link_text = f" ([Link]({tweet_url_display}))" if tweet_url_display else ""
                tweet_texts_for_display.append(f"**{header}**: {content_display}{image_description_text}{link_text}")

            raw_tweets_display_str = "\n\n".join(tweet_texts_for_display)
            if not raw_tweets_display_str:
                raw_tweets_display_str = "No new tweet content could be formatted for display."

            await send_progress(f"Formatting {len(new_tweets_to_process)} new tweets for display...")

            all_seen_tweet_ids_cache[home_key] = user_seen_tweet_ids.union(processed_tweet_ids_current_run)
            save_seen_tweet_ids(all_seen_tweet_ids_cache)
            logger.info(f"Updated and saved seen tweet IDs for home timeline. Total seen: {len(all_seen_tweet_ids_cache[home_key])}")

            if new_tweets_to_process and rcm.tweets_collection:
                tweet_docs_to_add = []
                tweet_metadatas_to_add = []
                tweet_ids_to_add = []
                seen_doc_ids: set[str] = set()
                for t_data in new_tweets_to_process:
                    if not t_data.tweet_url:
                        logger.warning(f"Skipping tweet storage for home timeline due to missing 'tweet_url'. Data: {t_data}")
                        continue

                    tweet_id_val = str(t_data.id or "")
                    if not tweet_id_val:
                        tweet_id_val = t_data.tweet_url.split("?")[0].split("/")[-1]
                    doc_id = f"tweet_home_{tweet_id_val}"

                    document_content = t_data.content or ''
                    if not document_content.strip():
                        logger.info(f"Skipping empty tweet from home timeline, ID: {doc_id}")
                        continue

                    if doc_id in seen_doc_ids:
                        logger.debug(f"Duplicate tweet doc_id detected in batch: {doc_id}")
                        continue

                    tweet_docs_to_add.append(document_content)
                    tweet_metadatas_to_add.append({'username': t_data.username, 'tweet_url': t_data.tweet_url})
                    tweet_ids_to_add.append(doc_id)
                    seen_doc_ids.add(doc_id)

                if tweet_ids_to_add:
                    try:
                        rcm.tweets_collection.add(
                            documents=tweet_docs_to_add,
                            metadatas=tweet_metadatas_to_add,
                            ids=tweet_ids_to_add
                        )
                        logger.info(
                            f"Successfully stored {len(tweet_ids_to_add)} new tweets from home timeline in ChromaDB."
                        )
                    except Exception as e_add_tweet:
                        logger.error(
                            f"Failed to store tweets from home timeline in ChromaDB: {e_add_tweet}",
                            exc_info=True,
                        )
            elif not rcm.tweets_collection:
                logger.warning(
                    "tweets_collection is not available. Skipping storage of home timeline tweets."
                )

            embed_title = "Recent Tweets from Home Timeline"
            raw_tweet_chunks = chunk_text(raw_tweets_display_str, config.EMBED_MAX_LENGTH)

            for i, chunk_content_part in enumerate(raw_tweet_chunks):
                chunk_title = embed_title if i == 0 else f"{embed_title} (cont.)"
                embed = discord.Embed(
                    title=chunk_title,
                    description=chunk_content_part,
                    color=config.EMBED_COLOR["complete"]
                )
                if i == 0:
                    if interaction.channel:
                        progress_message = await safe_message_edit(
                            progress_message,
                            interaction.channel,
                            content=None,
                            embed=embed,
                        )
                    else:
                        progress_message = await safe_message_edit(
                            progress_message,
                            interaction.channel or progress_message.channel,
                            content=None,
                            embed=embed,
                        )
                else:
                    await safe_followup_send(interaction, embed=embed)

            user_query_content_for_summary = (
                "Please analyze and summarize the main themes, topics discussed, and overall sentiment "
                f"from the recent tweets in my home timeline provided below. Extract key points and present a concise yet detailed overview. "
                f"Do not just re-list the tweets.\n\nRecent Tweets:\n{raw_tweets_display_str[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
            )
            user_msg_node = MsgNode("user", user_query_content_for_summary, name=str(interaction.user.id))

            # Build a minimal prompt using only the scraped tweets as context (no RAG, no prior channel history)
            prompt_nodes_summary = await _build_initial_prompt_messages(
                user_query_content=user_query_content_for_summary,
                channel_id=None,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
            )
            # Insert temporal system context to avoid outdated references
            insert_idx_sum3 = 0
            for idx, node in enumerate(prompt_nodes_summary):
                if node.role != "system":
                    insert_idx_sum3 = idx
                    break
                insert_idx_sum3 = idx + 1
            final_prompt_nodes_summary3 = (
                prompt_nodes_summary[:insert_idx_sum3]
                + [MsgNode("system", TEMPORAL_SYSTEM_CONTEXT)]
                + prompt_nodes_summary[insert_idx_sum3:]
            )

            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, final_prompt_nodes_summary3,
                title="Tweet Summary for Home Timeline",
                force_new_followup_flow=True,
                bot_user_id=bot_instance.user.id,
            )
            
            # Delete and recreate progress message to move to bottom after results are shown
            if progress_message:
                try:
                    await progress_message.delete()
                except (discord.NotFound, discord.HTTPException):
                    pass  # Message already deleted or webhook expired
                # Recreate the progress message at the bottom
                progress_message = await safe_followup_send(
                    interaction,
                    content="✅ Home timeline processing complete!",
                )
        except Exception as e:
            logger.error(
                f"Error in homefeed_slash_command: {e}", exc_info=True
            )
            error_content = (
                f"My tweet-fetching antenna is bent! Failed for home timeline. Error: {str(e)[:500]}"
            )
            try:
                if interaction.response.is_done():
                    await safe_followup_send(
                        interaction,
                        content=error_content,
                        ephemeral=True,
                        error_hint=" final error in homefeed",
                    )
                else:
                    if interaction.channel:
                        progress_message = await safe_message_edit(
                            progress_message,
                            interaction.channel,
                            content=error_content,
                            embed=None,
                        )
                    else:
                        progress_message = await safe_message_edit(
                            progress_message,
                            interaction.channel or progress_message.channel,
                            content=error_content,
                            embed=None,
                        )
            except discord.HTTPException:
                logger.warning("Could not send final error message for homefeed to user (HTTPException).")
        finally:
            if acquired_lock:
                scrape_lock.release()
                logger.debug("Scrape lock released for /homefeed")


    @bot_instance.tree.command(name="alltweets", description="Fetches tweets from all default accounts.")
    @app_commands.describe(
        limit="Number of tweets per account to fetch (max 100).",
        list_name="Optional saved list name to use instead of the default accounts.",
    )
    async def alltweets_slash_command(
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 100] = 50,
        list_name: str = "",
    ) -> None:
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("alltweets_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot get tweets.", ephemeral=True)
            return

        logger.info(
            f"Alltweets command initiated by {interaction.user.name}, limit {limit}."
        )
        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        scope_guild_id, scope_user_id = _twitter_scope_from_interaction(interaction)
        should_ephemeral = interaction.guild_id is not None
        if not interaction.response.is_done():
            try:
                await interaction.response.defer(ephemeral=should_ephemeral)
            except discord.HTTPException as exc:
                if getattr(exc, "status", None) == 404 and getattr(exc, "code", None) == 10062:
                    logger.warning(
                        "alltweets_slash_command: Interaction already invalid while deferring; continuing with fallbacks.",
                        exc_info=False,
                    )
                else:
                    raise

        await _ensure_default_twitter_list(scope_guild_id, scope_user_id)

        normalized_list_name = list_name.strip().lower()
        if normalized_list_name:
            handles = await bot_state_instance.get_twitter_list_handles(
                scope_guild_id,
                normalized_list_name,
                user_id=scope_user_id,
            )
            if not handles:
                await safe_followup_send(
                    interaction,
                    content=(
                        f"No saved list named `{normalized_list_name}` was found. "
                        "Use `/twitter_list_add` first to create it."
                    ),
                    ephemeral=should_ephemeral,
                    error_hint=" missing alltweets list",
                )
                return
            selected_accounts = handles
            list_descriptor = f"list `{normalized_list_name}`"
        else:
            stored_default = await bot_state_instance.get_twitter_list_handles(
                scope_guild_id,
                "default",
                user_id=scope_user_id,
            )
            selected_accounts = stored_default or DEFAULT_TWITTER_USERS
            list_descriptor = "default accounts"

        scrape_lock = bot_state_instance.get_scrape_lock()
        queue_notice = scrape_lock.locked()
        acquired_lock = False

        if queue_notice:
            await safe_followup_send(
                interaction,
                content="Waiting for other scraping tasks to finish before fetching tweets...",
                ephemeral=should_ephemeral,
                error_hint=" waiting for alltweets lock",
            )

        await scrape_lock.acquire()
        acquired_lock = True
        await safe_followup_send(
            interaction,
            content=f"Starting to scrape tweets from {list_descriptor}...",
            error_hint=" starting alltweets (queued)" if queue_notice else " starting alltweets",
        )

        try:
            any_new = False
            for username in selected_accounts:
                processed = await process_twitter_user(
                    interaction,
                    username,
                    limit,
                    source_command="/alltweets",
                )
                any_new = any_new or processed

            if not any_new:
                await safe_followup_send(
                    interaction,
                    content="No new tweets found for any default account.",
                    ephemeral=should_ephemeral,
                )

            user_msg = MsgNode("user", f"/alltweets (limit {limit}) [{list_descriptor}]", name=str(interaction.user.id))
            assistant_msg = MsgNode(
                "assistant",
                "Finished fetching tweets from default accounts.",
                name=str(bot_instance.user.id),
            )
            await bot_state_instance.append_history(interaction.channel_id, user_msg, config.MAX_MESSAGE_HISTORY)
            await bot_state_instance.append_history(interaction.channel_id, assistant_msg, config.MAX_MESSAGE_HISTORY)
            progress_note = None
            try:
                progress_note = await safe_followup_send(
                    interaction,
                    content="\U0001F501 Post-processing...",
                    ephemeral=should_ephemeral,
                    error_hint=" for alltweets post-processing",
                )
            except discord.HTTPException:
                progress_note = None

            start_post_processing_task(
                ingest_conversation_to_chromadb(
                    llm_client_instance,
                    interaction.channel_id,
                    interaction.user.id,
                    [user_msg, assistant_msg],
                    None,
                ),
                progress_message=progress_note,
            )
        except Exception as e:
            logger.error(f"Error in alltweets_slash_command: {e}", exc_info=True)
            await safe_followup_send(
                interaction,
                content=f"Failed to process tweets. Error: {str(e)[:500]}",
                error_hint=" in alltweets error",
            )
        finally:
            if acquired_lock:
                scrape_lock.release()
                logger.debug("Scrape lock released for /alltweets")


    @bot_instance.tree.command(name="groundnews", description="Scrapes Ground News 'My Feed' and summarizes new articles.")
    @app_commands.describe(
        limit="Number of articles to fetch (max 100)."
    )
    async def groundnews_slash_command(
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 100] = 50,
    ) -> None:
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("groundnews_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot scrape Ground News.", ephemeral=True)
            return

        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        scrape_lock = bot_state_instance.get_scrape_lock()
        queue_notice = scrape_lock.locked()
        acquired_lock = False
        if queue_notice:
            await interaction.response.send_message(
                "Waiting for other scraping tasks to finish before fetching Ground News...",
                ephemeral=True,
            )
            await scrape_lock.acquire()
            acquired_lock = True
            await safe_followup_send(
                interaction,
                content="Starting Ground News scraping...",
                error_hint=" starting groundmy",
            )
        else:
            await scrape_lock.acquire()
            acquired_lock = True
            await interaction.response.defer(ephemeral=False)
            await safe_followup_send(
                interaction,
                content="Starting Ground News scraping...",
                error_hint=" starting groundmy",
            )

        try:
            processed = await process_ground_news(interaction, limit)
            if not processed:
                await safe_followup_send(
                    interaction,
                    content="No new Ground News articles found.",
                    ephemeral=True,
                )
        except Exception as e:
            logger.error("Error in groundnews_slash_command: %s", e, exc_info=True)
            await safe_followup_send(
                interaction,
                content=f"Failed to process Ground News articles. Error: {str(e)[:500]}",
                error_hint=" in groundmy error",
            )
        finally:
            if acquired_lock:
                scrape_lock.release()
                logger.debug("Scrape lock released for /groundnews")


    @bot_instance.tree.command(name="groundtopic", description="Scrapes a Ground News topic page and summarizes new articles.")
    @app_commands.describe(
        topic="Topic page to scrape.",
        limit="Number of articles to fetch (max 100).",
    )
    @app_commands.choices(
        topic=GROUND_NEWS_TOPIC_CHOICES
    )
    async def groundtopic_slash_command(
        interaction: discord.Interaction,
        topic: str,
        limit: app_commands.Range[int, 1, 100] = 50,
    ) -> None:
        if not llm_client_instance or not bot_state_instance or not bot_instance or not bot_instance.user:
            logger.error("groundtopic_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot scrape Ground News.", ephemeral=True)
            return

        if interaction.channel_id is None:
            await interaction.response.send_message("Error: This command must be used in a channel.", ephemeral=True)
            return

        scrape_lock = bot_state_instance.get_scrape_lock()
        queue_notice = scrape_lock.locked()
        acquired_lock = False
        if queue_notice:
            await interaction.response.send_message(
                "Waiting for other scraping tasks to finish before fetching Ground News...",
                ephemeral=True,
            )
            await scrape_lock.acquire()
            acquired_lock = True
            await safe_followup_send(
                interaction,
                content="Starting Ground News scraping...",
                error_hint=" starting groundtopic",
            )
        else:
            await scrape_lock.acquire()
            acquired_lock = True
            await interaction.response.defer(ephemeral=False)
            await safe_followup_send(
                interaction,
                content="Starting Ground News scraping...",
                error_hint=" starting groundtopic",
            )

        try:
            processed = await process_ground_news_topic(interaction, topic, limit)
            if not processed:
                await safe_followup_send(
                    interaction,
                    content="No new Ground News articles found.",
                    ephemeral=True,
                )
        except Exception as e:
            logger.error("Error in groundtopic_slash_command: %s", e, exc_info=True)
            await safe_followup_send(
                interaction,
                content=f"Failed to process Ground News articles. Error: {str(e)[:500]}",
                error_hint=" in groundtopic error",
            )
        finally:
            if acquired_lock:
                scrape_lock.release()
                logger.debug("Scrape lock released for /groundtopic")


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

            chosen_celebrity = random.choice(["Keanu Reeves", "Dwayne 'The Rock' Johnson", "Zendaya", "Tom Hanks", "Margot Robbie", "Ryan Reynolds", "Morgan Freeman", "Secret MAGA Man"])

            ap_task_prompt_text = (
                f"You are an Associated Press (AP) photo caption writer with a quirky sense of humor. Your task is to describe the attached image in vivid detail, as if for someone who cannot see it. "
                f"However, here's the twist: you must creatively and seamlessly replace the main subject or a prominent character in the image with the celebrity **{chosen_celebrity}** or celeb they are associated in some way, weaving that association into the description. "
                f"Maintain a professional AP caption style (who, what, when, where, why - if inferable), but weave in {chosen_celebrity}'s or associate's presence naturally and humorously. "
                f"Start your response with 'AP Photo: {chosen_celebrity}'s...' "
                f"If the user provided an additional prompt, try to incorporate its theme or request into your {chosen_celebrity}-centric description: '{user_prompt if user_prompt else 'No additional user prompt.'}'"
            )

            user_content_for_ap_node = [
                {
                    "type": "text",
                    "text": user_prompt if user_prompt else "Describe this image with the AP Photo celebrity twist.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url_for_llm},
                },
            ]
            user_msg_node = MsgNode("user", user_content_for_ap_node, name=str(interaction.user.id))

            rag_query_for_ap = user_prompt if user_prompt else f"AP photo style description featuring {chosen_celebrity} for an image."
            synthesized_summary, raw_snippets = await retrieve_rag_context_with_progress(
                llm_client=llm_client_instance,
                query=rag_query_for_ap,
                interaction=interaction,
            )

            base_prompt_nodes = await _build_initial_prompt_messages(
                user_query_content=user_content_for_ap_node,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
                synthesized_rag_context_str=synthesized_summary,
                raw_rag_snippets=raw_snippets,
                max_image_history_depth=0
            )

            insert_idx = 0
            for idx, node in enumerate(base_prompt_nodes):
                if node.role != "system": insert_idx = idx; break
                insert_idx = idx + 1
            # Add temporal grounding to avoid outdated references in image captions
            final_prompt_nodes = (
                base_prompt_nodes[:insert_idx]
                + [MsgNode("system", TEMPORAL_SYSTEM_CONTEXT), MsgNode("system", ap_task_prompt_text)]
                + base_prompt_nodes[insert_idx:]
            )

            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, final_prompt_nodes,
                title=f"AP Photo Description ft. {chosen_celebrity} or someone associted with them.",
                synthesized_rag_context_for_display=synthesized_summary,
                bot_user_id=bot_instance.user.id,
                retrieved_snippets=raw_snippets,
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
                await safe_followup_send(
                    interaction,
                    content=f"An unexpected error occurred: {str(error)[:500]}",
                    ephemeral=True,
                    error_hint=" in clearhistory error handler",
                )

    @bot_instance.tree.command(name="pruneitems", description="Summarize and prune the oldest N chat history entries.")
    @app_commands.describe(limit="Number of oldest entries to summarize and prune")
    async def pruneitems_slash_command(
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 10],
    ):
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            pruned, summaries = await prune_oldest_items(limit)
            summary_text = "\n\n".join(summaries) if summaries else "No summary generated."

            # Create and send the summary in embeds, chunking if necessary
            title = f"Pruned {pruned} documents from chat history."
            if len(summary_text) > 4000: # A bit of buffer for the description
                chunks = chunk_text(summary_text, 4000)
                for i, chunk in enumerate(chunks):
                    embed = discord.Embed(
                        title=title + (f" (Part {i+1})" if len(chunks) > 1 else ""),
                        description=chunk,
                        color=config.EMBED_COLOR["complete"]
                    )
                    await safe_followup_send(interaction, embed=embed, ephemeral=True)
            else:
                embed = discord.Embed(
                    title=title,
                    description=summary_text,
                    color=config.EMBED_COLOR["complete"]
                )
                await safe_followup_send(interaction, embed=embed, ephemeral=True)

        except Exception as e:
            logger.error(f"Error in pruneitems_slash_command: {e}", exc_info=True)
            await safe_followup_send(
                interaction,
                content=f"Failed to prune items: {str(e)[:1900]}",
                ephemeral=True,
                error_hint=" while sending pruneitems result",
            )

    @bot_instance.tree.command(name="dbcounts", description="List the number of entries in each ChromaDB collection.")
    async def dbcounts_slash_command(interaction: discord.Interaction):
        try:
            await interaction.response.defer(thinking=True, ephemeral=True)
        except discord.NotFound:
            logger.warning("dbcounts_slash_command: Interaction expired before defer.")
            return
        except discord.HTTPException as exc:
            logger.warning(
                "dbcounts_slash_command: Failed to defer interaction: %s",
                exc,
            )
            return
        try:
            counts = rcm.get_collection_counts()
            if not counts:
                await safe_followup_send(
                    interaction,
                    content="No collection data available.",
                    ephemeral=True,
                )
                return
            lines = [f"{name}: {count}" for name, count in counts.items()]
            await safe_followup_send(
                interaction,
                content="\n".join(lines),
                ephemeral=True,
            )
        except Exception as e:
            logger.error(f"Error in dbcounts_slash_command: {e}", exc_info=True)
            await safe_followup_send(
                interaction,
                content=f"Failed to fetch counts: {str(e)[:500]}",
                ephemeral=True,
                error_hint=" while sending dbcounts result",
            )

    @bot_instance.tree.command(name="rss_podcast", description="Toggle auto-podcast after RSS/allrss chunks in this channel.")
    @app_commands.describe(enabled="True to enable; False to disable")
    async def rss_podcast_toggle(interaction: discord.Interaction, enabled: bool):
        try:
            if interaction.channel_id is None:
                await interaction.response.send_message("This command must be used in a channel.", ephemeral=True)
                return
            await bot_state_instance.set_podcast_after_rss_enabled(interaction.channel_id, bool(enabled))
            await interaction.response.send_message(
                f"Auto-podcast after RSS chunks is now {'ENABLED' if enabled else 'DISABLED'} in this channel.",
                ephemeral=True,
            )
        except Exception as e:
            logger.error("/rss_podcast toggle failed: %s", e, exc_info=True)
            try:
                await interaction.response.send_message(
                    f"Failed to update setting: {str(e)[:500]}", ephemeral=True
                )
            except discord.HTTPException:
                pass

    @bot_instance.tree.command(name="podcastthatshit", description="Triggers the podcast that shit instruction based on current chat history.")
    async def podcastthatshit_slash_command(interaction: discord.Interaction):
        if not all([llm_client_instance, bot_state_instance, bot_instance, bot_instance.user]):
            logger.error("podcastthatshit_slash_command: One or more bot components are None.")
            await interaction.response.send_message("Bot components not ready. Cannot process.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)

        try:
            user_query_content = "Podcast that shit"
            user_msg_node = MsgNode("user", user_query_content, name=str(interaction.user.id))

            prompt_nodes = await _build_initial_prompt_messages(
                user_query_content=user_query_content,
                channel_id=interaction.channel_id,
                bot_state=bot_state_instance,
                user_id=str(interaction.user.id),
            )

            await stream_llm_response_to_interaction(
                interaction, llm_client_instance, bot_state_instance, user_msg_node, prompt_nodes,
                title="Podcast: The Current Conversation"
            )
        except Exception as e:
            logger.error(f"Error in podcastthatshit_slash_command: {e}", exc_info=True)
            await safe_followup_send(
                interaction,
                content=f"My podcasting equipment seems to be on fire. Error: {str(e)[:1000]}",
                ephemeral=True,
                error_hint=" in podcastthatshit error",
            )
        finally:
            # Optionally re-enable TTS after this command completes (env-controlled)
            try:
                if config.PODCAST_ENABLE_TTS_AFTER and not config.TTS_ENABLED_DEFAULT:
                    config.TTS_ENABLED_DEFAULT = True
                    logger.info("TTS has been enabled after /podcastthatshit command (per PODCAST_ENABLE_TTS_AFTER).")
            except Exception:
                pass

    @bot_instance.tree.command(name="tts_delivery", description="Choose how Sam sends voice responses in this server.")
    @app_commands.describe(mode="audio, video, both, or off")
    @app_commands.choices(mode=TTS_DELIVERY_CHOICES)
    async def tts_delivery(interaction: discord.Interaction, mode: app_commands.Choice[str]):
        if not bot_state_instance:
            await interaction.response.send_message(
                "Bot state is not ready yet; try again shortly.",
                ephemeral=True,
            )
            return

        if interaction.guild_id is None:
            await interaction.response.send_message(
                "This command can only be used inside a server.",
                ephemeral=True,
            )
            return

        try:
            previous_mode = await bot_state_instance.get_tts_delivery_mode(interaction.guild_id)
            await bot_state_instance.set_tts_delivery_mode(interaction.guild_id, mode.value)

            labels = {
                "audio": "Audio only (MP3)",
                "video": "Video only (MP4)",
                "both": "Audio + video",
                "off": "Disabled",
            }

            await interaction.response.send_message(
                f"TTS delivery is now {labels.get(mode.value, mode.value)} (was {labels.get(previous_mode, previous_mode)}).",
                ephemeral=True,
            )
        except Exception as e:
            logger.error("/tts_delivery failed: %s", e, exc_info=True)
            try:
                await interaction.response.send_message(
                    f"Failed to update TTS delivery: {str(e)[:500]}",
                    ephemeral=True,
                )
            except discord.HTTPException:
                pass

    # Toggle inclusion of <think> thoughts in TTS audio
    @bot_instance.tree.command(name="tts_thoughts", description="Enable or disable TTS playback of <think> thoughts.")
    @app_commands.describe(enabled="True to include thoughts in TTS; False to skip thoughts")
    async def tts_thoughts(interaction: discord.Interaction, enabled: bool):
        try:
            prev = bool(getattr(config, "TTS_INCLUDE_THOUGHTS", False))
            config.TTS_INCLUDE_THOUGHTS = bool(enabled)
            await interaction.response.send_message(
                f"Thoughts TTS is now {'ENABLED' if enabled else 'DISABLED'} (was {'ENABLED' if prev else 'DISABLED'}).",
                ephemeral=True,
            )
        except Exception as e:
            logger.error("/tts_thoughts failed: %s", e, exc_info=True)
            try:
                await interaction.response.send_message(
                    f"Failed to update thoughts TTS: {str(e)[:500]}", ephemeral=True
                )
            except discord.HTTPException:
                pass

    # Toggle raw tweet output in scheduled alltweets (per server or per DM; default off = summary only)
    @bot_instance.tree.command(
        name="scheduled_tweets_display",
        description="Show or hide raw tweet text in scheduled alltweets (summary is always posted).",
    )
    @app_commands.describe(
        enabled="True to post tweet text embeds; False for summary only (default)."
    )
    async def scheduled_tweets_display(interaction: discord.Interaction, enabled: bool):
        if not bot_state_instance:
            await interaction.response.send_message(
                "Bot state not available.",
                ephemeral=True,
            )
            return
        # In servers use guild id; in DMs use negative user id so the setting is per-DM thread
        scope_id = interaction.guild_id if interaction.guild_id is not None else -interaction.user.id
        try:
            prev = await bot_state_instance.get_scheduled_alltweets_show_tweets(scope_id)
            await bot_state_instance.set_scheduled_alltweets_show_tweets(scope_id, enabled)
            where = "this server" if interaction.guild_id is not None else "DMs"
            await interaction.response.send_message(
                f"Scheduled alltweets in {where} will {'show' if enabled else 'hide'} raw tweet text (was {'show' if prev else 'hide'}). Summary is always posted.",
                ephemeral=True,
            )
        except Exception as e:
            logger.error("/scheduled_tweets_display failed: %s", e, exc_info=True)
            try:
                await interaction.response.send_message(
                    f"Failed to update setting: {str(e)[:500]}",
                    ephemeral=True,
                )
            except discord.HTTPException:
                pass
