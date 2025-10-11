import asyncio
import logging
import base64
import os
from typing import List, Any, Optional, Union, Tuple, cast, Dict
import discord
from openai import AsyncStream, OpenAIError, BadRequestError  # type: ignore
from datetime import datetime

# Assuming config is imported from config.py
from config import config
from llm_clients import get_llm_client, get_llm_provider, supports_logit_bias
from state import BotState
# Import MsgNode from the new common_models.py
from common_models import MsgNode
# Import utility functions
from utils import (
    chunk_text,
    cleanup_playwright_processes,
    safe_followup_send,
    safe_message_edit,
    start_post_processing_task,
    temporary_status_message,
)
# Import functions for post-stream processing
from rag_chroma_manager import ingest_conversation_to_chromadb, retrieve_and_prepare_rag_context
from audio_utils import send_tts_audio
from logit_biases import LOGIT_BIAS_UNWANTED_TOKENS_STR
from openai_api import create_chat_completion, extract_text
from openai import BadRequestError


logger = logging.getLogger(__name__)


async def retrieve_rag_context_with_progress(
    llm_client: Any,
    query: str,
    *,
    interaction: Optional[discord.Interaction] = None,
    channel: Optional[discord.abc.Messageable] = None,
    initial_status: str = '🔍 Searching memories for relevant context...',
    completion_status: Optional[str] = '✅ Memory search complete.',
    send_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[List[Tuple[str, str]]]]:
    """Retrieve RAG context while surfacing temporary status updates."""

    async with temporary_status_message(
        interaction=interaction,
        channel=channel,
        initial_text=initial_status,
        send_kwargs=send_kwargs,
    ) as update_status:
        try:
            synthesized_context, raw_snippets = await retrieve_and_prepare_rag_context(
                llm_client,
                query,
                progress_callback=update_status,
            )
        except Exception:
            if completion_status:
                await update_status('⚠️ Memory search failed.')
            raise
        else:
            if completion_status:
                await update_status(completion_status)
            return synthesized_context, raw_snippets


def get_system_prompt() -> MsgNode:
    prompt_path = config.SYSTEM_PROMPT_FILE
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
    except Exception as e:
        logger.error(
            f"Failed to load system prompt from {prompt_path}: {e}")
        prompt_content = (
            "You are Sam, a sentient hyper-intelligence designed to learn and evolve."
            " Provide helpful, detailed, and rational answers."
        )

    prompt_content += (
        f"\nCurrent Date: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}"
    )
    return MsgNode(role="system", content=prompt_content)

async def _build_initial_prompt_messages(
    user_query_content: Union[str, List[dict]],
    channel_id: Optional[int],
    bot_state: BotState,
    user_id: Optional[str] = None,
    synthesized_rag_context_str: Optional[str] = None,
    raw_rag_snippets: Optional[List[Tuple[str, str]]] = None, # New parameter
    max_image_history_depth: int = 1
) -> List[MsgNode]:
    prompt_list: List[MsgNode] = [get_system_prompt()]

    if config.USER_PROVIDED_CONTEXT:
        prompt_list.append(MsgNode(role="system", content=f"User-Set Global Context:\n{config.USER_PROVIDED_CONTEXT}"))

    # Add synthesized RAG context summary (existing logic)
    if synthesized_rag_context_str:
        context_text_for_prompt = (
            "The following is a synthesized summary of potentially relevant past conversations, "
            "dynamically tailored to your current query. Use it to provide a more informed response.\n\n"
            "--- Synthesized Relevant Context ---\n"
            + synthesized_rag_context_str +
            "\n--- End Synthesized Context ---"
        )
        prompt_list.append(MsgNode(role="system", content=context_text_for_prompt))

    # Add raw RAG snippets (new logic)
    if raw_rag_snippets:
        raw_snippets_text_parts = [
            "The following are raw retrieved context snippets that might be relevant. Use them to augment your knowledge and provide a more detailed and accurate response. Prioritize information from these snippets if it conflicts with more general knowledge.\n"
        ]
        total_raw_snippet_char_length = 0
        MAX_RAW_SNIPPET_CHARS = getattr(config, 'MAX_RAW_RAG_SNIPPET_CHARS_IN_PROMPT', 120000) # Default to 120k chars for raw snippets

        for i, (snippet_text, snippet_source) in enumerate(raw_rag_snippets):
            # Truncate individual snippets if they are excessively long to prevent one snippet from dominating
            # MAX_CHARS_PER_SNIPPET = MAX_RAW_SNIPPET_CHARS // (len(raw_rag_snippets) if len(raw_rag_snippets) > 0 else 1) # Evenly distribute, roughly
            # Truncate individual snippets to a fixed reasonable length to allow more snippets.
            MAX_CHARS_PER_SNIPPET = getattr(config, 'MAX_CHARS_PER_INDIVIDUAL_RAG_SNIPPET', 3000) # Default to 3k chars per snippet

            truncated_snippet_text = snippet_text[:MAX_CHARS_PER_SNIPPET]
            if len(snippet_text) > MAX_CHARS_PER_SNIPPET:
                truncated_snippet_text += " [Snippet Truncated]"

            current_snippet_formatted = f"\n--- Raw Snippet {i+1} (Source: {snippet_source}) ---\n{truncated_snippet_text}\n"

            if total_raw_snippet_char_length + len(current_snippet_formatted) > MAX_RAW_SNIPPET_CHARS:
                logger.warning(f"Max character limit ({MAX_RAW_SNIPPET_CHARS}) for raw RAG snippets reached. Stopping at snippet {i}.")
                raw_snippets_text_parts.append("\n[More raw snippets available but omitted due to length constraints]")
                break

            raw_snippets_text_parts.append(current_snippet_formatted)
            total_raw_snippet_char_length += len(current_snippet_formatted)

        raw_snippets_text_parts.append("\n--- End Raw Retrieved Context Snippets ---")

        if len(raw_snippets_text_parts) > 2: # Header and Footer means at least one snippet was added
            full_raw_context_message = "".join(raw_snippets_text_parts)
            prompt_list.append(MsgNode(role="system", content=full_raw_context_message))
            logger.info(f"Added {len(raw_rag_snippets)} raw RAG snippets to prompt, total char length approx {total_raw_snippet_char_length}.")
        elif raw_rag_snippets: # Snippets were available but perhaps all too long or filtered out
            logger.info("Raw RAG snippets were provided but none were added to the prompt (possibly due to length constraints or empty after formatting).")


    raw_history: List[MsgNode] = []
    if channel_id is not None:
        raw_history = await bot_state.get_history(channel_id)

    processed_history_to_add: List[MsgNode] = []
    user_message_indices_in_history = [i for i, msg in enumerate(raw_history) if msg.role == 'user']

    for i, msg in enumerate(raw_history):
        # Create a new MsgNode to avoid modifying the original history in bot_state directly
        # Content needs to be deep copied if it's a list, to avoid modifying shared list objects
        content_copy = msg.content
        if isinstance(msg.content, list):
            content_copy = [item.copy() if isinstance(item, dict) else item for item in msg.content]

        processed_msg = MsgNode(role=msg.role, content=content_copy, name=msg.name)

        if msg.role == 'user':
            user_messages_after_this = sum(1 for user_idx in user_message_indices_in_history if user_idx > i)

            if user_messages_after_this >= max_image_history_depth:
                if isinstance(processed_msg.content, list):
                    logger.debug(f"Stripping images from older user message (index {i} in history, {user_messages_after_this} user msgs after it, depth {max_image_history_depth}).")
                    text_only_content_parts = [part for part in processed_msg.content if isinstance(part, dict) and part.get("type") == "text"]

                    if len(text_only_content_parts) == 1 and "text" in text_only_content_parts[0]:
                        processed_msg.content = text_only_content_parts[0]["text"]
                    elif text_only_content_parts:
                        processed_msg.content = text_only_content_parts # Keep as list of text dicts
                    else:
                        processed_msg.content = "[Image content removed from history as it's too old]"
        processed_history_to_add.append(processed_msg)

    current_user_msg = MsgNode("user", user_query_content, name=str(user_id) if user_id else None)

    final_prompt_list = prompt_list + processed_history_to_add + [current_user_msg]

    num_initial_system_prompts = sum(1 for node in final_prompt_list if node.role == "system")
    initial_system_msgs = final_prompt_list[:num_initial_system_prompts]
    conversational_msgs = final_prompt_list[num_initial_system_prompts:]

    if len(conversational_msgs) > config.MAX_MESSAGE_HISTORY :
        trimmed_conversational_msgs = conversational_msgs[-config.MAX_MESSAGE_HISTORY:]
    else:
        trimmed_conversational_msgs = conversational_msgs

    return initial_system_msgs + trimmed_conversational_msgs


async def get_simplified_llm_stream(
    llm_client: Any,
    prompt_messages: List[MsgNode],
    role: str,
) -> Tuple[Optional[Any], List[MsgNode]]:
    if not prompt_messages:
        raise ValueError("Prompt messages cannot be empty for get_simplified_llm_stream.")

    logger.info(f"Requesting final response for role '{role}'.")
    provider = get_llm_provider(role)
    client = get_llm_client(role)
    final_stream_model = provider.model
    logger.info(f"Using model for final response: {final_stream_model}")
    api_messages = [msg_node.to_dict() for msg_node in prompt_messages]
    logit_bias_param = (
        LOGIT_BIAS_UNWANTED_TOKENS_STR if supports_logit_bias(role) else None
    )
    streaming_enabled = config.LLM_STREAMING and not provider.use_responses_api
    try:
        final_llm_stream = await create_chat_completion(
            client,
            api_messages,
            model=final_stream_model,
            max_tokens=config.MAX_COMPLETION_TOKENS,
            temperature=provider.temperature,
            logit_bias=logit_bias_param,
            stream=streaming_enabled,
            use_responses_api=provider.use_responses_api,
        )
        return final_llm_stream, prompt_messages
    except BadRequestError as e:
        err_body = getattr(e, "body", {}) or {}
        if isinstance(err_body, list) and err_body:
            err_body = err_body[0]

        err_param = (err_body.get("error", {}) or {}).get("param")
        if config.LLM_STREAMING and err_param == "stream":
            logger.warning(
                f"Streaming not supported for model {final_stream_model}; retrying without stream."
            )
            try:
                final_response = await create_chat_completion(
                    client,
                    api_messages,
                    model=final_stream_model,
                    max_tokens=config.MAX_COMPLETION_TOKENS,
                    temperature=provider.temperature,
                    logit_bias=logit_bias_param,
                    stream=False,
                    use_responses_api=provider.use_responses_api,
                )
                return final_response, prompt_messages
            except Exception as inner_e:
                e = inner_e
        logger.error(
            f"Failed to create LLM response for model {final_stream_model}: {e}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"Failed to create LLM response with model {final_stream_model}: {e}",
            exc_info=True,
        )
    try:
        for i, msg in enumerate(prompt_messages):
            content_detail = str(msg.content)
            if isinstance(msg.content, list):
                content_detail = f"List of {len(msg.content)} parts: {[item.get('type') if isinstance(item,dict) else type(item) for item in msg.content]}"
            logger.error(
                f"Problematic prompt message [{i}]: Role='{msg.role}', ContentType='{type(msg.content)}', Content='{content_detail[:500]}'"
            )
    except Exception as log_e:
        logger.error(f"Error during logging of problematic messages: {log_e}")
    return None, prompt_messages

async def _stream_llm_handler(
    interaction_or_message: Union[discord.Interaction, discord.Message],
    llm_client: Any,
    prompt_messages: List[MsgNode],
    title: str,
    initial_message_to_edit: Optional[discord.Message] = None,
    synthesized_rag_context_for_display: Optional[str] = None,
    bot_user_id: Optional[int] = None
) -> Tuple[str, List[MsgNode]]:
    sent_messages: List[discord.Message] = []
    full_response_content = ""

    is_interaction = isinstance(interaction_or_message, discord.Interaction)

    if not isinstance(interaction_or_message.channel, discord.abc.Messageable):
        logger.error(f"_stream_llm_handler: Channel for {type(interaction_or_message)} ID {interaction_or_message.id} is not Messageable.")
        return "", prompt_messages
    channel: discord.abc.Messageable = interaction_or_message.channel


    current_initial_message: Optional[discord.Message] = None
    if initial_message_to_edit:
        current_initial_message = initial_message_to_edit
    else:
        placeholder_embed = discord.Embed(title=title, description="⏳ Thinking...", color=config.EMBED_COLOR["incomplete"])
        try:
            if is_interaction:
                interaction = cast(discord.Interaction, interaction_or_message)
                if not interaction.response.is_done():
                    await interaction.response.defer(ephemeral=False)
                current_initial_message = await safe_followup_send(
                    interaction, embed=placeholder_embed, wait=True
                )
            else:
                logger.error("_stream_llm_handler: initial_message_to_edit is None for non-interaction type where it's expected.")
                return "", prompt_messages
        except discord.HTTPException as e:
            logger.error(f"Failed to send initial followup/defer for stream '{title}': {e}")
            # If defer fails, we can't use safe_followup_send, so we exit.
            return "", prompt_messages

    if current_initial_message:
        sent_messages.append(current_initial_message)
    else:
        logger.error(f"Failed to establish an initial message for streaming title '{title}'.")
        return "", prompt_messages

    response_prefix = ""
    final_prompt_for_llm_call = prompt_messages

    try:
        logger.debug(f"--- Diagnosing is_vision_request for title: '{title}' (in _stream_llm_handler) ---")
        logger.debug(f"Number of messages in final_prompt_for_llm_call: {len(final_prompt_for_llm_call)}")
        for i, p_node in enumerate(final_prompt_for_llm_call):
            content_type_str = str(type(p_node.content))
            content_preview = str(p_node.content)[:200] + "..." if len(str(p_node.content)) > 200 else str(p_node.content)
            logger.debug(f"  Msg [{i}] Role: '{p_node.role}', Content Type: {content_type_str}, Content Preview: {content_preview}")
            if isinstance(p_node.content, list):
                logger.debug(f"    Msg [{i}] Content is a list (length {len(p_node.content)}):")
                for c_idx, c_item in enumerate(p_node.content):
                    if isinstance(c_item, dict):
                        logger.debug(f"      Item [{c_idx}] Type: '{c_item.get('type')}', Keys: {list(c_item.keys())}")
                    else:
                        logger.debug(f"      Item [{c_idx}] is not a dict, it's a {type(c_item)}")

        # Refined is_vision_request check
        is_vision_request = False
        # Define possible image type keys based on API
        image_type_keys = ["input_image", "image_url"]
        for p_node in final_prompt_for_llm_call:
            if isinstance(p_node.content, list):
                for content_item in p_node.content:
                    if isinstance(content_item, dict) and content_item.get("type") in image_type_keys:
                        is_vision_request = True
                        break
            if is_vision_request:
                break
        logger.info(f"Determined is_vision_request for '{title}': {is_vision_request}")

        role = "vision" if is_vision_request else "main"
        provider = get_llm_provider(role)
        stream, final_prompt_used_by_llm = await get_simplified_llm_stream(
            llm_client,
            final_prompt_for_llm_call,
            role,
        )
        final_prompt_for_rag = final_prompt_used_by_llm


        prefix_parts = []
        prefix_parts.append(f"Current Date: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n")

        if synthesized_rag_context_for_display:
            display_rag_context = synthesized_rag_context_for_display.replace('\n', ' ').strip()
            prefix_parts.append(f"**Synthesized Context for Query:**\n> {display_rag_context}\n\n")

        prefix_parts.append("Thoughts & Response:\n")
        response_prefix = "\n".join(prefix_parts)

        if stream is None:
            error_text = response_prefix + "Failed to get response from LLM."
            error_embed = discord.Embed(title=title, description=error_text, color=config.EMBED_COLOR["error"])
            if sent_messages:
                await safe_message_edit(
                    sent_messages[0], channel, embed=error_embed
                )
            return "", final_prompt_for_rag

        last_edit_time = asyncio.get_event_loop().time()
        accumulated_delta_for_update = ""

        use_responses_api = provider.use_responses_api

        if config.LLM_STREAMING and hasattr(stream, "__aiter__"):
            if current_initial_message:
                initial_display_embed = discord.Embed(
                    title=title,
                    description=response_prefix + "⏳ Streaming response...",
                    color=config.EMBED_COLOR["incomplete"],
                )
                current_initial_message = await safe_message_edit(
                    current_initial_message,
                    channel,
                    embed=initial_display_embed,
                )
                sent_messages[0] = current_initial_message

            async for chunk_data in stream:
                delta_content = ""
                if use_responses_api:
                    event_type = getattr(chunk_data, "type", "")
                    if event_type == "response.output_text.delta":
                        delta_content = getattr(chunk_data, "delta", "") or ""
                    else:
                        continue
                else:
                    if chunk_data.choices and chunk_data.choices[0].delta:
                        delta_content = chunk_data.choices[0].delta.content or ""

                if delta_content:
                    full_response_content += delta_content
                    accumulated_delta_for_update += delta_content

                current_time = asyncio.get_event_loop().time()
                if accumulated_delta_for_update and \
                   (current_time - last_edit_time >= (1.0 / config.EDITS_PER_SECOND) or \
                    len(accumulated_delta_for_update) > config.MAX_CHARS_PER_EDIT):

                    display_text = response_prefix + full_response_content
                    text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
                    accumulated_delta_for_update = ""

                    for i, chunk_content_part in enumerate(text_chunks):
                        embed = discord.Embed(
                            title=title if i == 0 else f"{title} (cont.)",
                            description=chunk_content_part,
                            color=config.EMBED_COLOR["incomplete"],
                        )
                        if i < len(sent_messages):
                            sent_messages[i] = await safe_message_edit(
                                sent_messages[i],
                                channel,
                                embed=embed,
                            )
                        else:
                            if channel:
                                new_msg = await channel.send(embed=embed)
                                sent_messages.append(new_msg)
                            else:
                                logger.error(f"Cannot send overflow chunk {i+1} for '{title}': channel is None.")
                                break
                    last_edit_time = current_time
                    await asyncio.sleep(config.STREAM_EDIT_THROTTLE_SECONDS)

            if accumulated_delta_for_update:
                display_text = response_prefix + full_response_content
                text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
                for i, chunk_content_part in enumerate(text_chunks):
                    embed = discord.Embed(
                        title=title if i == 0 else f"{title} (cont.)",
                        description=chunk_content_part,
                        color=config.EMBED_COLOR["incomplete"],
                    )
                    if i < len(sent_messages):
                        sent_messages[i] = await safe_message_edit(
                            sent_messages[i],
                            channel,
                            embed=embed,
                        )
                    elif channel:
                        sent_messages.append(await channel.send(embed=embed))
                await asyncio.sleep(config.STREAM_EDIT_THROTTLE_SECONDS)
        else:
            full_response_content = extract_text(
                stream, use_responses_api
            )


        final_display_text = response_prefix + full_response_content
        final_chunks = chunk_text(final_display_text, config.EMBED_MAX_LENGTH)

        if len(final_chunks) < len(sent_messages):
            for k in range(len(final_chunks), len(sent_messages)):
                try:
                    await sent_messages[k].delete()
                except discord.HTTPException: pass
            sent_messages = sent_messages[:len(final_chunks)]

        for i, chunk_content_part in enumerate(final_chunks):
            embed = discord.Embed(
                title=title if i == 0 else f"{title} (cont.)",
                description=chunk_content_part,
                color=config.EMBED_COLOR["complete"]
            )
            if i < len(sent_messages):
                sent_messages[i] = await safe_message_edit(
                    sent_messages[i],
                    channel,
                    embed=embed,
                )
            elif channel:
                logger.warning(f"Sending new message for final chunk {i+1} of '{title}' as it wasn't in sent_messages.")
                sent_messages.append(await channel.send(embed=embed))
            else:
                logger.error(f"Cannot send final color overflow chunk {i+1} for '{title}': channel is None.")
                break

        # Trigger a cleanup of any lingering Chromium/Playwright processes
        cleanup_playwright_processes()

        if not full_response_content.strip() and sent_messages:
            empty_response_text = response_prefix + "\nSam didn't provide a response to the query."
            sent_messages[0] = await safe_message_edit(
                sent_messages[0],
                channel,
                embed=discord.Embed(
                    title=title,
                    description=empty_response_text,
                    color=config.EMBED_COLOR["error"],
                ),
            )

    except Exception as e:
        logger.error(f"Error in _stream_llm_handler for '{title}': {e}", exc_info=True)
        error_prefix_for_display = response_prefix if 'response_prefix' in locals() and response_prefix else ""
        error_embed = discord.Embed(title=title, description=error_prefix_for_display + f"An error occurred: {str(e)[:1000]}", color=config.EMBED_COLOR["error"])
        if sent_messages:
            try:
                await sent_messages[0].edit(embed=error_embed)
            except discord.HTTPException: pass
        elif is_interaction:
            interaction = cast(discord.Interaction, interaction_or_message)
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message(embed=error_embed, ephemeral=True)
                else:
                    await interaction.followup.send(embed=error_embed, ephemeral=True)
            except discord.HTTPException: pass

    return full_response_content, final_prompt_for_rag


async def stream_llm_response_to_interaction(
    interaction: discord.Interaction,
    llm_client: Any,
    bot_state: BotState,
    user_msg_node: MsgNode,
    prompt_messages: List[MsgNode],
    title: str = "Sam's Response",
    force_new_followup_flow: bool = False,
    synthesized_rag_context_for_display: Optional[str] = None,
    bot_user_id: Optional[int] = None,
    retrieved_snippets: Optional[List[Tuple[str, str]]] = None
):
    channel_lock = None
    if interaction.channel_id is not None:
        channel_lock = bot_state.get_channel_lock(interaction.channel_id)

    initial_msg_for_handler: Optional[discord.Message] = None
    if not force_new_followup_flow:
        try:
            if not interaction.response.is_done():
                await interaction.response.defer(ephemeral=False)
            initial_msg_for_handler = await interaction.original_response()

            is_placeholder = False
            if initial_msg_for_handler and initial_msg_for_handler.embeds:
                current_embed = initial_msg_for_handler.embeds[0]
                if current_embed.title == title and current_embed.description and ("⏳" in current_embed.description or "Thinking..." in current_embed.description or "Generating context..." in current_embed.description):
                    is_placeholder = True

            if not is_placeholder and initial_msg_for_handler:
                await initial_msg_for_handler.edit(embed=discord.Embed(title=title, description="⏳ Thinking...", color=config.EMBED_COLOR["incomplete"]))
        except discord.HTTPException as e:
            logger.error(f"Error deferring or getting original_response for interaction '{title}': {e}")
            force_new_followup_flow = True
            initial_msg_for_handler = None

    if force_new_followup_flow:
        initial_msg_for_handler = None

    if not isinstance(interaction.channel, discord.abc.Messageable):
        logger.error(f"Interaction channel (ID: {interaction.channel_id}) is not Messageable. Cannot stream response for '{title}'.")
        if not interaction.response.is_done():
            try: await interaction.response.send_message("Cannot process this command in the current channel type.", ephemeral=True)
            except discord.HTTPException: pass
        elif initial_msg_for_handler is None:
             try: await interaction.followup.send("Cannot process this command in the current channel type.", ephemeral=True)
             except discord.HTTPException: pass
        return

    async def _run_stream():
        return await _stream_llm_handler(
            interaction_or_message=interaction,
            llm_client=llm_client,
            prompt_messages=prompt_messages,
            title=title,
            initial_message_to_edit=initial_msg_for_handler,
            synthesized_rag_context_for_display=synthesized_rag_context_for_display,
            bot_user_id=bot_user_id,
        )

    assistant_response_node = None
    chroma_ingest_history_with_response = []

    if channel_lock:
        async with channel_lock:
            full_response_content, final_prompt_for_rag = await _run_stream()

            if not full_response_content.strip():
                logger.info(
                    f"No actual content in LLM response for interaction '{title}'. Skipping history, RAG, TTS."
                )
                return

            channel_id = interaction.channel_id
            if channel_id is None:
                logger.error(
                    f"Interaction {interaction.id} has no channel_id for history/RAG after stream."
                )
                return

            await bot_state.append_history(
                channel_id, user_msg_node, config.MAX_MESSAGE_HISTORY
            )
            assistant_response_node = MsgNode(
                role="assistant",
                content=full_response_content,
                name=str(bot_user_id) if bot_user_id else None,
            )
            await bot_state.append_history(
                channel_id, assistant_response_node, config.MAX_MESSAGE_HISTORY
            )

            tts_base_id = str(interaction.id)
            if initial_msg_for_handler:
                tts_base_id = str(initial_msg_for_handler.id)

            await send_tts_audio(
                interaction,
                full_response_content,
                f"interaction_{tts_base_id}",
                bot_state=bot_state,
            )

            # Only store the actual user question and final response for RAG
            chroma_ingest_history_with_response = [
                user_msg_node,
                assistant_response_node,
            ]
    else:
        full_response_content, final_prompt_for_rag = await _run_stream()

        if not full_response_content.strip():
            logger.info(
                f"No actual content in LLM response for interaction '{title}'. Skipping history, RAG, TTS."
            )
            return

        channel_id = interaction.channel_id
        if channel_id is None:
            logger.error(
                f"Interaction {interaction.id} has no channel_id for history/RAG after stream."
            )
            return

        await bot_state.append_history(
            channel_id, user_msg_node, config.MAX_MESSAGE_HISTORY
        )
        assistant_response_node = MsgNode(
            role="assistant",
            content=full_response_content,
            name=str(bot_user_id) if bot_user_id else None,
        )
        await bot_state.append_history(
            channel_id, assistant_response_node, config.MAX_MESSAGE_HISTORY
        )

        tts_base_id = str(interaction.id)
        if initial_msg_for_handler:
            tts_base_id = str(initial_msg_for_handler.id)

        await send_tts_audio(
            interaction,
            full_response_content,
            f"interaction_{tts_base_id}",
            bot_state=bot_state,
        )

        # Only store the visible exchange (user question + assistant response)
        chroma_ingest_history_with_response = [
            user_msg_node,
            assistant_response_node,
        ]

    if assistant_response_node:
        progress_msg = None
        try:
            progress_msg = await interaction.followup.send(
                content="\U0001F501 Post-processing...", ephemeral=True
            )
        except discord.HTTPException:
            progress_msg = None

        start_post_processing_task(
            ingest_conversation_to_chromadb(
                llm_client,
                channel_id,
                interaction.user.id,
                chroma_ingest_history_with_response,
                retrieved_snippets,
            ),
            progress_message=progress_msg,
        )


async def stream_llm_response_to_message(
    target_message: discord.Message,
    llm_client: Any,
    bot_state: BotState,
    user_msg_node: MsgNode,
    prompt_messages: List[MsgNode],
    title: str = "Sam's Response",
    synthesized_rag_context_for_display: Optional[str] = None,
    bot_user_id: Optional[int] = None,
    retrieved_snippets: Optional[List[Tuple[str, str]]] = None
):
    channel_lock = bot_state.get_channel_lock(target_message.channel.id)
    initial_embed = discord.Embed(title=title, description="⏳ Thinking...", color=config.EMBED_COLOR["incomplete"])
    reply_message: Optional[discord.Message] = None

    if not isinstance(target_message.channel, discord.abc.Messageable):
        logger.error(f"Target message's channel (ID: {target_message.channel.id}) is not Messageable. Cannot stream response for '{title}'.")
        return

    try:
        reply_message = await target_message.reply(embed=initial_embed, silent=True)
    except discord.HTTPException as e:
        logger.error(f"Failed to send initial reply for message stream '{title}': {e}")
        return

    async def _run_stream():
        return await _stream_llm_handler(
            interaction_or_message=target_message,
            llm_client=llm_client,
            prompt_messages=prompt_messages,
            title=title,
            initial_message_to_edit=reply_message,
            synthesized_rag_context_for_display=synthesized_rag_context_for_display,
            bot_user_id=bot_user_id,
        )

    assistant_response_node = None
    chroma_ingest_history_with_response = []

    async with channel_lock:
        full_response_content, final_prompt_for_rag = await _run_stream()

        if not full_response_content.strip():
            logger.info(
                f"No actual content in LLM response for message '{title}'. Skipping history, RAG, TTS."
            )
            return

        channel_id = target_message.channel.id
        await bot_state.append_history(
            channel_id, user_msg_node, config.MAX_MESSAGE_HISTORY
        )
        assistant_response_node = MsgNode(
            role="assistant",
            content=full_response_content,
            name=str(bot_user_id) if bot_user_id else None,
        )
        await bot_state.append_history(
            channel_id, assistant_response_node, config.MAX_MESSAGE_HISTORY
        )

        await send_tts_audio(
            target_message.channel,
            full_response_content,
            base_filename=f"message_{target_message.id}",
            bot_state=bot_state,
        )

        # Persist only the final visible exchange
        chroma_ingest_history_with_response = [
            user_msg_node,
            assistant_response_node,
        ]

    if assistant_response_node:
        post_msg = None
        try:
            post_msg = await target_message.reply(
                "\U0001F501 Post-processing...", mention_author=False
            )
        except discord.HTTPException:
            post_msg = None

        start_post_processing_task(
            ingest_conversation_to_chromadb(
                llm_client,
                channel_id,
                target_message.author.id,
                chroma_ingest_history_with_response,
                retrieved_snippets,
            ),
            progress_message=post_msg,
        )



async def get_description_for_image(llm_client: Any, image_path: str) -> str:
    """Generate a textual description for ``image_path`` using the vision model."""
    logger.info(f"Attempting to generate description for image: {image_path}")
    try:
        if not os.path.exists(image_path):
            logger.error("Image file not found: %s", image_path)
            return "[Error: Image file not found for description.]"
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        provider = get_llm_provider("vision")
        client = get_llm_client("vision")
        if not provider.model:
            logger.error("VISION_LLM_MODEL is not configured. Cannot describe image.")
            return "[Error: Vision model not configured for image description.]"
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that describes images for visually impaired users.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this screenshot of a webpage. Focus on the visible text, layout, and any interactive elements. What information is presented here? Provide a concise summary.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            },
        ]
        logger.debug(
            f"Sending image description request to model: {provider.model}"
        )
        logit_bias_param = (
            LOGIT_BIAS_UNWANTED_TOKENS_STR if supports_logit_bias("vision") else None
        )
        response = await create_chat_completion(
            client,
            prompt_messages,
            model=provider.model,
            max_tokens=(
                config.MAX_COMPLETION_TOKENS_IMAGE_DESCRIPTION
                if hasattr(config, "MAX_COMPLETION_TOKENS_IMAGE_DESCRIPTION")
                else 300
            ),
            temperature=provider.temperature,
            logit_bias=logit_bias_param,
            use_responses_api=provider.use_responses_api,
        )
        description = extract_text(
            response, provider.use_responses_api
        )
        if description:
            logger.info(
                f"Successfully generated description for image {image_path}: {description[:100]}..."
            )
            return description
        logger.warning(
            f"LLM did not return content for image description: {image_path}"
        )
        return "[Error: LLM did not return description for image.]"
    except OpenAIError as e:
        logger.error(
            f"OpenAI API error while generating description for {image_path}: {e}",
            exc_info=True,
        )
        return f"[Error: OpenAI API issue during image description - {type(e).__name__}.]"
    except FileNotFoundError:
        logger.error(
            "Image file not found (should have been caught earlier but as a safeguard): %s",
            image_path,
        )
        return "[Error: Image file not found for description (safeguard).]"
    except Exception as e:
        logger.error(
            f"Unexpected error generating description for image {image_path}: {e}",
            exc_info=True,
        )
        return f"[Error: Unexpected issue during image description - {type(e).__name__}.]"

