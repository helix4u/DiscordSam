import asyncio
import logging
import base64
import os
from typing import List, Any, Optional, Union, Tuple, cast
import discord
from openai import AsyncStream, OpenAIError # type: ignore
from datetime import datetime

# Assuming config is imported from config.py
from config import config
from state import BotState
# Import MsgNode from the new common_models.py
from common_models import MsgNode
# Import utility functions
from utils import chunk_text, cleanup_playwright_processes
# Import functions for post-stream processing
from rag_chroma_manager import ingest_conversation_to_chromadb
from audio_utils import send_tts_audio


logger = logging.getLogger(__name__)


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
    max_image_history_depth: int = 1
) -> List[MsgNode]:
    prompt_list: List[MsgNode] = [get_system_prompt()]

    if config.USER_PROVIDED_CONTEXT:
        prompt_list.append(MsgNode(role="system", content=f"User-Set Global Context:\n{config.USER_PROVIDED_CONTEXT}"))

    if synthesized_rag_context_str:
        context_text_for_prompt = (
            "The following is a synthesized summary of potentially relevant past conversations, "
            "dynamically tailored to your current query. Use it to provide a more informed response.\n\n"
            "--- Synthesized Relevant Context ---\n"
            + synthesized_rag_context_str +
            "\n--- End Synthesized Context ---"
        )
        prompt_list.append(MsgNode(role="system", content=context_text_for_prompt))

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
    is_vision_request: bool
) -> Tuple[Optional[AsyncStream], List[MsgNode]]:
    if not prompt_messages:
        raise ValueError("Prompt messages cannot be empty for get_simplified_llm_stream.")

    logger.info(f"Streaming final response. Vision request: {is_vision_request}")

    final_stream_model = config.VISION_LLM_MODEL if is_vision_request else config.LLM_MODEL
    logger.info(f"Using model for final streaming response: {final_stream_model}")
    try:
        api_messages = []
        for msg_node in prompt_messages:
            api_messages.append(msg_node.to_dict())

        final_llm_stream = await llm_client.chat.completions.create(
            model=final_stream_model,
            messages=api_messages,
            max_tokens=config.MAX_COMPLETION_TOKENS, stream=True, temperature=0.7,
        )
        return final_llm_stream, prompt_messages
    except Exception as e:
        logger.error(f"Failed to create LLM stream for final response with model {final_stream_model}: {e}", exc_info=True)
        try:
            for i, msg in enumerate(prompt_messages):
                content_detail = str(msg.content)
                if isinstance(msg.content, list):
                    content_detail = f"List of {len(msg.content)} parts: {[item.get('type') if isinstance(item,dict) else type(item) for item in msg.content]}"
                logger.error(f"Problematic prompt message [{i}]: Role='{msg.role}', ContentType='{type(msg.content)}', Content='{content_detail[:500]}'")
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
                current_initial_message = await interaction.followup.send(embed=placeholder_embed, wait=True)
            else:
                logger.error("_stream_llm_handler: initial_message_to_edit is None for non-interaction type where it's expected.")
                return "", prompt_messages
        except discord.HTTPException as e:
            logger.error(f"Failed to send initial followup/defer for stream '{title}': {e}")
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
        for p_node in final_prompt_for_llm_call:
            if isinstance(p_node.content, list):
                for content_item in p_node.content:
                    if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                        is_vision_request = True
                        break
            if is_vision_request:
                break
        logger.info(f"Determined is_vision_request for '{title}': {is_vision_request}")

        stream, final_prompt_used_by_llm = await get_simplified_llm_stream(
            llm_client, final_prompt_for_llm_call, is_vision_request
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
                await sent_messages[0].edit(embed=error_embed)
            return "", final_prompt_for_rag

        last_edit_time = asyncio.get_event_loop().time()
        accumulated_delta_for_update = ""

        if current_initial_message:
            initial_display_embed = discord.Embed(title=title, description=response_prefix + "⏳ Streaming response...", color=config.EMBED_COLOR["incomplete"])
            try:
                await current_initial_message.edit(embed=initial_display_embed)
            except discord.HTTPException as e:
                logger.warning(f"Failed to edit initial message with context prefix for '{title}': {e}")

        async for chunk_data in stream:
            delta_content = ""
            if chunk_data.choices and chunk_data.choices[0].delta:
                delta_content = chunk_data.choices[0].delta.content or ""

            if delta_content:
                full_response_content += delta_content
                accumulated_delta_for_update += delta_content

            current_time = asyncio.get_event_loop().time()
            if accumulated_delta_for_update and \
               (current_time - last_edit_time >= (1.0 / config.EDITS_PER_SECOND) or \
                len(accumulated_delta_for_update) > 200):

                display_text = response_prefix + full_response_content
                text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
                accumulated_delta_for_update = ""

                for i, chunk_content_part in enumerate(text_chunks):
                    embed = discord.Embed(
                        title=title if i == 0 else f"{title} (cont.)",
                        description=chunk_content_part,
                        color=config.EMBED_COLOR["incomplete"]
                    )
                    try:
                        if i < len(sent_messages):
                            await sent_messages[i].edit(embed=embed)
                        else:
                            if channel:
                                new_msg = await channel.send(embed=embed)
                                sent_messages.append(new_msg)
                            else:
                                logger.error(f"Cannot send overflow chunk {i+1} for '{title}': channel is None.")
                                break
                    except discord.HTTPException as e_edit_send:
                        logger.warning(f"Failed edit/send embed part {i+1} (stream) for '{title}': {e_edit_send}")
                last_edit_time = current_time
                await asyncio.sleep(config.STREAM_EDIT_THROTTLE_SECONDS)

        if accumulated_delta_for_update:
            display_text = response_prefix + full_response_content
            text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
            for i, chunk_content_part in enumerate(text_chunks):
                embed = discord.Embed(
                    title=title if i == 0 else f"{title} (cont.)",
                    description=chunk_content_part,
                    color=config.EMBED_COLOR["incomplete"]
                )
                try:
                    if i < len(sent_messages):
                        await sent_messages[i].edit(embed=embed)
                    elif channel:
                        sent_messages.append(await channel.send(embed=embed))
                except discord.HTTPException as e:
                    logger.warning(f"Failed final accumulated content edit/send for '{title}': {e}")
            await asyncio.sleep(config.STREAM_EDIT_THROTTLE_SECONDS)

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
                await sent_messages[i].edit(embed=embed)
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
            await sent_messages[0].edit(embed=discord.Embed(title=title, description=empty_response_text, color=config.EMBED_COLOR["error"]))

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
    bot_user_id: Optional[int] = None
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
                interaction, full_response_content, f"interaction_{tts_base_id}"
            )

            chroma_ingest_history_with_response = list(final_prompt_for_rag)
            chroma_ingest_history_with_response.append(assistant_response_node)
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

        await send_tts_audio(interaction, full_response_content, f"interaction_{tts_base_id}")

        chroma_ingest_history_with_response = list(final_prompt_for_rag)
        chroma_ingest_history_with_response.append(assistant_response_node)

    if assistant_response_node:
        await ingest_conversation_to_chromadb(
            llm_client,
            channel_id,
            interaction.user.id,
            chroma_ingest_history_with_response,
        )


async def stream_llm_response_to_message(
    target_message: discord.Message,
    llm_client: Any,
    bot_state: BotState,
    user_msg_node: MsgNode,
    prompt_messages: List[MsgNode],
    title: str = "Sam's Response",
    synthesized_rag_context_for_display: Optional[str] = None,
    bot_user_id: Optional[int] = None
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
        )

        chroma_ingest_history_with_response = list(final_prompt_for_rag)
        chroma_ingest_history_with_response.append(assistant_response_node)

    if assistant_response_node:
        await ingest_conversation_to_chromadb(
            llm_client,
            channel_id,
            target_message.author.id,
            chroma_ingest_history_with_response,
        )


async def get_description_for_image(llm_client: Any, image_path: str) -> str:
    """
    Generates a textual description for a given image using a vision-capable LLM.
    """
    logger.info(f"Attempting to generate description for image: {image_path}")
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return "[Error: Image file not found for description.]"

        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        b64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Ensure VISION_LLM_MODEL is appropriate for non-streaming, single image description
        # It might be the same as config.VISION_LLM_MODEL or a specific one if needed.
        # For now, we assume config.VISION_LLM_MODEL can handle this.
        if not config.VISION_LLM_MODEL:
            logger.error("VISION_LLM_MODEL is not configured. Cannot describe image.")
            return "[Error: Vision model not configured for image description.]"

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this screenshot of a webpage. Focus on the visible text, layout, and any interactive elements. What information is presented here? Provide a concise summary."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ]

        logger.debug(f"Sending image description request to model: {config.VISION_LLM_MODEL}")
        response = await llm_client.chat.completions.create(
            model=config.VISION_LLM_MODEL,
            messages=prompt_messages,
            max_tokens=config.MAX_COMPLETION_TOKENS_IMAGE_DESCRIPTION if hasattr(config, 'MAX_COMPLETION_TOKENS_IMAGE_DESCRIPTION') else 300, # Use a specific max_tokens for descriptions
            temperature=0.3 # Lower temperature for more factual descriptions
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            description = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated description for image {image_path}: {description[:100]}...")
            return description
        else:
            logger.warning(f"LLM did not return content for image description: {image_path}")
            return "[Error: LLM did not return description for image.]"

    except OpenAIError as e:
        logger.error(f"OpenAI API error while generating description for {image_path}: {e}", exc_info=True)
        return f"[Error: OpenAI API issue during image description - {type(e).__name__}.]"
    except FileNotFoundError:
        logger.error(f"Image file not found (should have been caught earlier but as a safeguard): {image_path}")
        return "[Error: Image file not found for description (safeguard).]"
    except Exception as e:
        logger.error(f"Unexpected error generating description for image {image_path}: {e}", exc_info=True)
        return f"[Error: Unexpected issue during image description - {type(e).__name__}.]"
