import asyncio
import io
import os
import logging
from typing import Optional, Union, Any 
import re # Ensures 're' is available

import discord
import aiohttp
from pydub import AudioSegment
import torch 
import whisper 
import gc

from config import config
from utils import clean_text_for_tts

logger = logging.getLogger(__name__)

# Global lock to ensure TTS requests are processed sequentially.
TTS_LOCK = asyncio.Lock()

WHISPER_MODEL: Optional[Any] = None

def load_whisper_model() -> Optional[Any]: 
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        try:
            logger.info("Loading Whisper model...")
            WHISPER_MODEL = whisper.load_model("large-v3-turbo") 
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Whisper model: {e}", exc_info=True)
            WHISPER_MODEL = None 
    return WHISPER_MODEL

async def tts_request(text: str, speed: Optional[float] = None) -> Optional[bytes]:
    if not text:
        return None
    if speed is None:
        speed = config.TTS_SPEED
    payload = {
        "input": text,
        "voice": config.TTS_VOICE,
        "response_format": "mp3",
        "speed": speed,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(config.TTS_API_URL, json=payload, timeout=90) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    logger.error(f"TTS request failed: status={resp.status}, response_text={await resp.text()}")
                    return None
    except asyncio.TimeoutError:
        logger.error("TTS request timed out.")
        return None
    except Exception as e:
        logger.error(f"TTS request error: {e}", exc_info=True)
        return None

async def _send_audio_segment(
    destination: Union[discord.abc.Messageable, discord.Interaction, discord.Message], 
    segment_text: str, 
    filename_suffix: str, 
    is_thought: bool = False,
    base_filename: str = "response"
):
    if not segment_text:
        return

    cleaned_segment = clean_text_for_tts(segment_text)
    if not cleaned_segment:
        logger.info(f"Skipping TTS for empty or fully cleaned '{filename_suffix}' segment.")
        return

    tts_audio_data = await tts_request(cleaned_segment)
    
    actual_destination_channel: Optional[discord.abc.Messageable] = None
    if isinstance(destination, discord.Interaction):
        if isinstance(destination.channel, discord.abc.Messageable):
            actual_destination_channel = destination.channel
        else:
            logger.warning(f"TTS: Interaction channel for {destination.id} is not Messageable.")
            return
    elif isinstance(destination, discord.Message):
        actual_destination_channel = destination.channel 
    elif isinstance(destination, discord.abc.Messageable): 
        actual_destination_channel = destination
    
    if not actual_destination_channel:
        logger.warning(f"TTS destination channel could not be resolved for type {type(destination)}")
        return

    if tts_audio_data:
        try:
            audio = AudioSegment.from_file(io.BytesIO(tts_audio_data), format="mp3")
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format="mp3", bitrate="128k")
            fixed_audio_data = output_buffer.getvalue()

            if len(fixed_audio_data) <= config.TTS_MAX_AUDIO_BYTES:
                file = discord.File(io.BytesIO(fixed_audio_data), filename=f"{base_filename}_{filename_suffix}.mp3")
                content_message = None
                if is_thought:
                    content_message = "**Sam's thoughts (TTS):**"
                elif filename_suffix in ["main_response", "full"]:
                    content_message = "**Sam's response (TTS):**"

                await actual_destination_channel.send(content=content_message, file=file)
                logger.info(f"Sent TTS audio: {base_filename}_{filename_suffix}.mp3 to Channel ID {actual_destination_channel.id}")
            else:
                bytes_per_second = 16000  # 128 kbps
                max_duration_ms = int((config.TTS_MAX_AUDIO_BYTES / bytes_per_second) * 1000)
                segments = [
                    audio[i : i + max_duration_ms]
                    for i in range(0, len(audio), max_duration_ms)
                ]
                logger.info(
                    "Audio segment '%s' exceeds size limit (%d > %d). Splitting into %d parts.",
                    filename_suffix,
                    len(fixed_audio_data),
                    config.TTS_MAX_AUDIO_BYTES,
                    len(segments),
                )

                for idx, segment in enumerate(segments, start=1):
                    part_buffer = io.BytesIO()
                    segment.export(part_buffer, format="mp3", bitrate="128k")
                    part_data = part_buffer.getvalue()
                    part_file = discord.File(
                        io.BytesIO(part_data),
                        filename=f"{base_filename}_{filename_suffix}_part{idx}.mp3",
                    )
                    content_message = None
                    if idx == 1:
                        if is_thought:
                            content_message = "**Sam's thoughts (TTS):**"
                        elif filename_suffix in ["main_response", "full"]:
                            content_message = "**Sam's response (TTS):**"
                    await actual_destination_channel.send(content=content_message, file=part_file)
                    logger.info(
                        "Sent TTS audio part %d/%d: %s_%s_part%d.mp3 to Channel ID %s",
                        idx,
                        len(segments),
                        base_filename,
                        filename_suffix,
                        idx,
                        actual_destination_channel.id,
                    )
                    await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error processing or sending TTS audio for '{filename_suffix}': {e}", exc_info=True)
    else:
        logger.warning(f"TTS request failed for '{filename_suffix}' segment, no audio data received.")

async def send_tts_audio(
    destination: Union[discord.abc.Messageable, discord.Interaction, discord.Message],
    text_to_speak: str,
    base_filename: str = "response"
) -> None:
    """Generate TTS audio and send it to the given destination.

    The global ``TTS_LOCK`` ensures that only one TTS request is processed at a
    time so audio playback doesn't overlap when multiple commands trigger TTS
    concurrently.
    """
    if not config.TTS_ENABLED_DEFAULT or not text_to_speak:
        return

    async with TTS_LOCK:
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        match = think_pattern.search(text_to_speak)

        if match:
            thought_text = match.group(1).strip()
            response_text = think_pattern.sub('', text_to_speak).strip()

            if config.TTS_INCLUDE_THOUGHTS and thought_text:
                logger.info("Found <think> tags. Sending thoughts and response for TTS.")
                await _send_audio_segment(destination, thought_text, "thoughts", is_thought=True, base_filename=base_filename)
                await asyncio.sleep(0.5)
            else:
                logger.info("Found <think> tags. Skipping thoughts for TTS as per config.")

            if response_text:
                await _send_audio_segment(destination, response_text, "main_response" if config.TTS_INCLUDE_THOUGHTS else "full", is_thought=False, base_filename=base_filename)
            else:
                logger.info("No user-facing content left for TTS after removing <think> section.")
        else:
            logger.info("No <think> tags found. Processing full text for TTS.")
            await _send_audio_segment(destination, text_to_speak, "full", is_thought=False, base_filename=base_filename)

def transcribe_audio_file(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found for transcription: {file_path}")
        return None
    
    if WHISPER_MODEL is None:
        logger.error("Whisper model is not loaded. Cannot transcribe audio.")
        if load_whisper_model() is None: 
             return None

    try:
        logger.info(f"Transcribing audio file: {file_path}")
        use_fp16 = torch.cuda.is_available()
        result = WHISPER_MODEL.transcribe(file_path, fp16=use_fp16) # type: ignore
        transcribed_text = result.get("text") if isinstance(result, dict) else None
        if transcribed_text:
             logger.info(f"Transcription successful for {file_path}.")
             return str(transcribed_text) 
        else:
            logger.warning(f"Whisper transcription for {file_path} did not return 'text'. Result: {result}")
            return None
    except Exception as e:
        logger.error(f"Whisper transcription failed for {file_path}: {e}", exc_info=True)
        return None
    finally:
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
