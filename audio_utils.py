import asyncio
import io
import os
import logging
import random
import tempfile
import textwrap
from pathlib import Path
from typing import Optional, Union, Any, TYPE_CHECKING
import re  # Ensures 're' is available
import threading
import discord
import aiohttp
from pydub import AudioSegment
import torch
import whisper
import gc
from config import config
from utils import clean_text_for_tts
if TYPE_CHECKING:
    from state import BotState
logger = logging.getLogger(__name__)
# Global lock to ensure TTS requests are processed sequentially.
TTS_LOCK = asyncio.Lock()
_MISSING_FONT_WARNING_EMITTED = False
def _create_rolling_subtitles(text: str, duration_seconds: float, width: int = 1280, height: int = 720, font_size: int = 42) -> str:
    """Create rolling subtitle ASS content with centered, faded effect.
    
    Args:
        text: The full text to display
        duration_seconds: Total duration of the audio/video
        width: Video width in pixels
        height: Video height in pixels
        font_size: Font size being used
        
    Returns:
        ASS content with time-synced rolling subtitles with proper styling
    """
    normalized = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
    
    # Calculate approximate characters per line based on video width
    usable_width = width - 96  # Account for margins
    chars_per_line = int(usable_width / (font_size * 0.6))
    chars_per_line = max(60, min(chars_per_line, 120))  # Double the words per segment
    
    # Max lines visible at once for rolling effect (more for faded context)
    max_visible_lines = 8
    
    # Split text into chunks that fit on one line
    words = normalized.split()
    line_chunks = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > chars_per_line and current_line:
            line_chunks.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length
    
    if current_line:
        line_chunks.append(" ".join(current_line))
    
    if not line_chunks:
        return "1\n00:00:00,000 --> 00:00:01,000\n \n"
    
    # Calculate timing for each chunk
    time_per_chunk = duration_seconds / len(line_chunks)
    # Ensure each chunk shows for at least 0.8 seconds for smooth reading
    time_per_chunk = max(2.0, time_per_chunk)  # Double time for double the words
    
    # Calculate margins for maximum screen usage (vision aid style)
    margin_lr = int(width * 0.03)  # Minimal 3% horizontal margin
    margin_v = int(height * 0.05)  # Minimal 5% vertical margin
    
    # Create ASS header with maximum screen usage styling
    # ASS color format: &HAABBGGRR (alpha, blue, green, red in hex)
    # ScaleX: 110 = wider/fuller text, ScaleY: 100 = normal height, Spacing: 0 = no extra char spacing
    # Alignment: 2 = bottom center
    ass_header = f"""[Script Info]
Title: Rolling Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Current,Arial Black,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,110,100,0,0,1,6,0,2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    def format_ass_time(seconds: float) -> str:
        """Format time for ASS format (H:MM:SS.CC)"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    
    ass_events = []
    for i, chunk in enumerate(line_chunks):
        start_time = i * time_per_chunk
        end_time = min((i + 1) * time_per_chunk, duration_seconds)
        
        start_str = format_ass_time(start_time)
        end_str = format_ass_time(end_time)
        
        # Build the subtitle text with inline styling for faded effect
        text_parts = []
        
        for j in range(max(0, i - 4), i):
            if j < len(line_chunks):
                text_parts.append(f"{{\\alpha&H00&\\1c&HFFFFFF&}}{line_chunks[j]}")
                text_parts.append(" ")  # Add blank line for more vertical spacing
        
        current_line_size = int(font_size * 1.0)  # Even bigger for vision aid
        text_parts.append(f"{{\\alpha&H00&\\1c&HFFFFFF&\\b1\\fs{current_line_size}}}{chunk}{{\\fs{font_size}\\b0}}")
        text_parts.append(" ")  # Add blank line after current for spacing
        
        for j in range(i + 1, min(i + 3, len(line_chunks))):
            text_parts.append(f"{{\\alpha&H00&\\1c&HFFFFFF&}}{line_chunks[j]}")
            text_parts.append(" ")  # Add blank line for more vertical spacing
        
        # Join with \\N (line break in ASS)
        dialogue_text = "\\N".join(text_parts)
        ass_events.append(f"Dialogue: 0,{start_str},{end_str},Current,,0,0,0,,{dialogue_text}")
    
    return ass_header + "\n".join(ass_events)
def _escape_subtitles_path(value: str) -> str:
    """Escape a filesystem path for use inside ffmpeg subtitle filters."""
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\\\:")
    return escaped.replace("'", "\\'")
def _escape_force_style(value: str) -> str:
    """Escape force_style values for ffmpeg subtitles filter."""
    return value.replace("'", "\\'")

async def _chunk_video_with_ffmpeg(video_bytes: bytes, max_bytes: int, base_filename: str, filename_suffix: str) -> list[tuple[bytes, str]]:
    """Chunk a video file using ffmpeg to fit size restrictions.
    
    Args:
        video_bytes: The full video data
        max_bytes: Maximum bytes per chunk
        base_filename: Base filename for chunks
        filename_suffix: Suffix for the file type
        
    Returns:
        List of tuples (chunk_bytes, chunk_filename)
    """
    import tempfile
    from pathlib import Path
    
    # Estimate duration from file size (rough approximation)
    # Assume ~500kbps average bitrate for MP4
    estimated_duration = len(video_bytes) / (500 * 1024)  # seconds
    bytes_per_second = len(video_bytes) / estimated_duration if estimated_duration > 0 else len(video_bytes)
    
    # Calculate how many chunks we need
    num_chunks = max(1, int(len(video_bytes) / max_bytes) + 1)
    chunk_duration = estimated_duration / num_chunks
    
    logger.info("Chunking video: %d bytes, estimated %.1f seconds, %d chunks of %.1f seconds each", 
               len(video_bytes), estimated_duration, num_chunks, chunk_duration)
    
    chunks = []
    
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        input_video = tmp_dir / "input.mp4"
        input_video.write_bytes(video_bytes)
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, estimated_duration)
            chunk_filename = f"{base_filename}_{filename_suffix}_part{i + 1}.mp4"
            output_video = tmp_dir / f"chunk_{i + 1}.mp4"
            
            # Use ffmpeg to extract a time segment
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(input_video),
                "-ss", str(start_time),
                "-t", str(end_time - start_time),
                "-c", "copy",  # Copy streams without re-encoding
                "-avoid_negative_ts", "make_zero",
                str(output_video)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error("ffmpeg chunking failed for chunk %d: %s", i + 1, 
                           stderr.decode("utf-8", errors="ignore") if stderr else "No error output")
                continue
                
            if output_video.exists():
                chunk_bytes = output_video.read_bytes()
                chunks.append((chunk_bytes, chunk_filename))
                logger.info("Created video chunk %d: %d bytes", i + 1, len(chunk_bytes))
            else:
                logger.error("Video chunk %d was not created", i + 1)
    
    return chunks
def _css_hex_to_ass_color(hex_color: str | None, default_ass: str) -> str:
    """Convert a CSS-style hex color (optionally with alpha) to ASS colour format."""
    if not hex_color:
        return default_ass
    value = hex_color.strip().lstrip("#")
    if len(value) not in {6, 8}:
        return default_ass
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        css_alpha = 255
        if len(value) == 8:
            css_alpha = int(value[6:8], 16)
        css_alpha = max(0, min(255, css_alpha))
        ass_alpha = 255 - css_alpha  # ASS uses inverse alpha
        ass_alpha = max(0, min(255, ass_alpha))
        return f"&H{ass_alpha:02X}{b:02X}{g:02X}{r:02X}&"
    except ValueError:
        return default_ass
def _format_srt_timestamp(duration_ms: int) -> str:
    """Return an SRT timestamp given a duration in milliseconds."""
    if duration_ms <= 0:
        return "00:00:01,000"
    hours = duration_ms // 3_600_000
    remainder = duration_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    seconds = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
async def _generate_tts_video(
    audio_bytes: bytes,
    audio_segment: AudioSegment,
    display_text: str,
    output_base: str,
) -> Optional[bytes]:
    """Render an MP4 with a solid background and burned-in subtitles."""
    if not audio_bytes:
        return None
    
    # Get video dimensions first
    # Calculate duration first (needed for rolling subtitles)
    duration_ms = int(len(audio_segment))
    duration_seconds = duration_ms / 1000.0
    
    width = max(320, int(getattr(config, "TTS_VIDEO_WIDTH", 1280)))
    height = max(320, int(getattr(config, "TTS_VIDEO_HEIGHT", 720)))
    fps = max(1, int(getattr(config, "TTS_VIDEO_FPS", 30)))
    font_size = max(12, int(getattr(config, "TTS_VIDEO_FONT_SIZE", 120)))  # Much larger for vision aid
    
    # Create rolling subtitles based on audio duration
    logger.info("Creating rolling subtitles for %.2f second video (%dx%d, font size %d)", 
               duration_seconds, width, height, font_size)
    # Force black background for centered, faded text effect
    background_color = "#000000"  # Pure black background
    text_color_hex = getattr(config, "TTS_VIDEO_TEXT_COLOR", "#FFFFFF")
    # Grey shadow color (semi-transparent grey)
    shadow_color_hex = "#808080CC"
    margin_v = max(20, int(getattr(config, "TTS_VIDEO_MARGIN", 96)))
    margin_lr = max(20, int(getattr(config, "TTS_VIDEO_TEXT_BOX_PADDING", 48)))
    font_path = getattr(config, "TTS_VIDEO_FONT_PATH", "")
    font_name = Path(font_path).stem if font_path else "Arial"
    if not font_name:
        font_name = "Arial"
    font_name = font_name.replace("'", "")
    text_color_ass = _css_hex_to_ass_color(text_color_hex, "&H00FFFFFF&")
    shadow_color_ass = _css_hex_to_ass_color(shadow_color_hex, "&HCC808080&")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            audio_path = tmp_dir / "input.mp3"
            video_path = tmp_dir / f"{output_base}.mp4"
            subtitle_path = tmp_dir / "captions.ass"
            audio_path.write_bytes(audio_bytes)
            logger.debug("Wrote audio file: %s (%d bytes)", audio_path, len(audio_bytes))
            logger.info("Audio file created for video generation: %d bytes, duration: %.2f seconds", 
                       len(audio_bytes), duration_seconds)
            
            # First, transcode the audio to a clean AAC format that works with x264/Discord
            transcoded_audio_path = tmp_dir / "audio_transcoded.aac"
            logger.info("Transcoding audio to AAC for compatibility...")
            transcode_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(transcoded_audio_path),
            ]
            transcode_process = await asyncio.create_subprocess_exec(
                *transcode_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            transcode_stdout, transcode_stderr = await transcode_process.communicate()
            
            if transcode_process.returncode != 0:
                logger.error("Audio transcoding failed: %s", 
                           transcode_stderr.decode("utf-8", errors="ignore") if transcode_stderr else "No error output")
                return None
            
            if not transcoded_audio_path.exists():
                logger.error("Transcoded audio file was not created")
                return None
            
            logger.info("Audio transcoded successfully: %d bytes", transcoded_audio_path.stat().st_size)
            
            # Verify audio file was created and has content
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                logger.error("Failed to create audio file or file is empty")
                return None
            
            # Test if ffmpeg can read the audio file and verify it has content
            try:
                test_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(audio_path)]
                test_process = await asyncio.create_subprocess_exec(
                    *test_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                test_stdout, test_stderr = await test_process.communicate()
                
                if test_process.returncode == 0:
                    import json
                    test_data = json.loads(test_stdout.decode("utf-8"))
                    audio_streams = [s for s in test_data.get("streams", []) if s.get("codec_type") == "audio"]
                    logger.info("Audio file verification: %d audio streams found", len(audio_streams))
                    if len(audio_streams) > 0:
                        stream = audio_streams[0]
                        logger.info("Audio stream: %s, %d Hz, %d channels", 
                                   stream.get("codec_name", "unknown"),
                                   int(stream.get("sample_rate", 0)),
                                   int(stream.get("channels", 0)))
                        
                        # Check if the audio stream has duration
                        duration = stream.get("duration", "0")
                        logger.info("Audio duration: %s seconds", duration)
                        
                        # Check if audio has actual content
                        if duration == "0" or duration == "0.000000":
                            logger.error("Audio file has no duration - this will cause silent video!")
                        else:
                            logger.info("Audio file has valid duration - should be audible in video")
                else:
                    logger.warning("ffprobe failed to read audio file: %s", test_stderr.decode("utf-8", errors="ignore"))
            except Exception as test_exc:
                logger.debug("Could not test audio file with ffprobe: %s", test_exc)
            
            # Generate rolling subtitles in ASS format
            ass_content = _create_rolling_subtitles(display_text, duration_seconds, width, height, font_size)
            subtitle_path.write_text(ass_content, encoding="utf-8")
            
            # Count subtitle entries for logging
            subtitle_count = ass_content.count('Dialogue:')
            logger.info("Created rolling ASS file with %d subtitle entries for %.2f seconds", 
                       subtitle_count, duration_seconds)
            
            # Verify the file was created successfully
            if not subtitle_path.exists():
                logger.error("Failed to create ASS file at %s", subtitle_path)
                return None
            
            # Use exact audio duration for video to ensure perfect sync
            color_source = (
                f"color=c={background_color}:s={width}x{height}:r={fps}:d={duration_seconds:.3f}"
            )
            logger.info("Creating video with duration: %.3f seconds to match audio", duration_seconds)
            style_parts = [
                "Alignment=8",  # Center text horizontally, bottom vertically
                f"Fontname={font_name}",
                f"Fontsize={font_size}",
                "BorderStyle=1",  # Outline with shadow
                "Outline=2",  # Thin outline
                "Shadow=3",  # Grey shadow offset
                f"PrimaryColour={text_color_ass}",  # White text
                "OutlineColour=&H00000000&",  # Black outline
                f"BackColour={shadow_color_ass}",  # Grey shadow
                f"MarginV={margin_v}",
                f"MarginL={margin_lr}",
                f"MarginR={margin_lr}",
            ]
            # For ASS files, we don't need force_style since styles are defined in the file
            vf_filter = f"subtitles=captions.ass"
            logger.debug("Using ffmpeg filter: %s", vf_filter)
            logger.debug("Working directory: %s", tmp_dir)
            logger.debug("Files in tmp_dir: %s", list(tmp_dir.iterdir()))
            # Create video and mux with transcoded audio for perfect sync
            # Step 1: Create video with subtitles (no audio yet)
            video_only_path = tmp_dir / "video_only.mp4"
            logger.info("Step 1: Creating video track with subtitles...")
            video_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                color_source,
                "-vf",
                vf_filter,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_only_path),
            ]
            
            logger.info("Video creation command: %s", " ".join(video_cmd))
            video_process = await asyncio.create_subprocess_exec(
                *video_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(tmp_dir),
            )
            video_stdout, video_stderr = await video_process.communicate()
            
            if video_process.returncode != 0:
                logger.error("Video creation failed: %s", 
                           video_stderr.decode("utf-8", errors="ignore") if video_stderr else "No error output")
                logger.error("Video command was: %s", " ".join(video_cmd))
                return None
            
            # Log any stderr output even on success
            if video_stderr:
                video_stderr_text = video_stderr.decode("utf-8", errors="ignore")
                if video_stderr_text.strip():
                    logger.info("Video creation stderr: %s", video_stderr_text)
            
            if not video_only_path.exists():
                logger.error("Video file was not created")
                return None
            
            logger.info("Video track created successfully: %d bytes", video_only_path.stat().st_size)
            
            # Step 2: Mux video and transcoded audio together
            logger.info("Step 2: Muxing video and audio together...")
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "info",
                "-y",
                # Input video (already encoded)
                "-i",
                str(video_only_path),
                # Input transcoded audio
                "-i",
                str(transcoded_audio_path),
                # Map both streams explicitly
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                # Copy both streams without re-encoding
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                # Use the video duration as reference
                "-fflags",
                "+genpts",
                # Output settings
                "-movflags",
                "+faststart",
                str(video_path),
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(tmp_dir),
            )
            stdout, stderr = await process.communicate()
            logger.info("Muxing command: %s", " ".join(cmd))
            logger.info("Video input: %s", video_only_path)
            logger.info("Audio input: %s", transcoded_audio_path)
            if stdout:
                stdout_text = stdout.decode("utf-8", errors="ignore")
                logger.info("ffmpeg stdout: %s", stdout_text)
            else:
                logger.debug("ffmpeg stdout: None")
            stderr_text: Optional[str] = None
            if stderr:
                try:
                    stderr_text = stderr.decode("utf-8", errors="ignore")
                except Exception:
                    stderr_text = None
            if process.returncode != 0:
                if stderr_text:
                    logger.error("Muxing failed: %s", stderr_text.strip())
                else:
                    logger.error("Muxing failed (no diagnostic output).")
                logger.error("Muxing command was: %s", " ".join(cmd))
                return None
            else:
                logger.info("Muxing completed successfully - video and audio combined")
                # Log any stderr output even on success
                if stderr_text and stderr_text.strip():
                    logger.info("Muxing stderr: %s", stderr_text)
            if not video_path.exists():
                logger.error("ffmpeg reported success but produced no MP4 for '%s'", output_base)
                return None
            
            # Verify the video file was created and has reasonable size
            video_size = video_path.stat().st_size
            logger.debug("Generated video file: %s (%d bytes)", video_path, video_size)
            
            if video_size < 1024:  # Less than 1KB is suspicious
                logger.warning("Generated video file is very small (%d bytes), may not contain audio", video_size)
            
            # Try to verify the video contains audio streams using ffprobe
            try:
                probe_cmd = [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams",
                    str(video_path)
                ]
                probe_process = await asyncio.create_subprocess_exec(
                    *probe_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                probe_stdout, _ = await probe_process.communicate()
                
                if probe_process.returncode == 0 and probe_stdout:
                    import json
                    probe_data = json.loads(probe_stdout.decode("utf-8"))
                    audio_streams = [s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"]
                    video_streams = [s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"]
                    
                    logger.debug("Video contains %d audio streams and %d video streams", 
                                len(audio_streams), len(video_streams))
                    
                    if len(audio_streams) == 0:
                        logger.warning("Generated video has no audio streams!")
                    elif len(video_streams) == 0:
                        logger.warning("Generated video has no video streams!")
                    else:
                        logger.info("Video generation successful - contains %d audio streams and %d video streams", 
                                   len(audio_streams), len(video_streams))
                        
                        # Log video stream details
                        if len(video_streams) > 0:
                            video_stream = video_streams[0]
                            logger.info("Video stream: %s, %dx%d, %s", 
                                       video_stream.get("codec_name", "unknown"),
                                       video_stream.get("width", 0),
                                       video_stream.get("height", 0),
                                       video_stream.get("duration", "unknown"))
                        
                        # Log audio stream details
                        if len(audio_streams) > 0:
                            audio_stream = audio_streams[0]
                            logger.info("Audio stream: %s, %d Hz, %d channels, %s duration", 
                                       audio_stream.get("codec_name", "unknown"),
                                       int(audio_stream.get("sample_rate", 0)),
                                       int(audio_stream.get("channels", 0)),
                                       audio_stream.get("duration", "unknown"))
                        
            except Exception as probe_exc:
                logger.debug("Could not probe video streams: %s", probe_exc)
            
            return video_path.read_bytes()
    except FileNotFoundError:
        logger.error("ffmpeg binary not found on PATH. Cannot render TTS video.")
        return None
    except Exception as exc:
        logger.error("Unexpected error while generating TTS video: %s", exc, exc_info=True)
        return None
    return None
WHISPER_MODEL: Optional[Any] = None
WHISPER_UNLOAD_TIMER: Optional[threading.Timer] = None
WHISPER_TTL_SECONDS = 60
WHISPER_LOCK = threading.Lock()
def _unload_whisper_model() -> None:
    """Unload the Whisper model to free VRAM."""
    global WHISPER_MODEL, WHISPER_UNLOAD_TIMER
    with WHISPER_LOCK:
        if WHISPER_MODEL is not None:
            WHISPER_MODEL = None
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            logger.info("Whisper model unloaded due to inactivity.")
        WHISPER_UNLOAD_TIMER = None
def _schedule_whisper_unload() -> None:
    """Schedule unloading of the Whisper model after the TTL."""
    global WHISPER_UNLOAD_TIMER
    if WHISPER_UNLOAD_TIMER is not None:
        WHISPER_UNLOAD_TIMER.cancel()
    WHISPER_UNLOAD_TIMER = threading.Timer(WHISPER_TTL_SECONDS, _unload_whisper_model)
    WHISPER_UNLOAD_TIMER.daemon = True
    WHISPER_UNLOAD_TIMER.start()
def load_whisper_model() -> Optional[Any]:
    """Load the Whisper model on demand and schedule its unloading."""
    global WHISPER_MODEL
    with WHISPER_LOCK:
        if WHISPER_MODEL is None:
            try:
                device = config.WHISPER_DEVICE
                if device:
                    logger.info(f"Loading Whisper model onto specified device: {device}...")
                else:
                    logger.info("Loading Whisper model with auto-device-detection...")
                WHISPER_MODEL = whisper.load_model("large-v3-turbo", device=device)
                logger.info("Whisper model loaded successfully.")
            except Exception as e:
                logger.critical(f"Failed to load Whisper model: {e}", exc_info=True)
                WHISPER_MODEL = None
        _schedule_whisper_unload()
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
    logger.info(f"Requesting TTS for {len(text)} characters.")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(config.TTS_API_URL, json=payload, timeout=config.TTS_REQUEST_TIMEOUT_SECONDS) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    logger.error(f"TTS request failed: status={resp.status}, response_text={await resp.text()}")
                    return None
    except asyncio.TimeoutError:
        logger.error(f"TTS request timed out after {config.TTS_REQUEST_TIMEOUT_SECONDS} seconds for {len(text)} characters.")
        return None
    except Exception as e:
        logger.error(f"TTS request error: {e}", exc_info=True)
        return None
async def _send_audio_segment(
    destination: Union[discord.abc.Messageable, discord.Interaction, discord.Message], 
    segment_text: str, 
    filename_suffix: str, 
    is_thought: bool = False,
    base_filename: str = "response",
    delivery_mode: str = "audio",
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
            logger.info("Received Kokoro TTS audio: %d bytes", len(tts_audio_data))
            audio = AudioSegment.from_file(io.BytesIO(tts_audio_data), format="mp3")
            logger.info("Loaded Kokoro TTS audio: %d ms duration, %d channels, %d Hz", 
                       len(audio), audio.channels, audio.frame_rate)
            
            # Ensure audio is in the right format for video generation
            # Convert to stereo if mono, and ensure consistent sample rate
            if audio.channels == 1:
                audio = audio.set_channels(2)  # Convert mono to stereo
            if audio.frame_rate != 44100:
                audio = audio.set_frame_rate(44100)  # Ensure 44.1kHz sample rate
            
            # Log original audio volume
            logger.info("Original Kokoro TTS audio: max_dBFS=%.1f, channels=%d, frame_rate=%d", 
                       audio.max_dBFS, audio.channels, audio.frame_rate)
            
            # Only apply minimal volume adjustment if needed
            if audio.max_dBFS == float('-inf') or audio.max_dBFS < -50:
                logger.warning("Audio appears to be very quiet (max_dBFS=%.1f), applying small boost", audio.max_dBFS)
                audio = audio + 10  # Small boost only
            elif audio.max_dBFS < -30:
                logger.info("Audio is quiet but audible, applying minimal boost")
                audio = audio + 5  # Minimal boost
            
            # Export directly as MP3 to preserve the original Kokoro TTS quality
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format="mp3", bitrate="192k")
            fixed_audio_data = output_buffer.getvalue()
            
            # Recreate the AudioSegment from the fixed data to ensure they match
            audio = AudioSegment.from_file(io.BytesIO(fixed_audio_data), format="mp3")
            
            logger.info("Processed audio for video: %d channels, %d Hz, %d bytes, max_dBFS=%.1f", 
                        audio.channels, audio.frame_rate, len(fixed_audio_data), audio.max_dBFS)
            normalized_mode = delivery_mode.lower()
            send_audio_files = normalized_mode in {"audio", "both"}
            send_video_file = normalized_mode in {"video", "both"}
            content_message: Optional[str] = None
            if is_thought:
                content_message = "**Sam's thoughts (TTS):**"
            elif filename_suffix in ["main_response", "full"]:
                content_message = "**Sam's response (TTS):**"
            content_sent = False
            if send_video_file:
                logger.info("Generating video with Kokoro TTS audio: %d bytes", len(fixed_audio_data))
                video_bytes = await _generate_tts_video(
                    fixed_audio_data,
                    audio,
                    cleaned_segment,
                    f"{base_filename}_{filename_suffix}",
                )
                if video_bytes is None:
                    logger.warning(
                        "Failed to create TTS video for %s_%s; falling back to audio.",
                        base_filename,
                        filename_suffix,
                    )
                    if normalized_mode == "video":
                        send_audio_files = True
                    send_video_file = False
                elif len(video_bytes) > config.TTS_MAX_VIDEO_BYTES:
                    logger.info(
                        "Generated TTS video exceeds size limit (%d > %d) for %s_%s. Chunking video.",
                        len(video_bytes),
                        config.TTS_MAX_VIDEO_BYTES,
                        base_filename,
                        filename_suffix,
                    )
                    
                    # Chunk the video using ffmpeg
                    video_chunks = await _chunk_video_with_ffmpeg(
                        video_bytes, 
                        config.TTS_MAX_VIDEO_BYTES, 
                        base_filename, 
                        filename_suffix
                    )
                    
                    if video_chunks:
                        logger.info("Sending %d video chunks", len(video_chunks))
                        for chunk_bytes, chunk_filename in video_chunks:
                            video_file = discord.File(
                                io.BytesIO(chunk_bytes),
                                filename=chunk_filename,
                            )
                            await actual_destination_channel.send(
                                content=content_message if not content_sent else None,
                                file=video_file,
                            )
                            content_sent = True
                            logger.info(
                                "Sent TTS video chunk: %s to Channel ID %s",
                                chunk_filename,
                                actual_destination_channel.id,
                            )
                    else:
                        logger.warning("Video chunking failed, falling back to audio")
                    send_video_file = False
                    if normalized_mode == "video":
                        send_audio_files = True
                else:
                    video_file = discord.File(
                        io.BytesIO(video_bytes),
                        filename=f"{base_filename}_{filename_suffix}.mp4",
                    )
                    await actual_destination_channel.send(
                        content=content_message if not content_sent else None,
                        file=video_file,
                    )
                    content_sent = True
                    logger.info(
                        "Sent TTS video: %s_%s.mp4 to Channel ID %s",
                        base_filename,
                        filename_suffix,
                        actual_destination_channel.id,
                    )
            if send_audio_files:
                if len(fixed_audio_data) <= config.TTS_MAX_AUDIO_BYTES:
                    file = discord.File(
                        io.BytesIO(fixed_audio_data),
                        filename=f"{base_filename}_{filename_suffix}.mp3",
                    )
                    await actual_destination_channel.send(
                        content=content_message if not content_sent else None,
                        file=file,
                    )
                    content_sent = True
                    logger.info(
                        "Sent TTS audio: %s_%s.mp3 to Channel ID %s",
                        base_filename,
                        filename_suffix,
                        actual_destination_channel.id,
                    )
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
                        part_content = None
                        if not content_sent:
                            part_content = content_message
                        await actual_destination_channel.send(
                            content=part_content,
                            file=part_file,
                        )
                        content_sent = True
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
    base_filename: str = "response",
    *,
    bot_state: Optional["BotState"] = None,
    delivery_mode: Optional[str] = None,
) -> None:
    """Generate TTS audio and send it to the given destination.
    The global ``TTS_LOCK`` ensures that only one TTS request is processed at a
    time so audio playback doesn't overlap when multiple commands trigger TTS
    concurrently.
    """
    if not config.TTS_ENABLED_DEFAULT or not text_to_speak:
        return
    resolved_mode = delivery_mode
    guild_id: Optional[int] = None
    if isinstance(destination, discord.Interaction):
        guild_id = destination.guild_id
    elif isinstance(destination, discord.Message):
        if destination.guild:
            guild_id = destination.guild.id
    elif isinstance(destination, discord.abc.Messageable):
        guild = getattr(destination, "guild", None)
        if guild:
            guild_id = getattr(guild, "id", None)
    if resolved_mode is None and bot_state and guild_id is not None:
        try:
            resolved_mode = await bot_state.get_tts_delivery_mode(guild_id)
        except Exception as exc:
            logger.error(
                "Failed to fetch TTS delivery mode for guild %s: %s",
                guild_id,
                exc,
                exc_info=True,
            )
            resolved_mode = None
    if resolved_mode is None:
        resolved_mode = getattr(config, "TTS_DELIVERY_DEFAULT", "audio")
    normalized_mode = resolved_mode.lower()
    if normalized_mode not in {"off", "audio", "video", "both"}:
        normalized_mode = getattr(config, "TTS_DELIVERY_DEFAULT", "audio")
    if normalized_mode == "off":
        logger.info(
            "TTS delivery mode is OFF%s; skipping TTS send.",
            f" for guild {guild_id}" if guild_id is not None else "",
        )
        return
    async with TTS_LOCK:
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        match = think_pattern.search(text_to_speak)
        if match:
            thought_text = match.group(1).strip()
            response_text = think_pattern.sub('', text_to_speak).strip()
            if config.TTS_INCLUDE_THOUGHTS and thought_text:
                logger.info("Found <think> tags. Sending thoughts and response for TTS.")
                await _send_audio_segment(
                    destination,
                    thought_text,
                    "thoughts",
                    is_thought=True,
                    base_filename=base_filename,
                    delivery_mode=normalized_mode,
                )
                await asyncio.sleep(0.5)
            else:
                logger.info("Found <think> tags. Skipping thoughts for TTS as per config.")
            if response_text:
                await _send_audio_segment(
                    destination,
                    response_text,
                    "main_response" if config.TTS_INCLUDE_THOUGHTS else "full",
                    is_thought=False,
                    base_filename=base_filename,
                    delivery_mode=normalized_mode,
                )
            else:
                logger.info("No user-facing content left for TTS after removing <think> section.")
        else:
            logger.info("No <think> tags found. Processing full text for TTS.")
            await _send_audio_segment(
                destination,
                text_to_speak,
                "full",
                is_thought=False,
                base_filename=base_filename,
                delivery_mode=normalized_mode,
            )
def transcribe_audio_file(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found for transcription: {file_path}")
        return None
    model = load_whisper_model()
    if model is None:
        logger.error("Whisper model could not be loaded. Cannot transcribe audio.")
        return None
    try:
        logger.info(f"Transcribing audio file: {file_path}")
        use_fp16 = torch.cuda.is_available()
        result = model.transcribe(file_path, fp16=use_fp16)  # type: ignore
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
