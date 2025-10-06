# TTS Video Generation Debug Guide

## Process Overview

The TTS video generation now works in **3 distinct steps**:

### Step 1: Audio Transcoding (Wait for TTS completion)
```
Kokoro TTS MP3 → AAC format
```
- **Input**: Original Kokoro TTS MP3 file
- **Output**: `audio_transcoded.aac`
- **Log to look for**: "Transcoding audio to AAC for compatibility..."
- **Success log**: "Audio transcoded successfully: X bytes"

### Step 2: Video Creation with Subtitles
```
Color background + Subtitles → x264 video
```
- **Input**: Generated color source + SRT subtitle file
- **Output**: `video_only.mp4`
- **Log to look for**: "Step 1: Creating video track with subtitles..."
- **Success log**: "Video track created successfully: X bytes"

### Step 3: Muxing (Combining audio + video)
```
video_only.mp4 + audio_transcoded.aac → final.mp4
```
- **Input**: Video from step 2 + Audio from step 1
- **Output**: Final MP4 with both audio and video
- **Log to look for**: "Step 2: Muxing video and audio together..."
- **Success log**: "Muxing completed successfully - video and audio combined"

## What to Check if Video is Missing

1. **Check Step 1 (Audio Transcoding)**:
   - Look for: "Audio transcoded successfully"
   - If missing: Check for "Audio transcoding failed" error

2. **Check Step 2 (Video Creation)**:
   - Look for: "Video track created successfully"
   - If missing: Check for "Video creation failed" error
   - Check: "Video creation stderr" for any warnings

3. **Check Step 3 (Muxing)**:
   - Look for: "Muxing completed successfully"
   - If missing: Check for "Muxing failed" error

## Common Issues

### No Video Track
- **Symptom**: Audio plays but no video shows
- **Likely cause**: Step 2 (video creation) failed
- **Check**: Look for "Video file was not created" error
- **Solution**: Check the video creation command in logs

### No Audio Track  
- **Symptom**: Video shows but no audio
- **Likely cause**: Step 1 (transcoding) or Step 3 (muxing) failed
- **Check**: Look for transcoding or muxing errors
- **Solution**: Check if audio_transcoded.aac was created

### File Size Issues
- **Check**: "Verify the video file was created and has reasonable size"
- **Min size**: Should be at least a few KB

## Key Log Messages

```
INFO: "Transcoding audio to AAC for compatibility..."
INFO: "Audio transcoded successfully: X bytes"
INFO: "Step 1: Creating video track with subtitles..."
INFO: "Video creation command: ffmpeg ..."
INFO: "Video track created successfully: X bytes"
INFO: "Step 2: Muxing video and audio together..."
INFO: "Muxing command: ffmpeg ..."
INFO: "Muxing completed successfully - video and audio combined"
INFO: "Video generation successful - contains X audio streams and X video streams"
```

## Expected Behavior

✓ All 3 steps complete successfully
✓ Final video contains both audio and video streams
✓ Video embeds properly in Discord
✓ TTS audio is audible
✓ Subtitles are visible

