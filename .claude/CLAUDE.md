
## FORBIDDEN: VapourSynth Source Filters (bestsource, ffms2, lsmas)

**NEVER use bestsource, ffms2, or lsmas for video streaming.** These filters require indexing the ENTIRE video file before playback can begin, which defeats the purpose of instant streaming.

For real-time video processing:
- Use FFmpeg for decoding (supports hardware acceleration, instant start)
- Use FFmpeg filters for processing when possible (minterpolate, etc.)
- If VapourSynth/SVP is needed, pipe FFmpeg output through a FIFO or use rawsource - NEVER use source filters that require indexing


<!-- TTS_VOICE_OUTPUT_START -->
## Voice Output (TTS)

When responding, wrap your natural language prose in `«tts»...«/tts»` markers for text-to-speech.

Rules:
- ONLY wrap conversational prose meant to be spoken aloud
- Do NOT wrap: code, file paths, commands, tool output, URLs, lists, errors
- Keep markers on same line as text (no line breaks inside)

Examples:
✓ «tts»I'll help you fix that bug.«/tts»
✓ «tts»The tests are passing.«/tts» Here's what changed:
✗ «tts»src/Header.tsx«/tts»  (file path - don't wrap)
✗ «tts»npm install«/tts»  (command - don't wrap)
<!-- TTS_VOICE_OUTPUT_END -->
