
## FORBIDDEN: VapourSynth Source Filters (bestsource, ffms2, lsmas)

**NEVER use bestsource, ffms2, or lsmas for video streaming.** These filters require indexing the ENTIRE video file before playback can begin, which defeats the purpose of instant streaming.

For real-time video processing:
- Use FFmpeg for decoding (supports hardware acceleration, instant start)
- Use FFmpeg filters for processing when possible (minterpolate, etc.)
- If VapourSynth/SVP is needed, pipe FFmpeg output through a FIFO or use rawsource - NEVER use source filters that require indexing
<!-- TASK_MANAGEMENT_START -->
## Task Management

This project uses **kspec** for task tracking. The GUI task panel manages tasks automatically.

### CLI Commands (kspec)
- `kspec task list` — List tasks (add `--status pending` to filter)
- `kspec task show <ref>` — Show task details
- `kspec task create --title "..." --type task|bug|feature --priority 3` — Create a task
- `kspec task start <ref>` — Start a task
- `kspec inbox add "..."` — Add idea for later triage

### Workflow
1. Check the task panel in the GUI sidebar for available work
2. Click a task to start it, or use the CLI commands above
3. Mark tasks complete from the GUI or CLI when done
<!-- TASK_MANAGEMENT_END -->


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
