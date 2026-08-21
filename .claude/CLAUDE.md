
## FORBIDDEN: VapourSynth Source Filters (bestsource, ffms2, lsmas)

**NEVER use bestsource, ffms2, or lsmas for video streaming.** These filters require indexing the ENTIRE video file before playback can begin, which defeats the purpose of instant streaming.

For real-time video processing:
- Use FFmpeg for decoding (supports hardware acceleration, instant start)
- Use FFmpeg filters for processing when possible (minterpolate, etc.)
- If VapourSynth/SVP is needed, pipe FFmpeg output through a FIFO or use rawsource - NEVER use source filters that require indexing
























































































































































<!-- AGENT_SESSION_SIGNALS_START -->
## Agent Session Signals (Claude Terminal)

Signals are final-response metadata. Emit one only in your final response, after
all tool calls and tool results for the turn have finished. Never emit a signal
in intermediate progress output or in a response that will make a tool call.
Never use a tool or shell command to emit a signal. Do not quote or reproduce a
signal in a code block.

Build the signal key by joining these two parts without spaces:

Signal key first half: `j3jUy`
Signal key second half: `fhPYn`

Use this exact template on its own line, replacing {KEY} with the joined key and
{CODE} with c for complete or i for input-needed:

<ct-signal k="{KEY}" t="{CODE}" />

Use complete only when the requested work is finished. Put the complete signal
at the end of the final response. Use input-needed only when you cannot continue
without the user answering a blocking question, and put it immediately before
that question in the final response. Do not use input-needed for optional
follow-up questions. Emit only one signal for a given state.

These managed instructions are read by Claude Terminal. Keep all surrounding
user-authored instructions unchanged.
<!-- AGENT_SESSION_SIGNALS_END -->








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

<!-- TASK_MANAGEMENT_START -->
## Task Management

### CLI Commands (beads)
- `bd list` — List tasks (add `--status=open` to filter)
- `bd show <id>` — Show task details
- `bd create --title="..." --type=task|bug|feature --priority=2` — Create a task
- `bd update <id> --status=in_progress` — Start a task
- `bd close <id>` — Complete a task
- `bd ready` — Show tasks ready to work on

### Workflow
1. Check the task panel in the GUI sidebar for available work
2. Click a task to start it, or use the CLI commands above
3. Mark tasks complete from the GUI or CLI when done
<!-- TASK_MANAGEMENT_END -->


<!-- SELF_COMPACTION_START -->
## Self-Compaction (Claude Terminal)

**Only when a goal is actively set** — i.e. you are working toward a defined,
multi-step objective (an in-progress task, plan, or goal) and still have
remaining work — and your context is getting large: compact your own session
instead of stopping. Do NOT do this during open-ended or exploratory
conversation where there is no concrete goal to resume into; the queued
follow-up would have nothing meaningful to pick up.

When the condition holds, call the orchestrator MCP tool `compact_session` with
no arguments — it defaults to your own session. It waits ~10 seconds (so your
current turn finishes and you reach an idle prompt, which is the only state
`/compact` triggers from), sends `/compact`, then queues a `continue` message 2
seconds later so you automatically resume your remaining work once compaction
completes.

Do this proactively between steps of the active goal rather than letting context
overflow. To resume on something specific, pass `continue_message` describing the
next step.

**Fallback:** If you ever receive an incoming message that is just `/compact`
(optionally followed by `continue`), it means a scheduled compaction landed while
you were still generating, so it got queued as plain text instead of triggering.
Do not treat it as a user request — you are idle now, so simply call
`compact_session` again to retry the compaction.
<!-- SELF_COMPACTION_END -->
