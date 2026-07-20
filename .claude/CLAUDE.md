
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

<!-- TASK_MANAGEMENT_START -->
## Task Management

@kspec-agents.md

This project uses **kspec** for task management. Full agent instructions are in `kspec-agents.md`.

### Do NOT parallelize tasks with your own subagents
- **Do not use the Agent tool to run multiple kspec tasks in parallel.** Parallelizing tasks is `kspec agent dispatch`'s job — it spawns isolated worker agents in their own worktrees. Spawning your own Claude subagents to "work on tasks in parallel" bypasses the whole dispatch system.
- If the user says "dispatch" or "launch agents" in a task-management context, they mean `kspec agent dispatch start` — the kspec daemon's own dispatch system, NOT Claude subagents.
- **Within a single task, the Agent tool is fine.** A worker (or an orchestrator working on one task at a time) can freely spawn research/exploration subagents as it normally would — code discovery, parallel searches, etc. The restriction is only against using subagents as an alternative to dispatch for cross-task parallelism.

### Task Workflow
1. `kspec task list` — See available tasks
2. `kspec task start <ref>` — Claim a task (marks it in_progress)
3. Do the work in this session
4. `kspec task submit <ref>` — Submit when done (creates PR/commit)
5. `kspec task complete <ref>` — Mark complete after merge

### Task Commands
- `kspec task list` — List tasks
- `kspec task add --title "..." --type task|bug|epic|spike|infra --priority 3` — Create a task
- `kspec task start <ref>` — Start a task
- `kspec task submit <ref>` — Submit completed work
- `kspec task complete <ref>` — Mark task done
- `kspec inbox add "..."` — Add idea for later triage

### Agent Dispatch — How to Use

Dispatch runs approved tasks as parallel worker agents in isolated git worktrees. Use it to chain a sequence of dependent tasks against an integration branch.

**Per task — MUST include all four:**
1. `kspec task add --title "..." --type task --priority 3 --spec-ref @spec-ref` — description must be self-contained (workers start with zero conversation context).
2. `kspec task todo add @task-ref "..."` — implementation checklist; this is the worker's definition of done.
3. `kspec item ac add @spec-ref --given "..." --when "..." --then "..."` — acceptance criteria on the spec.
4. `kspec task set @task-N --depends-on @task-(N-1)` — chain order.

**Run the chain:**
1. **Set the base branch.** Dispatch workers branch from `base_branch`. If the feature lives on an integration branch (not `main`), ensure kspec resolves that branch:
   - Simplest: create `kspec.config.yaml` with `base_branch: <integration-branch>` and NOTHING else. (Do NOT set `publication_mode` — see rules.)
   - Alternative: `git checkout <integration-branch>` then `kspec agent dispatch start`; kspec's fallback chain picks up the current branch. Less reliable — if `origin/HEAD` is set to main it may take precedence, and workers silently branch from main instead.
2. `kspec task set @task-1 --automation eligible` — only task 1. Not the whole chain.
3. `kspec agent dispatch start`.
4. When a task transitions to `pending_review` and the reviewer approves:
   - `git cherry-pick <worker-sha>` onto the integration branch
   - Run the build
   - `git tag recovery/<task-ref> <worker-sha>` (worktrees get pruned on completion; tags survive)
   - `kspec task set @task-next --automation eligible`
5. Repeat until the chain drains. The final cherry-pick is the shipping merge.

**Commands:**
- `kspec agent dispatch start | stop | status | watch`
- `kspec task list --status pending_review` — tasks awaiting review/merge
- `git diff <base>...<branch>` — three-dot diff to see only what the branch adds (two-dot shows bidirectional and misleads)

**Rules:**
- One task eligible at a time — mark the next one only after the current one's cherry-pick builds cleanly.
- Cherry-pick each approved step immediately; don't batch them at the end.
- Expect add/add conflicts on shared files between sibling tasks; combine both sides when compatible, don't `--theirs`/`--ours` wholesale.
- `kspec.config.yaml` may set `base_branch` — that's fine and often necessary. But NEVER set `publication_mode`; the default `auto` is the only one with a track record, and `manual_merge` is structurally broken (reviewer runs in its own worktree, can't write to the checked-out base_branch, marks task completed without merging, work gets cleaned up).

### Spec-First Development
- **Before creating tasks for new features, update the kspec spec first.** New behavior that doesn't exist in the spec must be added as spec items (modules, features, requirements) with acceptance criteria BEFORE creating tasks. Tasks reference specs via `spec_ref` — without a spec, there's no definition of what to build.
- Specs define WHAT to build. Tasks track the WORK of building it.
- Use `/kspec:writing-specs` to create spec items (modules, features, requirements, acceptance criteria)
- Use `/kspec:plan` to translate plans into specs and tasks

### Acceptance Criteria & Task Todos
- **Spec-level ACs** (given/when/then): Define "done" for a spec item. Tasks referencing that spec inherit its ACs.
  - Add: `kspec item ac add @spec-ref --given "context" --when "action" --then "expected result"`
  - Annotate tests with `// AC: @spec-ref ac-N` to link code to acceptance criteria
- **Task-level todos** (checklist): Concrete implementation steps on the task itself. Agents use these as their work checklist.
  - Add: `kspec task todo add @task-ref "Implement X"`
  - Mark done: `kspec task todo done @task-ref <id>`
- When creating tasks for dispatch, add todos as a checklist of what the agent must do
- Use `kspec task submit @ref` only when ALL ACs are satisfied and todos are complete

### Dependencies & Branch Strategy
- Dependencies control **execution order** — task 2 waits for task 1 to finish before starting
- Set dependencies: `kspec task set @task-2 --depends-on @task-1`
- **Order dependencies logically**: foundational tasks first, then tasks that build on them
- Dispatch respects `depends_on` — blocked tasks won't be picked up until dependencies complete
- Check readiness: `kspec tasks ready` shows only unblocked tasks
- The reviewer agent automatically merges each approved task into `base_branch` (see Dispatch Merge Configuration)

### Automation Eligibility
- Before dispatching, tasks must be marked automation-eligible
- Assess eligibility: `kspec tasks assess` or use `/kspec:triage-automation`
- Mark eligible: `kspec task set @ref --automation eligible`
- Requirements for automation eligibility:
  - Task has a clear spec_ref with acceptance criteria
  - Dependencies are properly set
  - Task scope is well-defined (no ambiguous "improve X" tasks)
  - Required context is available (no human decisions needed)
- `kspec tasks ready --eligible` shows tasks ready for automated dispatch

### Triage
- `/kspec:triage` — Triage inbox items, observations, and automation eligibility
- `/kspec:triage-inbox` — Process inbox items (promote to task/spec, merge, defer, delete)
- `kspec inbox add "idea"` — Capture ideas for later triage (not yet tasks)
<!-- TASK_MANAGEMENT_END -->
