# Task Work

Structured workflow for working on tasks. Full lifecycle from start through PR merge.

## When to Use

- Starting work on a ready task
- Continuing in-progress or needs_work tasks
- Ensuring consistent task lifecycle with notes and audit trail

**Not for:** Spec creation (use `{skill:writing-specs}`), plan translation (use `{skill:plan}`), or triage (use `{skill:triage}`).

## Inherit Existing Work First

Before starting new work, check for existing work:

```bash
kspec session start  # Shows active work at the top
```

Priority order:
1. **needs_work** — Fix cycle: address review feedback (highest priority)
2. **in_progress** — Continue work already started
3. **ready (pending)** — New work to start

Always inherit existing work unless explicitly told otherwise. This prevents orphaned tasks.

## Task Lifecycle

```
pending → in_progress → pending_review → completed
              ↓              ↓
          blocked        needs_work
                     (→ in_progress → pending_review)
```

| Command | Transition | When |
|---------|-----------|------|
| `kspec task start @ref` | → in_progress | Beginning work |
| `kspec task submit @ref` | → pending_review | Code done, PR created |
| `kspec task complete @ref --reason "..."` | → completed | PR merged |
| `kspec task block @ref --reason "..."` | → blocked | External blocker |

## CLI Lookups

Use CLI commands to find information. **Do NOT search `.kspec/` YAML files manually** — it wastes time and misses context that CLI commands provide (like inherited trait ACs).

| Need | Command |
|------|---------|
| Task details | `kspec task get @ref` |
| Spec + all ACs (own + inherited) | `kspec item get @ref` |
| Trait definition + ACs | `kspec item get @trait-slug` |
| Search by keyword | `kspec search "keyword"` |
| List by type | `kspec item list --type feature` |
| All traits | `kspec trait list` |
| Task's linked spec | `kspec task get @ref` → read `spec_ref` field |
| Task's linked plan | `kspec task get @ref` → if `plan_ref` is non-null, run `kspec plan get @plan-ref` |

**Key pattern:** When `kspec item get` output shows "Inherited from @trait-slug", run `kspec item get @trait-slug` to see the trait's ACs. One command — do not grep YAML files.

## Workflow Steps

### 1. Choose Task

```bash
kspec tasks ready                # All ready tasks
kspec tasks ready --eligible     # Automation-eligible only (loop mode)
```

### 2. Verify Work Is Needed

Before starting, check if work is already done:

```bash
# Check git history
git log --oneline --grep="feature-name"
git log --oneline -- path/to/relevant/files

# Check existing implementation
kspec item get @spec-ref  # View spec and ACs
```

If already implemented: verify tests pass, AC coverage exists, then `kspec task complete @ref --reason "Already implemented"`.

**Notes are context, not proof.** If a task has notes saying "already done" — verify independently. Run tests, check code, validate against ACs. If a task is in the ready queue, there's a reason.

### 3. Start Task

```bash
# Check task links first
kspec task get @ref

# If plan_ref is non-null, load implementation context from the plan
kspec plan get @plan-ref

# Then start work
kspec task start @ref
```

### 3.5 Branch Isolation (Required Before Edits)

Immediately after `kspec task start` and before any file edits, create or switch to a dedicated branch for that task. Do not keep implementing on a branch that is tied to another pending-review task.

```bash
# Example naming from task intent/slug
TASK_BRANCH="fix/<task-slug>"

# Create the branch if missing, otherwise switch to it
git checkout -b "$TASK_BRANCH" 2>/dev/null || git checkout "$TASK_BRANCH"
```

This keeps each task's commits scoped to its own PR and prevents cross-task contamination.

### 4. Work and Note

Read all ACs (own + trait) before implementing:

```bash
kspec item get @spec-ref  # Shows own ACs AND inherited trait ACs
```

Add notes during work, not just at the end:

```bash
# Good: explains decisions and context
kspec task note @ref "Using retry with exponential backoff. Chose 3 max retries based on API rate limits."

# Bad: no context
kspec task note @ref "Done"
```

Note when you:
- Discover something unexpected
- Make a design decision
- Encounter a blocker
- Complete a significant piece

For tasks that are missing standalone context (for example, generic derived notes), add one structured note before deep implementation work:

```bash
kspec task note @ref "Execution context:
- Background: why this task matters
- Scope: what is in/out for this task
- Files: exact files expected to change
- Verification: tests/commands that prove completion"
```

### 5. Commit

Include task and spec trailers:

```
feat: add user authentication

Implemented JWT-based auth with refresh tokens.

Task: @task-add-auth
Spec: @auth-feature
```

Trailers enable `kspec log @ref` to find related commits.

### 6. Local Review

Run quality checks before submitting. Verify:

- **Own AC coverage** — Each spec AC has a test annotated `// AC: @spec-ref ac-N`
- **Trait AC coverage** — Each inherited trait AC has a test annotated `// AC: @trait-slug ac-N`
- **Tests pass** — Full test suite, not just new tests
- **Code quality** — Matches existing patterns, no duplicated utilities
- **No regressions** — Existing tests still pass

```bash
kspec validate  # Reports uncovered trait ACs as warnings
```

### 7. Submit Task

```bash
kspec task submit @ref
```

Moves task to `pending_review`. Create PR after submitting.

### 8. Complete Task

After PR is merged:

```bash
kspec task complete @ref --reason "Merged in PR #N. Summary of what was done."
```

## Fix Cycle

When inheriting a `needs_work` task:

1. **Find the PR** — Check for review comments
   ```bash
   gh pr list --search "Task: @task-ref" --json number,url
   gh api repos/{owner}/{repo}/pulls/{number}/comments --jq '.[] | {path, line, body}'
   ```

2. **Fix findings** — Address MUST-FIX and SHOULD-FIX items

3. **Push fixes** — Commit with descriptive message
   ```bash
   git add <files> && git commit -m "fix: address review feedback

   Task: @task-slug"
   git push
   ```

4. **Re-submit** — `kspec task submit @ref` (back to pending_review)

You do NOT merge in a fix cycle. The reviewer handles merge decisions.

## Scope Management

### What's In Scope

Tasks describe expected outcomes, not rigid boundaries:

- **Tests need implementation?** Implementing missing functionality is in scope — the goal is verified behavior
- **Implementation needs tests?** Proving it works is always in scope

### When to Expand vs Escalate

**Expand** (do it yourself):
- Additional work is clearly implied by the goal
- Proportional to the original task
- You have the context

**Escalate** (capture separately):
- Scope expansion is major
- Uncertain about the right approach
- Outside your task's domain

**When you notice something outside your task:** Capture it separately (`kspec inbox add` or `kspec task note`). Don't fix it inline — even small detours compound into drift.

## AC Test Annotations

Link tests to acceptance criteria:

```javascript
// AC: @spec-item ac-N
it('should validate input', () => { ... });
```

```python
# AC: @spec-item ac-N
def test_validates_input():
    ...
```

Every AC should have at least one test with this annotation.

## Implementation Quality

Before submitting:

- **Search for existing utilities** — Don't duplicate helpers that already exist
- **Match neighboring file style** — Naming conventions, error handling, imports
- **Run full test suite** — Not just your new tests
- **Validate** — `kspec validate` for spec alignment

## Loop Mode

Autonomous task execution without human confirmation.

```bash
kspec tasks ready --eligible  # Only automation-eligible tasks
```

### Task Selection Priority

1. `needs_work` — Fix review feedback
2. `in_progress` — Continue existing work
3. Tasks that unblock others
4. Highest priority ready task

### Key Behaviors

- Verify work is needed before starting (prevent duplicates)
- Create/switch to a dedicated task branch before making code edits
- Decisions auto-resolve without prompts
- PR review handled externally (not this workflow)
- All actions are logged and auditable

### Blocking Rules

**Block only for genuine external blockers:**
- Human architectural decision needed
- Spec clarification required
- External dependency unavailable
- Formal `depends_on` blocker

**Do NOT block for:**
- Task seems complex (do the work)
- Tests are failing (fix them)
- Service needs running (start it)

After blocking:
```bash
kspec task block @ref --reason "Reason..."
kspec tasks ready --eligible  # Check for other work
# If tasks exist: work on the next one
# If empty: stop responding (agent dispatch exits automatically)
```

### Turn Completion

After creating a PR, **stop responding**. The agent dispatch engine continues automatically — it checks for remaining eligible tasks and exits when none remain.

**Do NOT call `end-loop`** after creating a PR. That ends ALL remaining iterations. It's a rare escape hatch for when work is stalling across multiple iterations.

## Command Reference

```bash
# Task lifecycle
kspec task start @ref
kspec task note @ref "..."
kspec task submit @ref
kspec task complete @ref --reason "..."
kspec task block @ref --reason "..."

# Task discovery
kspec tasks ready
kspec tasks ready --eligible
kspec task get @ref

# Validation
kspec validate
kspec validate --alignment

# Session context
kspec session start
```

## Integration

- **`{skill:writing-specs}`** — Create specs before deriving tasks
- **`{skill:plan}`** — Plans create specs that become tasks
- **`{skill:review}`** — Review checks AC coverage and code quality
- **`{skill:observe}`** — Capture friction found during task work
- **`{skill:reflect}`** — Session reflection after completing tasks
