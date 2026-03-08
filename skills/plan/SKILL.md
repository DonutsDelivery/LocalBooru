# Plan to Spec Translation

Translate approved plans into specs and tasks. Plans are durable artifacts — they persist in the shadow branch, link to derived work, and provide auditable planning history across sessions.

## When to Use

- After plan mode approval — turning an approved plan into trackable specs and tasks
- Creating specs for new features or multi-spec capabilities
- Translating design documents into the spec hierarchy

**Not for:** Raw ideas (use `kspec inbox add`), single spec creation (use `{skill:writing-specs}`), or triage (use `{skill:triage}`).

## Two Paths

### Import Path (Recommended for 3+ Specs)

Write a structured markdown document, then import it. All specs, tasks, and notes created atomically.

```bash
kspec plan import ./plan.md --module @target-module --dry-run  # Preview
kspec plan import ./plan.md --module @target-module             # Execute
```

### Manual Path (1-2 Specs)

Create plan record and specs incrementally via CLI.

```bash
kspec plan add --title "Plan Title" --content "Description" --status approved
kspec item add --under @parent --title "Feature" --type feature --slug slug
kspec item ac add @slug --given "..." --when "..." --then "..."
kspec derive @slug
```

### When to Use Which

| Situation | Path |
|-----------|------|
| Plan mode just approved, complex feature | Import |
| Adding a requirement to existing feature | Manual |
| Multiple related specs with parent/child | Import |
| Quick bug fix that needs spec coverage | Manual |
| Translating design doc with many specs | Import |
| Iterating on previously imported plan | Import (`--update`) |

## Three-Phase Workflow

### Phase 1: Design

Always start here — never skip research.

```bash
kspec workflow start @spec-plan-design
```

1. **Explore** — Read relevant code, understand current state
2. **Clarify** — Identify ambiguities, resolve with user
3. **Design** — Spec structure, AC coverage, trait selection
4. **Review** — Check for completeness and gaps

Design concludes by choosing import or manual path.

### Phase 2: Execute

Run the chosen workflow:

```bash
# Import path
kspec workflow start @spec-plan-import

# Manual path
kspec workflow start @spec-plan-manual
```

### Phase 3: Validate

After creating specs:

```bash
kspec validate              # Check spec quality
kspec validate --alignment  # Verify spec-task links
```

## Plan Document Format

The import parser extracts specs, tasks, and notes from this markdown structure:

````markdown
# Plan Title

## Specs

```yaml
- title: OAuth Provider Support
  slug: oauth-provider
  type: feature
  parent: "@auth"
  description: Support third-party OAuth providers for authentication
  traits:
    - trait-error-guidance
  acceptance_criteria:
    - id: ac-1
      given: User clicks sign-in with Google
      when: OAuth flow completes successfully
      then: User session is created with provider metadata
    - id: ac-2
      given: OAuth provider returns an error
      when: Error callback is received
      then: User sees descriptive error with retry option
  implementation_notes: |
    Use passport.js for OAuth. Per-spec notes go to this spec's derived task.

- title: Token Refresh
  slug: token-refresh
  type: requirement
  parent: "@oauth-provider"
  acceptance_criteria:
    - id: ac-1
      given: Access token is within 5 minutes of expiry
      when: User makes an API request
      then: Token is silently refreshed before the request proceeds
```

## Tasks

derive_from_specs: true

```yaml
- title: Write migration guide
  slug: migration-guide
  priority: 2
  tags:
    - docs
```

## Implementation Notes

General architecture notes. Attached to the plan record.
Use passport.js for OAuth, following existing auth patterns.
````

### Section Reference

| Section | Content | Notes |
|---------|---------|-------|
| `## Specs` | YAML code block — array of spec objects | **Must** use fenced code block (triple-backtick yaml) |
| `## Tasks` | `derive_from_specs: true` + optional manual tasks | Manual tasks get `plan_ref` but no `spec_ref` |
| `## Implementation Notes` | Plain text | Attached to plan record; per-spec notes use `implementation_notes` field |

### Spec Fields

| Field | Required | Description |
|-------|----------|-------------|
| `title` | Yes | Spec title |
| `slug` | No | Human-friendly ID (auto-generated if omitted) |
| `type` | No | `feature`, `requirement`, `constraint`, `decision` (default: `feature`) |
| `parent` | No | Parent ref (e.g., `"@parent-slug"`) |
| `description` | No | What and why |
| `acceptance_criteria` | No | Array of `{id, given, when, then}` |
| `traits` | No | Array of trait slugs (e.g., `trait-json-output`) |
| `implementation_notes` | No | Scoped to this spec's derived task |

## Trait Selection

Before writing specs, review available traits:

```bash
kspec trait list
kspec trait get @trait-json-output  # See inherited ACs
```

### Common Trait Applications

| Building... | Consider these traits |
|-------------|---------------------|
| CLI command with output | `@trait-json-output`, `@trait-semantic-exit-codes` |
| Destructive operation | `@trait-confirmation-prompt`, `@trait-dry-run` |
| List/search command | `@trait-filterable-list`, `@trait-json-output` |
| Shadow branch mutation | `@trait-shadow-commit` |
| User-facing error paths | `@trait-error-guidance` |
| Batch operations | `@trait-multi-ref-batch` |
| API endpoint | `@trait-api-endpoint`, `@trait-localhost-security` |
| WebSocket feature | `@trait-websocket-protocol` |

### Trait Naming in Plan Documents

Use the **full trait slug** — import only auto-prefixes `@`, not `@trait-`:

```yaml
# Wrong — resolves to nonexistent item
traits:
  - json-output

# Correct — full slug
traits:
  - trait-json-output
```

## YAML Pitfalls

Plan documents embed YAML that is parsed as structured data. Common issues:

### Block Scalars for Complex Text

Use `|` (literal block scalar) when AC text contains quotes, colons, or special characters:

```yaml
# Problem — mixed quoting breaks parser
acceptance_criteria:
  - id: ac-1
    when: "kspec foo" is run

# Solution — block scalar preserves content literally
acceptance_criteria:
  - id: ac-1
    when: |
      "kspec foo" is run
```

### Colons in Values

YAML treats `: ` (colon-space) as a key-value separator:

```yaml
# Problem — parsed as nested key
then: output shows time: 10:30

# Solution — quote the value
then: "output shows time: 10:30"
```

### Best Practice

Use block scalars (`|`) for ALL given/when/then text. This avoids every quoting issue:

```yaml
acceptance_criteria:
  - id: ac-1
    given: |
      user has an existing session
    when: |
      user runs "kspec session start"
    then: |
      session context shows: active tasks, recent notes, inbox count
```

## Always Dry-Run First

```bash
kspec plan import ./plan.md --module @target --dry-run
```

Dry-run catches:
- YAML syntax errors before partial state is created
- Missing parent refs
- Invalid trait references
- Duplicate slugs

## Post-Import Checklist

After importing, verify the results:

```bash
# Verify each spec has ACs
kspec item get @spec-slug

# Check trait coverage
kspec validate

# Set task dependencies (import doesn't infer these)
kspec task set @task-slug --depends-on @other-task

# Review plan record
kspec plan get @plan-slug
```

If derived tasks are too generic to execute without chat history, add a structured task note immediately:

```bash
kspec task note @task-slug "Execution context:
- Background: why this task exists
- Scope: concrete boundaries for this task
- Files: exact files/areas to touch
- Verification: commands/tests to run"
```

## Plan Lifecycle

```
draft → approved → active → completed
                     ↓
                  rejected
```

- **Import** auto-creates plan as `active`
- **Manual** creates plan as `approved`
- Mark completed when all derived work is done:

```bash
kspec plan set @plan --status completed
```

## Programmatic Alternative: Batch

For fully programmatic creation (scripts, agent pipelines):

```bash
kspec batch --commands '[
  {"command":"item add","args":{"under":"@parent","title":"Feature X","type":"feature","slug":"feature-x"}},
  {"command":"item ac add","args":{"ref":"@feature-x","given":"...","when":"...","then":"..."}},
  {"command":"derive","args":{"ref":"@feature-x"}}
]'
```

Atomic — all succeed or all roll back. Use `--dry-run` to preview.

## Command Reference

```bash
# Design phase
kspec workflow start @spec-plan-design

# Import path
kspec plan import <path> --module @module --dry-run   # Preview
kspec plan import <path> --module @module             # Create
kspec plan import <path> --module @module --update    # Re-import

# Manual path
kspec plan add --title "..." --content "..." --status approved
kspec plan get <ref>
kspec plan set <ref> --status <status>
kspec plan note <ref> "..."
kspec plan list

# Validation
kspec validate
kspec validate --completeness
kspec validate --alignment
```

## Integration

- **`{skill:writing-specs}`** — Spec authoring details (types, AC format, traits)
- **`{skill:task-work}`** — After specs are created, work on derived tasks
- **`{skill:triage}`** — Inbox items may trigger plan creation
- **`{skill:observe}`** — Friction during planning becomes observations
