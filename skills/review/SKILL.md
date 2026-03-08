# Review

Kspec-specific review concerns for verifying spec alignment, AC coverage, and trait coverage. A building block that projects reference in their own review skills and workflows.

## When to Use

- Before creating a PR — verify implementation meets spec
- As part of a project-specific local review workflow
- When reviewing code changes against acceptance criteria

**This is NOT a complete review workflow.** It covers kspec-specific quality gates (spec alignment, AC coverage, trait coverage, validation). Projects should wrap this in their own review skill that adds project-specific concerns (test commands, E2E patterns, coding standards).

## Spec Context Discovery

If a spec ref is not explicitly provided, discover it before proceeding with AC checks:

```bash
# 1. Check commit messages for Task: or Spec: trailers
git log --format='%B' main..HEAD | grep -E '^(Task|Spec):'

# 2. Check changed files for // AC: annotations pointing to specs
git diff main..HEAD | grep '// AC: @'

# 3. If a task ref is found, get its spec_ref
kspec task get @task-ref --json | jq '.spec_ref'

# 4. Search recent tasks matching the scope of changes
kspec tasks list | grep -i "<keywords from changed files>"
```

If a spec is found through any method, proceed with full AC validation below.
If no spec context is found after all discovery steps, skip AC coverage checks and focus on code quality and regression checks.

**Principle:** The absence of a trailer is a signal to look harder, not permission to skip validation.

## CLI Lookups

Use CLI commands to resolve specs and traits. **Do NOT search `.kspec/` YAML files manually.**

| Need | Command |
|------|---------|
| Spec + all ACs (own + inherited) | `kspec item get @spec-ref` |
| Trait definition + ACs | `kspec item get @trait-slug` |
| All traits on a spec | shown in `kspec item get @spec-ref` output |
| Search by keyword | `kspec search "keyword"` |
| All traits | `kspec trait list` |

**Resolving inherited traits:** When `kspec item get` shows "Inherited from @trait-slug", run `kspec item get @trait-slug` to see the full trait ACs. This is one command — never grep through `.kspec/modules/*.yaml` files.

## Spec Alignment

Implementation must match spec intent, not just pass tests.

### How to Verify

```bash
# Read the spec — all ACs (own + inherited)
kspec item get @spec-ref
```

For each AC, verify:
1. **Implementation exists** — Code handles the described behavior
2. **Test exists** — A test validates the behavior
3. **Behavior matches** — The test actually proves the AC, not just syntactically passes

### What to Flag

| Issue | Severity |
|-------|----------|
| AC has no implementation | MUST-FIX |
| AC has no test | MUST-FIX |
| Implementation deviates from spec | MUST-FIX |
| Undocumented behavior (not in any AC) | SHOULD-FIX |
| Spec is vague, implementation chose reasonable interpretation | Note it |

## Own AC Coverage

Every acceptance criterion on the spec MUST have at least one annotated test.

### Annotation Format

```javascript
// AC: @spec-ref ac-N
it('should validate input when given invalid data', () => { ... });
```

```python
# AC: @spec-ref ac-N
def test_validates_input():
    ...
```

### Checking Coverage

```bash
# Get all ACs for the spec
kspec item get @spec-ref

# Search for annotations in test files
# (adapt grep path to your project's test directories)
grep -rn "// AC: @spec-ref" tests/
```

Each AC listed in the spec output must have a corresponding annotation. Missing annotations are MUST-FIX.
Before accepting coverage, confirm each annotation matches the AC text from `kspec item get @spec-ref` (not only the `ac-N` label).

## Trait AC Coverage

When a spec implements traits, it inherits their ACs. Every inherited trait AC must also have test coverage.

### How It Works

```bash
# kspec item get shows inherited ACs under "Inherited from @trait-slug" sections
kspec item get @spec-ref
```

Each inherited AC needs a test annotated with the **trait's** ref, not the spec's ref:

```javascript
// AC: @trait-json-output ac-1
it('should output valid JSON with --json flag', () => { ... });
```

### Checking Coverage

```bash
# kspec validate reports uncovered trait ACs
kspec validate

# Search for specific trait annotations
grep -rn "// AC: @trait-json-output" tests/
```

Any "inherited trait AC(s) without test coverage" warning from `kspec validate` is a MUST-FIX blocker.

### When a Trait AC Doesn't Apply

If a trait AC genuinely doesn't apply to this spec, annotate it with a reason:

```javascript
// AC: @trait-json-output ac-3 — N/A: this command has no tabular output to format
```

The annotation must exist so coverage tooling can track it.
Annotations must be standalone line comments (`// AC:` or `# AC:`), not embedded inside block/JSDoc comments.

### No Traits?

If the spec has no traits (`kspec item get` shows no "Inherited from" sections), skip this step entirely.

## Validation Integration

```bash
kspec validate
```

Validation catches spec-level issues:
- Missing acceptance criteria on specs
- Broken references (dangling `@slug`)
- Missing descriptions
- Uncovered trait ACs (the most common review finding)
- Orphaned specs (no linked tasks)

**Exit codes:** `0` = clean, `4` = errors, `6` = warnings only.

Treat errors as MUST-FIX. Treat warnings as SHOULD-FIX (especially trait AC warnings).

## Review Checklist

Use this checklist when reviewing implementation against a spec:

### MUST-FIX (Blocks PR)

- [ ] Every own AC has at least one annotated test
- [ ] Every inherited trait AC has at least one annotated test (or N/A annotation)
- [ ] `kspec validate` reports no errors for this spec
- [ ] Implementation matches spec behavior (not just syntactically correct tests)
- [ ] No regressions — existing tests still pass

### SHOULD-FIX

- [ ] `kspec validate` warnings addressed (especially trait AC coverage)
- [ ] Undocumented behavior has spec coverage or is flagged
- [ ] Test annotations reference correct spec/trait refs

### SUGGESTION

- [ ] Tests are meaningful (would fail if feature breaks)
- [ ] Prefer E2E over unit where practical
- [ ] Tests run in isolation (temp dirs, not project repo)

## Severity Guide

| Finding | Severity | Action |
|---------|----------|--------|
| Missing own AC test annotation | MUST-FIX | Add test with `// AC: @spec-ref ac-N` |
| Missing trait AC test annotation | MUST-FIX | Add test with `// AC: @trait-slug ac-N` |
| `kspec validate` error | MUST-FIX | Fix the validation error |
| Implementation doesn't match spec | MUST-FIX | Fix implementation or update spec |
| `kspec validate` warning | SHOULD-FIX | Address warning |
| Undocumented behavior | SHOULD-FIX | Add AC or note deviation |
| Test doesn't prove its AC | SHOULD-FIX | Rewrite test |
| No E2E tests | SUGGESTION | Consider adding |

## Using in Project Reviews

This skill provides the kspec-specific gates. Wrap it in your project's review:

```
Project Review = kspec:review gates + project-specific gates
```

Project-specific gates to add in your own review skill:
- **Test commands** — How to run your test suite
- **Test patterns** — Project-specific test helpers and isolation patterns
- **Code style** — Naming, error handling, import conventions
- **E2E specifics** — How E2E tests work in your project
- **Regression check** — Full suite command and expectations

## Integration

- **`{skill:task-work}`** — Run review before submitting tasks
- **`{skill:writing-specs}`** — If review reveals spec gaps, update specs first
- **`kspec validate`** — Automated validation complements manual review
