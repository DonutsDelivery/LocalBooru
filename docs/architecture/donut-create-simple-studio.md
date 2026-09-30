# Donut Create: basic studio and advanced workflow

The default Create view is a website-style interface for people who do not know
ComfyUI. It exposes a prompt, the workflow's model choices, image shape, and a
Generate button beside a result gallery. Edit accepts a reference image and an
instruction. Less frequently used controls are collapsed. Setup remains separate
from everyday generation.

The Advanced editor opens the existing DonutUI/DonutNodes v5 panels. Both views
operate on the same live graph and saved draft. Returning to the basic view reads
the current settings; it must not load a preset or overwrite advanced changes.
Basic changes update only their specific widget bindings. Model selection is
limited to the current workflow's supported backend catalog; selecting a file
does not convert a Krea2 workflow into an arbitrary model architecture.

## Existing interfaces studied

- [SwarmUI advanced usage](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Advanced%20Usage.md)
  connects its Generate parameters to a custom workflow edited in ComfyUI. This
  informed the basic/advanced handoff. DMC keeps one live graph rather than
  requiring a second workflow import after editing.
- [ComfyUI App Mode](https://github.com/Comfy-Org/docs/blob/main/interface/app-mode.mdx)
  exposes selected workflow inputs and outputs and adapts to narrow screens.
  This informed selective controls and the result area.
- The user's Civitai screenshot informed a compact controls sidebar, a prominent
  generation action, and a spacious queue/result gallery.

These are design references. No wrapper code or extra backend runtime is bundled.

## Workflow connection

The scoped ComfyUI frame stays mounted while the basic UI is visible. It supplies
the existing v5 widget callbacks, subgraph bindings, prompt serialization and
generation preflight. The host sends a small set of typed commands: snapshot,
patch specific basic fields, upload a reference, generate, or explicitly load the
setup preset or latest submitted workflow. It cannot request arbitrary node changes or execute JavaScript.

The host checks the exact frame window, origin and session capability on replies.
The frame accepts commands only from its parent with the current capability and
pins that parent's origin during the handshake. Mutations are serialized so
generation sees acknowledged settings. Backend/session changes invalidate pending
commands and results. Failures are shown in the basic interface.

Reference uploads, job tracking, cancellation, output previews and gallery imports
use existing session-owned routes. Account credentials and local filesystem paths
are not put in frame messages. Results are saved only into an explicitly selected,
available image directory.

## Editing scope

The basic view has separate Create, Edit and Tuning tabs. Edit supports uploading
a reference, describing a change, and painting or erasing a selected area with
undo and clear actions. Selections use the existing DonutEditStudio normalized
stroke format, so the same mask remains editable in the v5 editor. Fine crop,
multiple references, canvas placement and the full v5 tuning controls remain
available through Advanced editor. Starting an edit must enable the editing
branch without discarding the generation recipe; returning to Create disables
that branch explicitly. Ordinary previewing never queues a generation.

## Verification limits

Use disposable sessions and synthetic uploads/jobs for behavior checks. Inspect
the serialized execution prompt to verify real widget/subgraph synchronization.
Do not use a user's library as test data or queue real GPU work merely to inspect
the interface. Browser rendering, native packaging and actual generation are
separate forms of evidence.

## Paired mobile workspace

Write-authorized desktop and paired mobile clients request the same private
workspace for a backend. The controller shares its scoped jobs, output previews,
cancellation, and latest successfully submitted workflow. One upstream WebSocket
per workspace fans out filtered execution events and previews to both clients;
bounded device buffers reconnect instead of delaying other devices. Legacy isolated sessions
remain separate. Switching backend selects a different workspace; expired
workspaces retain only their own latest recipe and reference ownership.

The basic layout switches to full-width Controls and Results on narrow screens.
Full application reopening bootstraps from the latest submitted graph, with local
draft fallback when no run exists. Closing and reopening the mounted studio keeps
the current unsubmitted draft. A later run exposes Load latest run; it does not replace an
in-progress edit. Queued jobs can be cancelled and an active job interrupted
through the same existing scoped queue controls.

The v5 execution graph uses the shared scene prompt for both creation and edit
instructions. Basic Prompt and Edit instruction intentionally bind to that same
executed input rather than to the unused EditStudio prompt output. Uploading a
replacement reference clears only its stale selection and crop geometry.
