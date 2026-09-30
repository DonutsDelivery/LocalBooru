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

Advanced is still a scoped editor: an imported workflow must use the supported
node packs and routes, and the basic controls bind specifically to v5. Entirely
different workflows are not generally supported by this addon yet. A future
custom workflow profile should check its node/model requirements, open in
Advanced by default, and optionally expose selected inputs as basic controls
without replacing the retained v5 recipe.

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

## Output directories and saved recipes

Create output directory creates or reuses an image-only **Created Images** folder
inside the selected library. It registers the folder through the regular directory
scan and watcher paths and selects it for saving. Other image directories remain
available. Creation requires write access, including for paired mobile clients.

The PNG remains byte-for-byte the backend output, including any embedded metadata.
Saving also writes `.donut-create/<image filename>.json` beside it: for example,
`Created Images/.donut-create/example.png.json`. The hidden companion contains a
versioned format, image SHA-256, the full editable workflow, executed prompt graph,
execution details, normalized prompt/seed/sampling fields and original PNG text.
It does not contain the studio capability or account credentials. Writes are staged
and atomically replaced; an invalid, mismatched or symlinked companion is rejected.
The database caches searchable fields; reindexing can recover those fields from
the verified companion. Keep the hidden folder with the images when moving a
directory or making a backup.

Gallery context menus and the image lightbox offer **Load workflow** when a recipe
is available. Lookup uses the mounted library, directory, image and expected file
hash, checks the current file and companion, and falls back to embedded PNG
workflow data for older images. It opens Create and loads that graph into the same
basic/advanced editor after readiness, without generating. Closing before readiness
cancels the delayed load. Missing backend models, node packs or reference assets
must still be supplied; the companion describes the recipe rather than embedding
those dependencies. Loading an image recipe does not change the shared latest run.

## Studio panes and controls

The expanded studio uses a large latest-result view in the center, prompt and
reference tools on the left, optional effects on the right, and models, LoRAs
and result history in a lower pane. Narrow screens expose these areas through
pane tabs. Disabled effects remain collapsed; turning one on reveals its
relevant settings. Advanced editor still exposes the complete live v5 graph.

Model controls include separate primary and secondary choices, a blend slider,
and LoRA add/remove with individual model and text strengths. Sampling exposes
the global sampler and scheduler. Optional effects include each upscale stage
and its scale/denoise, face detail, compatibility/TAB and decensor weights,
NAG, SDA and Tone Lab. Zero strength disables the effects that support it.
Choices come from the connected backend and preserve its workflow constraints.
Prompt brace choices are supported; arbitrary filesystem wildcard macros remain
outside the scoped studio interface.

Edit exposes the existing reference, crop, painted selection, mask, canvas
placement and outpainting capabilities through organized tools. Reference
guidance is also available from Create. Selections use DonutEditStudio's
normalized geometry so they remain editable in the v5 editor. Starting an edit
enables the editing branch without discarding the generation recipe; returning
to Create disables that branch explicitly. Ordinary previewing never queues a
generation.

The main prompt describes the complete image. When face detail is enabled,
an optional **Describe only the face** input supplies additional instructions
only to the face branch. It does not enter the base image or upscale prompts.
Executed metadata retains the main prompt separately from that face override.

## Live results and automatic actions

The controller registers temporary stage images only from its owned executed
events or history. The scoped image route serves those registered previews,
uploaded references and owned completed outputs. The bounded preview ledger
supports reconnecting devices; arbitrary backend temporary files stay outside
the studio. Filtered binary previews also stream through the existing shared
workspace socket without replacing the graph editor's connection.

Final save nodes are identified separately from intermediate outputs. **Save
output** imports only completed final images into the selected directory and
writes the same hidden companion used for manual saving. Intermediate and
live previews remain display-only. Errors remain visible and do not silently
change the destination.

**Run (Instant)** is explicit opt-in. It submits an acknowledged changed draft
when the owned queue becomes empty, with one pending submission at a time.
It does not repeatedly submit an unchanged draft. Errors, cancellation,
workflow loading and backend changes pause the automatic run behavior.

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

The basic layout switches to full-width pane tabs on narrow screens.
Full application reopening bootstraps from the latest submitted graph, with local
draft fallback when no run exists. Closing and reopening the mounted studio keeps
the current unsubmitted draft. A later run exposes Load latest run; it does not replace an
in-progress edit. Queued jobs can be cancelled and an active job interrupted
through the same existing scoped queue controls.

The v5 execution graph uses the shared scene prompt for both creation and edit
instructions. Basic Prompt and Edit instruction intentionally bind to that same
executed input rather than to the unused EditStudio prompt output. Uploading a
replacement reference clears only its stale selection and crop geometry.
