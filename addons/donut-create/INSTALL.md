# Donut Create runtime

Donut Create is an optional Images add-on. Its small Python controller runs
separately from a managed ComfyUI installation. Setup puts ComfyUI, its own
Python environment, the node packs, download cache, and models under the
add-on's private state directory. A registered existing local backend can also
update its required DonutNodes source pack through the ComfyUI status menu. This
updates the selected installation without replacing its models or Python environment.

The controller uses port **18009**. Managed ComfyUI listens on
**127.0.0.1:18010**, independently of the usual ComfyUI port 8188. An occupied
managed port is reported as a conflict; the add-on does not adopt that process
or fall back to 8188. Connecting to another ComfyUI installation requires
explicitly choosing Existing backend and entering its origin.

To manage a running local installation, choose **Connect existing ComfyUI**, enter
its localhost URL and leave **Detect and manage this local installation** enabled.
For an existing URL-only connection, the **ComfyUI** status menu offers **Manage
this ComfyUI**. Discovery does not stop or restart the backend. It identifies the
same-user Python process listening on that port, verifies its `main.py` and
`comfy` folder, and privately saves its Python executable, working directory,
launch arguments and relevant Python/GPU environment. Discovery installs nothing.
The separate **Update DonutNodes** action changes only the required node-pack
source after ComfyUI has been stopped. Start and Restart
reuse the original port and launch options. Stop and Restart target only the
verified process identity; an unrelated port owner is never terminated.

Discovery runs on the computer hosting the add-on, including when requested by
a paired phone. It accepts loopback addresses, not a remote ComfyUI URL. Linux
and Windows use the socket table; macOS uses `lsof`. Restricted permissions,
containers and shell/module launch wrappers can prevent discovery. Uncheck the
detection option for a URL-only connection, or use manual installation setup.
Older add-on environments install the small pinned `psutil` binary wheel at
controller startup; if offline, URL connections still work and repairing the
add-on retries this dependency. No administrator access is requested.

For manual setup, choose **Manage existing local installation** and enter its
ComfyUI folder on the hosting server. The controller detects `venv` or `.venv`
Python in that folder or its parent; an executable can also be selected explicitly.
Manual setup launches on port 18010 using the existing models and node packs.
Start, Stop and Restart are available through authorized routes to paired clients.
Stop a managed backend before changing configuration. Adopting the connected
backend preserves the active studio and saved draft. Adopted external processes
stay running when the controller closes. If the host reboots, the controller can
reattach a newly running process only when its installation, Python, working
directory and launch command still match the saved profile. Children launched
by DMC retain the existing shutdown behavior.

Drop a workflow JSON, or a PNG/WebP containing embedded workflow metadata, into
the studio to restore its settings. A plain image becomes an editing reference.
Dropping an image onto a reference card or canvas replaces that reference and
keeps the workflow. Imports pause Run Instant and never generate automatically;
unrecognized editable graphs open in the Advanced editor.

## Local prompt enhancement

Open **Prompt enhancer** below the image/face descriptions in the simple Create
or Edit controls. Choose **Download model**, then enable prompt enhancement.
**Preserver** is MiniCPM5-2B mondk Abliterated **Q6_K** (2.07 GB); **Rewriter** is
Qwen3.5-4B HauhauCS Aggressive **Q4_K_M** (2.71 GB). Each Create, Edit, Advanced,
and Run Instant submission resolves the scene and face briefs through the
backend wildcard processor before generating or reusing an enhancement.
Inline choices such as `{red|blue}` and nested choices select their options
using that image's actual execution seed, so alternatives become one scene
before enhancement. New LLM inference uses the same seed. Larger image
seeds map deterministically into llama.cpp's 32-bit sampling range. Fixed briefs,
model and seed repeat sampling; changing the seed allows new variations.

**Every run** is the default timing. **When prompts change** reuses the current
studio's last successful enhancement until the image brief, face brief, resolved
wildcard choices, or enhancer model changes. Loading another workflow starts
fresh. A reused result keeps its original enhancement seed while the image uses
the current run's seed; the preview identifies reuse. Failed or cancelled work
cannot supply a reusable result.

You can edit the next prompt and its generation settings while enhancement or
cancellation is ongoing. These edits stay in the next draft and leave the
captured run unchanged. Generate becomes available once the current submission
settles; completion cannot replace newer draft edits.

The whole-image style also goes to the enhancer. The enhancer writes it into the
scene prompt and into every non-empty face prompt, so the image and each face
refinement are conditioned on the same style. For that run only, the style box is
left empty so the style is not applied twice; Edit runs leave the style alone.
If the model leaves the style out, DMC puts it back in front. Wildcards in the
style resolve with the image's execution seed, like the other briefs.

Original editable briefs stay unchanged. Write explicit directions to invent,
choose or vary details when desired; fixed facts and literal trigger tokens must
remain intact. The resulting scene and face text conditions only that run and
appears under **Last run's enhanced prompts**, together with its model and seed.
When the face brief is empty, the enhancer derives one identifiable visible face
from the image brief. For several visible faces, describe their screen positions
or give facial descriptions explicitly in left-to-right order. The enhancer
returns up to 16 separate character descriptions, shown in the last-run preview.
It can associate characters through relative left/right and above/below ordering
without requiring a fixed grid. The owned DMC detailer matches those relationships
against detected face-box centers and applies independent conditioning only when
there is exactly one possible assignment. Ambiguous positions, contradictory
relationships or a face-count mismatch stop refinement before any character
prompt is applied. Left/right-only descriptions retain their existing ordering.
Newly inferred single-face runs also require exactly one detected face. Gaze direction does not determine screen position.

The face runtime belongs to DonutNodes and is downloaded with the required
node-pack version; DMC does not deploy a separate ComfyUI detailer implementation.
Managed setup installs that pack. For a registered existing local backend, open
the **ComfyUI** status menu, stop ComfyUI, choose **Update DonutNodes**, and start
ComfyUI again. The update targets the selected backend's custom-nodes folder.
Dirty Git checkouts, modified owned sources, unowned folders and unverified links are
preserved with an actionable error. A linked Git checkout is supported when its
repository root and origin identify the required DonutNodes source; the link
stays in place and the same local-change safeguards apply. Unverified linked
folders are preserved. URL-only remote backends must be updated on
their host. Ordinary workflows remain usable while the face runtime is missing;
enhanced face routing reports the node requirement before image submission. Multi-face identity editing/references, detailer wildcards and custom
conditioning hooks currently report an unsupported-mode error; there is no
implicit identity-reference matching. Original prompts stay editable and the
ordered descriptions used by a run remain in its workflow metadata.

There is no separate apply step. Enhancement failure, a changed workflow or
cancellation prevents submission. Closing Create or stopping ComfyUI cancels a
pending enhancement without cancelling already accepted image jobs. Model setup
stays available when an older or custom workflow lacks a prompt control; enable
requires both controls, while disabling remains available if a binding or model
becomes unavailable.

The add-on downloads its own pinned llama.cpp **b11457** CPU runtime and GGUF
weights using the byte counts, revisions, and SHA-256 checksums in
`enhancer_models.json`. It needs no Ollama installation or model-server URL.
Runtime assets are available for Linux x64/ARM64, Windows x64, and macOS
Intel/Apple Silicon; platform availability does not establish runtime acceptance
on every platform. Linux x64 is exercised with both models. macOS runtime binaries
are downloaded from upstream, not modified or re-signed by DMC.

Files live under `enhancer/` in the add-on state root. Setup is optional, requires
internet access and free disk space for the selected model and runtime, verifies
before activation, and retains partial downloads for retry. **Repair / verify
model** reuses verified files. Inference works offline afterward. It runs on the
computer hosting Create, including when controlled from a paired phone. CPU
inference avoids unloading ComfyUI's GPU models. An authenticated helper listens
only on loopback. After a successful enhancement, one CPU model stays loaded in
system RAM so later runs using that model skip loading its weights again. Switching
models stops the previous helper before loading the next one. Repair, inference
failure, cancellation, and add-on shutdown release it. Cancellation targets only
that studio's current enhancer operation; discarding an older result cannot stop
another studio's newer run.

Enhancer inputs and outputs are held in bounded memory, not logged or saved by
the enhancer. Submitted workflows retain original briefs and record that run's enhanced pair in workflow metadata.
Malformed, truncated, or incomplete output cannot change either workflow field.
The context is bounded; overly long briefs fail without truncating or applying.
Maximum input sizes are 16,000 characters for the image and 8,000 for the face.

llama.cpp is separately licensed under MIT; see its pinned
[release](https://github.com/ggml-org/llama.cpp/releases/tag/b11457) and
[source license](https://github.com/ggml-org/llama.cpp/blob/b11457/LICENSE).
Model weights retain their publishers' terms:
[MiniCPM GGUF](https://huggingface.co/mondk/MiniCPM5-2B-Abliterated-Uncensored-GGUF)
and [Qwen GGUF](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive).
They are downloaded on request and are not bundled or relicensed by DMC.

## Image backend setup

Desktop addon installation uses an existing Python **3.12** when available,
including standard Homebrew/python.org locations on macOS. Otherwise DMC
downloads a pinned, SHA-256 verified Python 3.12 runtime into its own application
data folder on Linux x86_64/ARM64, macOS Apple Silicon/Intel, and Windows x64.
It does not change the system Python or require shell configuration. Android
connects to a desktop for generation; it does not install ComfyUI on the phone.
Managed setup requires an internet
connection and space for the selected model files plus a 16 GiB environment
reserve. The reserve is an estimate for dependencies and download/build caches.

## Runtime choices

| Runtime | Supported host | Tensor packages | Requirement |
| --- | --- | --- | --- |
| CPU | Linux x86_64/ARM64, Windows x64, Apple Silicon macOS | PyTorch 2.9.1, torchvision 0.24.1, torchaudio 2.9.1 | Enough system memory for the selected models |
| CUDA | Linux x86_64, Windows x64 | Same versions, CUDA 12.8 wheels | NVIDIA GPU and a compatible NVIDIA driver |
| MPS | Apple Silicon macOS | Same versions from PyPI | Available Metal/MPS backend |

CPU is the conservative default outside Apple Silicon; MPS is the Apple Silicon
default. The installer checks the selected accelerator using the installed
PyTorch environment. CPU support does not make the 26 GB base diffusion model
small or guarantee practical generation speed. Windows and real Mac generation
are not certified by the synthetic installer tests.

The tensor package indexes are the [official PyTorch wheel indexes](https://download.pytorch.org/whl/)
and [PyPI](https://pypi.org/project/torch/2.9.1/). The full runtime and repository
pins are in [runtime.json](runtime.json).

## Model profiles

| Profile | Models | Model bytes | Workflow selection |
| --- | ---: | ---: | --- |
| `base` — Neutral starter (default) | 6 | 32,273,722,868 | Original v5 wiring; Krea2 base diffusion model, **Single model** mode, empty aesthetic LoRA stack, neutral teapot prompt |
| `workflow` — Original v5 model choices | 8 | 45,872,604,519 | Original primary/secondary two-model merge and enabled aesthetic LoRA, with a neutral prompt |
| `all` — Complete model catalog | 16 | 63,162,364,113 | Original v5 choices and every optional catalog weight |

The original v5 profile contains the Civitai finetuned diffusion model and its
enabled aesthetic LoRA. Setup never removes an enabled LoRA or changes profiles
to conceal an authentication failure. If those files are unavailable to the
account, the error remains visible. Choose the explicitly different Neutral
starter profile if that is the desired workflow.

The base and original workflow profiles download the files used by their enabled
stages. Disabled identity editing, SDA, SeedVR2, reference background removal,
SAM3 subject selection, and the alternative 2x VAE keep their v5 controls and
saved selections; their extra weights are optional until enabled. The Complete
model catalog installs those weights and the VAE Utils pack needed for its
alternative VAE. The studio checks active file selections before Run and reports
missing node types or models. Select the needed setup profile before enabling an
optional engine.

All downloads come from [model_sources.json](model_sources.json). A destination
is activated only after its exact byte count and SHA-256 match. Hugging Face
links use fixed repository revisions; provider-independent checksums also cover
the Civitai and SAM files.

An optional existing **models folder** may be supplied in setup. The installer
reads only matching catalog paths in that folder, verifies their hashes, and
reuses them through its own `extra_model_paths.yaml`. It does not alter, move,
or copy those existing weights. A differing existing file is preserved and
reported as a conflict. No existing user path is built into the add-on.

## Cancellation and recovery

Setup reports the current phase, file, byte progress, completed file count,
available disk space, and failures. Cancel stops the owned setup command tree;
add-on shutdown also stops its descendants. Completed verified files remain
reusable. Partial files remain in the hash-addressed download cache and Retry
uses HTTP ranges when the provider supports them. A provider ignoring ranges
restarts the partial safely. A checksum mismatch deletes the invalid partial
and leaves the final destination absent.

Existing conflicting source folders and model files are preserved. Move the
identified conflict aside before Retry. Changing CPU/CUDA/MPS when an environment
already exists similarly requires moving the managed `backend/venv` folder
aside; the installer does not overwrite an existing environment.

`hf_token` and `civitai_api_key` are optional setup inputs held in memory for that
attempt. They are never written to setup state, runtime configuration, receipt,
pip commands, or subprocess environments. Cross-origin redirects strip
Authorization. A Retry needing authentication requires entering the token again.
Account permissions and upstream availability still apply.

Readiness requires the selected files, the managed interpreter, pinned source
markers, the expected ComfyUI frontend API prefix support, and a real installed
engine import that registers all required backend node types. CUDA/MPS readiness
also requires the selected accelerator to be available. A second live capability
and model check occurs before the controller opens a studio session.

## Workflow and browser assets

New v5 recipes default to **Euler / Simple**, **Rebalance**, **NAG 0.45** and
**DecensorFix off**. The aesthetic LoRA in the original workflow profile uses
model and text strengths of **1 / 1**; the neutral starter retains its empty
LoRA stack. Saved workflows keep their own settings. Load setup preset explicitly
to start from these defaults.

[workflow.json](workflow.json) retains the v5 subgraphs, links, promoted inputs,
and panel paths. Its personal prompt/reference/preview state has been removed.
Prompt controls still target the three `DF_Text_Box` nodes, their nested
`DonutText` nodes, and `DonutPromptConditioning`; no replacement sampler or
synthetic generation path is used. v5's frontend VAE migrations remain enabled.

The default save path is relative to the owned output folder. Create runs use
lossless WebP, disable overwrite, and retain workflow and execution metadata.
The controller further scopes output paths to each studio's own session.

[assets/donut-create.js](assets/donut-create.js) adapts the DonutUI bootstrap,
bridge, and public drawing API to a normal browser. The controller configures its
scoped backend prefix before ComfyUI modules run. The adapter imports the real
ComfyUI app, loads v5 on first initialization, and retains studio drafts in local
browser storage under a stable backend identity. It preserves meaningful working
workflows already being loaded. **Load setup preset** is an explicit reset.
Browser renderers are bundled; the Tauri IPC plugin/settings loaders are absent.

## Provenance and verification

ComfyUI is pinned to
[`3c80da7f87ee359b2d06f107cb3c0797079dfbbb`](https://github.com/Comfy-Org/ComfyUI/commit/3c80da7f87ee359b2d06f107cb3c0797079dfbbb),
version 0.36.0 with frontend 1.53.6. The required node packs are DonutNodes,
bleh, Impact Pack, Impact Subpack, Derfuu, Krea2Edit, Krea2 NAG, and Seed Variance
Enhancer. Their exact public commits, archive byte counts and hashes were
verified on 2026-09-30 and are recorded in `runtime.json`. Registry extraction
directories were not mistaken for Git checkouts. Impact's unpinned SAM2
requirement is replaced by a fixed, verified SAM2 source; its optional CUDA
extension build is disabled.

DonutNodes workflow/catalog provenance:

- Source repository: [ComfyUI-DonutNodes](https://github.com/DonutsDelivery/ComfyUI-DonutNodes), commit `2806ebe12caada993e6305e3f441ce61ab22a11c`.
- Source workflow SHA-256: `f06309a40fff7f8ff82fc9d7c23303814f940898842fa3d98be40f3fd8525383`.
- Source model catalog SHA-256: `ed76e898953e8b9393118671581e416d05a081b0878cf572bce95c375a5cc179`.
- Source DonutUI wrapper version: 0.1.0. Per-file hashes for its bootstrap, bridge, and API are in `runtime.json`; the browser adaptation retains their attribution comments. DonutNodes' own license and third-party notices remain in the installed pinned source.

Run focused synthetic installer checks with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s addons/donut-create/tests -p test_installer.py -v
node --check addons/donut-create/assets/donut-create.js
node --experimental-vm-modules --test addons/donut-create/tests/browser_adapter.test.mjs
```

These checks cover profile selection, real graph bindings, checksum activation,
resume/cancel, conflicts, reuse, credential stripping, capability receipts,
disk failures, archive safety, and owned process cancellation. The browser contract
fixture covers edited prompts through Run, missing-model queue rejection, backend
draft retention, late stock-workflow initialization, and meaningful workflow
preservation. Their synthetic
fixtures stay outside the repository. They do not download model weights,
install backend dependencies, start ComfyUI, or establish GPU/Windows/Mac
generation quality. Desktop integration checks use the repository's isolated
app launcher.

## Output destination

Finalize selects one permanent destination. **Save output** defaults on and
remembers an explicit off choice. If no destination is selected, the first
saving-enabled run creates or reuses an Images-only **Created Images** directory
in the selected output library before queueing; opening Create alone creates no
folder. Saved library images are full-resolution lossless WebP; their hidden
workflow companions retain executed settings. Choose an image directory for automatic DMC imports, or **ComfyUI output
folder** to keep backend outputs without importing them into DMC. With saving disabled, runs
remain temporary until explicitly saved. Execution rewrites the known image save
sinks; the editable workflow and its generation settings remain intact.

DMC runs use ComfyUI temporary files for previews and transfer, rather than
writing a second permanent copy into its output folder. Temporary files remain
subject to ComfyUI’s temporary-directory cleanup. Earlier permanent outputs are
not deleted. Choosing a DMC directory later does not automatically copy existing
permanent ComfyUI results; an explicit **Save to library** remains available.

Older recognized v5 recipes gain the missing face-prompt textbox and face-only conditioning path in place when opened. Existing scene/style prompts, seeds, models, and generation settings remain intact; no image is generated. Custom face wiring and partial upgrades are left unchanged. When Face detail is enabled, the face textbox appears above Prompt enhancer, and its written draft is supplied alongside the main brief.

The basic studio offers **Workflow preset** → **Neutral starter** or **Original v5** → **Reset workflow**. Confirming replaces the draft prompts, references, model choices and workflow settings with the bundled preset; it keeps saved images and downloaded models and does not generate. Original v5 requires its original model and LoRA choices to be installed. Legacy v5 face repair follows the exposed scene/detail controls and executable face routes, including copied workflows with different IDs or missing release metadata; custom face wiring remains unchanged. The actual scene and face text used by the last enhanced run is shown under **Prompt enhancer**.
