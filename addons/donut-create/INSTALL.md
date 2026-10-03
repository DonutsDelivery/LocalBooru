# Donut Create runtime

Donut Create is an optional Images add-on. Its small Python controller runs
separately from a managed ComfyUI installation. Setup puts ComfyUI, its own
Python environment, the node packs, download cache, and models under the
add-on's private state directory. It does not install into another ComfyUI
installation or the system Python environment.

The controller uses port **18009**. Managed ComfyUI listens on
**127.0.0.1:18010**, independently of the usual ComfyUI port 8188. An occupied
managed port is reported as a conflict; the add-on does not adopt that process
or fall back to 8188. Connecting to another ComfyUI installation requires
explicitly choosing Existing backend and entering its origin.

To manage an existing installation, choose **Manage existing local installation**
and enter its ComfyUI folder on the hosting server. The controller detects a
`venv` or `.venv` Python in that folder or its parent; a Python executable can
also be selected explicitly. This mode uses the installation's existing models
and node packs without installing or downloading anything. Start, Stop and
Restart control only the child DMC launches on port 18010. They are available
through the same authorized routes to paired clients. URL-only connections
remain controlled by their host application. Stop the owned child before
changing backend configuration; restarting it preserves the studio's workflow
and draft.

Drop a workflow JSON, or a PNG/WebP containing embedded workflow metadata, into
the studio to restore its settings. A plain image becomes an editing reference.
Dropping an image onto a reference card or canvas replaces that reference and
keeps the workflow. Imports pause Run Instant and never generate automatically;
unrecognized editable graphs open in the Advanced editor.

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

The default save path is relative to the owned output folder. `DonutImageSave`
uses PNG, disables overwrite, and embeds workflow and execution metadata.
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

Finalize selects one permanent destination. With **Save output** enabled, choose
an image directory for automatic DMC imports, or **ComfyUI output folder** to
keep backend outputs without importing them into DMC. With saving disabled, runs
remain temporary until explicitly saved. Execution rewrites the known image save
sinks; the editable workflow and its generation settings remain intact.

DMC runs use ComfyUI temporary files for previews and transfer, rather than
writing a second permanent copy into its output folder. Temporary files remain
subject to ComfyUI’s temporary-directory cleanup. Earlier permanent outputs are
not deleted. Choosing a DMC directory later does not automatically copy existing
permanent ComfyUI results; an explicit **Save to library** remains available.
