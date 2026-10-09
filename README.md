# VastAIBatchJobs

Run one prompt against a Hugging Face GGUF model on a fresh VastAI GPU, retrieve the response into this repository, then destroy the instance. The controller runs locally; model weights and CUDA inference stay on the rented instance.

A **GGUF** file packages model weights and inference metadata, including tokenizer and architecture information. It is not an executable. This workflow builds a pinned CUDA version of [llama.cpp](https://github.com/ggml-org/llama.cpp) and uses its chat-completion server to run the model.

## Install

Local requirements: Python 3.10+, `ssh`, `scp`, and `ssh-keygen`. No local GPU, CUDA, or Docker is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The existing repository-root `.env` works without renaming its variables:

```dotenv
VAST_AI_API_KEY=<your VastAI API key>
HUGGING_FACE_ACCESS_TOKEN=<your Hugging Face read token>
```

If you do not already have `.env`, create it using `.env.example`. Do not overwrite an existing file or commit credentials. The Hugging Face token needs access to the selected file, including any required upstream model-license acceptance.

Process environment variables override the selected `.env`, including explicitly empty values. `--env-file` selects another file, which must exist. Cleanup requires only `VAST_AI_API_KEY`.

## Preview, then run

Start with an approximately 8B **instruction-tuned**, standalone `Q4_K_M` GGUF. An 8B Q4 model is typically around 5 GB, but GPU memory also needs to accommodate the inference context and runtime. Select an exact filename, not just a repository.

This example performs authenticated, read-only model/template/offer checks. It does **not** rent an instance, attach keys, or download weights:

```bash
python run_job.py run \
  --model-repo bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --model-file Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --prompt 'Explain GGUF in three short sentences.' \
  --dry-run
```

**Removing `--dry-run` rents a paid instance.** The run performs these steps:

1. Resolve the requested model revision to a commit and check access to the actual GGUF file.
2. Choose and revalidate an eligible GPU offer, then create a uniquely labeled instance.
3. Attach a fresh instance-specific SSH key and wait for SSH readiness.
4. Upload the worker, build the pinned CUDA runtime, and download only the selected GGUF.
5. Generate one non-streaming assistant response through a loopback-only server.
6. Retrieve and verify the response, destroy the instance, and confirm its absence.

For a longer or sensitive prompt, create a UTF-8 file and use `--prompt-file ./prompt.txt` instead of `--prompt`. The options are mutually exclusive. A prompt file avoids shell quoting/expansion and keeps prompt text out of local process arguments and shell history. Prompt size is limited to 1 MiB, but the model's token context can impose a much smaller practical limit.

## Inputs and defaults

`--model-repo`, `--model-file`, and either `--prompt` or `--prompt-file` are required.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model-revision` | `main` | Resolved and recorded as a full commit before rental |
| `--template-hash` | `661d064bbda1f2a133816b6d55da07c3` | Compatible CUDA development/SSH template |
| `--gpu-policy` | `prefer-a100` | Prefer any eligible A100; fall back to another compatible NVIDIA GPU. `a100-only` forbids fallback |
| `--max-hourly-price` | `4.00` | Advertised USD/hour cap, including the requested storage |
| `--timeout-minutes` | `180` | Job deadline covering preflight through response retrieval |
| `--disk-gb` | `64` | Instance disk allocation |
| `--context-size` | `4096` | Total model context, including input and output |
| `--max-tokens` | `1024` | Output token limit, strictly below context size |
| `--temperature` | `0.7` | Generation temperature between 0 and 2 |
| `--output-dir` | `<repository>/modeloutput` | Local output root |
| `--env-file` | `<repository>/.env` | Credential file |

The default template is pinned to `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`. A changed default image is rejected rather than silently trusted. Overrides must provide Python 3.10+, CUDA development tools, build tools (or root/`apt-get` to install missing ones), and noninteractive root SSH access.

Offer selection requires one NVIDIA GPU with at least **24,000 MB per-GPU VRAM** according to VastAI's API units, an AMD64 host, at least 16,000 MB system RAM, sufficient disk/rental duration, verified/rentable status, and reliability of at least 0.98. Template hardware filters also apply. The default CUDA 12.4 image admits compute capabilities 7.5 through 9.0, excluding newer Blackwell GPUs; CUDA 12.8+ overrides admit capabilities through 12.0.

An eligible A100 wins over a cheaper non-A100. Within the preference tier, the lowest hourly price wins. Searches follow full result pages rather than assuming the first mixed-GPU page contains all available A100s.

Default paths resolve relative to `run_job.py`, even when invoked from another directory. Explicit relative `--env-file`, `--output-dir`, and `--prompt-file` paths resolve against your current working directory.

## Output

Each run gets its own directory:

```text
modeloutput/<run-id>/
  response.txt
  completion.json
  metadata.json
  workflow.log
  remote.log
```

`response.txt` contains the exact UTF-8 assistant text, without the prompt or runtime banners. `completion.json` preserves the validated chat completion and finish reason. `metadata.json` records model/revision, template, offer price, instance ID, actual GPU, generation settings, timestamps, and independent job/cleanup statuses. It stores the prompt's SHA-256, not the prompt itself.

The controller retrieves files over SSH/SCP, verifies checksums and agreement between text and completion, then publishes each final artifact atomically. There is no mount of your local filesystem on the GPU instance. Earlier runs are preserved. A `length` finish reason produces an explicit truncation warning.

Successful execution exits `0` only after the response is retrieved **and destruction is confirmed**. Failures exit nonzero. A valid local response is preserved even if subsequent cleanup fails. Diagnostic logs are retrieved when possible; unavailable logs produce a warning.

## Cleanup and interrupted runs

Normal errors, deadlines, SIGINT, and SIGTERM trigger teardown. Cleanup has a separate five-minute budget, including bounded failure-diagnostic retrieval. Further SIGINT/SIGTERM signals are deferred during cleanup.

Recovery state and the temporary SSH identity live under `.vastai-runs/<run-id>/`. The workflow attaches the public key only to the new instance, never registers an account-wide key or modifies your existing SSH configuration. Local private/public key files are removed after confirmed cleanup.

If cleanup fails or the controller is interrupted, keep the recovery directory and run:

```bash
python run_job.py cleanup --run-id <run-id>
```

Use `--env-file` if your credential file is elsewhere. Recovery checks the recorded ID and unique label, reconciles uncertain allocations, and refuses unrelated instances. It is safe to repeat after confirmed deletion. It does not regenerate a missing response.

If creation's response is lost, the workflow reconciles its unique label instead of issuing another rental. If allocation remains uncertain and no instance is visible yet, cleanup reports that uncertainty instead of claiming success. Retry recovery and inspect the VastAI dashboard before starting another run.

**Keep the local controller awake and connected.** A local `finally` block cannot guarantee teardown after SIGKILL, laptop sleep/crash, complete network loss, or an unavailable VastAI control plane. The job deadline is not a provider-side rental expiry. There is no external watchdog in this version.

**Destroy, do not merely stop.** Stopped instances still incur storage charges. When automated cleanup cannot be confirmed, use recovery or destroy the identified instance in the VastAI dashboard.

## Billing, credentials, and scope

USD 4/hour for 180 minutes is a nominal USD 12 limit on the advertised hourly component, **not a total invoice cap**. The offer is repriced for the requested disk and rechecked immediately before creation. Model-download bandwidth, result-upload bandwidth, cleanup time, and provider pricing changes can add charges. The preview shows advertised bandwidth rates and estimated model-download cost.

Only the Hugging Face token and job request go to the remote worker, through encrypted SSH stdin. The VastAI key and `.env` remain local. The runtime listens only on `127.0.0.1`; no public inference endpoint is exposed. First SSH contact uses trust-on-first-use with an isolated known-hosts file; subsequent key changes are rejected. Rented hosts still need to be trusted with your prompt and read token.

Credential values and URL query strings are redacted from collected diagnostics. Local result/state directories are private, and final artifact files use mode `0600`. `.env`, `.venv`, `.vastai-runs`, and default `modeloutput` are ignored by Git. If you choose another output path inside the repository, ignore it separately; do not commit sensitive prompt files or generated results.

This is a synchronous, one-prompt workflow, not training, an agent/tool runner, an interactive session, or a persistent endpoint. Use a standalone text GGUF supported by the pinned runtime. Split GGUFs, multimodal sidecars, and adapters are unsupported; common filename patterns are rejected before rental. Differently named unsupported architectures or files fail explicitly during remote loading. CPU fallback and partial GPU-layer offload are not accepted as successful GPU execution.

Runtime source is pinned to llama.cpp **v0.6.0**, commit `8345f333951c661d166b00e6f9362e553768f292`. Each fresh instance builds `llama-server` for its GPU, so setup time is part of the paid run.

## Local verification

```bash
python -m unittest discover -s tests
python -m compileall -q run_job.py vast_client.py remote_job.py job_utils.py
python run_job.py --help
```

Tests use fake cloud/SSH/downloader interfaces, temporary directories, fake clocks, and a local HTTP fixture. They never read your actual `.env`, rent a GPU, download real weights, or require CUDA. They cover teardown, uncertain allocation, cancellation, recovery isolation, artifact integrity, credential redaction, full GPU-offload evidence, and price/VRAM boundaries.

Live read-only template, model metadata, offer pagination, and CLI dry-run checks have been exercised. **No paid GPU smoke test has been performed**; mocked tests are not proof that remote SSH, CUDA compilation, and real model inference work together on a particular host. A paid smoke test requires separate approval.
