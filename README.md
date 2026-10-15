# VastAIBatchJobs

Run a single prompt or an action-capable agent against a Hugging Face GGUF model on a fresh VastAI GPU, retrieve the results into this repository, then destroy the instance. The controller runs locally; model weights, CUDA inference, and agent actions stay on the rented instance.

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

For a longer or sensitive prompt, create a UTF-8 file and use `--prompt-file ./research.md` instead of `--prompt`. Markdown needs no conversion; the whole file becomes the prompt, and referenced local files are not automatically uploaded. The options are mutually exclusive. A prompt file avoids shell quoting/expansion and keeps prompt text out of local process arguments and shell history. Prompt size is limited to 1 MiB, but the model's token context can impose a much smaller practical limit.

## Agent actions

`run` remains a single model response. **`agent` executes actions**, observes their results, and asks the model for the next action until it finishes or reaches a limit. It uses schema-constrained JSON decisions, not model-specific native function calls.

```bash
python run_job.py agent \
  --model-repo unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF \
  --model-file Q3_K_S/Qwen3-Coder-480B-A35B-Instruct-Q3_K_S-00001-of-00005.gguf \
  --prompt-file ./examples/hardhat-research.md \
  --verify-script ./examples/verify_hardhat_research.py \
  --agent-env ALCHEMY_API_KEY \
  --agent-env ETHERSCAN_API_KEY \
  --gpu-policy a100-only \
  --gpu-count 8 \
  --max-hourly-price 7.52 \
  --max-bandwidth-price 0.015625 \
  --timeout-minutes 60 \
  --disk-gb 400 \
  --context-size 65536 \
  --max-steps 40 \
  --max-tokens 4096
```

This is a paid run. Add `--dry-run` to inspect the model/template/offer without executing actions or renting an instance. The supplied [Hardhat task](examples/hardhat-research.md) instructs the agent to create and execute its own scripts on a local Ethereum mainnet fork, read native ETH balances for two randomly sampled verified contracts, fetch deployed bytecode through Alchemy, obtain verified Solidity through Etherscan, and write a source-grounded report. **Alchemy `eth_getCode` returns bytecode, not verified Solidity.** No transactions, deployment, or wallet credentials are needed.

The larger agent example uses Qwen3-Coder-480B-A35B, a 480B-total/35B-active mixture-of-experts model, with approximately 207 GB of Q3_K_S GGUF weights across five files. It requires a multi-GPU instance such as eight 40 GB A100s. Active parameters do not reduce the memory needed to fully offload all model weights. This lower-precision choice reduces download cost versus its approximately 290 GB Q4_K_M set, but model size and quantization do not guarantee reasoning accuracy. Earlier 8B, 30B/3B-active MoE, and 32B coder attempts did not complete this scenario reliably; inspect actual execution and source evidence.

At an observed eight-A100 quote of $7.5126/hour and $0.015625/GB ingress, the Q3_K_S weights cost approximately $3.23 to download. The 60-minute example deadline plus five-minute cleanup costs approximately $8.14 at that hourly quote, before other traffic and billing updates. This is a cost estimate, not an invoice guarantee; recheck the dry-run quote and obtain sufficient approval before renting.

The real 480B test verified all five shards, all eight A100s, 63/63 GPU-offloaded layers and actual 65,536-token serving capacity. However, cold setup/download/loading consumed nearly the entire 43-minute budgeted run: the agent completed only its `package.json` write and began `npm install` before timing out. No research scripts or report were produced. Partial evidence was retrieved and instance destruction independently confirmed. Updated recorded charges were $9.312 when checked, subject to updates. This is a successful model-loading check, **not a successful Hardhat research result**; the longer example timeout needs a separately approved larger budget.

A separate Qwen3-Coder-Next **80B Q4_K_S** test on a 48 GB RTX A6000 executed both Hardhat/source scripts and retrieved genuine Dai/WETH9 Solidity. Its finish was rejected because it saved Etherscan `result[0]` instead of the complete API envelope; its report also contains incorrect prose and declaration/modifier details. It reached the decision limit without an accepted finish. A targeted continuation never reached SSH and was destroyed.

A subsequent **30B/3B-active Q4_K_M** continuation on an RTX 3090 reached SSH too late in its 40-minute window and timed out during Node bootstrap, before loading the model or executing agent actions. All four instances were independently confirmed absent. Their combined recorded charges were $9.543, leaving $0.457 of the renewed $10 authorization, subject to billing updates. Official CUDA-only template overrides are now supported to avoid the unused PyTorch environment, but faster startup on a rented host has not yet been demonstrated. Another paid run requires fresh aggregate billing and a complete-cost check, including cleanup, traffic and late-billing reserves. **End-to-end source-grounded research is not yet verified.** Partial artifacts remain diagnostic evidence, not a successful result.

The research example uses a 65,536-token context because retained script/tool/source history exceeded 32,768 tokens in a real attempt. It reserves 4,096 output tokens per decision so the source-evidence report tables fit. The pinned server caps usable slots to the GGUF's training context even if `--context-size` requests more; the worker checks `/props` after startup and fails before generation if the actual single-slot capacity is smaller. Increasing the allocation alone did not extend the tested Qwen2.5-Coder model's 32,768-token limit. The workflow also fails explicitly instead of silently discarding evidence when history plus reserved output exceeds the verified limit. Keep metadata output compact, avoid dumping bytecode/ABI, and choose context capacity supported by both the model and GPU.

Add `ALCHEMY_API_KEY` and `ETHERSCAN_API_KEY` to the selected `.env` for this example. Each `--agent-env NAME` forwards only that named variable to command processes over SSH stdin. Process-environment precedence is unchanged. Missing/empty selected variables fail before rental. The VastAI/Hugging Face credentials and identity/interpreter variables cannot be forwarded this way. Values are not included in model messages or metadata.

Available actions are `run` with an argv array, `read_file`, `write_file`, `list_files`, and `finish`. CLI arguments are not implicitly shell-expanded. File tools support only relative paths inside the workspace, reject symlinks/hardlinks, and create missing parent directories for writes. `read_file` returns a byte `next_offset` for paginated UTF-8 reading. Node **v22.23.3**, npm, and Python are available remotely; the Node archive is checked against its pinned official SHA-256. Package installs run without root privileges.

| Agent option | Default | Meaning |
| --- | --- | --- |
| `--input-dir` | None | Snapshot a small local input directory into the remote workspace |
| `--verify-script` | None | Require a caller-supplied Python verifier to pass before accepting `finish` |
| `--agent-env NAME` | None | Repeat to select task variables explicitly |
| `--max-steps` | `30` | Model decisions, including finish attempts; maximum 200 |
| `--command-timeout-seconds` | `120` | Wall-clock limit per tool invocation; maximum 600 |
| `--max-artifact-mb` | `64` | Returned workspace byte limit, in MiB; maximum 1024 |
| `--context-size` | `16384` | Agent history and reserved output tokens |
| `--max-tokens` | `2048` | Tokens per model decision |
| `--temperature` | `0` | Agent generation temperature |

Other instance/model/output defaults match `run`. Input snapshots are limited to 64 MiB and 5,000 files. Symlinks, hardlinks, special files, common credential paths, and dependency directories are refused. Known credential values are checked before upload; this cannot detect every possible secret, so review selected input files yourself. Relative `--input-dir` paths resolve against the caller's current directory.

`--gpu-count` selects exactly that many GPUs in one instance and is revalidated before rental. Per-GPU VRAM is not aggregate VRAM; offers must have at least 24 GB per GPU and enough combined memory for the weight files. Context/cache and runtime allocations require additional headroom. Every exposed device is checked against the selected GPU family and count. The server retains one inference slot and full layer offload across the available devices.

For split GGUFs, select shard `00001` with `--model-file`. Preflight derives the complete ordered set, checks every file at the same resolved revision, and requires each shard's authenticated size/SHA-256. Disk/download estimates use the whole set. The worker downloads and verifies every shard in the same directory before starting llama.cpp with the first file. Non-first, malformed, incomplete, or mixed-revision sets fail explicitly; standalone GGUF behavior is unchanged.

`--verify-script` snapshots a trusted, nonempty standard-library Python script of at most 16 KiB before rental, matching the tool's per-argument byte limit. It runs remotely in the workspace with the same non-root identity, approved task variables, and tool limits whenever the model requests `finish`. A nonzero exit, timeout, or tool error rejects finish and returns the observation to the model for correction. The immutable script text is sent through SSH stdin, not copied into the agent-writable workspace or exposed as a local process argument; metadata records its SHA-256. It is not executed on your Mac. Relative paths resolve against the caller's directory.

The supplied verifier checks the Hardhat task's exact contract count, decoded byte lengths, same-block balances/bytecode, hashes, preserved verified source, and report references against fresh API reads. Saved Etherscan evidence must retain the complete response envelope, including `status` and `result`, not only `result[0]`. A malformed saved response gets a repair-and-rerun diagnostic distinct from a failing fresh API request. Each contract's report table must quote its real function declarations and match their named modifiers, including inherited overrides; invented ownership functions and a false `auth` claim on Dai's `burn` are rejected. These checks are tailored to the included legacy-contract pool, not a general Solidity parser. Prose reasoning and caller-condition explanations still need direct review, and the verifier does not prove arbitrary generated code safe. Without a verifier, completion means the model finished and artifacts were retrieved, not that every task claim was independently confirmed.

Commands run as a dedicated non-root Linux user with cleared groups/capabilities, no-new-privileges, a clean environment, resource limits, and a private workspace/home. Standard output and error are each bounded to 16 KiB; truncation is explicit. Remaining processes owned by that identity are killed after each action, so background services are not preserved. The model server uses a private API-key file unavailable to tool processes. **This is not a full hostile-code sandbox or an egress firewall.** Tools have internet access and can use approved task keys; trust the task, dependencies, scripts, and rented host. No local filesystem is mounted.

Native Tokio worker pools are bounded to two threads. On many-core hosts, larger
pools can exceed the tool identity's 64-process/thread limit: a Hardhat script
can abort while its CLI still exits zero. `TOKIO_WORKER_THREADS` is reserved and
cannot be forwarded with `--agent-env`. Check actual output and expected files,
not only the CLI exit code.

Agent output:

```text
modeloutput/<run-id>/
  response.txt
  agent-report.json
  agent-transcript.json
  artifacts/
    <generated scripts, outputs, source evidence, and report>
  metadata.json
  workflow.log
  remote.log
```

`response.txt` is the summary from the final JSON decision, not an OpenAI completion envelope. The report records outcome, limits, decisions, and the workspace manifest/checksums. The transcript preserves decisions, model completions, and actual tool observations; it can contain private task/file content. There is no `completion.json` for agent mode. Dependencies/caches named `node_modules`, `.git`, `.venv`, `venv`, `__pycache__`, `.cache`, or `.npm` are intentionally excluded and listed in the manifest. Keep the returned package manifest/lockfile to reinstall dependencies.

Credential-bearing files or unsafe/oversized archives are refused rather than silently copied. Generated script bytes are not rewritten. Tool failures are returned to the model for correction, with a recovery hint after three consecutive failures; five consecutive failures stop the run explicitly. Malformed or token-limited decisions never execute; context overflow and exhausted step/job budgets fail explicitly. Failed runs preserve available partial scripts, outputs, and transcripts before teardown. Finishing alone is not evidence that every task claim is correct: inspect the execution transcript and source references.

## Inputs and defaults

`--model-repo`, `--model-file`, and either `--prompt` or `--prompt-file` are required.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model-revision` | `main` | Resolved and recorded as a full commit before rental |
| `--template-hash` | `661d064bbda1f2a133816b6d55da07c3` | Compatible CUDA development/SSH template |
| `--offer-id` | None | Select one eligible marketplace offer; never silently fall back |
| `--gpu-policy` | `prefer-a100` | Prefer any eligible A100; fall back to another compatible NVIDIA GPU. `a100-only` forbids fallback |
| `--max-hourly-price` | `4.00` | Advertised USD/hour cap, including the requested storage |
| `--max-bandwidth-price` | None | Optional advertised USD/GB cap for both ingress and egress, rechecked before creation |
| `--timeout-minutes` | `180` | Job deadline covering preflight through response retrieval |
| `--gpu-count` | `1` | Exact GPUs in one instance; every GPU needs at least 24 GB VRAM |
| `--disk-gb` | `64` | Instance disk allocation |
| `--context-size` | `4096` | Total model context, including input and output |
| `--max-tokens` | `1024` | Output token limit, strictly below context size |
| `--temperature` | `0.7` | Generation temperature between 0 and 2 |
| `--output-dir` | `<repository>/modeloutput` | Local output root |
| `--env-file` | `<repository>/.env` | Credential file |

The default template is pinned to `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`. A changed default image is rejected rather than silently trusted. Overrides may also use official `nvidia/cuda` development tags such as `12.4.1-devel-ubuntu22.04`. They must provide Python 3.10+, CUDA development tools, build tools, and noninteractive root SSH access. On root Debian/Ubuntu images with `apt-get`, bootstrap can install missing Python, Python venv support, and build tools before invoking the worker. Python versions below 3.10 and missing CUDA toolkits still fail explicitly. CUDA-only images avoid downloading an unused PyTorch environment, but startup time depends on the host and image cache.

Offer selection requires exactly `--gpu-count` NVIDIA GPUs, each with at least **24,000 MB per-GPU VRAM** according to VastAI's API units, an AMD64 host, at least 16,000 MB system RAM, sufficient disk/rental duration, verified/rentable status, and reliability of at least 0.98. Combined advertised VRAM must cover the weight files, with additional headroom needed for context/runtime memory. Template hardware filters also apply. The default CUDA 12.4 image admits compute capabilities 7.5 through 9.0, excluding newer Blackwell GPUs; CUDA 12.8+ overrides admit capabilities through 12.0.

An eligible A100 wins over a cheaper non-A100. Within the preference tier, the lowest hourly price wins. Searches follow full result pages rather than assuming the first mixed-GPU page contains all available A100s.

Use `--offer-id` to select a different host after a marketplace/SSH failure. This is
the rentable offer ID, not a previously created instance ID. The selected offer
still passes template, GPU, duration, storage, and price checks and is revalidated
before rental. An unavailable or ineligible selection fails without falling back.
Confirm the previous run's cleanup before starting another paid attempt.

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

The Hugging Face token, job request, and any explicitly selected agent task variables go to the remote worker through encrypted SSH stdin. The VastAI key and `.env` remain local. The Hugging Face token is used by the privileged downloader, not injected into agent command processes. The runtime listens only on `127.0.0.1`; no public inference endpoint is exposed. First SSH contact uses trust-on-first-use with an isolated known-hosts file; subsequent key changes are rejected. Rented hosts still need to be trusted with your prompt and selected credentials.

Credential values and URL query strings are redacted from collected diagnostics. Local result/state directories are private, and final artifact files use mode `0600`. `.env`, `.venv`, `.vastai-runs`, and default `modeloutput` are ignored by Git. If you choose another output path inside the repository, ignore it separately; do not commit sensitive prompt files or generated results.

Both modes are synchronous, one-task-per-instance workflows, not training, interactive sessions, or persistent endpoints. `run` produces one response; `agent` can perform bounded multi-step actions. Use a supported standalone text GGUF or the first shard of a complete split set. Multimodal sidecars and adapters remain unsupported; common filename patterns are rejected before rental. Differently named unsupported architectures or files fail explicitly during remote loading. CPU fallback and partial GPU-layer offload are not accepted as successful GPU execution.

The worker sets server verbosity to `4`: the pinned runtime hides its model-loader and GPU-offload diagnostics at the default level `3`. After healthy startup, it waits briefly for the asynchronous logger to report nonzero full layer offload from this server, not an earlier log entry. Missing, zero, or partial offload still fails before generation and triggers teardown.

Runtime source is pinned to llama.cpp **v0.6.0**, commit `8345f333951c661d166b00e6f9362e553768f292`. Each fresh instance builds `llama-server` for its GPU, so setup time is part of the paid run.

## Local verification

```bash
python -m unittest discover -s tests
python -m compileall -q run_job.py vast_client.py remote_job.py job_utils.py agent_job.py agent_tools.py agent_utils.py
python run_job.py --help
```

Tests use fake cloud/SSH/downloader interfaces, temporary directories, fake clocks, and local HTTP fixtures. Agent tests create and execute real local test scripts and inspect their outputs; Linux privilege setup is verified separately through its contracts. Tests never read your actual `.env`, rent a GPU, download real weights, or require CUDA. They cover teardown, uncertain allocation, cancellation, recovery isolation, artifact integrity, credential redaction, full GPU-offload evidence, price/VRAM boundaries, actual multi-step actions, and negative tool/archive/context/step-limit boundaries.

Live read-only template, model metadata, offer pagination, and CLI dry-run checks have been exercised. The default-verbosity failure from an earlier paid run is reproduced by a local regression.

**A real end-to-end A100 smoke test completed successfully on 2026-10-15 (UTC).** It used `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` / `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` on an A100-SXM4-40GB with the pinned template/runtime. All **33/33 layers** were offloaded to the GPU, and the exact response `GPU test successful.` was retrieved and verified locally with finish reason `stop`. Instance destruction was independently confirmed through the VastAI API. The run took approximately 12 minutes, including a fresh CUDA build, with A100-only selection, a USD 1/hour cap, a 30-minute deadline, 64 output tokens, and temperature 0.

Further paid tests require separate approval. This result verifies the tested model/template/runtime/GPU combination, not every supported model or future marketplace offer.
