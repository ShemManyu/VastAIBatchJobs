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
  --model-repo unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  --model-file Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --model-revision b17cb02dd882d5b6ab62fc777ad2995f19668350 \
  --template-hash 5ec457ba5e023038cdcfe7d4aab5c207 \
  --prompt-file ./examples/hardhat-research.md \
  --verify-script ./examples/verify_hardhat_research.py \
  --agent-env ALCHEMY_API_KEY \
  --agent-env ETHERSCAN_API_KEY \
  --gpu-policy prefer-a100 \
  --max-hourly-price 0.60 \
  --max-bandwidth-price 0.003 \
  --timeout-minutes 30 \
  --disk-gb 64 \
  --context-size 49152 \
  --max-steps 60 \
  --max-tokens 4096
```

This is a paid run. Add `--dry-run` to inspect the model/template/offer without executing actions or renting an instance. The supplied [Hardhat task](examples/hardhat-research.md) instructs the agent to create and execute its own scripts on a local Ethereum mainnet fork, read native ETH balances for two randomly sampled verified contracts, fetch deployed bytecode through Alchemy, obtain verified Solidity through Etherscan, and write a source-grounded report. **Alchemy `eth_getCode` returns bytecode, not verified Solidity.** No transactions, deployment, or wallet credentials are needed.

Start with the single-GPU Qwen3-Coder-30B-A3B example above: its Q4_K_M download is approximately 18.56 GB. A 24 GB GPU still needs additional memory for context/cache/runtime, and actual full offload and serving context are checked before actions. Hourly and bandwidth limits are quote caps, not an all-in dollar-budget guarantee. Account for startup, model/dependency traffic, storage and cleanup before renting.

Qwen3-Coder-480B-A35B is an optional larger model, not the smoke-test default. Its 480B-total/35B-active mixture of experts needs approximately 207 GB of Q3_K_S weights across five files and a multi-GPU instance such as four 80 GB or eight 40 GB A100s. Its repository is `unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF`; select `Q3_K_S/Qwen3-Coder-480B-A35B-Instruct-Q3_K_S-00001-of-00005.gguf` and size GPU count, disk, time and budget explicitly. Active parameters do not reduce the memory needed to fully offload all weights. Model size and quantization do not guarantee reasoning accuracy.

At an observed eight-A100 quote of $7.5126/hour and $0.015625/GB ingress, the Q3_K_S weights alone cost approximately $3.23 to download. A 60-minute deadline plus five-minute cleanup costs approximately $8.14 at that hourly quote, before other traffic and billing updates. This is a cost estimate, not an invoice guarantee; recheck the dry-run quote and obtain sufficient approval before renting.

The real 480B test verified all five shards, all eight A100s, 63/63 GPU-offloaded layers and actual 65,536-token serving capacity. However, cold setup/download/loading consumed nearly the entire 43-minute budgeted run: the agent completed only its `package.json` write and began `npm install` before timing out. No research scripts or report were produced. Partial evidence was retrieved and instance destruction independently confirmed. Recorded charges were $9.312: $5.672 GPU, $0.205 storage, $3.344 download and $0.091 upload, subject to updates. This is a successful model-loading check, **not a successful Hardhat research result**.

A separate Qwen3-Coder-Next **80B Q4_K_S** test on a 48 GB RTX A6000 executed both Hardhat/source scripts and retrieved genuine Dai/WETH9 Solidity. Its finish was rejected because it saved Etherscan `result[0]` instead of the complete API envelope; its report also contains incorrect prose and declaration/modifier details. It reached the decision limit without an accepted finish. A targeted continuation never reached SSH and was destroyed.

A subsequent **30B/3B-active Q4_K_M** continuation on an RTX 3090 reached SSH too late in its 40-minute window and timed out during Node bootstrap. A CUDA-only Titan RTX attempt reached SSH after about 15 minutes and installed the prerequisites, but failed before model download because Ubuntu 22.04's CMake 3.22 cannot interpret CUDA architecture `native`. The worker now queries the actual devices and passes explicit numeric architectures instead. A faster RTX 3090 then completed CUDA setup, full 49/49 offload and both real research scripts, but failed after 27 decisions when retained history exhausted its 49,152-token context. Its report had not passed verification. All six instances were independently confirmed absent. Their last recorded combined charges were $9.672 under that previous $10 authorization, subject to updates. **End-to-end source-grounded research is not yet verified.** Partial artifacts remain diagnostic evidence, not a successful result.

A new-budget 30B continuation on one 80 GB A100 successfully reran both research scripts, fully offloaded 49/49 layers, and recovered twice from measured context exhaustion using the compact ledger. It nevertheless reached 60 decisions without an accepted report: its generated extractor confused Solidity keywords with named modifiers, and ineffective report rewrites repeated the same legitimate verifier failure. The instance was independently confirmed absent, with $0.276 recorded when checked, subject to updates. Context recovery worked; report correctness did not.

A subsequent 80B continuation on an 80 GB A100 reran both scripts and passed the declaration/data gate in 36 decisions with 49/49 offload and 65,536 context. Independent checks confirmed real stdout, fresh Alchemy/Etherscan data, private artifact hashes and destruction. Direct source review nevertheless rejected its report: Dai `push` and WETH9 `transfer` incorrectly required recipient allowances, and contract-purpose/state-transition prose was missing. Updated recorded charges were $0.586, bringing the first two continuations to $0.862, subject to updates. Mechanical verification alone is not research acceptance. The example verifier now detects the known self-funded source calls and requires their balance-only caller formula before accepting finish.

A later 80B repair exposed a separate host issue: npm and both research commands silently aborted with `SIGABRT`, while synchronous Node file reads still worked. The then-current artifact-only checker accepted pre-existing data even though no new research execution succeeded. That result was rejected, not counted as completed research. A model-free diagnostic isolated per-UID process accounting on the shared host; randomized mapped UIDs fix the native tools without relaxing resource limits. The caller checker now reads protected current-run command evidence and rejects prior-file printouts as execution proof. Original rejected artifacts remain unchanged.

After those fixes, a **480B Q3_K_S continuation on four 80 GB A100s** completed 19 decisions. All five shards, all four actual device inventories/model buffers, 63/63 offload and 65,536-token serving capacity were verified. npm, the exact Hardhat command and the source command genuinely succeeded under random UID 1682537860 with the original resource caps. Returned stdout matched the data, fresh APIs confirmed the balances/sources, and retrieval/destruction succeeded. This continued the unchanged earlier workspace, not a fresh-from-scratch research run.

Direct review still rejected the report: it mislabeled Dai's legacy permit as EIP-2612 and mentioned a nonexistent WETH9 `push`. The renderer consumes the prose notes but ignores its extracted declaration/modifier arrays, and the report omits the requested wei presentation despite retaining exact wei in JSON. Original model artifacts and the completed controller status remain unchanged; a separate manual-review record distinguishes real execution success from report acceptance.

A separate reviewer-authored `reviewed-report.md` beside the returned artifacts corrects the source claims, includes exact wei/ETH and covers all 11+6 named functions. Its source declarations, observed amounts and unchanged raw artifact hashes were checked. It is not model-authored output or evidence of autonomous report accuracy. No further rental was started for these corrections.

All 24 registered allocations were independently confirmed absent. The four-A100 continuation's updated recorded charge was $2.608, bringing all five instances under the renewed authorization to $3.862 when checked, subject to updates. No raw model report from this authorization has yet passed complete source review.

The research example uses a 49,152-token context and reserves 4,096 output tokens per decision. The pinned server caps usable slots to the GGUF's training context even if `--context-size` requests more; the worker checks `/props` after startup and fails before generation if the actual single-slot capacity is smaller. Increasing the allocation alone did not extend the tested Qwen2.5-Coder model's 32,768-token limit.

Before sending an oversized agent prompt, the worker replaces older steps with an explicitly marked action/status ledger, retaining the system instructions, original task and newest observation. It remeasures the rendered prompt, including reserved output, before generation. Original steps remain unchanged in the full transcript and are available to the model with `{"action":"read_history","step":1,"offset":0}`. History reads use bounded UTF-8 byte pagination, do not execute old actions, and redact known credentials before calculating offsets. Compaction events are recorded in the transcript and counted in the agent report. If the original task, ledger and newest observation still cannot fit, the run fails explicitly rather than silently dropping them. Keep metadata output compact and avoid dumping bytecode, ABI or license text into context.

Add `ALCHEMY_API_KEY` and `ETHERSCAN_API_KEY` to the selected `.env` for this example. Each `--agent-env NAME` forwards only that named variable to command processes over SSH stdin. Process-environment precedence is unchanged. Missing/empty selected variables fail before rental. The VastAI/Hugging Face credentials and identity/interpreter variables cannot be forwarded this way. Values are not included in model messages or metadata.

Available actions are `run` with an argv array, `read_file`, `read_history`, `write_file`, `list_files`, and `finish`. CLI arguments are not implicitly shell-expanded. File tools support only relative paths inside the workspace, reject symlinks/hardlinks, and create missing parent directories for writes. `read_file` and `read_history` return a byte `next_offset` for paginated UTF-8 reading. History access is limited to earlier completed steps in the current run, not arbitrary files or other runs. Node **v22.23.3**, npm, and Python are available remotely; the Node archive is checked against its pinned official SHA-256. Package installs run without root privileges.

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

The verifier receives one argument, `sys.argv[1]`, naming a temporary root-owned, read-only JSON file outside the writable workspace. It contains the current run's real command records: `step`, `action` and `observation`. Known credentials are redacted, and the file is removed after verification. Use this evidence to distinguish actual successful execution from a model's claims or pre-existing output files. Verifiers that do not need command history can ignore the argument.

The supplied verifier requires successful current-run execution and complete nonempty stdout from both exact research commands, then checks contract count, decoded byte lengths, same-block balances/bytecode, hashes, preserved verified source, and report references against fresh API reads. Printing a prior JSON file does not count as executing its generating script. Saved Etherscan evidence must retain the complete response envelope, including `status` and `result`, not only `result[0]`. Each contract's report table must quote its real function declarations and match their named modifiers, including inherited overrides; invented ownership functions and a false `auth` claim on Dai's `burn` are rejected. Contract sections must also contain nonempty `Purpose:` and `State transitions:` paragraphs. These checks are tailored to the included legacy-contract pool, not a general Solidity parser or prose grader. Prose accuracy and remaining caller-condition explanations still need direct review, and the verifier does not prove arbitrary generated code safe. Without a verifier, completion means the model finished and artifacts were retrieved, not that every task claim was independently confirmed.

Commands run as a dedicated non-root Linux user with cleared groups/capabilities, no-new-privileges, a clean environment, resource limits, and a private workspace/home. Standard output and error are each bounded to 16 KiB; truncation is explicit. Remaining processes owned by that identity are killed after each action, so background services are not preserved. The model server uses a private API-key file unavailable to tool processes. **This is not a full hostile-code sandbox or an egress firewall.** Tools have internet access and can use approved task keys; trust the task, dependencies, scripts, and rented host. No local filesystem is mounted.

Native Tokio worker pools are bounded to two threads. On many-core hosts, larger
pools can exceed the tool identity's 64-process/thread limit: a Hardhat script
can abort while its CLI still exits zero. `TOKIO_WORKER_THREADS` is reserved and
cannot be forwarded with `--agent-env`. Check actual output and expected files,
not only the CLI exit code.

Tool identities use random unused UIDs within the container's mapped UID range,
not a reused system UID such as 999. A model-free control on a shared host
reproduced silent `SIGABRT` from asynchronous Node filesystem/DNS calls and npm
with UID 999 under the 64-thread limit. Raising address space did not fix it;
a distinct UID fixed npm and native Hardhat execution under the original
64-thread/16-GiB limits. Those limits remain unchanged. A native asynchronous
Node probe now rejects a broken tool runtime before the model's first action.
High-UID account creation uses `--no-log-init` to avoid oversized login records.

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

Credential-bearing files or unsafe/oversized archives are refused rather than silently copied. Generated script bytes are not rewritten. Tool failures are returned to the model for correction, with a recovery hint after three consecutive failures; five consecutive failures stop the run explicitly. Five identical finish rejections also stop the run even if successful file edits or other actions intervene. A changed finish diagnostic resets that separate counter, allowing genuine verification progress. Malformed or token-limited decisions never execute; unrecoverable context overflow and exhausted step/job budgets fail explicitly. Failed runs preserve available partial scripts, outputs, and transcripts before teardown. Finishing alone is not evidence that every task claim is correct: inspect the execution transcript and source references.

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

The default template is pinned to `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`. A changed default image is rejected rather than silently trusted. Overrides may also use official `nvidia/cuda` development tags such as `12.4.1-devel-ubuntu22.04`. They must provide Python 3.10+, CUDA development tools, build tools, and noninteractive root SSH access. On root Debian/Ubuntu images with `apt-get`, bootstrap can install missing Python, Python venv support, and build tools before invoking the worker. The worker queries every GPU's compute capability and passes explicit numeric CUDA architectures to CMake, avoiding the unsupported `native` value on Ubuntu 22.04's CMake 3.22. Python versions below 3.10, missing CUDA toolkits, and incomplete/malformed capability inventories still fail explicitly. CUDA-only images avoid downloading an unused PyTorch environment, but startup time depends on the host and image cache.

Offer selection requires exactly `--gpu-count` NVIDIA GPUs, each with at least **24,000 MB per-GPU VRAM** according to VastAI's API units, an AMD64 host, at least 16,000 MB system RAM, sufficient disk/rental duration, verified/rentable status, and reliability of at least 0.98. Combined advertised VRAM must cover the weight files, with additional headroom needed for context/runtime memory. Template hardware filters also apply. The default CUDA 12.4 image admits compute capabilities 7.5 through 9.0, excluding newer Blackwell GPUs; CUDA 12.8+ overrides admit capabilities through 12.0. GPU-name verification accepts the marketplace's `RTX PRO 6000 Max-Q` alias for NVIDIA's longer Blackwell Max-Q product name, without accepting the non-Max-Q workstation or server variants in its place.

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
