# GPU native workflow completion (in progress)

This run separates CUDA workflow evidence from the earlier Ascend and CPU runs. It preserves all historical records. Preparing an adapter, installing a dependency, importing a package, or discovering an entrypoint does not complete a native GPU workflow.

## Measured CUDA results

The local RTX 5060 supplement has **22 native CUDA cases**, with independently executed CPU references and retained portable arrays. These are single-GPU cases. They complete seven additional selected library lanes; together with the retained TensorRT-LLM evidence, eight NVIDIA-capable lanes have complete selected native workflow evidence. Four-rank DDP is still pending.

- TRL SFTTrainer, LLaMAFactory, ms-swift SFT and Axolotl: each ran tiny Qwen2 random weights and the pinned Qwen3-0.6B checkpoint in FP32 and BF16 modes. Sixteen native data-preprocessing/train/SGD cases produced 80 original comparison groups. See `trl/summary.json`, `factory/summary.json`, `swift/summary.json` and `axolotl/summary.json` with four per-case ZIPs in each directory.
- LMDeploy PyTorch CUDA and llama.cpp GGML_CUDA: two pinned pretrained models per backend. Each case retains a prefill and two real single-token cached decodes, full-vocabulary logits and an independently loaded CPU reference. Four cases produced 8 overall and 24 stage comparisons. The portable replay index contains the 8 overall groups. See `inference/native-summary.json` and its 16.1 MB indexed evidence ZIP. llama.cpp offloaded 31/31 SmolLM2 layers and 25/25 Qwen layers; this is actual CUDA offload.
- ms-swift RL: tiny FP32 and Qwen3-0.6B BF16 each completed native generation, synthetic execution-only rewards, group advantages, objective, full-model backward/clipping and native AdamW updates. Independent CPU computation produced 8 original comparison groups. Numerical tensor comparison covers 98,304 sampled coordinates out of 19,709,312 and 751,632,384 native parameter elements respectively. Complete parameter inventories remain available. See `swift-rl/results.json` and its two small indexed ZIPs.

All numerical acceptance values remain `null`. For example, the pretrained BF16 RL native objective is zero while the independent CPU objective is about 0.03235. The measured difference is retained, not presented as an accuracy pass. Synthetic rewards do not establish response quality or production RL performance.

Both Swift RL cases condition on the actual native old-policy log-probability anchors, stored as FP32 `[2,4]` tensors. The independent CPU computation recomputes the current policy on the fixed generated histories; it does not recompute both old and current policy scores. `swift-rl/*-fixed-inputs.zip` adds the original-bit input tensors, masks, rollout/group identities, rewards, objective/optimizer settings, and a typed loader. Its independent CPU forward/backward reproduces the retained reference loss and full clipping norm. These small supplements preserve the original comparison ZIPs and do not add native cases or numerical acceptance decisions. See `swift-rl/fixed-inputs-index.json` for hashes, input scope and verification.

The SFT objective uses the actual native effective denominator, which can differ from the number of valid shifted target tokens. The CPU execution replays that denominator. Consequently, native objective and selected-token NLL are distinct observations. FP32/BF16 names denote requested execution modes; original parameter/logit/gradient storage dtypes are retained explicitly.

The pretrained SFT checkpoint's eight recorded files were independently matched to the immutable Hugging Face revision after execution. `SOURCE_IDENTITY_AUDIT.json` preserves this verification separately from the original collector's conservative `local_source_revision_verified=false` field. `HARDWARE_AUDIT.json` supplies the same-host GPU identity where the Axolotl completion receipt omitted its marketing model.

Public exports replace private text paths and machine identifiers. Every protected tensor/array file remains byte-identical to its raw capture. ZIP indices, SHA256 values and replay instructions are retained. `sft-environment-and-native-logs.zip` contains actual commands, native stdout, environment resolution/check/freeze records, and failed preparation attempts. The old read-only TRL environment has unrelated installed-package metadata conflicts (`pip check` failed); its actual TRL cases succeeded. The three new SFT environments pass their own checks. See `sft-guide-notes.md` for the exact boundary.

## Remaining work

OpenRLHF, slime, NeMo RL, AReaL, ROLL and official NVIDIA Megatron Bridge still need their selected full CUDA workflow and independent reference. The four-rank TRL run and per-rank audit also remain. Native imports, resolved locks, local CUDA IPC failures, source builds and earlier NPU execution do not close these items.

The selected MindIE-LLM, MindSpeed-RL and historical MindSpeed-LLM routes require Ascend hardware and remain separately identified. A newer MindSpeed-LLM FSDP2 CUDA initialization branch has been inspected but has not been validated as a replacement route.

## Completion requirement

A completed case must retain the selected framework's actual workflow, effective model and library revisions, CUDA execution evidence, ordered native inputs, and an independently executed numerical reference. Training cases require a nonzero gradient and stored parameter update from the framework-owned optimizer path. Inference cases require a native prefill followed by at least two single-token cached decode calls. Completion describes evidence coverage; numerical acceptance remains `null` without a user-defined threshold.

The first remote SFT case will use four native DDP ranks. Single-GPU local cases and rank-local CPU references are labeled separately. Matching a replicated rank's gradients does not validate a four-rank CPU topology.

## Starting progress

At the initial audit, TensorRT-LLM had retained complete CUDA evidence for two models. Five lanes had earlier GPU component evidence but still needed their complete selected workflows. Eleven lanes had no complete CUDA numerical workflow; three of these use selected Ascend-specific routes. The exact per-lane state and remaining boundary are in `GPU_PROGRESS.json`.

The earlier 17-lane completion report primarily covers NPU routes. It is not a claim that those same 17 lanes completed CUDA execution. Likewise, earlier llama.cpp CPU/CANN results and ms-swift RL CPU results do not complete their CUDA lanes.

## Execution and recovery

The remote plan uses the existing four-H100 ACP configuration and persistent storage. Its existing launcher can run a real command with `--training-command`; the old four-hour performance benchmark is not required for native training. The new workload ends naturally and retains logs, checkpoints, exit receipts and captures. It does not use dummy GPU load or an idle holding command to keep access open.

The portable launch source is preparation, not measured evidence. It installs each conflicting framework stack in a separate environment, fixes source/model identities, runs finite native cases, and recomputes CPU references independently. Current core comparison scripts are included byte-for-byte. Framework-specific adapters remain case code; no operator is replaced with a fabricated implementation.

Use `source-launch-v3.zip` with its `SOURCE-v3.json` hash for new jobs. The first restored four-GPU worker validated four visible GPUs and opened the existing SSH tunnel, then exited because the retained Ubuntu image lacked `python3-venv/ensurepip`. Version 3 installs that missing OS package, clears inherited image `PYTHONPATH`, and starts the real workload through the existing launcher. The replacement job is submitted; it is not counted as native numerical completion until its captures and independent comparisons finish. Bridge v3 uses the dereferenced NVIDIA source commit `7af9601c9da746d21481194356295bfe23df3737`; the older `717c62de…` identifier denotes an annotated tag object.

Use the existing private platform guide for account, image, mount and SSH settings. Those details and authentication files are deliberately excluded from this public record. Once the source archive is published, extract it into a fresh persistent directory, set `PRECISION_WORK_ROOT`, and run `bash launch_all.sh` through the existing real-training launcher. The restored job explicitly selects `PRECISION_GROUPS='sft bridge'`; the combined launcher's unused default group list is not a validated matrix. Later RL and Slime adapters use separately frozen runtime payloads with their own source identities, locks and manifests.

Subsequent preparation exposed an inherited, unresolvable extra package index and slow per-file environment installation on the shared filesystem. Fresh attempts clear the stale package index and use node-local isolated environments with persistent source, model, log and lock records. Git's ownership guard on the shared filesystem is handled with a task-specific configuration listing only the exact official source checkouts. These are environment recovery steps, not completed precision measurements. Superseded attempts remain recorded.

## Starting blockers and provenance corrections

- The old four-hour job had ended and its SSH reverse forwarding was closed. This is not evidence of a framework failure.
- The console session had expired. Authentication and allocation must recover before remote measurements can proceed.
- The old llama.cpp identifier `757cad021…` denotes an annotated tag object. The newly prepared adapter records the dereferenced source commit `620d2225cc436ab0b001e3200e590e3b286ef9ab` as well as the tag identity.
- MindIE-LLM and the historically selected MindSpeed-RL/MindSpeed-LLM routes depend on Ascend hardware. Current MindSpeed-LLM also contains a CUDA/NCCL initialization branch in its newer FSDP2 entrypoint; this does not establish complete CUDA support or measured execution. That alternative must be pinned and tested separately before any CUDA completion claim.

## Publication

New measured cases and their portable evidence are committed to `validation/gpt-6-cold-skill-20261019` first. Only then are corresponding records added to the site's evaluation history using immutable source-commit links. A prepared launcher or unresolved environment is not published as a completed precision evaluation.

The 22 single-GPU cases were published at `a9fe8dff25de1c3bf806955b06e85d56a9c0a26b`. Website commit `65a0d984c97d12166928baf58f2a735ac7c5e4b2` added 14 entries containing their 120 original comparison groups and 96 portable replay groups. The deployed site contains 57 evaluations and preserves the previous 43 records and nine repair verification records. `PUBLICATION.json` records the validation and deployment checks. This publication does not close the remaining H100 workflows.
