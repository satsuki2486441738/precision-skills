# GPU native workflow completion

This run separates CUDA workflow evidence from the earlier Ascend and CPU runs. It preserves all historical records. Preparing an adapter, installing a dependency, importing a package, or discovering an entrypoint does not complete a native GPU workflow.

## Completion requirement

A completed case must retain the selected framework's actual workflow, effective model and library revisions, CUDA execution evidence, ordered native inputs, and an independently executed numerical reference. Training cases require a nonzero gradient and stored parameter update from the framework-owned optimizer path. Inference cases require a native prefill followed by at least two single-token cached decode calls. Completion describes evidence coverage; numerical acceptance remains `null` without a user-defined threshold.

The first remote SFT case will use four native DDP ranks. Single-GPU local cases and rank-local CPU references are labeled separately. Matching a replicated rank's gradients does not validate a four-rank CPU topology.

## Starting progress

At the initial audit, TensorRT-LLM had retained complete CUDA evidence for two models. Five lanes had earlier GPU component evidence but still needed their complete selected workflows. Eleven lanes had no complete CUDA numerical workflow; three of these use selected Ascend-specific routes. The exact per-lane state and remaining boundary are in `GPU_PROGRESS.json`.

The earlier 17-lane completion report primarily covers NPU routes. It is not a claim that those same 17 lanes completed CUDA execution. Likewise, earlier llama.cpp CPU/CANN results and ms-swift RL CPU results do not complete their CUDA lanes.

## Execution and recovery

The remote plan uses the existing four-H100 ACP configuration and persistent storage. Its existing launcher can run a real command with `--training-command`; the old four-hour performance benchmark is not required for native training. The new workload ends naturally and retains logs, checkpoints, exit receipts and captures. It does not use dummy GPU load or an idle holding command to keep access open.

The portable launch source is preparation, not measured evidence. It installs each conflicting framework stack in a separate environment, fixes source/model identities, runs finite native cases, and recomputes CPU references independently. Current core comparison scripts are included byte-for-byte. Framework-specific adapters remain case code; no operator is replaced with a fabricated implementation.

Use the existing private platform guide for account, image, mount and SSH settings. Those details and authentication files are deliberately excluded from this public record. Once the source archive is published, extract it into a fresh persistent directory, set `PRECISION_WORK_ROOT`, and run `bash launch_all.sh` through the existing real-training launcher. The launcher defaults to SFT, inference, then the first two RL lanes. Additional RL adapters are added only after their native entrypoints and capture boundaries are ready.

## Starting blockers and provenance corrections

- The old four-hour job had ended and its SSH reverse forwarding was closed. This is not evidence of a framework failure.
- The console session had expired. Authentication and allocation must recover before remote measurements can proceed.
- The old llama.cpp identifier `757cad021…` denotes an annotated tag object. The newly prepared adapter records the dereferenced source commit `620d2225cc436ab0b001e3200e590e3b286ef9ab` as well as the tag identity.
- MindIE-LLM and the historically selected MindSpeed-RL/MindSpeed-LLM routes depend on Ascend hardware. Current MindSpeed-LLM also contains a CUDA/NCCL initialization branch in its newer FSDP2 entrypoint; this does not establish complete CUDA support or measured execution. That alternative must be pinned and tested separately before any CUDA completion claim.

## Publication

New measured cases and their portable evidence are committed to `validation/gpt-6-cold-skill-20261019` first. Only then are corresponding records added to the site's evaluation history using immutable source-commit links. A prepared launcher or unresolved environment is not published as a completed precision evaluation.
