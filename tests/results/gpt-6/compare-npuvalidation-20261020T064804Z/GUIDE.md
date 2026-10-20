# What was done and how to resume

## Scope and status

The request covers 16 libraries in 17 task lanes: ms-swift has separate SFT and RL lanes. This follow-up prioritizes actual Ascend execution for the complete list. It has **zero retrieved and verified native framework passes** at this checkpoint. Some detached jobs may have advanced after connectivity was lost; their outcomes are unknown. The record does not assert that they failed or that all requested tests are complete.

The starting skill/repository snapshot is `24486c13301bc3d5d4cfbf2ddb2e298febff6af3`, on `validation/gpt-6-cold-skill-20261019`. The earlier cold evaluation and its evidence are preserved. These workers initially prepared their assigned recipes without the parent findings, then received actual environment information for this coordinated follow-up. This follow-up is not another independent cold trial or a model ranking.

## Observed host and environment preparation

Actual authenticated inspection found aarch64 Linux, 16 logical Ascend910_9382 devices, 64 GiB per device, driver 25.5.2, CANN 9.1.0, Python 3.12.13, Torch 2.9.0+cpu and TorchNPU 2.9.0.post2. The installed Transformers version was 5.3.0. vLLM distribution metadata was `0.18.0+empty` and vLLM-Ascend metadata was `0.0.0`; those placeholders do not establish an inference engine's readiness.

The first noninteractive Torch import failed while reading the CANN version. Sourcing the vendor `set_env.sh` before Python initialization resolved it without changing packages. Device enumeration then succeeded, and a real NPU square operation executed. This initialization failure is retained as an environment observation, not a precision-skill defect.

The official [Ascend compatibility table at a fixed commit](https://github.com/Ascend/pytorch/blob/cf04f800cdebb3bdb650085d746b5433e869f145/COMPATIBILITY.en.md) pairs CANN 9.1.0 with TorchNPU product 26.1.0, including package `2.9.0.post6` for Torch 2.9.0 and `2.7.1.post8` for Torch 2.7.1. The original post2 package belongs to its CANN 9.0 row. Task-private environments therefore selected the matching package revisions; a successful base-runtime probe was not used as proof of a fully supported framework combination. Exact framework/Python compatibility still requires execution.

Persistent task storage has roughly 1.3 TB available, while the writable container root is limited to 10 GiB. Environment installations, caches, temporary builds, private source copies and results were directed to persistent storage. The source model pool includes pretrained Qwen3-0.6B. Planned tiny Qwen2/Qwen3 models are explicitly seeded random fixtures and do not measure pretrained capability.

## Actual stages and unresolved results

- **AReaL:** the official `ascend-v1.0.4` source was selected near the host's Torch/vLLM versions. Private installation completed. The first native entry failed before engine startup because `tenacity` was missing; the second attempt included a repair and was launched. Its result is unknown. The source recipe's CANN 9.0/post2 combination differs from the host CANN 9.1/post6 attempt.
- **ROLL:** private dependency preparation completed. The native FSDP2/vLLM RLVR adapter was prepared; its upload failed when transport became unavailable. No native rollout/update was observed.
- **OpenRLHF:** the Ascend route being prepared is the explicitly identified `zhuo97/OpenRLHF` community adaptation. Its installer was launched but its result was not retrieved. Its published older A2/CANN 8.1 recipe differs from this A3/CANN 9.1 attempt. A future community-fork pass must not be counted as an untouched upstream OpenRLHF pass.
- **MindSpeed-LLM:** a private Torch 2.7.1+cpu / TorchNPU 2.7.1.post8 / Transformers 5.2.0 environment installed successfully. The first native attempt exposed an inherited torchvision 0.24/Torch 2.7 mismatch. Private CPU torchvision 0.22.1 repaired that dependency. The repaired native one-step pretraining attempt was launched; its outcome and tensors remain uncollected.
- **MindIE-LLM:** private Torch 2.9.0+cpu / TorchNPU 2.9.0.post6 / Transformers 4.51.3 imports succeeded. Native ATB library construction was started from the official source because available Python bindings did not match host Python 3.12. A failed CMake pin was replaced with an available private CMake version. The second build outcome is unknown. No generation is claimed.
- **MindSpeed-RL:** exact source archives and native conversion, dataset, rollout and optimizer collectors were prepared. They were not executed. Ray device mapping must be verified before launching the original training CLI.
- **Slime:** its official NPU deployment patch recipe exists; it is not classified CUDA-only. Exact source and binary dependencies were transferred, and a private Python 3.11 / Torch 2.8 installation began. The published CANN 8.5 recipe differs from actual CANN 9.1, so device ABI and native training remain unverified. Patch-format and source-context repairs were checked locally and are disclosed in the detailed progress record; they are not remote numerical results.
- **TRL SFTTrainer, ms-swift SFT/RL, LLaMAFactory and Axolotl:** parent-owned private installations were launched. Last retrieved logs show Torch/NPU/framework dependency stages, with no native entrypoint completion. Complete SFT adapters were subsequently prepared locally, but are explicitly unexecuted.
- **llama.cpp and LMDeploy:** official CANN/Ascend routes were identified. NPU builds and inference were not started in this checkpoint. Their earlier CPU/CUDA cases do not satisfy this NPU request.
- **NeMo RL, upstream Megatron Bridge and TensorRT-LLM:** selected upstream sources and support requirements were inspected. NeMo/Bridge paths use CUDA/NCCL, and TensorRT-LLM's [support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html) lists NVIDIA GPUs. Prepared native-startup checks were not executed before endpoint loss. A support finding is recorded separately from an observed startup failure. The Slime-prescribed Bridge fork and NPU patch are a separate identity.

## Adapters and completion criteria

SFT adapters invoke `SFTTrainer.train`, the complete LLaMAFactory `run_exp` pipeline, or ms-swift's `sft_main` pipeline. They retain native tokenization, loss, backward and optimizer execution. Passive Trainer callbacks copy prepared inputs, native logits/loss, initial trainable parameters, gradients at optimizer entry, optimizer state and stored post-step parameters. Tiny full training and pretrained LoRA are separate planned cases. The CPU FP32 reference reloads the source checkpoint and equal initial trainable state, then performs its own forward/backward/SGD step; it never executes using native gradients or updates. These adapters passed AST checks only and still require their first actual NPU execution and any resulting repairs.

RL completion requires actual generated trajectories, rewards, advantages, native objective inputs/loss, a real optimizer update, rank completion and successful return of the original trainer. Synthetic response or step-verifier rewards are for exercising the training path, not reasoning-quality measurements. Zero group variance and unchanged stored parameters cannot silently count as an update.

Inference completion requires actual device-backed model execution and generated/scored outputs from the selected engine. A CPU fallback, import, help command, editable install, tensor operation or successfully built library alone does not satisfy native model inference. Capture native storage dtypes and identify any derived FP32 arrays. Precision comparisons use `thresholds: null`; no numerical acceptance is implied by artifact validation or computed metrics.

## Deployment friction and reusable candidates

Remote GitHub DNS resolution failed, so exact-pin source archives were downloaded locally and transferred through authenticated SSH stdio with SHA256 validation. The endpoint provides neither SFTP nor a remote SCP executable. This is a transport limitation; source download failure is not framework incompatibility.

Private MindIE build configuration omits a vendor CMake block that attempts a global kernel ASLR change through sudo. This is a scoped build-configuration change, with before/after provenance, rather than a native operator rewrite. Global driver/CANN configuration, unrelated processes and original model files were not modified.

Private launcher defects corrected during preparation included partial Git checkout recovery, repeated observer injection, overwritten retry logs, environment PATH selection and incomplete archive traversal exclusions. Nine local recovery regressions passed for the Slime/NeMo launcher; they are not hardware/framework tests. Useful reusable candidates are staged resume journals, copy-only native observers, bounded evidence collectors, and a generic hash-verified stdio artifact transport. They remain case helpers/proposals; no new precision-skill core tool is claimed here.

## Evidence and recovery order

`preparation-evidence.zip` contains sanitized local probes, last-retrieved installation status, three group progress records, source provenance, prepared adapters and validation output. `EXPORT.json` records each original/exported SHA256 and redaction transforms. Exact raw logs remain private. Public path placeholders are `/REMOTE_WORKSPACE`, `/REMOTE_HOME` and `/SHARED_MODELS`; restore actual case paths before any rerun. The archive contains no private key, SSH endpoint, credential store, model weights or environment tree.

This is not a numerical replay bundle: it has no `REPLAY.json`, no captured native model tensor and no precision metric that could be replayed into a native pass. Archive hash checks establish byte consistency only.

After the SSH endpoint is restored, first retrieve the existing detached job status and complete logs for AReaL attempt 02, MindSpeed-LLM attempt 02, MindIE build attempt 02, community OpenRLHF, Slime and the parent installs. Do not launch duplicates or infer completion from elapsed time. Reuse retained environments, caches and build objects; fix the actual remaining failure and use fresh output directories. Execute the remaining full native flows, export validated arrays/state/rank evidence, compute matched comparisons with the skill tools, then publish a new completed-run record linked from this checkpoint. Preserve these checkpoint bytes and the earlier historical results.

Repository validation for this checkpoint: the Linux unittest suite passed all 130 tests. This verifies repository tooling, not the unexecuted NPU adapters or interrupted framework jobs.
