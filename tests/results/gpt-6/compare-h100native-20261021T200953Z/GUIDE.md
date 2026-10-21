# H100 native CUDA continuation

This is an append-only continuation of [the 22-case CUDA recovery](../compare-gpurecovery-20261021T180045Z/GUIDE.md), using the existing four-H100 allocation. The selected workflows run their native forward, backward, clipping and optimizer paths. Import probes do not count as completed experiments. Numerical acceptance and thresholds remain **unset**; a completed workflow is not an accuracy pass.

## Published scope in this snapshot

- **Slime: 2 experiments**, seeded tiny Qwen3 and pinned Qwen3-0.6B; **14 comparisons**. Each uses one H100 FSDP2 training worker and a separate H100 SGLang rollout worker. The complete official controller performs generation, grouped rewards, four training microbatches, clipping, AdamW and a second NCCL weight synchronization. Weight versions `[1, 2]`, the router response and orderly worker shutdown are retained.
- **NeMo RL: 2 experiments**, tiny and pretrained **untied-config variants**; **8 comparisons**. The official `run_grpo_math.py` route performs HF generation, reward/group construction and one AdamW update. The selected `FSDP1PolicyWorker` DP=1 branch explicitly skips FSDP wrapping. This is a single-GPU workflow, not evidence of sharded or multi-GPU NeMo training.
- **TRL SFTTrainer: 4 experiments**, tiny/pretrained × FP32/BF16; **16 native rank captures and 80 comparisons**. All four H100 ranks participate in each native job. Each rank has an independent single-process CPU reference. Sixteen rank captures do not mean sixteen independent experiments or CPU DDP.
- Total: **8 native experiments, 20 retained native capture cases, 102 comparison/replay groups**. Together with the earlier recovery this adds 30 native CUDA experiments. The earlier TensorRT-LLM evidence is preserved separately and is not counted again.

The [progress snapshot](GPU_PROGRESS.json) distinguishes completed scopes, unresolved native runs and selected Ascend-only routes. [RUN_VALIDATION.json](RUN_VALIDATION.json) records independent verification of the exact public bytes. Each archive has an index and replay specification; portable comparison replay requires NumPy and the repository tools, not Torch, pickle or execution of archive code.

## Conditions that change the interpretation

### Slime

Official source: `0104a9e922ecd3cea768f7c2520e7d1a122afdc9`. The recorded stack includes Torch 2.9.1+cu128, Transformers 4.57.1, SGLang 0.5.9 and Ray 2.54. The native forward uses BF16 with FP32 parameter storage. The CPU reference independently recomputes its own **old** and current policies from the original checkpoint and actual generated histories. It does not consume native current scores, gradients or updates to construct a reference.

Each experiment retains four trajectories, two groups and 24 valid actions. The tiny model retains all 25 parameter tensors and 361,344 elements. The pretrained model runs full-model backward/clipping/AdamW but compares six complete tensors, **1,051,904 / 596,049,920 elements**. Uncaptured coordinates are not numerical coverage. Current FSDP, old FSDP and SGLang rollout scores remain three separate comparison boundaries.

Actual `clip_grad_norm_` return values determine clipping. Additional FP64 full-gradient norms are diagnostics only. For example, pretrained CPU clipping returned `60.67780303955`, while its FP64 diagnostic norm was `60.6939123357`; silently substituting the diagnostic changes the reference update. Initial ratio=1 and near-zero group-averaged loss still produce nonzero gradients and an optimizer update. Synthetic rewards validate execution, not task quality.

Evidence: [summary](slime/summary.json), [typed input index](slime/INPUTS-INDEX.json), [tiny arrays](slime/slime-tiny.zip), [pretrained arrays](slime/slime-pretrained.zip), [tiny fixed inputs and CPU recomputation](slime/slime-tiny-inputs.zip), [pretrained fixed inputs and CPU recomputation](slime/slime-pretrained-inputs.zip).

### NeMo RL

Official source: `26f651ea490ef2a343e7f298b9aa54bc7c5da67c`. The recorded stack includes Torch 2.6+cu124, Transformers 4.51.3 and Ray 2.43. Both paths use a derived config with exactly one change, `tie_word_embeddings=False`. The original pinned checkpoint config and weights remain unchanged. Original/derived config hashes, the exact diff and source weight identity are retained in each archive's `INPUTS/derivation.json` and `provenance/derivation.json`. This is not a claim about the original tied configuration.

Each experiment has four trajectories, one group, 32 actions and one optimizer update. The reference performs a full forward/backward and full-model clipping, with independent AdamW arithmetic on the recorded coordinates. The comparison retains the first twelve optimizer parameters, up to 8,192 coordinates per parameter: **65,856 tiny / 67,840 pretrained elements**, not an assumed 98,304. Both native and CPU policy objectives use fixed recorded native old-policy anchors. This differs from Slime's independently recomputed old-policy condition.

The original CPU clipping return scalar was not retained; no diagnostic FP64 norm is relabeled as that return value. Actual captured post-clipping gradients and updates remain available. Earlier DTensor generation and missing-worker-hook attempts have separate receipts and do not count as closed numerical evidence.

Evidence: [public export index](nemo/public-export-index.json), [earlier attempts](nemo/previous-attempts.json), [tiny arrays](nemo/nemo-tiny-untied-fp32.zip), [pretrained arrays](nemo/nemo-pretrained-untied-fp32.zip), [tiny fixed inputs](nemo/nemo-tiny-untied-fp32-fixed-inputs.zip), [pretrained fixed inputs](nemo/nemo-pretrained-untied-fp32-fixed-inputs.zip).

### TRL four-rank DDP

Each native four-rank job starts from identical initial values and identical ordered tokens, labels and masks. The audit checks synchronized gradient replicas for exact equality and records the native/CPU gradient scalar and norm ratios. It does not generalize this fixture to heterogeneous rank-local batches. Tiny Qwen2 uses full training; pinned Qwen3-0.6B uses LoRA q_proj/v_proj, rank 4. Every trainable coordinate is retained, including zeros and unchanged coordinates. Frozen tensors are checked separately.

The original effective objective denominator is replayed independently; the trainer's item count and valid shifted target count are distinct. Native objective loss is not relabeled as selected-token mean NLL. Requested BF16 trainer precision does not imply all stored parameters, gradients or logits are BF16; the original captured dtypes remain in the evidence. CPU references use world_size=1 and independently execute forward/backward/SGD for each paired native rank.

Evidence: [all-rank summary](trl-ddp/summary.json), [archive hashes and replay counts](trl-ddp/PUBLICATION-INDEX.json), [DDP scope and recovery notes](trl-ddp/guide-notes.json). Each of the sixteen indexed ZIPs contains native and CPU arrays, five comparator reports, exact input/checkpoint identities and the relevant DDP audit.

## Recovery bugs and reusable safeguards

- NCCL initialization failed in one library stack. The working Slime route and OpenRLHF recovery explicitly record `NCCL_CUMEM_ENABLE=0`, the actually loaded NCCL library and physical/logical device mapping. A runtime transport fix is not a numerical operator replacement.
- A shell helper assigned its argument to an outer framework variable. Declaring helper variables local fixed dispatch; the failed attempt ran no native training. The frozen numerical collector SHA remained unchanged.
- A weights-only reader rejected the `TorchVersion` string subclass after native DDP training had already finished. The CPU export recovery allowlisted only that metadata class for the owned files. It did not rerun GPU training, change arithmetic or reserialize historical private PT bytes.
- Passive hooks attached in the controller did not survive a serialized Ray worker. Hooks were installed on the actual worker in a fresh attempt, with original failures retained separately.
- Public packaging excludes raw PT, `__pycache__` and `.pyc`. Compiled Python can embed a private original filename even when text files are sanitized. The corrected Slime export preserves every numerical file byte-for-byte; unpublished copies containing compiled files were never pushed.
- Typed NumPy input archives preserve tensor dtype/shape/bits, list/scalar structure and model derivations. They support independent CPU recomputation without distributing private pickle files. Comparison-only replay is explicitly distinguished from rerunning the model.
- Repository `evidence_bundle.py` checks retained-file integrity; `replay_bundle.py` resolves all references inside the retained archive and applies extraction limits. Public receipts omit private local paths. Numeric arrays and checkpoint bytes are protected from textual redaction.

## Work still open

OpenRLHF, AReaL, ROLL and Megatron Bridge have not yet closed their selected H100 native-plus-reference workflows in this snapshot. Their active attempts are not included in the completed counts. AReaL's first native-returning attempt used an effective attention boundary different from the CPU reference; those results are incomparable until a fresh matched-condition run completes. ROLL hit its strict dataset/batch assertion before an optimizer step. OpenRLHF recovered NCCL initialization and is continuing through the native reward API. Bridge is completing real CUDA dependency builds before native conversion/checkpoint/training.

The previously selected MindSpeed-RL, MindIE-LLM and MindSpeed-LLM paths require Ascend. The newer MindSpeed-LLM FSDP2 CUDA candidate was checked separately at LLM `8fdb88fde799ea050f687fede0efd1b236dc058a` / FSDPTurbo `54c02abb1890ec3f682ea5489f3d3cd94dedcfe6`. Its 112 declared packages pass the dependency check, but the actual selected native entrypoint fails before argument parsing on an undeclared eager Core import. Independent module traces also retain unconditional `torch_npu` import and a Triton/Ascend binary namespace collision. Adding the documented Core branch encounters incompatible NumPy requirements. The [65-file boundary archive](mindspeed-cuda/mindspeed-cuda-pinned-boundary.zip) and [index](mindspeed-cuda/PUBLIC-INDEX.json) retain exact pins, original source bytes, resolver failures and runtime traces. GPU initialization and optimizer steps are both zero: this is an unclosed pinned candidate, not a completed precision experiment or a blanket claim that every revision/backend lacks CUDA support.

Results are pushed to the original validation branch before immutable links are added to the website's Evaluation history. Existing Repair styling and all historical evaluations are preserved. Only meaningful two-sided recorded measurements receive comparison bars; thresholds remain unset.
