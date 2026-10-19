# Native RL framework checks

Two native execution paths completed: OpenRLHF Actor/PolicyLoss on CUDA, and a genuinely constructed ms-swift GRPOTrainer with public compute_loss/backward on CPU. Five other frameworks received bounded official-source, native-entrypoint and backend prerequisite probes. No framework rollout or full training loop is claimed.

Run: `compare-rlmatrix-20261019T124818Z`. Start: 2026-10-19 12:48:18 UTC. Finish: 2026-10-19T13:17:21.521388+00:00.
Environment: Python 3.11.17, Torch 2.8.0+cu128, Transformers 4.57.0, RTX 5060 8151 MiB; private environment `$ISOLATED_ENV` references the base read-only site-packages and owns its dependency overrides.

## Executed comparisons

### tiny_qwen2_random

- Loss absolute difference: 1.043081284e-07.
- Action scores: 7/7 paired, MAE 2.043587821e-07, P99 4.768371582e-07, maximum 4.768371582e-07 nats at `{'sample': 'trajectory-0', 'position': 10, 'token': 42}`.
- Gradients: 22,816 paired elements, L2 error 7.751143217e-07, RMSE 5.13151978e-09, cosine 0.999999999999972, maximum 2.384185791e-07 at `{'parameter': 'model.embed_tokens.weight', 'index': 843}`.
- Stored explicit-SGD updates: 22,816 paired elements, L2 error 7.401590273e-09, RMSE 4.900103872e-11, cosine 0.999999999995759, maximum 3.725290298e-09.
- Captured stages: `{'model_scoring': True, 'native_policy_loss': True, 'backward': True, 'explicit_torch_sgd_step': True, 'rollout': False, 'native_trainer_step': False}`.
- Gradient report: `$RL_RUN/tiny_qwen2_random/explicit_disabled_views/gradients_comparison.json`.

### smollm2_135m_pretrained

- Loss absolute difference: 1.862645149e-07.
- Action scores: 7/7 paired, MAE 5.449567522e-07, P99 9.536743164e-07, maximum 9.536743164e-07 nats at `{'sample': 'trajectory-0', 'position': 11, 'token': 2}`.
- Gradients: 134,515,008 paired elements, L2 error 4.868455184e-05, RMSE 4.197645953e-09, cosine 0.999999999999094, maximum 1.14440918e-05 at `{'parameter': 'model.embed_tokens.weight', 'index': 28999}`.
- Captured stages: `{'model_scoring': True, 'native_policy_loss': True, 'backward': True, 'explicit_torch_sgd_step': False, 'rollout': False, 'native_trainer_step': False}`.
- Gradient report: `$RL_RUN/smollm2_135m_pretrained/explicit_disabled_views/gradients_comparison.json`.

### SmolLM2-135M-Instruct pretrained

- Loss absolute difference: 2.384185791e-07.
- Action scores: 7/7 paired, MAE 1.702989851e-06, P99 3.786087036e-06, maximum 3.814697266e-06 nats at `{'sample': 'trajectory-1', 'position': 8, 'token': 50}`.
- Gradients: 134,515,008 paired elements, L2 error 0.0002460203798, RMSE 2.121220002e-08, cosine 0.999999999986897, maximum 4.625320435e-05 at `{'parameter': 'model.embed_tokens.weight', 'index': 9415}`.
- Captured stages: `{'native_model_loader': True, 'trainer_constructed': True, 'native_action_scoring': True, 'native_compute_loss': True, 'backward': True, 'rollout': False, 'native_training_step': False, 'native_optimizer_step': False, 'trainer_train_loop': False}`.
- Gradient report: `$RL_RUN/ms_swift_smollm2_complete_group/gradient_comparison.json`.

The tiny Qwen2 checkpoint is random-initialized with seed 20261019; it is not a pretrained model. The SmolLM2 cases use the existing pretrained checkpoint. All matched pairs verify exact initial FP32 parameter hashes. Old policy scores and advantages are injected fixed tensors and explicitly recorded in trajectory.json. Swift complete group membership is recorded, but native reward/group normalization is not executed.

## Framework stage coverage and blockers

- **openrlhf**, source SHA `06ab51e75c0e8c5b4d6b66912996b2a611e3dc2f`: native CLI return code 0; runtime probe `completed`.
  Native Actor + PolicyLoss model execution; tiny Qwen2 explicit Torch SGD. No Ray/DeepSpeed training orchestration.
- **slime**, source SHA `a4202e0595467d9374d9a68d45a34f45c3f8d230`: native CLI return code 1; runtime probe `failed`.
  Native Megatron loss import fails: megatron is not installed.
  Native train.py --help reaches the SGLang dependency and fails: sglang is not installed.
  No Megatron or SGLang backend was fabricated or provisioned.
- **nemo_rl**, source SHA `31a7f8b801e8f4cdb71ebe948a25585ca80b5ddc`: native CLI return code 1; runtime probe `failed`.
  Pinned main requires Python >=3.13.14,<3.14 and Torch >=2.11,<2.14; actual Python is 3.11.17 and Torch is 2.8.0+cu128.
  Native GRPO entrypoint raises a Pydantic TypedDict-on-Python<3.12 error.
  Actual backend probes fail for nemo_automodel, megatron.core and transformer_engine.
- **ms_swift**, source SHA `2efb43ffe61294735be6816abca991fb2ca16011`: native CLI return code 0; runtime probe `completed`.
  Real native get_model_processor and GRPOTrainer constructor, public compute_loss, native scorer and backward. No train(), training_step(), rollout or optimizer.
- **areal**, source SHA `3b93679888e9f4929f6f155f13c6a286a62dfd72`: native CLI return code 1; runtime probe `failed`.
  Pinned main requires Torch >=2.9.1,<2.11 and Transformers >=5.0,<=5.3; actual 2.8.0+cu128/4.57.0 do not satisfy these.
  The native PPOTrainer/entrypoint dependency frontier is missing flask after bounded repairs.
  Actual SGLang and torch_memory_saver backend imports fail.
- **roll**, source SHA `d3506ea9e280ae720fb74df9f82db6efac1c72d5`: native CLI return code 1; runtime probe `failed`.
  Native RLVR pipeline/entrypoint import fails for transfer_queue.
  Actual selected backend prerequisites megatron.core and quack are absent.
  Its source has a Torch 2.8/vLLM 0.11 dependency file, but the distributed pipeline was not provisioned.
- **mindspeed_rl**, source SHA `26c21e64f84ae8cd26de9483292349a1850099cf`: native CLI return code 1; runtime probe `failed`.
  Native trainer/GRPO entrypoint fails for torch_npu after bounded repairs.
  Actual torch_npu, mindspeed and megatron.core imports fail.
  Available hardware is an NVIDIA RTX 5060 under WSL; no Ascend NPU or Ascend runtime is supplied.

## Precision scope

- Identical checkpoint files and exact FP32 initial parameter hash per matched pair.
- Fixed 2 trajectories, 7 active action targets including 2 EOS targets; native mask retained.
- Identical injected detached old scores and advantages per pair; no reward model or rollout recomputation.
- Full parameter inventory including tied-weight mappings; no gradient clipping or accumulation beyond one batch.
- Tiny Qwen2 has a stateless matched explicit SGD(lr=0.001, momentum=0) step with stored before/after tensors.
- All comparison contracts thresholds=null; no pass/fail precision threshold is invented.

- OpenRLHF uses the native Actor score transform/Triton cross_entropy and PolicyLoss; reference uses Transformers logits, torch.log_softmax, and an independent clipped PPO formula.
- Swift uses its native model loader, real GRPOTrainer and TRL selective_log_softmax; reference uses direct Transformers loading/log_softmax and independent sequence-mean GRPO reduction.
- Native Swift forwards logits_to_keep=5/use_cache=False as supported by the model; full-sequence reference forwards the same input IDs/positions/masks. The separate row-retention probe did not reduce the initial rescoring discrepancy; no cause is asserted.
- OpenRLHF cases use token-mean PPO; Swift uses its actual sequence-mean GRPO reduction. These cases are independently matched to their references, not cross-framework loss comparisons.

## Friction and preserved attempts

- DeepSpeed source build selected CUDA compatibility probes even with DS_BUILD_OPS=0; CUDA_HOME is absent. Supported DS_ACCELERATOR=cpu plus DS_BUILD_OPS=0 built the real package. No fake backend imports were used. Evidence: `install_native_prerequisites.log and repair_install_cpu_deepspeed.log`.
- Installed real FlashAttention exposes OpenRLHF Triton cross_entropy; CPU Actor scoring fails with CPU tensor pointer error. Unmodified native scoring/backward succeeded on CUDA under shared flock. Evidence: `native_cpu_smoke.log and native_gpu_replay.log`.
- Nested null for known-disabled clipping/optimizer was treated as an unknown state by conditions.known. Original reports/captures were preserved; fresh explicit_disabled_views use disabled/not-executed strings and validate. Evidence: `compare_captured.py and tensor_comparison_summary.json`.
- Requesting installed distribution metadata for an uninstalled source checkout made the first successful OpenRLHF symbol import report fail at metadata lookup. Editable no-deps installation provided genuine metadata; final ordered probe completed. Evidence: `after_install_openrlhf_runtime.json and post_native_runtime.json`.
- ms-swift top-level requirements allow TRL>=0.15,<1 but GRPOConfig requires TRL>=0.26. Initial actual constructor attempt failed with TRL0.25.1; retry with 0.26.2 constructed the real trainer and executed its loss/backward. Evidence: `native_ms_swift_replay.log and native_ms_swift_exact_capture.log`.
- First Swift attempt captured a no-grad rescoring call separately from the loss forward. A new exact-capture attempt uses a pass-through native-scorer wrapper during public compute_loss; prior attempts remain preserved. Evidence: `native_swift.py and ms_swift_smollm2_exact_capture/`.
- The final Swift replay gives both trajectories the identical eight-token prompt, making one complete two-completion GRPO group. Earlier fixed-batch attempts with distinct prompts are preserved as component probes and are not the final group comparison. Evidence: `ms_swift_smollm2_complete_group/trajectory.json`.
- The official OpenRLHF v0.9.0 Transformers 4.57.0 pin is yanked on PyPI for setup issues; its installed wheel executed. Ray 2.59 differs from the framework full-trainer pin 2.48 and no Ray trainer was run. These are subset native-component checks, not a fully provisioned trainer environment. Evidence: `repair_install_cpu_deepspeed.log and final/deeper native entrypoint logs`.

## Evidence and reusable adapters

- `results.json` contains the exact source pins, package/runtime paths, command arguments, comparison metrics, stage distinctions and evidence paths.
- `post_native_runtime.json` completed ordered imports, selected symbol checks and real Swift GRPOConfig construction with TRL0.26.2.
- `framework_matrix.json` and `framework_matrix_deeper.json` preserve native CLI/backend attempts and source prerequisite evidence. All logs and earlier failed captures remain in place.
- `native_openrlhf.py` and `native_swift.py` retain unmodified native loss operators and map their observations to the skill artifact interfaces.
- Full gradient arrays are in per-case native/reference directories; fresh metadata views refer to the original arrays without changing their bytes.
- Downloads stayed well below the 2 GB allocation by reusing Torch/model resources; largest explicit download was the 244.2 MiB official FlashAttention wheel. Exact aggregate network bytes were not independently metered.

Useful reusable candidates: exact native-scorer pass-through capture, fixed-trajectory action-coordinate export, prerequisite/stage coverage matrix, and explicit-disabled metadata state conventions.

Export scope: NPY/model/shard arrays are omitted; this archive cannot fully replay gradients. Raw originals remain preserved in the private run. See EXPORT.json inside rl-evidence.zip for source/exported hashes and redaction transforms.
