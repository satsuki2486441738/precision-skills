# How to read and replay this evaluation

This guide explains the local evaluation and its exported evidence. It was written during parent consolidation after the cold runs finished; it was not an input to those runs. The sibling [README.md](README.md) is the consolidated report. Consolidated and remote follow-up status belongs in the sibling [results.json](results.json).

The detailed packages are [SFT report](sft-report.md), [SFT results](sft-results.json), and [SFT evidence](sft-evidence.zip); [inference report](inference-report.md), [inference results](inference-results.json), and [inference evidence](inference-evidence.zip); and [RL report](rl-report.md), [RL results](rl-results.json), and [RL evidence](rl-evidence.zip). Read each report together with its machine-readable results because stage coverage and dependency limitations differ by case.

The cold evaluation started from skill snapshot `741afa4acba9664edfdffe8deb681fcbea3739dc` (`741afa4`). Three task agents used separate pinned worktrees and isolated environments. They read the repository instructions, precision skill, selected task recipe, and tool interfaces, then prepared bounded native adapters and environments. They did not consult Git history, existing repository tests/results, parent findings, or other task worktrees during the cold evaluation. Official framework source, documentation, actual imports, dependency metadata, and native entrypoint probes were allowed evidence. The agents reused the supplied model/runtime resources without changing the shared base environment, and serialized local GPU work through a shared lock.

The local host was WSL Ubuntu with an RTX 5060 8 GB GPU. Some executed components ran on CPU. These local measurements establish the recorded entrypoint, model, numerical settings, coordinates, and stages; they do not establish broad hardware or framework support.

There are **17 task lanes across 16 unique libraries**. A lane is a framework within a task. ms-swift appears in both SFT and RL; LMDeploy's PyTorch and TurboMind backends belong to one inference lane.

- SFT has six lanes: TRL, LLaMAFactory, ms-swift, Axolotl, Megatron Bridge, and MindSpeed-LLM. The first three executed native trainer components; the other three stopped at recorded prerequisites/imports.
- Inference has four lanes: llama.cpp, LMDeploy, TensorRT-LLM, and MindIE-LLM. llama.cpp and both selected LMDeploy backends executed; TensorRT-LLM and MindIE-LLM have prerequisite probes only.
- RL has seven lanes: OpenRLHF, Slime, NeMo RL, ms-swift, AReaL, ROLL, and MindSpeed-RL. OpenRLHF and ms-swift executed selected native components; the other five stopped at recorded prerequisites or entrypoints.

That gives **seven native-executed task lanes across six unique libraries**: TRL, LLaMAFactory, ms-swift, llama.cpp, LMDeploy, and OpenRLHF. Ten lanes have no native numerical capture. “Blocked” identifies the stage actually reached and the concrete limitation in the bounded local environment: for example, an import error, missing backend, incompatible dependency range, unavailable Ascend runtime, or failed native warmup. It is not a numerical precision verdict or a claim that the framework cannot run in another supported environment. An import or `--help` result alone does not count as native numerical execution.

The **15 primary local model/backend cases** are six SFT, six inference, and three RL cases. Reference executions, extra reference precisions, repeated attempts, diagnostic captures, and metric reports are not additional primary cases.

- The six SFT cases are three trainer components multiplied by two model initializations: pretrained SmolLM2-135M-Instruct and deterministic random-init tiny Qwen2. Each selected case executes one FP32 SGD step with 14 active targets, including two EOS targets. TRL uses `SFTTrainer.train()`; LLaMAFactory uses `CustomSeq2SeqTrainer.train()` with its native SFT collator/default template; ms-swift uses its native `Seq2SeqTrainer.train()` component and template/collator. Full framework CLIs, raw-text preprocessing, and the full LLaMAFactory `run_sft` workflow were not exercised.
- The six inference cases are llama.cpp CPU F32, LMDeploy TurboMind CUDA FP16, and LMDeploy PyTorch CUDA FP16, each on pretrained SmolLM2-135M-Instruct and pretrained Qwen2.5-0.5B-Instruct. Each case scores three fixed-history targets. llama.cpp preserves its native cache through one prefill and two fixed-token decode calls. LMDeploy uses three separate full-prefix requests per model, with zero cached-decode coverage. A separate Transformers CUDA FP16 Smol reference supports same-weight-dtype comparisons; it is not a seventh primary inference case.
- The three RL cases are OpenRLHF Actor/PolicyLoss on random-init tiny Qwen2, OpenRLHF Actor/PolicyLoss on pretrained SmolLM2, and a genuinely constructed ms-swift GRPOTrainer on pretrained SmolLM2. They score fixed trajectories using recorded detached old scores and advantages. The tiny OpenRLHF case also executes an explicit Torch SGD update. The pretrained OpenRLHF and Swift cases stop after backward. Swift uses one complete two-completion prompt group and its public `compute_loss`; native reward/group normalization, rollout, `training_step()`, optimizer steps, and full trainer loops were not executed.

The tiny Qwen2 checkpoints in SFT and RL are deterministic random initializations, not pretrained-model evaluations. They are separate task fixtures and should not be assumed identical across tasks. Inference uses two pretrained checkpoints only. Checkpoint/config/tokenizer hashes, initial parameter identities, fixed token IDs, labels, masks, positions, and effective settings are recorded in the corresponding evidence.

The references perform their own execution. In SFT, the reference forward/backward and SGD step run before native training; gradients are cleared and initial parameter clones are restored with exact equality asserted. Captured native gradients or updates are not loaded into the reference. The two sides still share the Hugging Face model/loss implementation and native batch preparation, so agreement tests independent execution/data flow rather than an independent mathematical implementation of the model or loss. RL references use separately expressed matched PPO or GRPO objectives; OpenRLHF's token-mean PPO and Swift's sequence-mean GRPO are separate matched cases, not interchangeable cross-framework objectives. Inference compares native full-vocabulary observations against native Transformers references on identical fixed token histories.

Every precision comparison uses **`thresholds: null`**. `no_thresholds` means metrics were computed for the declared coordinates; it does not mean a precision acceptance threshold was met. `validated` means the selected artifact schema, values, identities, rank records, and coverage passed their checks. It is distinct from native execution completion and numerical acceptance. Zero observed SFT error on these small selected cases is a measurement, not a general correctness proof. The inference normalization tolerance of `1e-4` checks finite-precision probability mass before explicit FP64 renormalization; it is not an accuracy acceptance threshold.

The reports distinguish three kinds of findings:

- **Adapter errors that were corrected:** initial TRL batch-row IDs did not preserve logical sample identity after native shuffling; a fresh capture now asserts equal prepared inputs. Earlier Swift rescoring was outside the actual loss forward; the final selected capture observes the native scorer during public `compute_loss`. Earlier captures remain available and do not replace the selected authoritative cases.
- **Confirmed API or representation friction:** LMDeploy PyTorch's default `async_get_logits` helper returns a temperature-transformed last row. All 49,152 values in each of three default rows exactly equal the neutral native row divided by FP16 `0.8`. Final comparisons use explicit neutral settings through the native request path; processed results remain diagnostic-only. Strict known-condition checks also rejected native configuration nulls and known-disabled nested nulls. Fresh metadata views use explicit identities/disabled states while retaining the original evidence.
- **Environment and coverage limits:** dependency installation, backend initialization, network retrieval, and hardware availability are reported at the stage reached. In particular, LMDeploy PyTorch needed isolated Triton 3.5.1 after the Torch 2.8/Triton 3.4 native warmup failed; this conflicts with Torch's Triton pin and is exploratory execution coverage. Broader LMDeploy requirements are also incomplete. RL's successful native subsets do not establish fully provisioned distributed trainer environments. These limitations are not silently converted into successful precision results.

The SFT results list no confirmed skill implementation bug. Metadata/schema friction, external framework behavior, adapter mistakes, and numerical differences should retain their separate classifications. Useful future tool candidates include stable logical sample/batch mapping, exact pass-through native-scorer capture, explicit disabled-state conventions, staged prerequisite/coverage manifests, dtype-aware logit serialization, and a score-boundary contract that rejects unknown or nonidentity output transforms for raw-score capture. These are proposals; this evaluation did not implement or ship those reusable tools. The inference archive includes a more detailed helper proposal, not an implementation.

The evidence archives contain redacted source, compact observations, provenance, and failed/superseded attempts. Each `EXPORT.json` records raw/exported hashes and transformations; originals remain preserved outside the public export. No private host, network, account, or subscription identifiers are needed to interpret the results. SFT's `AUDIT.json` selects the authoritative 18 score/gradient/update comparison reports and 24 single-side tensor validation reports. Superseded summary status fields are provisional.

For offline inspection, copy the three ZIP files into a scratch directory and extract them with Python's standard-library ZIP support. This also supports the inference archive's LZMA ZIP compression:

```sh
python -m zipfile -e sft-evidence.zip sft-evidence
python -m zipfile -e inference-evidence.zip inference-evidence
python -m zipfile -e rl-evidence.zip rl-evidence
```

The inference archive includes all 30 full-vocabulary FP32 logprob rows, contracts/manifests/environment/status records, exact comparison tools, and six raw rows for the default-temperature diagnostic. With Python and NumPy installed, run from the scratch directory:

```sh
python inference-evidence/replay.py --output inference-replay
```

The output directory must be new. The exported replay was verified from a fresh extraction: nine comparison pairs, nine validated side manifests, and three exact temperature-division diagnostic rows, with metrics identical to the original reports. It replays captured arrays only. Model weights, GGUF files, native executables, and external environments are excluded, so archive replay does not rerun the native frameworks.

The SFT archive supports recomputing six selected compact target-score comparisons with the included standard-library toolchain. For example, the corrected TRL tiny-Qwen2 case can be recomputed from the scratch directory with:

```sh
python sft-evidence/toolchain/compare_target_scores.py --contract sft-evidence/runs/compare-sftmatrix-20261019T130222Z/trl-tiny-qwen2/scores-contract.json --side-a sft-evidence/runs/compare-sftmatrix-20261019T130222Z/trl-tiny-qwen2/pytorch/scores.json --side-b sft-evidence/runs/compare-sftmatrix-20261019T130222Z/trl-tiny-qwen2/trl/scores.json --output sft-score-replay.json
```

Use `AUDIT.json`'s selected case mapping for the other five cases, including corrected/re-exported views. The six compact score comparisons were recomputed during export. NPY tensor arrays and model/optimizer weights are excluded, so the SFT archive alone cannot replay gradients, stored updates, or native training. Recorded tensor metrics and validation results remain auditable.

For the RL archive, the supplied verifier checks exported member hashes:

```sh
python rl-evidence/verify_export.py rl-evidence.zip
```

This is hash verification, not RL objective, gradient, update, rollout, or trainer replay. NPY/model/shard arrays are excluded. Full RL numerical replay needs the original captured arrays; native reruns additionally need the original models, compatible dependencies, restored path mappings, and the recorded framework entrypoints. Neither the SFT nor RL archive alone reproduces native training.

The separate [gpt-6-luna cold case](../../gpt-6-luna/compare-luna-native-20261019T140106Z/README.md) tests another configured agent model on actual llama.cpp CPU inference. Parent review found that native F16 KV storage was omitted, the reference used one full-sequence teacher-forced forward while native used three fresh prefix requests, and some superseded inputs/build commands were not preserved. Corrected exported metadata and replay are distinguished from original cold-agent artifacts. A single case with different settings does not establish an agent-model ranking.

The remote supplement created a four-H100-80GB instance and a new AFS project folder. Final visible console output for `compare-four-gpu-20261019T140808Z` reported completed four-rank NCCL and native TRL SFTTrainer execution. It used a locally created random tiny Qwen2 for one distributed step, with no matched precision reference or RL training. Dependency failures and repairs, the case-probe status fixes, and four synthetic status regression cases are recorded separately. Local case counts exclude this supplement. The instance later appeared stopped; the cause is unknown. Root [results.json](results.json) records console-transcription provenance and the retained remote file locations; detailed original rank JSON remains on AFS.

To rerun the synthetic launcher-status checks after extracting the parent evidence beside the public probe:

```sh
python -m zipfile -e parent-evidence.zip parent-evidence
python parent-evidence/remote/test_remote_probe_status.py remote_four_gpu_probe.py
```

These four checks simulate status files and subprocess exits; they do not execute Torch, NCCL, or training. The native remote run is separate evidence.
