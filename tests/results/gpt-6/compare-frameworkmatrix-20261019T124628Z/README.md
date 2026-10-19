# Expanded native framework validation

This round covers **17 task lanes across 16 unique libraries**. Native execution completed in **7 lanes across 6 unique libraries**, producing **15 primary local cases: 6 SFT, 6 inference, and 3 RL**. The remaining **10 lanes have bounded prerequisite or import evidence and zero native numerical coverage**. These counts distinguish trainer steps, model/loss components, inference execution, and prerequisite probes.

The tested skill snapshot is `741afa4acba9664edfdffe8deb681fcbea3739dc`. Collection began in pinned checkouts with cold agents created using `fork_turns="none"`; parent review, corrected captures, and public packaging followed collection. This snapshot merged `origin/main` at `4e68f9650d48f5cf2ff86e13e9b645ff2a8c420e` into the validation branch. No new skill defect was confirmed this round; two fixes from an earlier round were already on the branch.

The agent identity is known only as the **GPT-6 family**; the exact backend is unknown. This cohort varies the models being scored or trained. A separate [gpt-6-luna cold inference case](../../gpt-6-luna/compare-luna-native-20261019T140106Z/README.md) adds another configured agent model; different tasks/settings and one sample per agent do not support a controlled agent-model ranking.

## Read the evidence

- [Practical guide](GUIDE.md)
- SFT: [report](sft-report.md), [results](sft-results.json), [evidence ZIP](sft-evidence.zip)
- Inference: [report](inference-report.md), [results](inference-results.json), [evidence ZIP](inference-evidence.zip)
- RL: [report](rl-report.md), [results](rl-results.json), [evidence ZIP](rl-evidence.zip)
- [Consolidated machine-readable results](results.json)
- [Portable four-GPU probe](remote_four_gpu_probe.py)
- [Parent review and audit evidence](parent-evidence.zip) — intake, regressions, archive audit and remote observation provenance.

The group results and selected archive mappings are authoritative for individual cases. Historical attempts remain available; superseded summaries are not acceptance results. All final precision contracts use `thresholds: null`.

## What actually ran

**SFT: 6 lanes, 3 native lanes, 6 primary cases.** TRL 0.24.0, LLaMAFactory 0.9.5, and ms-swift 3.10.3 each executed one cold FP32 SGD step on pretrained SmolLM2 and deterministic random tiny Qwen2. Each case has 14 active targets, including two EOS targets. The 18 selected score, gradient, and stored-update comparisons observed zero error; 24 selected single-side tensor validation reports are retained. Zero observed error is not a precision acceptance decision.

These are fixed-token native trainer components. LLaMAFactory specifically ran `CustomSeq2SeqTrainer.train()` with its native collator/default template, without exercising the full `run_sft` workflow, CLI, model-loading workflow, or raw-text preprocessing. The reference executed its own forward/backward/update before native training; parameters were restored and checked exactly. Both sides share the Hugging Face model/loss implementation and native batch preparation, so this validates independent execution and data flow, not independent model/loss mathematics. Axolotl, Megatron Bridge, and MindSpeed-LLM remain blocked prerequisite lanes.

**Inference: 4 library lanes, 2 native lanes, 6 primary cases.** llama.cpp ran CPU F32 prefill plus two fixed-token decode steps on pretrained SmolLM2 and Qwen2.5. LMDeploy ran TurboMind and PyTorch CUDA FP16 scoring on both models, giving four additional cases. LMDeploy used three fresh prefix requests per case; its cached-decode coverage is zero. The PyTorch backend execution is exploratory because Triton 3.5.1 differs from Torch 2.8's Triton 3.4 metadata pin. TensorRT-LLM and MindIE have no native inference coverage.

llama.cpp maximum target-logprob errors against Transformers CPU F32 were about `3.10e-6` and `2.98e-6` nats. The neutral LMDeploy comparisons declared kernel and weight-dtype differences; their target MAE ranges from about `0.00245` to `0.00565` nats. Nine comparison pairs include supplemental reference comparisons; temperature diagnostics are separate and excluded. They do not increase the six-case primary count.

**RL: 7 lanes, 2 native component lanes, 3 primary cases.** OpenRLHF ran its native Actor scorer, PPO PolicyLoss, and full backward on pretrained SmolLM2 and random tiny Qwen2. Only tiny Qwen2 also received an explicit Torch SGD update. ms-swift constructed its real GRPOTrainer and ran its native scorer, public `compute_loss`, and full backward on a complete group of two SmolLM2 completions sharing one fixed prompt. Old scores and advantages were injected and recorded. No RL rollout, reward/group normalization, framework `training_step`, framework optimizer step, or full RL training loop is claimed.

The OpenRLHF SmolLM2 case paired all `134,515,008` gradient elements, with RMSE `4.20e-9`. The tiny Qwen2 case paired `22,816` gradient/update elements; stored-update RMSE was `4.90e-11`. The final ms-swift GRPO component case paired all `134,515,008` gradients, with loss difference `2.38e-7` and gradient RMSE `2.12e-8`. slime, NeMo RL, AReaL, ROLL, and MindSpeed-RL remain blocked prerequisite lanes.

Primary counts exclude retries, superseded captures, model conversion, Transformers references, offline replay, the separate remote smoke, and the separate gpt-6-luna case. ms-swift is counted once as a library but twice as a task lane. LMDeploy's two engines belong to one library lane.

## Models and comparison limits

Inference used two pretrained checkpoints: `HuggingFaceTB/SmolLM2-135M-Instruct` at `12fd25f77366fa6b3b4b768ec3050bf629380bac`, and `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`. SFT and RL additionally used explicitly random tiny Qwen2 checkpoints. The SFT tiny model has `20,768` elements; the RL tiny model has `22,816`. They are distinct test configurations and are not a shared checkpoint across groups.

Matched pairs retain checkpoint/input/state identities and native masks/reductions. Counts and metrics apply only to the captured stages and coordinates. Source inspection, successful imports, and `--help` completion do not establish numerical execution or training success.

## Findings and implementation status

- **LMDeploy score boundary:** the default PyTorch `async_get_logits` path used temperature `0.8`. All `49,152` logits in each of three returned rows exactly equaled the neutral native rows divided by FP16(`0.8`). This is `147,456` exactly checked values. Default captures remain diagnostic-only; final neutral captures use explicit native generation settings.
- **Dependency qualification:** Triton 3.4 warmup failed for this LMDeploy/Torch/GPU combination. Isolated Triton **3.5.1** executed but conflicts with Torch's metadata pin, so that execution is marked exploratory.
- **ms-swift RL prerequisite:** the broad top-level TRL requirement did not capture GRPOConfig's empirical `TRL >=0.26` requirement. TRL 0.25.1 failed configuration; 0.26.2 enabled real trainer construction and loss/backward execution.
- **Collector correction:** a TRL batch-row mapping error was in the task adapter. Corrected captures assert actual forward inputs and restore the independently executed reference state. It was not a skill-tool defect.
- **Artifact conditions:** nested `null` means unknown for required execution conditions. Known-disabled clipping/optimizer states use explicit disabled/not-executed values in fresh metadata views; original captures are preserved.

Proposed helpers remain **unimplemented**: a score-boundary guard, fixed-token/timeline capture support, and a stage-coverage/evidence-packaging normalizer. Existing [runtime_probe.py](../../../../skills/precision-skill/scripts/runtime_probe.py) and [inference_capture.py](../../../../skills/precision-skill/scripts/inference_capture.py) should be reused where their interfaces fit.

## Verification and replay

The parent reports **130 Linux unit tests passed in 5.334 seconds at `741afa4`**. An independent public-only audit validated all three ZIPs: safe relative member paths, CRCs, exported hashes, coverage coherence, and unchanged public inputs. It found no additional issue. The audit verified `1,219` indexed exported member hashes across the archives; recorded private-original hashes were format-checked, not independently re-read from the public bundles.

The inference archive supports captured-array offline replay with Python and NumPy. The independent audit disabled networking and reproduced nine comparisons, 18 metric groups, nine validated side manifests, and all three exact temperature-diagnostic rows, with metrics identical to the published results. This replay performs no native model execution.

The SFT and RL archives omit NPY/model/optimizer payloads and cannot fully replay gradients or native training alone. Their compact scores, per-tensor reports, coverage, logs, collectors, and source/exported hash transformations are retained. The inference ZIP uses LZMA; if needed, extract it using `python -m zipfile -e inference-evidence.zip NEW_DIRECTORY`.

## Remote follow-up

A separate four-H100-80GB instance and AFS folder `<REMOTE_PROJECT>` were created. Console output for `compare-four-gpu-20261019T140808Z` reported both `nccl_four_rank_completed: true` and `native_sft_trainer_completed: true`, with status `completed`. The probe uses a seeded random tiny Qwen2 and native TRL SFTTrainer for one distributed SGD step. This is additional execution coverage, with no matched precision reference, pretrained-model result, or RL rollout claim.

Earlier fresh runs retained two concrete blockers: unavailable optional dependencies after an interrupted install, then PyArrow26 requiring NumPy2 while the image supplied NumPy1.26.4. Isolated PyArrow21 enabled the final run, but conflicts with inherited cudf/pylibcudf requirements; those components were not executed. The shared base environment was unchanged.

Parent review corrected two status hazards in the case-specific remote probe: NCCL alone could produce exit0, and a rank could retain a completed nested SFT status after a later assertion failed. The published probe requires each rank's final completion and a zero torchrun exit; assertions precede completion recording. Four synthetic launcher-only regression scenarios pass. These are case-probe fixes, not precision-skill core defects.

Remote results are explicitly **console-observed transcriptions**. Original rank JSON, native metrics, and logs remain on AFS; they were not downloaded or independently re-read before the instance again appeared stopped (`0/1`). Its stop cause is unknown. Cross-rank gradient/update hash equality was not observed and is not claimed. See `remote/observations.json` inside the parent evidence archive and the root results for provenance and limits.

The separate gpt-6-luna export also passed an independent fresh extraction, hash verification, two side validations and exact replay of its three targets/distributions. Its cold metadata required parent corrections; incomplete superseded-input/build provenance is reported explicitly. These findings concern agent artifact collection and do not establish a precision-skill core defect.
