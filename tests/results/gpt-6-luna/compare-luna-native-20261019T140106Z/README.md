# Bounded clean-context native inference case

Configured agent: **gpt-6-luna**, reasoning effort **high**, `fork_turns="none"`, with a 15-minute dispatch budget. Parent dispatch pinned skill snapshot `741afa4acba9664edfdffe8deb681fcbea3739dc`. This is one bounded native case, not the larger multi-framework cohort. Task selection, context and prepared resources differ, so these results do not support a fair ranking of agent models. See the broader [evaluation guide](../../gpt-6/compare-frameworkmatrix-20261019T124628Z/GUIDE.md) for the separate cohort's scope.

The case used pretrained SmolLM2-135M-Instruct, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, vocabulary 49,152. llama.cpp C API at source revision `757cad021b357f417065ddbc41a6e8f87c1f8136` executed on CPU with F32 GGUF weights and F32 native logits, two threads. Transformers 4.57.1/Torch 2.8.0+cu128 provided an eager CPU F32 reference. CUDA availability in the runtime probe is not evidence of CUDA execution; the collector puts the model and inputs on CPU. No native model or training run was repeated during packaging.

Three targets were scored: prompt IDs `[504,3575,282,4649,314]`, fixed continuation `[7042,314,260]`, prediction rows 4,5,6. Each side supplies three full-vocabulary FP32 natural-log-softmax rows, with unchanged array bytes in the archive.

**Actual execution differs from the original cache-off labels.** Native llama.cpp uses F16 K/V storage within each prefix prefill, confirmed by native log line 1165. It clears native memory before each of three separate prefix requests; no cache state is reused across those requests. The reference instead performs one teacher-forced causal forward over all eight tokens and selects rows 4,5,6. It does not independently execute each prefix. The reference collector does not explicitly pass `use_cache=False`; the checkpoint default is true, and returned cache storage was not captured. It also does not set or capture TF32 flags; the actual reference forward is on CPU. F16 KV and different forward shapes are declared implementation differences. No matched cache-dtype or shape probe was run, so no cause is assigned to the measured numerical differences.

Fresh exported contracts/environment/manifests under `corrected/` declare these actual conditions. The available original metadata remains under `historical/` and must not be treated as authoritative where it claims cache-off execution or equal per-prefix forward boundaries. Review source/log/hash findings are under `review/`. Fresh offline comparison of the unchanged observations reproduces the original metrics exactly:

- Targets: 3/3 paired; reference NLL 3.162992219130198 versus native 3.162873069445292 nats/token; absolute NLL difference 0.00011914968490600586.
- Target logprob MAE 0.000430444876352946 nats; P99 0.0006144928932189941; maximum 0.0006175041198730469 at target position 6, token 314.
- Distributions: top-1 agreement 3/3; mean tail-merged JS 9.087503155271076e-8 nats; mean probability RMSE 1.943635834200819e-5; maximum probability error0.00013007220856880997 at prediction position 6, token 260.

All precision contracts retain `thresholds: null`. `no_thresholds` means metrics were computed, not that a precision acceptance criterion was met. The `5e-5` normalization tolerance validates FP32 mass rounding and is separate from precision acceptance; renormalization remains `none`. Both corrected side manifests validate.

Available failed reports record an initial `1e-5` mass tolerance failure, a target manifest schema mismatch, and an incorrect runtime symbol syntax. **The original attempt/build history is incomplete:** the earlier tolerance contract and schema 2 target manifests were overwritten by their corrections, `native-build.log` is empty, and no exact compile command is present in the available source. Initial compiler errors survive only as summary narrative. The archive includes the available artifacts and this limitation; it does not claim every original input or build attempt was retained.

[results.json](results.json) contains settings, metrics, corrections and provenance limits. [evidence.zip](evidence.zip) contains every available contract/manifest/status, all six full-vocabulary logprob rows, native raw logits, small logs/source, pinned source/config provenance, reviewed findings and the comparison tools. Model weights/GGUF, native executables and external environments are excluded. Personal paths and host/account identifiers are replaced by explicit placeholders. `EXPORT.json` records original/exported SHA256 hashes, transformations and separately derived corrected views. The private originals remain unchanged.

Copy `evidence.zip` into a scratch directory and run:

```sh
python -m zipfile -e evidence.zip evidence
python evidence/verify_export.py evidence.zip
python evidence/replay.py --output NEW_REPLAY_DIRECTORY
```

Python and NumPy are sufficient; the output directory must be new. A fresh extraction was verified: all exported/derived hashes match, both manifests validate, and target/distribution metrics equal the originals. This is captured-array replay only. It does not rebuild or execute llama.cpp/Transformers or reproduce the original native run.

Archive: 1694934 bytes; SHA256 `49d433192e2a94ec09b5d1e6301d41aca384499c72a03ca2e418dfbcd7b6367c`.
