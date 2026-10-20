# Bugs fixed and reusable precision tools

This maintenance update is dated **2026-10-21, Asia/Shanghai**; run directory timestamps use UTC. It turns demonstrated collection/export/replay failures from the native evaluation into installable skill code. No native model execution was rerun and the earlier framework case counts are unchanged. Historical archives remain byte-for-byte intact. [results.json](results.json) lists the changes and [VALIDATION.json](VALIDATION.json) records final checks and hashes.

## Fixes now shipped in the skill

1. **Logical coordinates could pass through Python type coercion.** `validate_history` previously accepted boolean/float token and position IDs and ignored supplied decode `prefix_length`. A temporary reproducer accepted `[True, 2.0]`, `[False, True]` and a decode prefix of 999. It now rejects those inputs, retains repeated integer position rows for MRoPE, and checks any supplied decode prefix. Existing adapters may still omit a decode prefix; that omission cannot establish its actual native value.
2. **Target-score exports were not bound to arrays.** `export_records` previously accepted an empty sample ID, missing array, invalid row and an arbitrary target logprob. It now loads the selected NumPy/safetensors row, checks row/vocabulary/history/normalization and verifies the target value. `None` is valid only for an actual `-inf` target. Invalid sample/rank/boundary metadata and nonfloating capture tensors are rejected before files are written.
3. **Portable replay could read unindexed external data.** Checking only the top-level contract/manifest names left their rank and array references unrestricted. A reproduced replay returned `matched=True` using an external array absent from the ZIP. The new runner validates every consumed environment/status and tensor/distribution array reference before launching tools. References must remain within extracted evidence; contained `../` references remain supported. General source-side comparators retain their existing path interfaces.
4. **Archive integrity must remain active under optimized Python.** Luna's historical bundled verifier used `assert`, which Python `-O` disables. The new archive tool uses explicit checks for hashes, sizes, exact member coverage, duplicate/nonportable paths and fresh extraction. Strict JSON parsing also rejects duplicate keys, nonfinite constants and overflow literals such as `1e999`, including ignored metadata. Historical helper bytes were not rewritten.
5. **Windows long paths failed on a real retained archive.** Deep TensorRT callback paths failed during extraction despite valid members. The archive tool now uses Windows extended-length paths for filesystem operations while keeping readable receipt paths. Tests exercise packing, transformation, verification, extraction, collisions and failure cleanup beyond 300 characters; no system settings are changed.

The implementation and permanent regressions are in [inference_capture.py](../../../../skills/precision-skill/scripts/inference_capture.py), [test_inference_capture.py](../../../test_inference_capture.py), [test_evidence_bundle.py](../../../test_evidence_bundle.py) and [test_replay_bundle.py](../../../test_replay_bundle.py). The temporary reproducer was removed after its before/after result was recorded.

## Reusable tools extracted

**[evidence_bundle.py](../../../../skills/precision-skill/scripts/evidence_bundle.py)** provides standard-library `pack`, `verify` and `extract` APIs and CLI commands. It streams files, records original/exported SHA256 and byte counts, and verifies the complete index before extraction. Optional literal replacements operate on valid UTF-8 text across extensions, including logs, YAML and shell files. Known numeric/model containers, invalid UTF-8 and NUL-containing files are unchanged. Raw `.f16`, `.bf16`, `.f32`, `.f64`, NumPy, safetensors, Torch and GGUF files are protected explicitly, even if their bytes happen to look like text.

Stage only intended files. Replacement specifications stay outside that directory; patterns and replacement values are not copied into the generated index. Filenames are preserved and privacy review remains a separate task. There is no automatic IP redaction: NVIDIA version strings such as `10.14.1.48` must not be damaged by a generic address regex. The tool does not certify that every secret has been found.

**[replay_bundle.py](../../../../skills/precision-skill/scripts/replay_bundle.py)** reuses verified extraction and invokes the installed comparison/validation CLIs. It preserves explicit side order, rejects malformed replay cases, and does not execute bundled Python. The old [tests/results entrypoint](../replay.py) is a thin compatibility wrapper. Both archive/replay tools default to a 16 GiB expanded-byte cap, with `--max-expanded-bytes` available; archive index JSON is additionally capped at 16 MiB. Complete `FILES.json` inventories are supported explicitly. Other legacy schemas require an explicit conversion rather than silent inference.

From the repository root:

```sh
python skills/precision-skill/scripts/evidence_bundle.py pack STAGED_FILES NEW.zip --replacements PRIVATE_REPLACEMENTS.json
python skills/precision-skill/scripts/evidence_bundle.py verify NEW.zip
python skills/precision-skill/scripts/evidence_bundle.py extract NEW.zip NEW_DIRECTORY
python skills/precision-skill/scripts/replay_bundle.py --bundle NEW.zip --output-root runs
```

Omit `--replacements` when no text transformation is needed. `REPLAY.json` selects `compare_tensors`, `compare_distributions`, `compare_target_scores` or `validate`, with explicit input paths and expected report subsets. Floating replay assertions use relative/absolute tolerance `1e-12`; these are retained-metric checks, not model precision thresholds. See the maintained [tool interface guide](../../../../skills/precision-skill/references/comparison.md#portable-evidence) and module docstrings for schemas and API arguments.

## Inference operations added

`derive_logprobs(raw_rows, dtype='float64')` provides a stable explicit offline log-softmax. Source arrays remain unchanged. Unrepresentable input/output conversion is rejected rather than returning NaNs or inventing zero-probability entries. `FixedHistory(..., logprob_dtype='float64')` derives from copied native values on CPU and records that policy. Its existing native-device FP32 default is preserved. Native parameter/cache arithmetic does not become FP64, and exported dtype must not be presented as measured native dtype.

`validate_generation_scores(context, generation, decode)` checks exact identity of the last context row with the first generation row, plus all supplied decode rows. Together with strict histories and array-bound target export, it provides reusable protection against the TensorRT observation error. It cannot establish uninstrumented physical positions, cache dtype or loaded-weight equality.

The retained TensorRT-LLM 1.2.0 case has a genuine upstream callback indexing defect with `return_context_logits=True`: its first callback exposes the first prompt row. Native returned generation scores correctly use the last prompt row. The new checker rejects that callback row for both captured models and validates the authoritative API rows and all three decode rows. No TensorRT vendor code or numerical output was patched. The [original evidence](../compare-trtcomplete-20261020T134000Z/guide-notes.md) remains the source of that finding.

## Metadata corrections and case-specific work

The inference recipe now distinguishes clearing cache between requests from disabling cache within a forward, and records actual forward shapes/cache observations. This addresses the [Luna review](../../gpt-6-luna/compare-luna-native-20261019T140106Z/README.md): native F16 KV prefix forwards and a single full-sequence CPU reference were previously described too loosely. The corrected historical metadata stays authoritative; no matched-cache/shape probe or causal explanation for its error is invented.

Training guidance separates requested precision, measured parameter/main-gradient/master storage, exported dtype and offline reduction dtype. It preserves native clipping norms/coefficients alongside independent full-coordinate diagnostics, and distinguishes sampled from full-model norms. NeMo's measured CPU postclip norm around 0.50012248 versus NPU 0.49999997 remains a real observation under the same configured 0.5 policy. No new norm calculator or rescaling was added; existing stable FP64 tensor reductions are reused.

Ray resource routing, rank-environment cleanup, short IPC paths, framework singleton/process lifecycle, CUDA/MPI dependencies, conversion layouts and EOS/template corrections remain in task adapters. Their assumptions depend on the actual backend and source version. They were not turned into a global launcher, an automatic environment repair tool or silent native-operator replacements. The [framework completion Guide](../compare-remainingcomplete-20261020T140716Z/GUIDE.md) retains exact routes and unresolved upstream/hardware limits.

## Retained-artifact validation

- The complete repository suite passes 167 tests on Linux with no skips; Windows passes 167 with five environment/privilege skips. The new archive suite also runs under `-O`. Skill metadata, documented interfaces and local links/anchors are checked separately.
- The old compatibility command replays all eight original synthetic cases. The installed runner replays all six TensorRT native/reference comparisons. Actual context/generation/decode rows validate, both incorrect first callbacks are rejected, all eight target exports bind to their arrays, and the explicitly derived FP64 rows equal the retained values exactly.
- The NeMo pretrained archive's complete `FILES.json` verifies 557 indexed files. This is archive verification, not another GRPO run.
- The [portable Luna supplement](luna-portable-index.json) explicitly converts its legacy inventory to the new format after verifying 67 source-export and 17 derived records. All 21 selected members, including seven numerical payloads, are byte-identical. The original ZIP and reviewed cache/shape differences remain unchanged. Four installed-tool operations match: two comparisons and two distribution validators. Target metrics are exact; three JS aggregate differences peak at 3.5051427732354825e-17 and remain visible in the index, within the declared replay tolerance.

Replay the converted Luna bundle directly:

```sh
python skills/precision-skill/scripts/replay_bundle.py --bundle tests/results/gpt-6/compare-skillhardening-20261020T180218Z/luna-portable-evidence.zip --output-root runs
```

This is explicit case conversion using generic primitives, not built-in support for every historical inventory schema. No bundled Python, model weights or framework environments are needed. All numerical contracts still have null acceptance thresholds. These changes improve collection validity, portability and evidence reuse; they do not add native cases or turn earlier measurements into numerical acceptance passes.
