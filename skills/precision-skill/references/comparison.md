# Compare artifacts

## Tools

Select tools by operation and supported input semantics. The tool library lives in `scripts/`; command-line entrypoints and importable Python functions are both tools. Prepare NumPy for array tools when needed; target-score tools use the standard library.

| Operation | Tool entrypoint | Inputs and outputs |
|---|---|---|
| Check runtime initialization / interfaces | [runtime_probe.py](../scripts/runtime_probe.py), CLI / Python `probe_runtime`, `inspect_runtime` | Ordered imports, symbols and task checks → actual module identity, distribution metadata and diagnostics in a fresh interpreter or native worker |
| Capture fixed-history inference / export scores | [inference_capture.py](../scripts/inference_capture.py), Python `FixedHistory`, `derive_logprobs`, `validate_history`, `validate_generation_scores`, `export_records` | Native rows and fixed tokens → declared logprob derivation, history/score-row checks and verified distribution/target-score records |
| Compare target/action scores | [compare_target_scores.py](../scripts/compare_target_scores.py), CLI | Contract and paired score records → target-score metrics, including EOS and cached execution |
| Compare full distributions | [compare_distributions.py](../scripts/compare_distributions.py), CLI | Contract and paired manifests → top-100 union/tail metrics |
| Compare distributions across hosts | [query_distributions.py](../scripts/query_distributions.py), CLI / Python | Source manifests → inventories and union queries → comparison; full arrays stay at source |
| Compare tensors | [compare_tensors.py](../scripts/compare_tensors.py), CLI | Contract and paired manifests → observation, gradient, postclip-gradient or stored-update metrics |
| Export logical tensors / compare replica views | [tensor_artifacts.py](../scripts/tensor_artifacts.py), CLI / Python | Per-rank inventories → logical manifests; explicit view pairs → comparisons |
| Compare assistant-body scores | [compare_scores.py](../scripts/compare_scores.py), CLI | Single-turn cache-off score artifacts with execution evidence → metrics |
| Validate artifacts | [validate_artifacts.py](../scripts/validate_artifacts.py), CLI | Contract and one side's tensor/distribution manifest → validation and coverage |
| Package / verify / extract evidence | [evidence_bundle.py](../scripts/evidence_bundle.py), CLI / Python `pack`, `verify`, `extract` | Curated files and optional literal text replacements → indexed source/export hashes, streamed verification and fresh extraction |
| Replay retained comparisons | [replay_bundle.py](../scripts/replay_bundle.py), CLI / Python `replay` | Indexed ZIP with `REPLAY.json` → selected skill CLI reports and expected-subset checks, preserving explicit side order |
| Plan msProbe module capture | [msprobe_capture.py](../scripts/msprobe_capture.py), Python `plan_modules` | Native module inventory, selected paths and output sizes → L0 configuration, observation selection and payload estimate; execution uses the [layerwise integration](layerwise.md#integrate-and-verify) |

Supporting modules provide [artifact readers](../scripts/artifact_io.py), [execution-condition checks](../scripts/conditions.py) and [target-score arithmetic](../scripts/score_metrics.py). Reuse them inside tools and task adapters; their module docstrings own the interfaces. Task adapters own native masks, logical mappings, losses and objective reductions.

Run from the skill directory. The score, distribution and tensor comparators share `--contract`, `--side-a`, `--side-b` and `--output`; the validator uses `--manifest` for one side's input. Use fresh output paths. The [synthetic tensor example](../examples/tensors.py) generates, validates and compares a small pair of artifacts. For a worked collection task, see the [fixed-history inference example](../examples/fixed-history-inference.md).

```sh
python scripts/compare_tensors.py --contract gradients.json --side-a a.json --side-b b.json --output report.json
```

Read the selected interface before collection. Distribution matrix rows belong inside each `scores` descriptor, e.g. `scores: {path: rows.npy, row: 3}`; multirow arrays require an explicit row. Put equal conditions in `common` and task-selected differences in `side_conditions`; each rank records its actual settings and imported versions. [Condition checks](../scripts/conditions.py) compare these against the contract-owned rank inventory. For task-selected different initial parameter representations, declare per-side `state`; common `state` retains exact before-value checks. Implement missing task-specific integration and reuse the shared arithmetic.

Use `--help` for CLI usage. When adapting inputs, extract only the selected module docstring with Python `ast`; distribution/tensor schemas share [artifact_io.py](../scripts/artifact_io.py). Inspect implementation functions as needed.

Contracts set `thresholds: null`. Outputs contain metrics, coordinate counts and unavailable comparisons. Exit codes: 0=comparison metrics computed successfully, 1=unavailable comparison, 2=CLI/output error.

## Portable evidence

Stage only the intended public files in a separate directory. Keep model/environment trees, credentials and private filenames out of that selection. The archive tool streams unchanged payloads and records each source/export SHA256 and byte count. Optional replacements are explicit literal pairs applied to valid UTF-8 text across extensions, including logs, YAML and shell files; keep the replacement specification outside the staged directory. Protected numeric/model extensions (listed in the docstring), NUL-containing files and invalid UTF-8 are unchanged. It does not detect every possible binary format; use untransformed packing for other raw formats. Paths are preserved, so filename review and privacy scanning remain the caller's responsibility. The tool neither discovers every secret nor certifies privacy. Avoid broad automatic IP replacement: public package versions can have the same spelling as an IPv4 address.

From the skill directory:

```sh
python scripts/evidence_bundle.py pack STAGED_FILES NEW.zip --replacements PRIVATE_REPLACEMENTS.json
python scripts/evidence_bundle.py verify NEW.zip
python scripts/evidence_bundle.py extract NEW.zip NEW_DIRECTORY
python scripts/replay_bundle.py --bundle NEW.zip --output-root runs
```

Omit `--replacements` when no text changes are needed. `EXPORT.json` covers every member except itself. Verification checks unique portable paths, complete coverage, hashes and any declared byte counts; extraction requires a new output directory. The default expanded-byte cap is 16 GiB and index JSON is capped at 16 MiB. Use `--max-expanded-bytes` to select a different archive/replay cap. Older supported `EXPORT.json` lists may omit sizes; their receipts disclose ZIP-metadata-derived lengths. An explicit `--index FILES.json` supports a complete plain path-to-hash/size inventory. Other legacy layouts require an explicit conversion/adapter; their archived verifier scripts are not executed by these tools.

Review compiled Python and raw serialized payloads separately: `.pyc` embeds original filenames, and pickle metadata can contain private paths even when tensor bytes are safe. Prefer typed numeric arrays plus an explicit input tree when those payloads are unnecessary for public replay. Preserve private originals and their hashes; do not redact protected binary bytes while claiming identity. Verify public sidecar receipts as well as archive members, since a verifier's local output path may itself be private.

`REPLAY.json` contains nonempty `cases`, each with unique `name`, `operation`, `inputs` and a nonempty `expected` report subset. Operations are `validate` (contract/manifest) and `compare_tensors`, `compare_distributions`, `compare_target_scores` (contract/side_a/side_b). Use relative paths, preserving the contract's side identities and order. Expected floating values use `rel_tol=abs_tol=1e-12`; these are retained-metric replay tolerances, not model acceptance thresholds. Replaying uses installed tools, or an explicitly selected `--scripts` directory, and does not execute bundled Python. Artifact references consumed by those tools must resolve inside the extracted evidence. A successful replay does not repeat native training/generation, prove source authenticity or supply missing raw arrays.

## Coordinates

Pair scores by sample/prediction position/token and tensors by logical tensor/element. For `stage: observations`, declare each tensor's logical sampling `coordinates` in the contract and both manifests; retain sample/token selection, boundary, invocation and execution stage as applicable. Native names map to this identity in the adapter. Reconstruct logical coordinates from shards, packed batches and fused projections; retain tied-weight mappings and reduction tags. Check expected coordinates, masks and ranks, and verify converted initial values. Exclude padding, deduplicate equal replicas, and retain differing replicas as explicit views. Distinguish absent gradients from present zeros.

## Target-token scores

For two executions' logprobs $\ell_{a,i},\ell_{b,i}$ over $N$ valid targets, report each side's NLL and their absolute difference:

$$
\mathrm{NLL}_{s} = -\frac{1}{N}\sum_{i=1}^{N} \ell_{s,i}
$$

For $e_i=|\ell_{a,i}-\ell_{b,i}|$, report MAE, P99 and maximum with its coordinate.

Use the global valid-target count. Native loss retains its implemented reduction. Percentiles use all paired errors, linearly interpolated at $a(N-1)$; P99 uses $a=0.99$.

## Tensors

For paired tensors $a$ and $b$, use stable FP64 reductions:

| Metric | Definition |
|---|---|
| L2 error | $\lVert a-b\rVert_2$ |
| RMSE | $\lVert a-b\rVert_2/\sqrt{N}$ |
| Cosine | $a^\top b/(\lVert a\rVert_2\lVert b\rVert_2)$ |
| Maximum absolute error | $\max_i|a_i-b_i|$ |

Global metrics concatenate coordinates. Retain counts and signal/error norms. Actual updates are $\theta_{after}-\theta_{before}$, promoting stored values to at least FP32 before subtraction.

Cosine is undefined when either norm is zero. Represent undefined metrics as null with a reason.

Use finite target scores/tensors; distribution logprobs permit `-inf` for zero probability.
