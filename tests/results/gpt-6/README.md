# GPT-6 skill validation records

Maintainer: [YuZhengYYDS](https://github.com/YuZhengYYDS).

This directory records parent inspection, clean-context skill execution,
reproduced bugs, reusable operations and verification evidence. `gpt-6` is the
model-family label supplied to this Codex session; a more specific backend model
identifier was not exposed. Child agents inherited the model selection.

- [2026-10-19 cold training trial and artifact fixes](compare-skillvalidation-20261019T083016Z/README.md)
- [2026-10-19 expanded native framework evaluation](compare-frameworkmatrix-20261019T124628Z/README.md)
- [Expanded evaluation guide](compare-frameworkmatrix-20261019T124628Z/GUIDE.md)
- [2026-10-20 NPU native follow-up checkpoint (incomplete)](compare-npuvalidation-20261020T064804Z/README.md)
- [NPU preparation and recovery guide](compare-npuvalidation-20261020T064804Z/GUIDE.md)
- [2026-10-20 original-instance recovery and native evidence](compare-npurecovery-20261020T103823Z/README.md)
- [Native execution, failures and replay guide](compare-npurecovery-20261020T103823Z/GUIDE.md)
- [2026-10-20 remaining native workflows and references completed](compare-remainingcomplete-20261020T140716Z/README.md)
- [Completion, numerical differences, unfinished scope and replay guide](compare-remainingcomplete-20261020T140716Z/GUIDE.md)
- [Reusable evidence replay utility](replay.py)

The cold worker starts with `fork_turns="none"`, receives a task and a pinned
checkout, and does not read parent findings, repository tests or history. This
isolates conversation context, not the runtime's system instructions or tools.
Parent review and later replay are recorded separately from the cold outcome.

## Replay a retained bundle

From the repository root, with Python and NumPy:

```sh
python -B tests/results/gpt-6/replay.py \
  --bundle tests/results/gpt-6/compare-skillvalidation-20261019T083016Z/evidence.zip \
  --output-root runs
```

The utility checks exported SHA256 hashes, extracts into a fresh UTC run, calls
the existing artifact tools and verifies the assertions in `REPLAY.json`.
Assertions check retained artifact results; earlier bundles contain synthetic
examples and later native bundles identify their actual capture provenance.
They are not model precision thresholds. The bundle index checks consistency,
not independent authenticity.

For another bundle, retain `EXPORT.json` with `files` entries containing `path`
and `exported_sha256`, and `REPLAY.json` with `cases`. Each case declares a unique
`name`, `operation` (`compare_tensors`, `compare_distributions`,
`compare_target_scores` or `validate`), relative `inputs`, and an
`expected` nested subset of the tool result. Numeric assertions use FP64 replay
tolerance (`rel_tol=1e-12`, `abs_tol=1e-12`). An optional `--scripts` selects another
checkout's `skills/precision-skill/scripts` directory for before/after replay.
