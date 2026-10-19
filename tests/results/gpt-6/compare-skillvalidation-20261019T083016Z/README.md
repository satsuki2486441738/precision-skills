# Cold skill execution and artifact-tool fixes, 2026-10-19

The clean-context worker completed the synthetic training task on the unmodified
`main` snapshot. Its 95 listed arithmetic, storage, coverage and behavior checks
were verified. Parent inspection independently reproduced and fixed two other
artifact-tool defects. The fixed public suite passes all 119 tests on Linux/WSL.

This is an offline CPU artifact-interface trial. No real model, autograd,
GPU/NPU kernel, distributed process group or live msProbe hook was executed.
All precision contracts retain `thresholds: null`.

## Repository progress inspected before delegation

- `main`: `f7325d2e919e04368b0e7d9d59299486c1ff85fa`, defining skill orchestration
  and callable tool responsibilities. This is the pinned cold-worker baseline.
- `validation/main-tests-20261019`: `602c8e5647a326066ed5ff46086ce97670f7e25b`,
  recording 116 public tests at `b8c9d481af3a954d6613f4fdb5643727d5eb84ab`.
  Linux passed 116/116; Windows had one directory-symlink privilege error.
  This branch and current main each have one commit absent from the other.
- `docs/resource-reuse-npu-validation`: `320b0a85e9700960171d2c92d0538afd0148b202`,
  an ancestor 46 commits behind current main, with no unique commit to integrate.
- The authenticated GitHub account was `YuZhengYYDS`, with repository push access.

The new publication branch is `validation/gpt-6-cold-skill-20261019`.
Historical branches and runs are preserved.

## Cold execution protocol and findings

The worker was launched with `fork_turns="none"` into an independent detached
worktree at the baseline commit. It received only the concrete synthetic task,
skill/worktree/runtime locations, write boundaries and a 12-minute budget.
It did not read parent findings, tests, Git history or another worktree.
System instructions and tool availability remain part of its environment.

The task used the repository's `precision-skill`: prescribed FP32 gradients,
native NumPy FP16 versus FP32 parameter storage, actual stored before/after
updates, two logical ranks, sharded tensors, an absent gradient and a present
zero gradient. Declared numerical/state differences were accepted. Four exports,
four single-side validations and two normal comparisons succeeded on the first
integration attempt; both deliberate negative comparisons returned exit 1.

- Gradients: 6 paired elements, L2 error 1, RMSE 0.408248290463863,
  cosine 0.9939990885479664; maximum error 1 at `synthetic_model.weight[3]`.
- Stored updates: 8 paired elements, L2 error 0.3535533905932738, RMSE 0.125,
  cosine approximately 0.9812567051925867; maximum error 0.25 at weight index 0
  (tied with index 3). The FP16 first-coordinate update rounds away in storage.
- Omitting the complete zero-gradient entry returned `not_covered`.
  Changing rank 1's accumulation setting returned `incomparable`.

No confirmed tool bug was discovered by this cold task. It found three useful
reporting gaps: inventory failures omit missing names/counts; setting failures
omit side/rank/nested-field/expected/actual details; report-field wrappers need
discovery from the first output. Its initial expectation of retaining unaffected
tensor metrics for a whole-inventory omission did not hold. This is recorded as
a documentation/diagnostic limitation, separately from the verified checks.

## Parent-discovered bugs and fixes

**BUG-001: ambiguous Safetensors headers were accepted.** A header containing
two `x` definitions silently selected the later value (99 instead of 1).
Nested duplicate keys and nonfinite JSON header literals were also accepted.
The format disallows duplicate keys; see the
[upstream format specification](https://github.com/safetensors/safetensors#format).
`artifact_io.parse_json` now shares the existing strict JSON rules between
artifact documents and Safetensors headers. This is a targeted parser fix,
not a claim of comprehensive Safetensors format validation.

**BUG-002: non-object distribution positions escaped diagnostic handling.**
`positions: [null]` raised `AttributeError` from comparison, validation and query
inventory entrypoints. CLI report files were not written. The shared coordinate
reader now rejects non-object records explicitly; comparison and validation
return `incomparable`, while the query API raises the expected `ArtifactError`.

Both defects were reproduced before editing. Three permanent regression methods
cover duplicate/nested keys, nonfinite header literals, five malformed position
types in contracts/manifests, and diagnostic CLI output. Their pre-fix failures
and post-fix success are retained. The temporary reproducer used stdin and left
no temporary source file. Independent review found no actionable fix issue.

## Reusable operations

- Implemented: `artifact_io.parse_json(data)` accepts text/bytes and enforces
  unique keys and finite numbers, removing a permissive second decoding path.
- Implemented: [evidence replay utility](../replay.py) checks archive hashes,
  restores portable artifacts, invokes existing tools and compares nested result
  subsets. It supports another checkout through `--scripts`.
- Reused: rank inventory export, absence handling, stored-update promotion,
  condition validation and tensor arithmetic already existed; no parallel
  exporter or metric framework was needed.
- Candidate, not implemented: structured inventory and condition differences
  with missing names/elements and side/rank/nested-field/expected/actual fields.
  This should extend existing condition/inventory checks.

Replay-tool review caught a list-boundary assertion inconsistency during
development. Recursive list checking fixed it before publication. Smoke checks
cover valid replay, modified array hashes, incorrect expected values and nested
list assertions. This development issue is separate from the two baseline bugs.

## Validation and evidence

- Linux/WSL: 119/119 tests passed, no skips (Python 3.11.17 / NumPy 2.2.6).
- Windows: 118 passed, 1 existing `WinError 1314` directory-symlink privilege
  error, no skips (Python 3.12.14 / NumPy 2.3.5). No privilege setting changed.
- Targeted artifact suite: 42/42 passed, independently repeated by the reviewer.
- Original synthetic tensor example: expected L2 4, RMSE sqrt(8), cosine 0.6.
- Skill metadata passed the bundled validator in WSL; all 52 original local
  documentation links/anchors passed. Windows lacked optional PyYAML for the
  metadata checker, so the existing WSL environment was reused.
- Retained cold artifacts: 8 replay cases matched on both baseline and fixed
  code, including four single-side validations and the two expected negatives.

[results.json](results.json) contains the parent ledger, scope and hashes.
[evidence.zip](evidence.zip) contains the 79 cold-run files, original worker
report, parent reproduction/test logs, and replay assertions. JSON references
within the cold artifacts were made relative; personal path prefixes were
replaced with placeholders. Array bytes are unchanged. `EXPORT.json` records
original/exported hashes and transformations; the worker's `artifact_index.json`
retains its original-byte meaning. Source hashes reflect tested working-tree
bytes; Git line-ending normalization may change blob hashes. No historical
local run was rewritten.

Use the command in the [model directory guide](../README.md#replay-a-retained-bundle)
to replay the bundle. The archived collector/verifier scripts document the
original task and expect its original run layout; use the portable replay tool
for the exported bundle. Hash checks demonstrate consistency, not independent
execution authenticity.
