# Main-branch validation: 2026-10-19

The public suite at commit `b8c9d481af3a954d6613f4fdb5643727d5eb84ab`
passed all **116 tests on Linux/WSL**, with no failures, errors or skips.
The tested scripts and tests are unchanged. This branch adds the validation
record and its publication allowlist.

- `test_artifact_tools.py`: 39 tests.
- `test_compare_scores.py`: 58 tests.
- `test_shared_tools.py`: 19 tests.

Windows executed the same 116 tests: 115 passed and one errored before its
assertions because creating a directory symlink requires a privilege unavailable
to that process (`WinError 1314`). The failed test was
`test_compare_scores.CompareTests.test_evidence_paths_resolve_directory_symlinks`.
It passed in the subsequent complete WSL run. The Windows failure was retained;
no test was removed or skipped and no host privilege setting was changed.

The Windows run used Python 3.12.14 / NumPy 2.3.5; WSL used Python 3.11.17 /
NumPy 2.2.6. Unittest reported 7.248 seconds on Windows and 2.266 seconds on
Linux. The machine-readable record separately reports whole-process wall time.

The documented synthetic tensor example also passed: two paired elements,
L2 error 4, RMSE sqrt(8), cosine 0.6 and maximum absolute error 4 at `weight[1]`.
Both single-side validations returned `validated`; the comparison returned
`no_thresholds`. Skill metadata validation and all 46 local Markdown links at
the tested commit passed.

## Evidence

[verification.json](verification.json) records the exact commit, environments,
every Linux test ID, outcomes, tested Python source hashes and archive digest.
[evidence.zip](evidence.zip) contains both test logs, metadata/link checks and
the synthetic example's contracts, manifests, arrays and reports.

`EXPORT.json` distinguishes exact copies from derived files and records source
and exported SHA256 for each retained artifact. The Windows log replaces local
path prefixes with placeholders and normalizes line endings; the example report
replaces its source paths with archive-relative paths. Test outcomes and metric
values are preserved. Original private files remain unchanged. These hashes
check consistency, not independent authenticity of execution.

## Reproduce

From the repository root, use a Python environment with NumPy. Linux supports
the symlink regression without the Windows privilege requirement:

```sh
python -B -m unittest discover -s tests -v
python -B skills/precision-skill/examples/tensors.py --output-root /writable/output
```

To replay the retained example, choose fresh extraction and report paths:

```sh
python -m zipfile -e tests/results/2026-10-19/evidence.zip replay-evidence
python -B skills/precision-skill/scripts/compare_tensors.py \
  --contract replay-evidence/example/contract.json \
  --side-a replay-evidence/example/a/manifest.json \
  --side-b replay-evidence/example/b/manifest.json \
  --output replay-report.json
```

The public tests cover offline artifact validation, numerical reductions,
coordinate/condition handling, command-line behavior and msProbe module planning
with fake model objects. They do not execute real models, GPU/NPU kernels,
distributed communication or live msProbe capture hooks. No public test entry
remains unexecuted at the recorded commit.
