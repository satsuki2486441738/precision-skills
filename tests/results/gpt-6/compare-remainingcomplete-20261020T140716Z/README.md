# Remaining framework completion

Maintainer: [YuZhengYYDS](https://github.com/YuZhengYYDS). Branch: `validation/gpt-6-cold-skill-20261019`.

All 16 requested libraries /17 task lanes now have a completed native workflow through the explicitly selected source route and an independent reference. The lineage contains34 canonical paired cases and138 comparison groups:136 publicly portable,2 pretrained Bridge full tensor groups retained at private source. Validator calls, repeated captures and the zero-reward NeMo smoke case are excluded from those counts. No numerical acceptance thresholds were supplied.

Read [GUIDE.md](GUIDE.md) for what ran, exact completion boundaries, numerical differences, replay commands and unfinished scope. [results.json](results.json) lists every lane and [VALIDATION.json](VALIDATION.json) owns final hashes/checks. New evidence supplements are [Slime](../compare-slimereference-20261020T123737Z/index.json), [NeMo worker](../compare-nemocomplete-worker-v3-20261020T131017Z/RESULTS.json), [NeMo tiny CLI](../compare-nemoofficialcli-v2-20261020T134600Z/RESULTS.json), [NeMo pretrained CLI](../compare-nemopretrained-20261020T135930Z/RESULTS.json), [Bridge](../compare-bridgecomplete-20261020T134310Z/INDEX.json), and [TensorRT-LLM](../compare-trtcomplete-20261020T134000Z/INDEX.json).

Bridge uses a disclosed NPU fork and NeMo uses disclosed NPU adapters. TensorRT-LLM ran on the local NVIDIA GPU. Historical untouched-upstream Ascend failures remain unchanged in the [previous checkpoint](../compare-npurecovery-20261020T103823Z/GUIDE.md). This is not an all-NPU, all-version, scaling or numerical-pass claim.
