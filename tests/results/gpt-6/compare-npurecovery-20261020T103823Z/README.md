# Original-instance recovery and native NPU evaluation

Maintainer: [YuZhengYYDS](https://github.com/YuZhengYYDS). Branch: `validation/gpt-6-cold-skill-20261019`. Starting repository snapshot: `16346fc21a41ff87c1d7a0603f4a6fbb33ac8c32`.

The original 16-device Ascend notebook was repaired and reused. No replacement instance was created. This run separates completed native model execution, independent reference comparisons, partial native failures and hardware/dependency boundaries. Read [results.json](results.json) for the per-lane outcome and evidence cutoff; a computed metric is not a numerical acceptance verdict.

At this freeze, 14 of 17 source-scoped task lanes completed native execution, covering 28 canonical cases. Twenty-seven cases have independent references and 107 core comparison operations; Slime completed a real native optimizer step but has no independent CPU reference or numerical comparison. Untouched upstream NeMo RL training, Megatron Bridge provider/training and TensorRT-LLM generation remain incomplete at their recorded CUDA/NCCL dependency boundaries. All numerical acceptance thresholds remain undefined. Community/vendor adaptations retain distinct source identities and do not establish untouched upstream support. The [completed and unfinished lane ledger](results.json), [Guide](GUIDE.md) and [validation/hash inventory](VALIDATION.json) make these limits explicit.

- [Guide: execution, recovery, limitations and replay](GUIDE.md)
- [SFT: 16 canonical cases and measured metrics](sft/matrix.json), [22-volume evidence index](sft/archive-index.json)
- [MindSpeed-LLM: native pretraining and independent reference evidence](mindspeed-llm/archive-index.json)
- [llama.cpp: two native CANN/CPU inference cases and six replay operations](llama-native-evidence.zip)
- [ms-swift GRPO: native rollout/update, independent reference and five replay operations](swift-grpo-evidence.zip)
- [AReaL: native FSDP/vLLM GRPO and four replay operations](areal-grpo-evidence.zip)
- [MindIE-LLM: native ATB generation and three replay operations](mindie/archive-index.json)
- [LMDeploy: two native Ascend/clean-CPU inference cases and six replay operations](lmdeploy-native-evidence.zip)
- [ROLL: native NPU GRPO, independent reference and five replay operations](roll-grpo-evidence.zip)
- [OpenRLHF community Ascend adaptation: native GRPO, independent reference and five replay operations](openrlhf-community-grpo-evidence.zip)
- [MindSpeed-RL: native rollout/actor update, independent reference and five comparisons](mindspeed-rl/archive-index.json)
- [Slime native completion and three untouched upstream boundaries](slime-and-upstream-boundaries.zip), [source-scoped status](slime-upstream-results.json)
- [Earlier RL native/dependency/observer failures](rlar-failure-history.zip), [parameter ownership/counting supplement](parameterization-supplement.json)
- [Third-party attribution](attribution-v2.zip), [MindSpeed-RL license supplement](attribution-mindspeed-rl.zip)

The earlier [interrupted checkpoint](../compare-npuvalidation-20261020T064804Z/README.md) is preserved unchanged. Its missing old job outcomes remain unknown; current successful runs do not retroactively establish them. This continuation is coordinated execution after the earlier cold-context trials, not a new independent cold trial or a model ranking.
