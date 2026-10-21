# Native CUDA inference supplement

Run: `compare-inferencecuda-20261021T182139Z`. Two pinned pretrained checkpoints; four native CUDA model/backend cases and two independent CPU references completed.

Every case covers one prefill distribution and two native cached-decode distributions at fixed history lengths 6, 7, and 8. Each row retains the full source vocabulary before sampling. Native raw dtypes and forward/cache observations are in `native-summary.json`; `thresholds` is null.

- SmolLM2-135M-Instruct / lmdeploy: JS mean=4.34946098e-06; top-1=3/3; target logprob MAE=0.00448760312, max=0.00706919843.
- SmolLM2-135M-Instruct / llama: JS mean=7.77482116e-07; top-1=3/3; target logprob MAE=0.00194275737, max=0.00328233801.
- Qwen2.5-0.5B-Instruct / lmdeploy: JS mean=2.90957395e-06; top-1=3/3; target logprob MAE=0.00474834139, max=0.00904679203.
- Qwen2.5-0.5B-Instruct / llama: JS mean=4.44365343e-07; top-1=3/3; target logprob MAE=0.00124315145, max=0.00229175486.

Six side manifests validate. Eight overall comparisons and 24 stage comparisons compute metrics; eight portable overall comparisons are replayed. Numeric acceptance was not assessed.

The original callback-exhaustion attempt and its diagnostic source/logs are retained separately. The canonical run has no captured Python traceback or RuntimeError. TurboMind is not counted as newly cached; TensorRT keeps its earlier real two-model evidence; the selected MindIE route remains an Ascend hardware requirement.

GPU collection finished and the GPU was released before this CPU-only finalization.
