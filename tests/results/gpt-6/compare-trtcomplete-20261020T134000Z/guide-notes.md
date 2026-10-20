# TensorRT-LLM continuation

Both the seeded tiny Llama and pretrained Qwen2.5-0.5B-Instruct completed real `tensorrt_llm.LLM.generate` calls and clean shutdown on the local RTX 5060. This is the official 1.2.0 wheel PyTorch backend with TRTLLM attention, TP1, unquantized requested FP16 loading. It is not an Ascend port or a serialized-engine build. Independent CPU FP32 eager forwards consume only the same source weights and the native token histories; they do not read native scores.

Each case captures one prefill plus three cached decode distributions. Six actual comparator CLIs ran, all with `thresholds: null` and status `no_thresholds`. Native and CPU greedy argmax agree at all eight predictions. Pretrained raw-logit RMSE is 0.09504748, max absolute error 0.79280019; target logprob MAE is 0.03503267. These errors are observations, not an acceptance pass. Native converted weights and cache tensors were not directly copied. Raw outputs are FP32; derived logprobs are explicitly offline FP64 log-softmax.

An upstream callback coordinate defect was reproduced in both models with `return_context_logits=True`: the first logits-processor callback exposes the first prompt row, whereas the actual first-generation distribution is the final prompt row. All three decode callback rows match the returned generation logits exactly. The canonical export uses unchanged returned API generation rows, independently checks the final-context/first-generation identity, and preserves the initially rejected exporter and every raw callback. No library numerical source was patched. See `independent-audit/final-semantic-audit.json` and the retained installed source for exact lines and evidence.

Private runtime setup repaired missing Python/OpenMPI libraries, installed CUDA 13.1 compiler headers, and pinned PyArrow 21 for NumPy 1.26 compatibility. Torch is 2.9.1+cu130; runtime API is CUDA 13.1 and driver API 13.2, driver 595.95. `pip check` passed. The ModelOpt/Transformers warning and optional MPI-plugin warnings remain visible; quantization was not used. Whole-wheel downloads were not retained, so only actual installed library/RECORD hashes are asserted. The actual wheel multi-component LICENSE is included unchanged.

`EXPORT.json` covers every member other than itself, with original/exported SHA256. Binary arrays are unchanged; only private paths/hosts and JSON formatting may differ. NVIDIA version strings `10.14.1.48`, `10.4.0.35`, `10.4.1.81` are public package versions, not connection addresses. Failed setup and exporter attempts remain labelled auxiliary evidence. `REPLAY.json` replays the six final comparisons through the shared utility; it does not rerun the GPU model.

From the repository root:

```sh
python -B tests/results/gpt-6/replay.py --bundle tests/results/gpt-6/compare-trtcomplete-20261020T134000Z/tensorrt-llm-native-and-reference.zip --output-root runs
```
