Actual native inference scoring completed on two pretrained checkpoints: SmolLM2-135M-Instruct (Llama family, pinned 12fd25f77366fa6b3b4b768ec3050bf629380bac) and Qwen2.5-0.5B-Instruct (Qwen2 family, pinned 7ae557604adf67be50417f59c2c2f167def9a775). Each used one short fixed sequence and 3 targets. No random-init model was used.

llama.cpp v0.7.0, SHA 757cad021b357f417065ddbc41a6e8f87c1f8136, ran native CPU F32 weights and KV cache through prefill and 2 fixed-token decode steps. Maximum target logprob difference versus native Transformers CPU F32 was 3.10e-6 nats on Smol and 2.98e-6 on Qwen. LMDeploy 0.18.0 TurboMind and PyTorch both captured full-vocabulary native CUDA FP16 rows on RTX5060, using3 fresh prefix requests per model. Their cached-decode coverage is 0.

- Smol / neutral PyTorch: target MAE 0.00560415 nats; mean tail-merged JS 4.98125e-6.
- Smol / TurboMind: target MAE 0.00564758 nats; mean JS 3.63877e-6.
- Qwen / neutral PyTorch: target MAE 0.00245885 nats; mean JS 3.30953e-6.
- Qwen / neutral TurboMind: target MAE 0.00244645 nats; mean JS 5.51413e-6.

All final comparisons use thresholds=null. Different native kernels and FP16 versus FP32 settings are declared. An additional native Transformers CUDA FP16 Smol reference supplies same-weight-dtype comparisons. Nine side manifests validated; nine comparison pairs replayed with exactly the original metrics.

The important confirmed API mismatch is preserved separately: LMDeploy PyTorch AsyncEngine.async_get_logits constructs GenerationConfig with default temperature 0.8. Every one of 49152 logits in each of 3 returned last rows exactly equals the explicit-neutral native logits divided by FP16(0.8). The processed capture is diagnostic-only under diagnostics/default-temperature/. The corrected native path uses Session.request_handle and AsyncEngine.safe_run with GenerationConfig(max_new_tokens=0, output_logits='all', temperature=1.0, repetition_penalty=1.0, top_p=1.0, top_k=1). Neutral final captures remain separate.

Dependency limits are explicit. Torch 2.8.0+cu128 pins Triton 3.4.0, and LMDeploy 0.18 on that combination failed native warmup at gdc_launch_dependents with unexpected _semantic. Isolated Triton 3.5.1 allowed native PyTorch execution but conflicts with Torch's metadata pin; this is exploratory coverage. LMDeploy's required xgrammar>=0.2.0 conflicts with inherited 0.1.25; rdkit, accelerate, apache-tvm-ffi0.1.11, flash-linear-attention, peft and tilelang0.1.11 are absent. See runtime-metadata-dependencies.json for exact requirements and installed versions. The original shared base environment remains Torch 2.8/Triton 3.4.

TensorRT-LLM had an actual import probe failure (tensorrt_llm absent); its requested LLM entrypoint was not resolved. The retained PyPI metadata shows a wheel stub for 1.2.1; no TensorRT engine ran. MindIE failed ordered initialization at missing torch_npu, with no Ascend devices/CANN/MindIE runtime. Both have zero numerical coverage. Official pinned source, native worker paths/versions and exact source hashes are retained with personal paths and host redacted.

The archive includes all 30 full logprob rows, contracts, manifests, environments/statuses, small logs and capture source, plus exact skill comparator/validator implementations. It excludes model weights/GGUF, native executables and optional raw-logit dumps except six rows needed for the exact default-temperature diagnostic. Array bytes are unchanged. EXPORT.json records raw and exported SHA256 hashes and every transformation. Artifact references are relative to each JSON location. Native capture source is archival; only captured-array offline replay is promised.

After extraction, run `python replay.py --output /path/to/a-new-directory` with Python and NumPy. This was verified from a fresh archive extraction: nine comparisons, nine validated side manifests, three exact temperature-division diagnostic rows, with target/distribution metrics identical to the originals. No native model/runtime execution is implied by replay success.

Proposed generic helper (not implemented): validate a score-semantics input contract before export. Require full unpadded vocabulary, observed/native/storage dtypes, exact API and row mapping, model-defined scale/softcap, effective score-affecting transforms and boundary evidence. Reject unknown boundaries or nonidentity sampling transforms for raw-score captures; distinguish sampling selection from changes to returned rows. A default-versus-neutral native probe can reveal this temperature aliasing issue. Do not silently invert masking/truncation or call the result raw. The full proposed contract is in HELPER-PROPOSAL.txt.

Archive: `inference-evidence.zip`, 7594354 bytes, SHA256 `96b9cf9f1be4584ce0ba41279aa8ffeca2434b6697db6907b0383d4af2e7e166`.

The ZIP uses LZMA compression (method 14). Extract with Python standard-library zipfile if the desktop ZIP handler does not support it: `python -m zipfile -e inference-evidence.zip NEW_DIRECTORY`. All 419 exported artifact hashes were verified against EXPORT.json.
