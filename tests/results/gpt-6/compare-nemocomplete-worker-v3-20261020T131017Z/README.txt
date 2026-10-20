NeMo RL adapted native worker continuation. See RESULTS.json for scope, metrics and limitations.
Extract nemo-worker-evidence.zip into a fresh folder. Run python replay_core.py --skill /path/to/precision-skills/skills/precision-skill --output /fresh/replay to recompute all six reports through actual repository CLIs.
CPU model replay: in Python3.12 with torch2.9.0+cpu, transformers4.57.1, numpy1.26.4, accelerate and safetensors, run python replay_cpu_reference.py --output /fresh/cpu-reference. It reloads the same safe initial checkpoint and fixed trajectory in a fresh process, uses an independent explicit objective, and stores new safe outputs.
The canonical native capture calls the real worker API locally with Ray modified_class and an official Ascend API migration adapter. The official examples/run_grpo.py orchestrator and multi-rank FSDP are not covered by this supplement.

EXPORT.json distinguishes private original and exported hashes. FILES.json verifies exported bytes. ORIGINAL-CONTENT-MANIFEST.json and RAW-CAPTURE-FILES.json are retained original capture provenance, not verification inventories for redacted text.

Source identity matches pinned Git after CRLF normalization. Native/reference NumPy1.26.4 and artifact exporter NumPy2.3.5 are recorded separately.
