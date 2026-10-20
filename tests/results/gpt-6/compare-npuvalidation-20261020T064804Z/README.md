# NPU native follow-up: incomplete checkpoint

This is a **progress checkpoint, not a completed evaluation**. None of the 17 requested task lanes has a retrieved, verified native numerical completion in this NPU run. Dependency installation, an actual NPU tensor operation, started jobs and prepared adapters are recorded separately. The earlier [CPU/CUDA evaluation](../compare-frameworkmatrix-20261019T124628Z/README.md) remains historical evidence; its successful cases are not NPU results.

SSH authentication succeeded and the actual host was inspected: aarch64, 16 logical Ascend910_9382 devices with 64 GiB each, driver 25.5.2, CANN 9.1.0 and Python 3.12.13. Four disjoint device groups were assigned to the parent and three follow-up agents. A real NPU square operation returned `[1, 4, 9]` from `[1, 2, 3]`. This proves runtime execution only.

Task-owned environments, source copies, wheel caches and detached logs were created on persistent storage with approximately 1.3 TB available. The container root filesystem has a 10 GiB limit. The shared Python environment, original framework sources, model weights, driver and CANN installation were retained.

AReaL and ROLL private dependency stages completed. AReaL's second native attempt and MindSpeed-LLM's repaired one-step attempt were launched; their results were not retrieved. MindIE's second native library build and the community OpenRLHF/Slime private installs also had unknown outcomes when the SSH TCP endpoint began refusing connections. The parent independently confirmed the transport failure. Its cause is unknown; it is neither a framework failure nor a successful run.

Read [GUIDE.md](GUIDE.md) for the executed preparation, actual failures, support distinctions, evidence boundaries and recovery order. [results.json](results.json) is the 17-lane ledger. The [preparation archive](preparation-evidence.zip) contains sanitized progress records, adapters and provenance with per-file export hashes. It contains no native numerical result or replay assertion that could establish a completed framework test.
