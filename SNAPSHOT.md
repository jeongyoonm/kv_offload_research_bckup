# Late-admission prototype snapshot

This repository is a source-only snapshot of the latest committed late-admission prototype from `/root/jeongyoon/vllm/vllm_glm53_late_opt_20260928` at commit `8425112e1445bb9168c0eea8fbd5e3e310cca9fe` (`Record paired 100 ms extension TP8 results`). It has a new, independent Git history.

The implementation includes asynchronous H2D submission, copy pipelining, early KV-load submission and readiness, independent worker preparation for late admission, candidate-ready and load-only early seals, and the bounded productive-predecessor seal extension. The vLLM package, native sources, build metadata, and tests are included so the code remains a usable project checkout. Historical experiment archives, raw benchmark artifacts, model weights, and the original repository history are omitted.

The source commit is the snapshot boundary. Uncommitted changes in the original working tree are not included.
