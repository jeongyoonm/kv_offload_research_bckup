# D2H driver descriptor cap sweep

Branch: `experiment/store-driver-chunk-sweep-20261009`.
Parent: `6aeab52`.
Artifacts: short roots under `/mnt/localnvme/jeongyoon`.

The existing compiled CUDA path supports a process-wide maximum descriptor count
through `VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS`. The default zero submits one
4,752-descriptor batch per GLM store job; 256, 512 and 1,024 produce smaller
calls on the same ordered D2H stream. No source-access ordering changes occur.
Native H2D uses its own 8 MiB / depth-two submission mechanism and is unaffected.
The cap is cached per process, so each allocation has one fixed cap. Run matching
seeds and GPU-fence scheduler settings, including a fresh cap-zero control.

Startup command explicitly sets the cap with `env`, overriding the inherited
collector default. Record this command and compare CPU wrapper wall time, GPU
gaps, cache restoration, batching, throughput and TTFT. The first implementation
uses an existing opt-in C++ mechanism and changes no model kernels.
