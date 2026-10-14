Workload and replay harness backup
=================================

The main repository snapshot preserves all tracked experiments and their imports.
This directory supplies additional sources and configuration inputs which were
stored outside the vLLM checkout. MANIFEST.json records every added payload hash
and its original location. Existing absolute paths are preserved in the scripts.

Homogeneous / heterogeneous CPU-hit workloads
--------------------------------------------
experiments/load_only_overlap_20261011/workloads.py defines the prefix/suffix grid,
profile-selected transfer-heavy/compute-heavy/balanced mixtures, and evolving
cache workloads. evaluate.py launches baseline, worker-only, or full serving
arms, primes CPU prefixes, resets GPU state, and validates actual cache outcomes.
cache_client.py issues the requests; select_classes.py selects mixture classes.

Use the evaluate.py CLI after configuring the environment and dependency paths:
  .venv/bin/python -m experiments.load_only_overlap_20261011.evaluate --help

Select --suite grid for homogeneous CPU hits, --suite heterogeneous for mixtures,
or --suite evolving for normal cache evolution. For heterogeneous runs pass
--classes research_harnesses/frozen_configs/load_only_overlap_20261011/profiled_classes.json.
Choose --server-kind baseline or optimized; optimized supports --arms worker full.
Use a fresh --output directory and set --concurrencies / --requests / --repeats.
Prefix priming/reset and measured requests have different accounting scopes;
the original experiment README.txt documents them precisely.

Inferact
--------
experiments/agentic_1m_20261012/prepare.py --kind inferact --sessions N prepares
the recorded conversations at a 1M-token total-context limit, with recorded
output lengths. replay.py accepts --root, --arm baseline|full, and --spec.
server.py and campaign.py preserve launch choices, cache sizing and pacing.
experiments/real_traces_20261012/ preserves the earlier bounded-context adapter,
Mooncake adapter, replay tests, and diagnostics. These are distinct protocols.

launch_helpers/trace_characterization_20261012/download_traces.py preserves the
trace download recipe. Raw Inferact conversations and prepared .u32 token pools
remain external; their original dataset manifests and hashes identify inputs.

TraceLab
--------
tracelab_runner/ contains the actual Rust replay sources from commit 98ab008
(model-call slots independent of tool waits; one-second HTTP idle expiry),
Cargo.lock, CSV exporter, utilities, examples, and original licenses/documentation.

Build from the backup repository root:
  cargo build --release --locked \
    --manifest-path research_harnesses/tracelab_runner/replay/Cargo.toml \
    --bin session_runner

The resulting binary is under research_harnesses/tracelab_runner/replay/target/release/.
Use --max-in-flight-requests for model-call concurrency; tool waits do not occupy
those slots. The official-agent variant is separately preserved on
experiment/tracelab-runner/official. Request-slot and keepalive variants are on
experiment/tracelab-runner/request-slots and experiment/tracelab-runner/keepalive.
Each branch is a TraceLab source repository snapshot, rather than a vLLM tree.

experiments/tracelab_runner_20261013/ has official replay and CSV preparation.
experiments/tracelab_request_slots_20261013/ has the independent-call-slot harness.
experiments/tracelab_arrival_intensity_20261013/ prepares reproducible arrival
pacing variants without changing request shapes, session ordering or tool waits.
experiments/tracelab_concurrency_20261013/ and tracelab_capacity_sweep_20261013/
drive concurrency and pressure comparisons. The newest fix comparison is under
experiments/cached_dispatch_horizon_20261014/.

Rebind BINARY/RUNNER to the saved runner and ROOT/SOURCE to fresh paths before
starting historical campaigns. Historical DEADLINE values and old owner/plan
markers must not be reused for a new allocation. The frozen trace manifests and
session metadata are in frozen_configs/. The trace database and enwik9 corpus
remain external; CSV-export and corpus instructions are included in the runner.

Shared configuration and native dependencies
--------------------------------------------
frozen_configs/frozen_predictor.json.gz is the exact 51,048,930-byte frozen_A
predictor used by the CPU-hit and agentic campaigns, compressed for backup.
Its uncompressed SHA256 is
f159f32d226894457fb56b3195c47d4ff6e2a5d982d8d600b1970ea000294f96.
Decompress it to a chosen path and bind STATE/STATE_PATH to that JSON file.
frozen_configs/load_only_environment.json preserves the shared launch template;
update its filesystem paths for a new machine. Cache budgets are aggregate CPU
budgets; GPU KV budgets are per rank.

Native transfer sources are in csrc/libtorch_stable/async_batch_* and their build
integration is in the repository. The execution observer source is
experiments/coalesced_notify_20261004/observer.cpp; its rebuild helper is
experiments/dataset_coverage_20261007/build_observer.py. The fast forest evaluator
source and CMake input are in experiments/predictor_fast_forest_20261006/.
Compiled .so files, CUDA/toolchain installations, and the virtual environment are
not bundled. Preserve the original model selector, weights and serving config
when comparing baseline and prototype.
