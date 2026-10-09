# Ascending physical assignment for descriptor merging

Live D2H traces from the first two measured pairs in `sc1` show no descriptor
merges: 4,752 address pairs remain 4,752. Reversed physical GPU order cannot
merge against ascending CPU destinations. This proposal sorts the newly
selected GPU blocks before attaching them to a request's logical block table.
It preserves the selected set, reference accounting and pool capacity at each
allocation. It does not relocate any existing live KV bytes.

Both arms enable store-aware allocation, global GPU reuse fences and bounded
H2D/D2H descriptor merging. Only `ascending_gpu_block_assignment` differs; its
default is false. CPU tests cover fragmented pools, pending-store avoidance,
all-pending fallback, selected sets, references, exhaustion and freeing.
Existing real-GPU coalescing tests must pass before serving. Serving validates
restored token counts, zero GPU cache hits and preemptions, one output token,
frozen predictor state and all-rank preparation coverage. Actual merge rates
and serving gains are experimental gates; do not infer them from layout alone.

Four separately primed/reset 75-request episodes per arm give 300 requests per
arm. Original selector, model, FP8 KV, cache sizes and scheduler policy remain
the same as the matched store experiments.
