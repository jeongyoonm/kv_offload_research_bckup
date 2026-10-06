# CPU cache initialization experiments

The default remains `cpu_cache_layout="block"` and
`cpu_cache_population="rank"`. The experimental options are passed through
`kv_connector_extra_config`.

## Per-worker cache files

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 2130300354560,
    "cpu_cache_layout": "private",
    "async_h2d_submission": true,
    "async_h2d_chunk_bytes": 134217728
  }
}
```

`private` creates one mmap file per worker. Each file contains that worker's
tensor slots for every logical CPU block. The worker populates contiguous ranges
and registers only its own mapping with CUDA. Files are unlinked after the
mapping barrier, and the mappings stay alive until transfers drain and the worker
releases them.

The CPU budget is server-wide. Logical block capacity matches the default layout
when rank-boundary padding fits that budget. Very small caches can have fewer
blocks because each worker mapping needs page-aligned boundaries. The GLM TP8
2.13 TB configuration preserves the logical block count.

This option supports direct, sharded CPU offloading with single-node TP and the
`mp` executor. Replicated, canonical and secondary-tier layouts are rejected
because they require shared byte-layout contracts. The process THP policy and
NUMA policy are inherited unchanged.

## Alternative experiments

| Option | Values | Default |
| --- | --- | --- |
| `cpu_cache_layout` | `block`, `worker`, `private` | `block` |
| `cpu_cache_population` | `rank`, `contiguous` | `rank` |
| `cpu_cache_population_workers` | Divisor of worker count | All workers |
| `cpu_cache_population_chunk_bytes` | At least 4096 bytes | 256 MiB |
| `cpu_cache_population_barrier` | Boolean | `false` |

`worker` keeps a shared file, puts each worker's blocks in a contiguous range,
and registers that range. It has the same compatibility restrictions as
`private`.

`contiguous` population assigns disjoint ranges to selected initializers and
waits for all workers before returning. Initializers are spread across the worker
indices. In the interleaved `block` layout, assigning complete rows can change
NUMA placement relative to the ranks that transfer those rows. Check transfer
performance before selecting this policy. For `private`, each worker must
initialize its own file.

## Measurement

The initializer logs mapping/population and registration times separately.
`SharedOffloadRegion.init_stats` contains monotonic phase timestamps, elapsed
times, call counts and byte counts. Registration success is also recorded.

Run an isolated eight-worker cell without loading a model:

```bash
PYTHONPATH="$PWD" .venv/bin/python tools/benchmark_cpu_cache_init.py \
  --capacity-gib 64 --layout private --thp disabled --cuda --async-h2d \
  --output /mnt/localnvme/cpu_cache_init_20261007/example.json
```

Replace `--capacity-gib 64` with `--capacity-bytes 2130300354560` for the full
GLM cache. Use `--layout block` for the default comparison, and omit `--cuda`
to measure population alone. `--registration-concurrency` controls concurrent
registration calls in this harness.

The harness records NUMA placement and hugepage coverage, checks first/middle/last
CPU blocks, and verifies byte-exact transfers through the strided KV handlers.
It checks GPU occupancy before starting and stops if a persistent new GPU-device
holder appears. Initialization time excludes transfer validation and teardown.

Run the GLM TP8 startup and CPU-restore canary:

```bash
.venv/bin/python tools/run_glm_cache_init_canary.py \
  --output /mnt/localnvme/cpu_cache_init_20261007/glm_canary
```

The canary retains the 2.13 TB CPU budget and disables THP only in its server
process and descendants. It uses an isolated cache directory and port, resets
GPU prefix state while retaining CPU entries, and checks external cache hits and
cold/restore outputs. It stops its own server when finished.
