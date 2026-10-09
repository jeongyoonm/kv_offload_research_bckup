# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
from typing_extensions import override

from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


def _all_workers_barrier() -> None:
    """Block until every worker rank has reached this point (gloo cpu group).

    A superset of the node-local mmap openers suffices: once the barrier
    releases, every worker sharing the region file has mapped it."""
    from vllm.distributed.parallel_state import (
        get_inner_dp_world_group,
        get_world_group,
    )

    try:
        group = get_inner_dp_world_group()
    except AssertionError:
        group = get_world_group()
    group.barrier()


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU blocks requested by each "
                    "KV offload prepare_store call."
                ),
                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = config.parallel.world_size
        self.memory_layout = self.extra_config.get("cpu_cache_layout", "block")
        if self.memory_layout not in ("block", "worker", "private"):
            raise ValueError("cpu_cache_layout must be block, worker or private")
        if self.memory_layout != "block" and (
            config.replicated_layout
            or config.canonical_layout
            or type(self) is not CPUOffloadingSpec
        ):
            raise ValueError("worker layout supports direct sharded CPU caching only")
        self.population_mode = self.extra_config.get("cpu_cache_population", "rank")
        self.population_workers = self.extra_config.get(
            "cpu_cache_population_workers", world_size
        )
        self.population_chunk_bytes = self.extra_config.get(
            "cpu_cache_population_chunk_bytes", 256 * 1024 * 1024
        )
        self.population_barrier = self.extra_config.get(
            "cpu_cache_population_barrier", False
        )
        if self.population_mode not in ("rank", "contiguous"):
            raise ValueError("cpu_cache_population must be rank or contiguous")
        if (
            type(self.population_workers) is not int
            or not 1 <= self.population_workers <= world_size
            or world_size % self.population_workers
        ):
            raise ValueError("cpu_cache_population_workers must divide world_size")
        if (
            type(self.population_chunk_bytes) is not int
            or self.population_chunk_bytes < self.BLOCK_SIZE_ALIGNMENT
        ):
            raise ValueError("cpu_cache_population_chunk_bytes must be at least a page")
        if type(self.population_barrier) is not bool:
            raise ValueError("cpu_cache_population_barrier must be a boolean")
        self.num_blocks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0
        self.replicated_layout = config.replicated_layout and self._uses_shared_region()
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            num_copies = 1 if self.replicated_layout else world_size
            kv_bytes_per_block = config.worker_kv_bytes_per_block * num_copies
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_chunk // num_copies

            # calculate num_blocks
            aligned_kv_bytes_per_chunk = round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk
            if self.memory_layout != "block":
                # Keep the same logical capacity; only rank-boundary padding
                # changes. Small caches can need one fewer block to fit.
                while num_copies * round_up(
                    self.num_blocks * self.cpu_page_size_per_worker,
                    self.BLOCK_SIZE_ALIGNMENT,
                ) > int(cpu_bytes_to_use):
                    self.num_blocks -= 1

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            # or |--- B0 (single copy) ---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        async_h2d_submission = self.extra_config.get("async_h2d_submission", False)
        if not isinstance(async_h2d_submission, bool):
            raise ValueError("async_h2d_submission must be a boolean")
        if async_h2d_submission and not current_platform.is_cuda():
            raise RuntimeError("async_h2d_submission requires CUDA")
        async_h2d_chunk_bytes = self.extra_config.get(
            "async_h2d_chunk_bytes", 128 * 1024 * 1024
        )
        if type(async_h2d_chunk_bytes) is not int or async_h2d_chunk_bytes <= 0:
            raise ValueError("async_h2d_chunk_bytes must be a positive integer")
        self.async_h2d_submission = async_h2d_submission
        self.async_h2d_chunk_bytes = async_h2d_chunk_bytes
        self.async_d2h_submission = self.extra_config.get("async_d2h_submission", False)
        if not isinstance(self.async_d2h_submission, bool):
            raise ValueError("async_d2h_submission must be a boolean")
        if self.async_d2h_submission and not current_platform.is_cuda():
            raise RuntimeError("async_d2h_submission requires CUDA")
        self.async_d2h_chunk_bytes = self.extra_config.get(
            "async_d2h_chunk_bytes", 8 * 1024 * 1024
        )
        if (
            type(self.async_d2h_chunk_bytes) is not int
            or self.async_d2h_chunk_bytes <= 0
        ):
            raise ValueError("async_d2h_chunk_bytes must be a positive integer")
        self.async_h2d_max_descriptors = self.extra_config.get(
            "async_h2d_max_descriptors", 4096
        )
        if (
            type(self.async_h2d_max_descriptors) is not int
            or self.async_h2d_max_descriptors < 0
        ):
            raise ValueError("async_h2d_max_descriptors must be nonnegative")
        self.force_h2d_dma = self.extra_config.get("force_h2d_dma", False)
        if not isinstance(self.force_h2d_dma, bool):
            raise ValueError("force_h2d_dma must be a boolean")
        ring_paths = self.extra_config.get("_early_load_completion_ring_paths", ())
        rank = self.config.parallel.rank
        self.async_h2d_completion_ring_path = (
            ring_paths[rank] if rank < len(ring_paths) else ""
        )
        self.async_h2d_completion_session = int(
            self.extra_config.get("_early_load_completion_session", 0)
        )

        self.cache_policy_module_path: str | None = self.extra_config.get(
            "cache_policy_module_path"
        )

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must be offered for
            # storage before it is eligible for CPU offloading.  Values < 2
            # disable filtering (a threshold of 1 equals no filter; 0 is the
            # default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def _uses_shared_region(self) -> bool:
        """Whether the worker CPU buffer is the shared mmap region (vs a private
        per-rank tensor); replicated-layout dedup is gated on this being True."""
        return current_platform.is_cuda_alike()

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        mmap_region: SharedOffloadRegion | None = None
        # num_blocks == 0 would size the region to zero bytes, which cannot be
        # mmap'd; fall back to the tensor path (empty tensors) as before.
        if self._uses_shared_region() and self.num_blocks > 0:
            # Replicated layout puts all ranks on slot 0 (single MLA copy);
            # otherwise each rank takes its own slot by physical device index.
            if self.replicated_layout:
                rank = 0
            else:
                world_size = self.config.parallel.world_size
                rank = torch.accelerator.current_device_index() % world_size
                if self.memory_layout != "block":
                    rank = self.config.parallel.rank
            mmap_region = SharedOffloadRegion(
                engine_id=self.config.engine_id,
                num_blocks=self.num_blocks,
                rank=rank,
                kv_bytes_per_block=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
                barrier=_all_workers_barrier,
                population_mode=self.population_mode,
                population_workers=self.population_workers,
                population_chunk_bytes=self.population_chunk_bytes,
                num_workers=self.config.parallel.world_size,
                process_rank=self.config.parallel.rank,
                population_barrier=self.population_barrier,
                memory_layout=self.memory_layout,
            )
        try:
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_blocks=self.num_blocks,
                mmap_region=mmap_region,
                async_h2d_submission=self.async_h2d_submission,
                async_h2d_chunk_bytes=self.async_h2d_chunk_bytes,
                async_h2d_max_descriptors=self.async_h2d_max_descriptors,
                async_h2d_completion_ring_path=(self.async_h2d_completion_ring_path),
                async_h2d_completion_session=self.async_h2d_completion_session,
                force_h2d_dma=self.force_h2d_dma,
                async_d2h_submission=self.async_d2h_submission,
                async_d2h_chunk_bytes=self.async_d2h_chunk_bytes,
            )
        except Exception:
            if mmap_region is not None:
                mmap_region.cleanup()
            raise

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
