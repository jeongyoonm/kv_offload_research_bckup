# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trusted local fixed-work replay for the timing-interference experiment.

Only one-token, text-only, no-preemption CPU-offload traces are accepted.
The real executor/connector execute the frozen dispatches and dependency edges.
The scheduler does not choose new batches during replay.
"""

import dataclasses
import enum
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from vllm.v1.kv_offload import timing_diagnostics as diagnostic


def content_digest(value):
    """Compare frozen values independently of pickle memoization and aliases."""

    def canonical(obj):
        kind = f"{type(obj).__module__}.{type(obj).__qualname__}"
        if isinstance(obj, enum.Enum):
            return [kind, obj.name]
        if obj is None or isinstance(obj, (str, int, bool)):
            return [kind, obj]
        if isinstance(obj, float):
            return [kind, obj.hex()]
        if isinstance(obj, bytes):
            return [kind, obj.hex()]
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise ValueError("object arrays are unsupported in controlled replay")
            return [kind, str(obj.dtype), list(obj.shape), obj.tobytes().hex()]
        if isinstance(obj, (list, tuple)):
            return [kind, [canonical(item) for item in obj]]
        if isinstance(obj, (set, frozenset)):
            return [kind, sorted((canonical(item) for item in obj), key=repr)]
        if isinstance(obj, dict):
            return [
                kind,
                sorted(
                    ((canonical(k), canonical(v)) for k, v in obj.items()), key=repr
                ),
            ]
        if dataclasses.is_dataclass(obj):
            return [
                kind,
                [
                    (f.name, canonical(getattr(obj, f.name)))
                    for f in dataclasses.fields(obj)
                ],
            ]
        if isinstance(obj, msgspec.Struct):
            return [
                kind,
                [
                    (name, canonical(getattr(obj, name)))
                    for name in obj.__struct_fields__
                ],
            ]
        if hasattr(obj, "__dict__"):
            return [kind, canonical(vars(obj))]
        raise ValueError(f"unsupported controlled manifest value: {kind}")

    encoded = json.dumps(canonical(value), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def connector(core):
    return core.scheduler.connector.connector_scheduler


def cpu_snapshot(core):
    manager = connector(core).manager
    blocks = getattr(manager._policy, "blocks", None)
    if blocks is None:
        raise ValueError("controlled replay requires the CPU LRU cache")
    if any(block.ref_cnt != 0 for block in blocks.values()):
        raise ValueError("controlled replay requires drained CPU cache references")
    return {block.block_id: key for key, block in blocks.items()}


def manifest_path():
    # No caller-supplied pickle paths: this is a trusted, startup-selected artifact.
    return Path(os.environ["VLLM_TIMING_REPLAY_MANIFEST"])


class Capture:
    def __init__(self, core):
        self.initial_cpu = cpu_snapshot(core)
        self.gpu_blocks = core.scheduler.kv_cache_manager.block_pool.num_gpu_blocks
        self.actions = []
        self.loads = {}
        self.requests = set()

    def dispatch(self, output, shapes):
        if len(self.actions) >= 8192:
            raise RuntimeError("controlled capture capacity exceeded")
        if (
            output.scheduled_encoder_inputs
            or output.scheduled_spec_decode_tokens
            or output.pending_structured_output_tokens
            or output.preempted_req_ids
        ):
            raise ValueError("unsupported controlled replay shape")
        for request in output.scheduled_new_reqs:
            if (
                request.mm_features
                or request.prompt_embeds is not None
                or request.sampling_params.max_tokens != 1
            ):
                raise ValueError("controlled replay requires text and one output token")
            self.requests.add(request.req_id)
        metadata = output.kv_connector_metadata
        for job, transfer in metadata.load_jobs.items():
            self.loads[transfer.req_id] = job
        dependencies = [
            self.loads[r] for r in output.num_scheduled_tokens if r in self.loads
        ]
        if set(dependencies).intersection(metadata.load_jobs):
            raise ValueError("same-dispatch load dependency is unsupported")
        self.actions.append(
            dict(
                kind="dispatch",
                payload=pickle.dumps(output),
                dependencies=dependencies,
                shapes=shapes,
            )
        )

    def wait(self, step):
        self.actions.append(dict(kind="wait", step=step))

    def finish(self, core):
        final = cpu_snapshot(core)
        # Reuse/eviction of an initial slot requires a richer lifetime mapping.
        if any(final.get(slot) != key for slot, key in self.initial_cpu.items()):
            raise ValueError("CPU slot eviction makes this capture unsupported")
        if len(self.requests) != 300:
            raise ValueError("controlled capture must contain exactly 300 requests")
        data = dict(
            version=1,
            gpu_blocks=self.gpu_blocks,
            initial_cpu=self.initial_cpu,
            final_cpu=final,
            actions=self.actions,
            requests=self.requests,
        )
        payload = pickle.dumps(data)
        path = manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
        return dict(
            sha256=hashlib.sha256(payload).hexdigest(),
            actions=len(self.actions),
            requests=len(self.requests),
        )


def remap_cpu(data, core):
    current = cpu_snapshot(core)
    by_key = {key: slot for slot, key in current.items()}
    missing = set(data["initial_cpu"].values()) - set(by_key)
    if missing:
        raise ValueError(
            f"controlled replay is missing {len(missing)} primed CPU blocks"
        )
    mapping = {slot: by_key[key] for slot, key in data["initial_cpu"].items()}
    extra = {
        slot: key for slot, key in data["final_cpu"].items() if slot not in mapping
    }
    manager = connector(core).manager
    if len(extra) > manager._get_num_free_blocks():
        raise ValueError(
            "controlled replay requires scratch CPU capacity without eviction"
        )
    allocated = manager._allocate_blocks(list(extra.values()))
    mapping.update({slot: block.block_id for slot, block in zip(extra, allocated)})
    return mapping


def replay(core):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
    )
    from vllm.v1.core.sched.output import SchedulerOutput

    payload = manifest_path().read_bytes()
    data = pickle.loads(payload)
    if data["version"] != 1:
        raise ValueError("unsupported controlled manifest")
    if data["gpu_blocks"] != core.scheduler.kv_cache_manager.block_pool.num_gpu_blocks:
        raise ValueError("controlled replay GPU allocation arena changed")
    mapping = remap_cpu(data, core)
    options = core.early_load_readiness.experiment_options
    enabled = options["timing_diagnostic_mode"] != "baseline"
    scheduler_connector = connector(core)
    job_ids = sorted(
        {
            job
            for action in data["actions"]
            if action["kind"] == "dispatch"
            for jobs in (
                pickle.loads(action["payload"]).kv_connector_metadata.load_jobs,
                pickle.loads(action["payload"]).kv_connector_metadata.store_jobs,
            )
            for job in jobs
        }
    )
    jobs = {job: scheduler_connector._generate_job_id() for job in job_ids}
    # GPU cache is reset by the collector before entering this method. Its block
    # numbers are an isomorphic allocation arena, with captured reuse dependencies.
    outputs = []
    max_step = 0
    for action in data["actions"]:
        if action["kind"] != "dispatch":
            continue
        output = pickle.loads(action["payload"])
        metadata = output.kv_connector_metadata
        for old, transfer in metadata.load_jobs.items():
            transfer.src_spec.block_ids[:] = [
                mapping[int(i)] for i in transfer.src_spec.block_ids
            ]
        for old, transfer in metadata.store_jobs.items():
            transfer.dst_spec.block_ids[:] = [
                mapping[int(i)] for i in transfer.dst_spec.block_ids
            ]
        metadata.load_jobs = {jobs[j]: t for j, t in metadata.load_jobs.items()}
        metadata.store_jobs = {jobs[j]: t for j, t in metadata.store_jobs.items()}
        metadata.jobs_to_flush = {jobs[j] for j in metadata.jobs_to_flush or ()}
        metadata.experiment_options = dict(options)
        output.deferral_timing = enabled and output.total_num_scheduled_tokens > 0
        output.deferral_rpc_timing = enabled
        max_step = max(max_step, output.readiness_timeline_step)
        normalized = pickle.loads(pickle.dumps(output))
        inverse_cpu = {new: old for old, new in mapping.items()}
        inverse_jobs = {new: old for old, new in jobs.items()}
        norm_meta = normalized.kv_connector_metadata
        for transfer in norm_meta.load_jobs.values():
            transfer.src_spec.block_ids[:] = [
                inverse_cpu[int(i)] for i in transfer.src_spec.block_ids
            ]
        for transfer in norm_meta.store_jobs.values():
            transfer.dst_spec.block_ids[:] = [
                inverse_cpu[int(i)] for i in transfer.dst_spec.block_ids
            ]
        norm_meta.load_jobs = {
            inverse_jobs[j]: t for j, t in norm_meta.load_jobs.items()
        }
        norm_meta.store_jobs = {
            inverse_jobs[j]: t for j, t in norm_meta.store_jobs.items()
        }
        original = pickle.loads(action["payload"])
        norm_meta.jobs_to_flush = original.kv_connector_metadata.jobs_to_flush
        norm_meta.experiment_options = original.kv_connector_metadata.experiment_options
        normalized.deferral_timing = original.deferral_timing
        normalized.deferral_rpc_timing = original.deferral_rpc_timing
        if content_digest(normalized) != content_digest(original):
            raise ValueError("normalized controlled dispatch differs from capture")
        outputs.append(output)
    completed_loads: set[int] = set()
    futures: dict[int, Any] = {}
    dispatch_index = 0
    wall_start = time.time_ns()
    start = time.monotonic_ns()
    samples = []
    selected = []
    for action in data["actions"]:
        if action["kind"] == "wait":
            future = futures.pop(action["step"])
            diagnostic.emit(
                "engine_future_wait_begin", step=action["step"], future_id=id(future)
            )
            result = future.result()
            diagnostic.emit(
                "engine_future_wait_end", step=action["step"], future_id=id(future)
            )
            if result is None:
                raise RuntimeError("controlled replay returned no model output")
            if result.sampled_token_ids is not None:
                samples.extend(result.sampled_token_ids)
            continue
        needed = {jobs[j] for j in action["dependencies"]}
        deadline = time.monotonic() + 30
        while not needed <= completed_loads:
            for completion in core.early_load_readiness.drain():
                completed_loads.add(completion.job_id)
                diagnostic.emit(
                    "controlled_kv_receipt",
                    job_id=completion.job_id,
                    publication_ns=completion.completion_ns,
                )
            if time.monotonic() >= deadline:
                raise RuntimeError("controlled KV dependency did not complete")
            if not needed <= completed_loads:
                time.sleep(0.0005)
        output = outputs[dispatch_index]
        dispatch_index += 1
        step = output.readiness_timeline_step
        diagnostic.emit(
            "diagnostic_dispatch",
            step=step,
            selected=dict(output.num_scheduled_tokens),
            request_shapes=action["shapes"],
        )
        selected.append(len(output.num_scheduled_tokens))
        future = core.model_executor.execute_model(output, non_block=True)
        if output.total_num_scheduled_tokens:
            future = core.model_executor.sample_tokens(None, non_block=True)
        if step in futures:
            raise RuntimeError("duplicate controlled future")
        futures[step] = future
    if futures:
        raise RuntimeError("controlled capture contains unconsumed futures")
    end = time.monotonic_ns()
    wall_end = time.time_ns()
    # Flush stores and clear the worker request cache after the measured region.
    cleanup = SchedulerOutput.make_empty()
    cleanup.readiness_timeline_step = max_step + 1
    cleanup.finished_req_ids = set(data["requests"])
    cleanup.kv_connector_metadata = OffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={},
        jobs_to_flush=set(jobs.values()),
        experiment_options=dict(options),
    )
    core.model_executor.execute_model(cleanup, non_block=False)
    core.model_executor.collective_rpc("synchronize_device")
    core.early_load_readiness.drain()
    if len(samples) != 300 or sum(selected) != 300:
        raise RuntimeError("controlled output/token accounting mismatch")
    return dict(
        start_ns=start,
        wall_start_ns=wall_start,
        wall_end_ns=wall_end,
        end_ns=end,
        requests=len(samples),
        selected=selected,
        sampled_token_ids=samples,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        normalized_manifest_equal=True,
        throughput=300e9 / (end - start),
        latency_definition="fixed executor replay; no client arrival latency",
    )
