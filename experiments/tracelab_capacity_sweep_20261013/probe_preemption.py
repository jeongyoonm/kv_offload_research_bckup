# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol probe: early load readiness, delayed retirement, preemption."""

import json

from experiments.tracelab_capacity_sweep_20261013.analyze import ROOT
from tests.v1.kv_connector.unit.offloading_connector.utils import RequestRunner
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus


def probe(early_ready, retire_before_preemption):
    runner = RequestRunner(4, 100, blocks_per_chunk=3, worker_count=8)
    connector = runner.connector_scheduler
    connector._maximal_prefix_lookup = lambda keys, context, *_: len(keys)
    runner.new_request(token_ids=[0] * 24)
    output = runner.scheduler.schedule()
    jobs = output.kv_connector_metadata.load_jobs
    assert len(jobs) == 1
    job_id = next(iter(jobs))
    req_id = connector._jobs[job_id].req_id
    assert (
        runner.scheduler.requests[req_id].status == RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    if early_ready:
        assert runner.scheduler.apply_early_kv_load_completion(job_id) == req_id
        if retire_before_preemption:
            connector.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=OffloadingWorkerMetadata(
                        completed_jobs={job_id: 8}
                    )
                )
            )
        runner.scheduler.schedule()
        assert runner.scheduler.requests[req_id].status == RequestStatus.RUNNING
    pending_load = job_id in connector._jobs
    runner.scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks = 0
    assertion = None
    try:
        output = runner.scheduler.schedule()
        preempted = sorted(output.preempted_req_ids or ())
    except AssertionError as error:
        import traceback

        assertion = traceback.format_exception(type(error), error, error.__traceback__)
        preempted = None
    return dict(
        early_ready=early_ready,
        retired=retire_before_preemption,
        pending_load=pending_load,
        assertion=assertion,
        preempted=preempted,
    )


def main():
    results = [probe(False, False), probe(True, False), probe(True, True)]
    assert results[0]["assertion"] is None
    assert any(
        "assert self._jobs[any_jid].is_store" in line
        for line in results[1]["assertion"]
    )
    assert results[2]["assertion"] is None and results[2]["preempted"]
    destination = ROOT / "cpu_preemption_probe.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
