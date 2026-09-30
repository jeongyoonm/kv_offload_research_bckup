# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative admission predicate; never resolves futures or mutates outputs."""


def qualifies(queue, ordinary_capacity: int) -> bool:
    if ordinary_capacity != 2 or len(queue) != 2:
        return False
    compute = 0
    zero = False
    for _, output, _ in queue:
        metadata = output.kv_connector_metadata
        # Unknown connectors cannot be classified as empty transfers.
        if metadata is not None and (
            not hasattr(metadata, "load_jobs") or not hasattr(metadata, "store_jobs")
        ):
            return False
        if output.pending_structured_output_tokens or getattr(
            output, "kv_cache_block_copies", None
        ):
            return False
        if not output.total_num_scheduled_tokens and getattr(
            metadata, "store_jobs", {}
        ):
            return False
        compute += int(output.total_num_scheduled_tokens > 0)
        zero |= output.total_num_scheduled_tokens == 0
    return zero and compute < 2
