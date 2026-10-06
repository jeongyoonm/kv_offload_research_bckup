# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a captured indexer row without model execution or CPU KV transfers."""

import argparse
import json
from pathlib import Path

import torch
from cache_init_dma_audit import _stable_indices

from vllm import _custom_ops as ops
from vllm.platforms import current_platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    captured = torch.load(args.captured, weights_only=True)
    device = torch.device(current_platform.device_type, 0)
    logits = captured["logits"].reshape(1, -1).to(device)
    starts = torch.tensor([captured["start"]], dtype=torch.int32, device=device)
    ends = torch.tensor([captured["end"]], dtype=torch.int32, device=device)
    topk = captured["indices"].numel()
    indices = torch.empty((1, topk), dtype=torch.int32, device=device)
    reference = None
    changed_order = []
    changed_membership = []
    oracle = _stable_indices(logits, starts, ends, topk).cpu()
    for _ in range(args.repeats):
        ops.top_k_per_row_prefill(
            logits, starts, ends, indices, 1, logits.stride(0), logits.stride(1), topk
        )
        observed = indices.cpu()
        if reference is None:
            reference = observed
        changed_order.append(int((observed != reference).sum().item()))
        changed_membership.append(
            int((observed.sort().values != reference.sort().values).sum().item())
        )
        assert torch.equal(observed.sort().values, oracle.sort().values)
    unchanged = torch.equal(logits.cpu().reshape(-1), captured["logits"])
    assert unchanged
    stable_changed = []
    for _ in range(10):
        observed = _stable_indices(logits, starts, ends, topk).cpu()
        stable_changed.append(int((observed != oracle).sum().item()))
    assert max(stable_changed) == 0
    result = {
        "logits_unchanged": unchanged,
        "repeats": args.repeats,
        "ordered_changed_elements": changed_order,
        "membership_changed_elements": changed_membership,
        "stable_oracle_changed_elements": stable_changed,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
