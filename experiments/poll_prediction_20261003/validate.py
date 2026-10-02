# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate preserved raw AIPerf usage and cache/transfer accounting."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from run import ROOT, run

usage = run.usage


def validate_root(root):
    rows = {}
    for path in root.glob("**/result.json"):
        result = json.loads(path.read_text())
        raw = usage(path.parent / "replay/replay_raw.jsonl")
        observed = {str(k): v for k, v in Counter(r["cached"] for r in raw).items()}
        before = json.loads((path.parent / "metrics_before.json").read_text())
        after = json.loads((path.parent / "metrics_after.json").read_text())
        hit_tokens = sum(r["cached"] for r in raw)
        external = (
            after["vllm:external_prefix_cache_hits"]
            - before["vllm:external_prefix_cache_hits"]
        )
        errors = []
        for key in (
            "measurement_jit_lines",
            "runtime_error_lines",
            "unexpected_gpu_pids",
        ):
            if result.get(key):
                errors.append(key)
        if observed != result["expected_cached_distribution"]:
            errors.append("unexpected cached token distribution")
        if result["gpu_hits"] or result["preemptions"]:
            errors.append("GPU residency or preemption affected replay")
        if external != hit_tokens or result["h2d_bytes"] != hit_tokens * 381600:
            errors.append("external cache tokens and transferred bytes disagree")
        if len(raw) != sum(result["expected_cached_distribution"].values()):
            errors.append("raw response count mismatch")
        outputs = {}
        completion_counts = Counter()
        count = sum(result["expected_cached_distribution"].values())
        manifest_path = result.get("workload_manifest")
        if not manifest_path:
            manifest_path = (
                ROOT
                / "workloads"
                / f"{path.parent.parent.name}_n{count}"
                / "manifest.json"
            )
        manifest = json.loads(Path(manifest_path).read_text())
        hit_positions = set(manifest["hit_positions"])
        for line in (path.parent / "replay/replay_raw.jsonl").read_text().splitlines():
            item = json.loads(line)
            text = []
            conversation = item["metadata"]["conversation_id"]
            index = int(conversation.split("_")[1])
            for response in item.get("responses", []):
                for packet in response.get("packets", []):
                    try:
                        value = json.loads(packet["value"])
                    except (ValueError, KeyError, TypeError):
                        continue
                    text.extend(
                        choice.get("text", "") for choice in value.get("choices", [])
                    )
                    if value.get("usage"):
                        details = value["usage"]
                        completion_counts[details["completion_tokens"]] += 1
                        expected_cached = (
                            manifest["prefix_tokens"] if index in hit_positions else 0
                        )
                        if details["prompt_tokens"] != manifest["prompt_tokens"]:
                            errors.append(
                                f"unexpected prompt length for {conversation}"
                            )
                        if (
                            details["prompt_tokens_details"]["cached_tokens"]
                            != expected_cached
                        ):
                            errors.append(
                                f"unexpected hit/miss position: {conversation}"
                            )
            outputs[conversation] = "".join(text)
        if len(outputs) != count:
            errors.append("replay conversation IDs were not unique")
        if dict(completion_counts) != {1: len(raw)}:
            errors.append("expected exactly one completion token per request")
        digest = hashlib.sha256(
            json.dumps(outputs, sort_keys=True).encode()
        ).hexdigest()
        (path.parent / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
        result.update(
            cached_distribution=observed,
            accepted=not errors,
            validation_errors=errors,
            external_hit_tokens=external,
            outputs_sha256=digest,
        )
        path.write_text(json.dumps(result, indent=2) + "\n")
        (path.parent / "usage.json").write_text(json.dumps(raw, indent=2) + "\n")
        rows[str(path.parent)] = result
        print(path.parent.name, result["accepted"], observed, errors)
    (root / "validated_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "validation_complete.json").write_text(
        json.dumps(
            {
                "cells": len(rows),
                "accepted": sum(r["accepted"] for r in rows.values()),
                "failed_paths": [name for name, r in rows.items() if not r["accepted"]],
            },
            indent=2,
        )
        + "\n"
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    validate_root(args.root)


if __name__ == "__main__":
    main()
