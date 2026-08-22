#!/usr/bin/env python3
"""Poll and validate a public TraceLab Hugging Face release."""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request


SERVER_ROOT = "https://datasets-server.huggingface.co"


def fetch(endpoint: str, parameters: dict[str, object]) -> tuple[int, dict]:
    query = urllib.parse.urlencode(parameters)
    request = urllib.request.Request(
        f"{SERVER_ROOT}/{endpoint}?{query}",
        headers={"User-Agent": "TraceLab-HF-release-check/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        response_text = error.read().decode()
        try:
            return error.code, json.loads(response_text)
        except json.JSONDecodeError:
            return error.code, {"error": response_text}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="UW-SyFI/TraceLab")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--load-dataset", action="store_true")
    parser.add_argument("--expected-default", type=int, default=665_453)
    parser.add_argument("--expected-tool-calls", type=int, default=743_819)
    parser.add_argument("--expected-timing-events", type=int, default=2_688_829)
    arguments = parser.parse_args()
    expected_counts = {
        "default": arguments.expected_default,
        "tool_calls": arguments.expected_tool_calls,
        "timing_events": arguments.expected_timing_events,
    }

    for attempt_number in range(1, arguments.max_attempts + 1):
        parquet_status, parquet_payload = fetch("parquet", {"dataset": arguments.repo_id})
        print(json.dumps({
            "attempt": attempt_number,
            "status": parquet_status,
            "files": len(parquet_payload.get("parquet_files", [])),
            "pending": parquet_payload.get("pending", []),
            "failed": parquet_payload.get("failed", []),
        }), flush=True)
        if parquet_payload.get("failed"):
            raise RuntimeError(f"HF conversion failed: {parquet_payload}")
        if parquet_status == 200 and len(parquet_payload.get("parquet_files", [])) == 3 and not parquet_payload.get("pending"):
            break
        time.sleep(arguments.poll_interval_seconds)
    else:
        raise TimeoutError("HF Parquet conversion did not finish")

    valid_status, valid_payload = fetch("is-valid", {"dataset": arguments.repo_id})
    if valid_status != 200:
        raise RuntimeError(f"is-valid failed: {valid_status} {valid_payload}")
    for capability in ("preview", "viewer", "search", "filter"):
        if valid_payload.get(capability) is not True:
            raise RuntimeError(f"capability {capability} is not ready: {valid_payload}")

    size_status, size_payload = fetch("size", {"dataset": arguments.repo_id})
    if size_status != 200 or size_payload.get("failed"):
        raise RuntimeError(f"size failed: {size_status} {size_payload}")
    actual_counts = {
        config["config"]: config["num_rows"]
        for config in size_payload["size"]["configs"]
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"Hub counts {actual_counts} != {expected_counts}")

    for config_name in expected_counts:
        row_status, row_payload = fetch("first-rows", {
            "dataset": arguments.repo_id,
            "config": config_name,
            "split": "train",
        })
        if row_status != 200 or not row_payload.get("rows"):
            raise RuntimeError(f"first-rows failed for {config_name}: {row_payload}")

    print(json.dumps({"capabilities": valid_payload, "rows": actual_counts}), flush=True)
    if arguments.load_dataset:
        from datasets import load_dataset

        for config_name, expected_count in expected_counts.items():
            load_arguments = (
                (arguments.repo_id,)
                if config_name == "default"
                else (arguments.repo_id, config_name)
            )
            dataset = load_dataset(*load_arguments, revision=arguments.revision, split="train")
            print(f"{config_name}: {dataset.num_rows} rows", flush=True)
            if dataset.num_rows != expected_count:
                raise RuntimeError(f"{config_name}: {dataset.num_rows} != {expected_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
