# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reserve mirrors before spending a fixed ten-hour experiment budget."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from experiments.agentic_1m_20261012.common import (
    DATA,
    DEADLINE,
    LIMIT,
    MEASURE_CUTOFF,
    ROOT,
    SOURCE,
    log,
    save,
)
from experiments.agentic_1m_20261012.prepare import Writer, finish


def probe_data():
    if (DATA / "long_context_probe.json").exists():
        return
    writer = Writer("long_context_probe")
    rng = np.random.default_rng(20261012)
    sessions = []
    for index, length in enumerate((131072, 262144, 524288, LIMIT - 16)):
        row = writer.row(
            rng.integers(128, 31700, length, dtype=np.uint32),
            16,
            session=index,
            index=0,
        )
        sessions.append(dict(session=index, requests=[row]))
    finish(
        "long_context_probe",
        sessions,
        writer,
        adapter="Disjoint capability probe, not a trace performance cell.",
    )


def select_windows():
    """Freeze finite contiguous segments before observing serving performance."""
    for provider in ("codex", "claude"):
        name = "tracelab_" + provider
        data = json.loads((DATA / (name + ".json")).read_text())
        if (DATA / (name + "_windows.json")).exists():
            continue
        import random

        rng = random.Random(20261013 + (provider == "claude"))
        windows = []
        for session in data["sessions"]:
            rows = session["requests"]
            start = rng.randrange(max(1, len(rows) - 64 + 1))
            windows.append(
                dict(
                    session=session["session"],
                    requests=rows[start : start + 64],
                    source_start=start,
                    source_requests=len(rows),
                )
            )
        selected = dict(
            data,
            sessions=windows,
            sampling=(
                "Fixed seeded contiguous windows, up to64 calls/session. Complete "
                "request lengths and preconstructed token identities preserved. "
                "Cold-start finite windows; original full cohorts archived. "
                "Windows identical across all arms/concurrencies/pacing."
            ),
        )
        rows = [r for s in windows for r in s["requests"]]
        selected.update(
            requests=len(rows),
            input_tokens=sum(r["input_tokens"] for r in rows),
            output_tokens=sum(r["output_tokens"] for r in rows),
        )
        selected["input_distribution"] = {
            str(q): float(np.quantile([r["input_tokens"] for r in rows], q / 100))
            for q in (0, 50, 90, 95, 99, 100)
        }
        save(DATA / (name + "_windows.json"), selected)
        log(
            "window_cohort_frozen",
            dataset=name,
            requests=len(rows),
            input_tokens=selected["input_tokens"],
            output_tokens=selected["output_tokens"],
            distribution=selected["input_distribution"],
            reason=(
                "A single source session has6294 Codex calls and another has1273 "
                "Claude calls; complete replay would be dominated by one "
                "serialized agent and cannot support the10-hour sweep. Freeze "
                "finite source windows before timing results; do not alter "
                "requests during intensity adjustment."
            ),
        )


def wait_ready(root, proc=None):
    start = time.time()
    while not (root / "ready.json").exists():
        if (root / "stopped.json").exists() or (proc and proc.poll() is not None):
            raise RuntimeError(f"Startup failed: {root}")
        if time.time() - start > 1800 or time.time() > MEASURE_CUTOFF:
            raise TimeoutError(f"Startup did not become ready: {root}")
        time.sleep(2)


def start_server(root, arm):
    with (ROOT / (root.name + "_supervisor.log")).open("w") as output:
        proc = subprocess.Popen(
            [
                str(SOURCE / ".venv/bin/python"),
                "-u",
                "-m",
                "experiments.agentic_1m_20261012.server",
                "--root",
                str(root),
                "--arm",
                arm,
            ],
            cwd=SOURCE,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
    wait_ready(root, proc)
    return proc


def stop_server(root):
    (root / "stop.request").touch()
    deadline = time.time() + 90
    while not (root / "stopped.json").exists():
        if time.time() > deadline:
            raise TimeoutError(f"Shutdown failed: {root}")
        time.sleep(2)


def run_cell(root, arm, spec):
    for active in (root / "cells").glob("*"):
        if (active / "responses.jsonl").exists():
            while not (active / "result.json").exists():
                if time.time() > MEASURE_CUTOFF or (root / "stopped.json").exists():
                    raise RuntimeError(f"Prior cell did not finish: {active}")
                time.sleep(2)
    existing = root / "cells" / spec["id"]
    if existing.exists():
        while not (existing / "result.json").exists():
            if time.time() > MEASURE_CUTOFF or (root / "stopped.json").exists():
                raise RuntimeError(f"Existing cell did not finish: {existing}")
            time.sleep(2)
        result = json.loads((existing / "result.json").read_text())
        assert result["accepted"], existing
        return result
    path = ROOT / "specs" / (arm + "_" + spec["id"] + ".json")
    save(path, spec)
    timeout = min(10800, max(60, int(MEASURE_CUTOFF - time.time())))
    output_path = ROOT / (arm + "_" + spec["id"] + ".log")
    with output_path.open("w") as output:
        subprocess.run(
            [
                str(SOURCE / ".venv/bin/python"),
                "-u",
                "-m",
                "experiments.agentic_1m_20261012.replay",
                "--root",
                str(root),
                "--arm",
                arm,
                "--spec",
                str(path),
            ],
            cwd=SOURCE,
            stdout=output,
            stderr=output,
            check=True,
            timeout=timeout,
        )
    return json.loads((root / "cells" / spec["id"] / "result.json").read_text())


def preflight(root, arm):
    if (root / "preflight_complete.json").exists():
        return
    probe_data()
    spec = dict(
        id="capability_cold",
        dataset="long_context_probe",
        sessions=4,
        concurrency=1,
        measurement="capability",
    )
    result = run_cell(root, arm, spec)
    assert result["accepted"]
    restore = dict(spec, id="capability_cpu_restore", reset_external=False)
    result = run_cell(root, arm, restore)
    assert result["h2d_jobs"] == 4 and result["accepted"], result
    save(root / "preflight_complete.json", dict(cold=True, restore=True))


def main():
    baseline = Path("/mnt/localnvme/jeongyoon/a1mb8")
    full = Path("/mnt/localnvme/jeongyoon/a1mf3")
    owned = [baseline]
    results, selected = [], []
    save(
        ROOT / "campaign.json",
        dict(
            start=DEADLINE - 36000,
            deadline=DEADLINE,
            max_model_len=LIMIT,
            concurrencies=[8, 16, 32],
            arms=["baseline", "full"],
            planned_repeats=2,
            mirror_budget_priority=True,
            source=str(SOURCE),
        ),
    )
    try:
        wait_ready(baseline)
        preflight(baseline, "baseline")
        for name in (
            "inferact_preparation_complete.json",
            "tracelab_preparation_complete.json",
        ):
            while not (ROOT / name).exists():
                if time.time() > MEASURE_CUTOFF:
                    raise TimeoutError("Dataset preparation exceeded campaign budget")
                time.sleep(2)
        select_windows()
        specs = [
            dict(
                id=f"{dataset}_n{n}_r0",
                dataset=dataset,
                concurrency=n,
                min_interval_seconds=0,
                repeat=0,
            )
            for n in (32, 16, 8)
            for dataset in (
                "inferact",
                "tracelab_codex_windows",
                "tracelab_claude_windows",
            )
        ]
        if (ROOT / "primary_plan_mixed.json").exists():
            specs = json.loads((ROOT / "primary_plan_mixed.json").read_text())
        save(ROOT / "primary_plan.json", specs)
        reserved = 0
        prior_durations = {}
        for spec in specs:
            expected = (
                prior_durations.get(spec["dataset"], 1200)
                * (32 / spec["concurrency"]) ** 0.4
            )
            needed = reserved + expected * 2.2 + 900
            if time.time() + needed > MEASURE_CUTOFF:
                log(
                    "cell_deferred_before_first_arm",
                    spec=spec,
                    reserved_seconds=needed,
                    remaining_seconds=MEASURE_CUTOFF - time.time(),
                )
                continue
            result = run_cell(baseline, "baseline", spec)
            prior_durations[spec["dataset"]] = result["seconds"]
            results.append(dict(root=str(baseline), **result))
            selected.append(spec)
            reserved += result["seconds"] * 1.1
            save(ROOT / "selected_primary.json", selected)
            save(
                ROOT / "progress.json",
                dict(results=results, mirror_seconds_reserved=reserved),
            )
        stop_server(baseline)
        owned.append(full)
        start_server(full, "full")
        preflight(full, "full")
        for spec in selected:
            result = run_cell(full, "full", spec)
            results.append(dict(root=str(full), **result))
            save(ROOT / "progress.json", dict(results=results))
        # Reverse-order repeats are optional only after every selected mirror exists.
        repeats = []
        reserved = 0
        for spec in selected:
            reference = next(
                r
                for r in results
                if r["arm"] == "baseline" and r["spec"]["id"] == spec["id"]
            )
            duration = reference["seconds"]
            if time.time() + reserved + duration * 2.3 + 1200 > MEASURE_CUTOFF:
                continue
            second = dict(spec, id=spec["id"].replace("_r0", "_r1"), repeat=1)
            result = run_cell(full, "full", second)
            results.append(dict(root=str(full), **result))
            repeats.append(second)
            reserved += duration * 1.2
            save(ROOT / "selected_repeats.json", repeats)
        stop_server(full)
        if repeats:
            second_baseline = Path("/mnt/localnvme/jeongyoon/a1mb9")
            owned.append(second_baseline)
            start_server(second_baseline, "baseline")
            preflight(second_baseline, "baseline")
            for spec in repeats:
                result = run_cell(second_baseline, "baseline", spec)
                results.append(dict(root=str(second_baseline), **result))
                save(ROOT / "progress.json", dict(results=results))
            stop_server(second_baseline)
        save(
            ROOT / "completed.json",
            dict(time=time.time(), results=results, owned=[str(r) for r in owned]),
        )
        log("campaign_complete", cells=len(results), matched_pairs=len(results) // 2)
    except BaseException as error:
        log("campaign_failed", error=repr(error))
        save(
            ROOT / "failure.json",
            dict(time=time.time(), error=repr(error), results=results),
        )
        raise
    finally:
        for root in owned:
            if (root / "owner.json").exists() and not (root / "stopped.json").exists():
                stop_server(root)


if __name__ == "__main__":
    main()
