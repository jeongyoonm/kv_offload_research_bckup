"""Sequential serving campaign; preserve failed evidence and stop on failed gates."""
import json
import subprocess
import time
from pathlib import Path

SOURCE = Path("/mnt/localnvme/jeongyoon/vllm_diverse_cache_20261011")
CAMPAIGN = Path("/mnt/localnvme/jeongyoon/load_only_overlap_20261011")
DEADLINE = 1791738420  # 17:07 UTC: watchdog cleanup starts.


def accepted(root, expected):
    p = root / "summary.json"
    d = json.loads(p.read_text())
    assert d["complete"] and d["stopped"], p
    assert len(d["results"]) == expected, p
    assert all(r["accepted"] for r in d["results"]), p
    return d


def execute(rootname, kind, suite, expected, options):
    assert time.time() < DEADLINE - 300, "Campaign deadline"
    root = Path("/mnt/localnvme/jeongyoon") / rootname
    command = [str(SOURCE / ".venv/bin/python"), "-m",
               "experiments.load_only_overlap_20261011.evaluate",
               "--output", str(root), "--server-kind", kind, "--suite", suite,
               "--idle-target", "4", "--seconds", str(int(DEADLINE - time.time() - 60)),
               *options]
    print(json.dumps(dict(event="launch", root=str(root), command=command)), flush=True)
    with (CAMPAIGN / f"{rootname}.log").open("x") as log:
        subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    d = accepted(root, expected)
    print(json.dumps(dict(event="completed", root=str(root), cells=expected,
                          seconds=d["seconds"])), flush=True)


while True:
    root = Path("/mnt/localnvme/jeongyoon/dc11oi")
    if (root / "summary.json").exists():
        accepted(root, 32)
        if (CAMPAIGN / "profiled_classes.json").exists():
            break
    assert time.time() < DEADLINE - 600
    time.sleep(20)

classes = ["--classes", str(CAMPAIGN / "profiled_classes.json")]
paired = ["--requests", "300", "--repeats", "2", "--concurrencies", "32"]
execute("dc11bx", "baseline", "heterogeneous", 4,
        [*paired, *classes, "--seed-start", "12511"])
execute("dc11ox", "optimized", "heterogeneous", 8,
        [*paired, *classes, "--seed-start", "12511"])
execute("dc11be", "baseline", "evolving", 4,
        [*paired, "--seed-start", "12601"])
execute("dc11oe", "optimized", "evolving", 8,
        [*paired, "--seed-start", "12601"])
classes_data = json.loads((CAMPAIGN / "profiled_classes.json").read_text())
selected = [f"c{c}_q{q}" for c, q in classes_data.values()]
assert len(set(selected)) == 3
sweep = ["--requests", "64", "--repeats", "1", "--concurrencies",
         "1", "2", "4", "8", "16", "64", "--seed-start", "12501", "--ids", *selected]
execute("dc11bc", "baseline", "grid", 18, sweep)
execute("dc11oc", "optimized", "grid", 36, sweep)
remaining = [f"c{c}_q{q}" for c in (4096,16384,32768,65536)
             for q in (128,512,2048,8192) if f"c{c}_q{q}" not in selected]
sampled = ["--requests", "64", "--repeats", "1", "--concurrencies",
           "1", "16", "64", "--seed-start", "12501", "--ids", *remaining]
execute("dc11bs", "baseline", "grid", 39, sampled)
execute("dc11os", "optimized", "grid", 78, sampled)
print(json.dumps(dict(event="campaign_queue_complete")), flush=True)
