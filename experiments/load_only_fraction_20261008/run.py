# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse the matched deadline experiment with proportional load-only sealing."""

import argparse
import importlib.util
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_load_only_fraction_20261008")
spec = importlib.util.spec_from_file_location(
    "deadline_comparison", SOURCE / "experiments/offset_deadline_ab_20261008/run.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.ROOT = driver.collector.ROOT = ROOT
driver.collector.BASE = "http://127.0.0.1:18412"
original_options = driver.original_options


def options(active=False):
    result = original_options(active)
    if active:
        result["late_seal_load_only_ready_percent"] = 50
    return result


def server_command():
    command, config = driver.server_command()
    command[command.index("--port") + 1] = "18412"
    return command, config


driver.original_options = options
driver.collector.server_command = server_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-profiles", action="store_true")
    args = parser.parse_args()
    if args.print_profiles:
        profiles = {}
        for policy in driver.POLICIES:
            driver.CURRENT_POLICY = policy
            profiles[policy] = driver.options(True)
        print(json.dumps(profiles, indent=2))
        return
    driver.main()


if __name__ == "__main__":
    main()
