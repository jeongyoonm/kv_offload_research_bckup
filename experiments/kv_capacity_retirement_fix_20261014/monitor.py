# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use identical GPU and metrics sampling for every capacity-fix variant."""

from experiments.kv_capacity_retirement_fix_20261014.campaign import ROOT
from experiments.tracelab_arrival_intensity_20261013 import monitor

if __name__ == "__main__":
    monitor.ROOT = ROOT
    monitor.EXTRA_METRIC_PREFIXES = ("vllm:iteration_tokens_total_",)
    monitor.main()
