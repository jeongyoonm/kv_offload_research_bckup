# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample unchanged serving metrics for both dispatch-horizon variants."""

from experiments.cached_dispatch_horizon_20261014.campaign import ROOT
from experiments.tracelab_arrival_intensity_20261013 import monitor

if __name__ == "__main__":
    monitor.ROOT = ROOT
    monitor.EXTRA_METRIC_PREFIXES = ("vllm:iteration_tokens_total_",)
    monitor.main()
