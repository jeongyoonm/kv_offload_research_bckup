# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the existing five-second GPU and ten-second cache monitor unchanged."""

from experiments.tracelab_arrival_intensity_20261013 import monitor
from experiments.tracelab_capacity_sweep_20261013.analyze import ROOT

if __name__ == "__main__":
    monitor.ROOT = ROOT
    monitor.EXTRA_METRIC_PREFIXES = ("vllm:iteration_tokens_total_",)
    monitor.main()
