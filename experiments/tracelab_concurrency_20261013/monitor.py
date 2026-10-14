# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse the matched TraceLab cache/GPU monitor with a separate artifact root."""

from experiments.tracelab_arrival_intensity_20261013 import monitor
from experiments.tracelab_concurrency_20261013.campaign import ROOT

if __name__ == "__main__":
    monitor.ROOT = ROOT
    monitor.main()
