# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen setup comparison after adding one mixed-batch training allocation."""

from experiments.setup_prediction_20261011 import snapshot_serving

if __name__ == "__main__":
    snapshot_serving.NEW_DIRECTORY = "mixed_models/models"
    snapshot_serving.NEW_ARM = "mixed_mean"
    snapshot_serving.main()
