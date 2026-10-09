# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse the existing gap audit with explicit GPU-fence and store API markers."""

from experiments.h2d8_deadline_bias_20261009 import report


def main():
    report.ARMS = ("control", "gpu")
    report.PAIRS.update(
        store_reuse_fence_enqueue_begin="store_reuse_fence_enqueue_end",
        store_descriptors_begin="store_descriptors_end",
        store_stream_dependencies_begin="store_stream_dependencies_end",
        store_copy_enqueue_begin="store_copy_enqueue_end",
    )
    report.main()


if __name__ == "__main__":
    main()
