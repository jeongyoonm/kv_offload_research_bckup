# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the structural proxy's actual continuity and discontinuity semantics."""

import numpy as np

from experiments.agentic_1m_20261012.prepare import structural_prompt


def test_growth_reuses_only_observed_prefix_without_inventing_prior_history():
    previous = np.arange(128, 228, dtype=np.uint32)
    result, keep = structural_prompt(previous, 150, 120, np.random.default_rng(1))
    assert keep == 100 and len(result) == 150
    np.testing.assert_array_equal(result[:100], previous)


def test_changed_suffix_retains_only_the_declared_prefix():
    previous = np.arange(128, 228, dtype=np.uint32)
    result, keep = structural_prompt(previous, 110, 64, np.random.default_rng(1))
    assert keep == 64
    np.testing.assert_array_equal(result[:64], previous[:64])
    assert not np.array_equal(result[64:100], previous[64:])


def test_context_shrink_resets_identity_instead_of_claiming_a_cache_hit():
    previous = np.arange(128, 228, dtype=np.uint32)
    result, keep = structural_prompt(previous, 80, 75, np.random.default_rng(1))
    assert keep == 0 and len(result) == 80
    assert not np.array_equal(result, previous[:80])
