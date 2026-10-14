# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze workload families before observing execution labels."""

import hashlib
import itertools

import numpy as np


def family(block, shapes, loads=0, timing="new", hit=100):
    geometry = sorted([list(s) for s in shapes])
    identity = repr((block, geometry, loads, timing, hit))
    return dict(
        id=hashlib.sha256(identity.encode()).hexdigest()[:12],
        block=block,
        shapes=geometry,
        background_loads=loads,
        background_timing=timing,
        hit_percent=hit,
    )


def feasible(shapes):
    return (
        len(shapes) <= 32
        and sum(s for s, c in shapes) <= 61440
        and all(s > 0 and c >= 0 and s + c < 65536 for s, c in shapes)
    )


def build():
    candidates = [
        family("geometry", [(s, c)] * b)
        for c, s, b in itertools.product(
            [0, 4096, 8192, 16384, 32768, 49152, 61440],
            [128, 512, 2048, 8192],
            [1, 2, 4, 8, 16],
        )
        if feasible([(s, c)] * b)
    ]

    def vector(f):
        sh = f["shapes"]
        s, c = zip(*sh)
        return [
            sum(s),
            len(sh),
            sum(q * (p + q / 2) for q, p in sh),
            sum(c),
            min(s),
            max(s),
            min(c),
            max(c),
            sum((q + 127) // 128 * 128 for q in s),
        ]

    x = np.log1p(np.array([vector(f) for f in candidates]))
    x = (x - x.min(0)) / np.maximum(x.max(0) - x.min(0), 1e-9)
    chosen = []
    # Force marginal coverage before adding farthest-point combinations.
    for axis, values in [
        ("context", [0, 4096, 8192, 16384, 32768, 49152, 61440]),
        ("suffix", [128, 512, 2048, 8192]),
        ("batch", [1, 2, 4, 8, 16]),
    ]:
        for v in values:
            eligible = [
                i
                for i, f in enumerate(candidates)
                if (
                    f["shapes"][0][1]
                    if axis == "context"
                    else f["shapes"][0][0]
                    if axis == "suffix"
                    else len(f["shapes"])
                )
                == v
                and i not in chosen
            ]
            if eligible:
                chosen.append(eligible[0])
    while len(chosen) < 32:
        distance = ((x[:, None, :] - x[chosen][None, :, :]) ** 2).sum(2).min(1)
        distance[chosen] = -1
        chosen.append(int(distance.argmax()))
    geometry = [candidates[i] for i in chosen]
    composition = []
    for b, ratio, kind in itertools.product(
        [4, 8], [0.5, 0.75], ["context", "suffix", "joint", "cross"]
    ):
        k = int(b * ratio)
        if kind == "context":
            a, z = (512, 4096), (512, 49152)
        elif kind == "suffix":
            a, z = (512, 16384), (2048, 16384)
        elif kind == "joint":
            a, z = (512, 4096), (2048, 49152)
        else:
            a, z = (2048, 4096), (512, 49152)
        composition.append(family("composition", [a] * k + [z] * (b - k)))
    transfer = [
        family("transfer", [(s, c)] * b, jobs, age)
        for c, s, b in itertools.product([8192, 49152], [512, 2048], [2, 8])
        for jobs, age in [(0, "new"), (2, "new"), (8, "new"), (2, "old"), (8, "old")]
    ]
    familiar = [
        family(
            "familiar", [(p - (p * f // 100 // 64 * 64), p * f // 100 // 64 * 64)] * b
        )
        for p, f in [(8192, 95), (61440, 95), (61440, 99)]
        for b in [1, 2, 4, 8]
        if feasible([(p - (p * f // 100 // 64 * 64), p * f // 100 // 64 * 64)] * b)
    ]
    tests = [
        family("intermediate", [(s, c)] * b)
        for c, s, b in itertools.product(
            [12288, 24576, 40960, 57344, 63488], [256, 1024], [3, 6]
        )
    ]
    tests += [
        family("mixture", [(512, 4096)] + [(2048, 49152)] * 5),
        family("mixture", [(2048, 4096)] * 5 + [(512, 49152)]),
    ]
    tests += [
        family("transfer_test", [(1024, 24576)] * 6, n, age)
        for n in [1, 4]
        for age in ["new", "old"]
    ]
    tests += [
        family("realistic", [(1024, 15360)] * 8, hit=h) for h in [0, 25, 50, 75, 100]
    ]
    # Entire combination families are absent from new training additions.
    combination = [geometry.pop(i) for i in [27, 19, 11, 3]]
    for f in combination:
        f["block"] = "combination"
    return dict(
        version=1,
        context_limit=65536,
        batch_tokens=61440,
        train=geometry + composition + transfer + familiar,
        challenge=tests + combination,
        splits=dict(
            train_seeds=[701, 702, 703],
            development_seeds=[801, 802, 803],
            final_seeds=[901, 902, 903],
        ),
        allocation_seconds=14400,
        phase_seconds=dict(pilot=1800, collection=7200, shadow_test=2700, live=2700),
    )
