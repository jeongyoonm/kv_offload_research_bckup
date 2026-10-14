# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic cache workload definitions; profiling precedes class selection."""

import gzip
import hashlib
import json
import random
from pathlib import Path

from experiments.native_scheduler_ab_20261010.run import collector

PREFIXES = (4096, 16384, 32768, 65536)
SUFFIXES = (128, 512, 2048, 8192)
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)


def homogeneous_grid():
    return [
        dict(id=f"c{c}_q{q}", family="homogeneous", shapes=[[c, q]])
        for c in PREFIXES
        for q in SUFFIXES
    ]


def heterogeneous(classes):
    order = ("transfer_heavy", "compute_heavy", "balanced")
    assert set(classes) == set(order)
    return [
        dict(
            id="heterogeneous_equal",
            family="heterogeneous",
            shapes=[classes[name] for name in order],
            weights=[1, 1, 1],
        ),
        dict(
            id="heterogeneous_transfer_skew",
            family="heterogeneous",
            shapes=[classes[name] for name in order],
            weights=[3, 1, 1],
        ),
    ]


def build(path, spec, count, seed):
    """Cache token payloads once; every policy consumes the same requests."""
    path = Path(path)
    if path.exists():
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
        assert data["seed"] == seed and data["spec"] == spec
        assert len(data["requests"]) == count
        return data
    requests = []
    shapes = spec["shapes"]
    weighted = [
        shape
        for shape, weight in zip(shapes, spec.get("weights", [1] * len(shapes)))
        for _ in range(weight)
    ]
    for index in range(count):
        context, suffix = weighted[index % len(weighted)]
        prefix, tail = collector.prompt(context, suffix, seed * 100000 + index)
        requests.append(
            dict(index=index, context=context, suffix=suffix, prefix=prefix, tail=tail)
        )
    data = dict(spec=spec, seed=seed, requests=requests)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        json.dump(data, stream)
    return data


def evolving(path, count, seed, *, prefix_pool=192, context=16384, suffix=512):
    """Reuse hot/cyclic prefixes and inject unique misses without cache resets."""
    path = Path(path)
    if path.exists():
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    prefixes = [
        collector.prompt(context, 0, seed * 100000 + index)[0]
        for index in range(prefix_pool)
    ]
    rng = random.Random(seed)
    warmup = []
    for index in range(prefix_pool):
        _, tail = collector.prompt(0, suffix, seed * 200000 + index)
        warmup.append(dict(index=index, prefix=prefixes[index], tail=tail))
    requests = []
    for index in range(count):
        draw = rng.random()
        if draw < 0.25:
            slot, kind = rng.randrange(8), "hot_reuse"
            prefix = prefixes[slot]
        elif draw < 0.85:
            slot, kind = 8 + index % (prefix_pool - 8), "cyclic_reuse"
            prefix = prefixes[slot]
        else:
            slot, kind = None, "new_prefix"
            prefix, _ = collector.prompt(context, 0, seed * 300000 + index)
        _, tail = collector.prompt(0, suffix, seed * 400000 + index)
        requests.append(
            dict(index=index, prefix=prefix, tail=tail, intended_kind=kind, slot=slot)
        )
    data = dict(
        seed=seed,
        prefix_pool=prefix_pool,
        context=context,
        suffix=suffix,
        warmup=warmup,
        requests=requests,
        scope="Fixed-budget evolving caches; intended reuse is not a realized hit.",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        json.dump(data, stream)
    return data


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
