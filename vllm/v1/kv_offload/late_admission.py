# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded admission protocol model; not enabled by the serving engine.

Ownership precedes runtime integration: reservations do not imply execution, and
only a single authoritative seal determines membership on all ranks.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    request_id: str
    version: int
    tokens: int


@dataclass(frozen=True)
class Seal:
    generation: int
    batch: int
    candidates: tuple[Candidate, ...]
    active_mask: int

    @property
    def active(self) -> tuple[Candidate, ...]:
        return tuple(
            c for i, c in enumerate(self.candidates) if self.active_mask & (1 << i)
        )


class AdmissionWindow:
    """Single-authority transition model with explicit per-rank ownership.

    The caller serializes transitions. A production transport must enforce this
    contract with process-safe publication; a Python object is not that transport.
    """

    def __init__(
        self,
        generation: int,
        batch: int,
        candidates: tuple[Candidate, ...],
        ranks: int,
        token_budget: int,
        candidate_cap: int = 4,
    ):
        if generation < 1 or batch < 0 or ranks < 1 or token_budget < 0:
            raise ValueError("invalid window bounds")
        if not 0 <= len(candidates) <= candidate_cap <= 4:
            raise ValueError("candidate capacity exceeded")
        if len({c.request_id for c in candidates}) != len(candidates):
            raise ValueError("duplicate candidate")
        if any(c.tokens <= 0 or c.version < 0 for c in candidates):
            raise ValueError("invalid candidate")
        self.generation, self.batch = generation, batch
        self.candidates, self.ranks, self.token_budget = candidates, ranks, token_budget
        self.staged_mask = self.consumed_mask = self.retired_mask = 0
        self.ready_masks = [0] * len(candidates)
        self.cancelled_mask = 0
        self.sealed: Seal | None = None
        self.failure: str | None = None

    def _check(self, generation: int, batch: int, rank: int | None = None):
        if (generation, batch) != (self.generation, self.batch):
            raise ValueError("stale admission identity")
        if rank is not None and not 0 <= rank < self.ranks:
            raise ValueError("invalid rank")
        if self.failure:
            raise RuntimeError(self.failure)

    def staged(self, generation: int, batch: int, rank: int):
        self._check(generation, batch, rank)
        self.staged_mask |= 1 << rank

    def ready(self, generation: int, batch: int, rank: int, slot: int, version: int):
        self._check(generation, batch, rank)
        if not 0 <= slot < len(self.candidates):
            raise ValueError("invalid candidate slot")
        if version != self.candidates[slot].version:
            raise ValueError("stale candidate version")
        if self.sealed is None:
            self.ready_masks[slot] |= 1 << rank

    def cancel(self, generation: int, batch: int, slot: int) -> bool:
        self._check(generation, batch)
        if not 0 <= slot < len(self.candidates):
            raise ValueError("invalid candidate slot")
        if self.sealed is not None:
            # Post-commit abort must use ordinary ordered output retirement.
            return False
        self.cancelled_mask |= 1 << slot
        return True

    def propose_seal(self, generation: int, batch: int) -> Seal:
        self._check(generation, batch)
        if self.sealed is not None:
            raise RuntimeError("already sealed")
        if not self.staged_mask:
            raise RuntimeError("no rank has reached seal boundary")
        remaining, active = self.token_budget, 0
        for i, candidate in enumerate(self.candidates):
            if (
                self.ready_masks[i] == (1 << self.ranks) - 1
                and not self.cancelled_mask & (1 << i)
                and candidate.tokens <= remaining
            ):
                active |= 1 << i
                remaining -= candidate.tokens
        return Seal(self.generation, self.batch, self.candidates, active)

    def commit(self, proposal: Seal, apply):
        """Commit scheduler accounting before publishing immutable membership.

        Any callback failure poisons the window: never publish partial accounting,
        discard a scheduler mutation, or continue with a per-rank fallback.
        """
        self._check(proposal.generation, proposal.batch)
        if proposal != self.propose_seal(self.generation, self.batch):
            raise ValueError("proposal changed before commitment")
        try:
            apply(proposal.active)
        except BaseException as error:
            self.failure = f"admission commitment failed: {error}"
            raise
        self.sealed = proposal

    def consume(self, generation: int, batch: int, rank: int) -> Seal:
        self._check(generation, batch, rank)
        if self.sealed is None or not self.staged_mask & (1 << rank):
            raise RuntimeError("rank is not prepared for sealed execution")
        self.consumed_mask |= 1 << rank
        return self.sealed

    def retire(self, generation: int, batch: int, rank: int):
        self._check(generation, batch, rank)
        if not self.consumed_mask & (1 << rank):
            raise RuntimeError("cannot release unconsumed reservation")
        if self.retired_mask & (1 << rank):
            raise RuntimeError("duplicate retirement")
        self.retired_mask |= 1 << rank

    @property
    def reclaimable(self) -> bool:
        return not self.failure and self.retired_mask == (1 << self.ranks) - 1
