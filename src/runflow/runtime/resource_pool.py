from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from runflow.policies import ResourcePolicy


@dataclass
class ResourcePool:
    """Small async resource allocator.

    It allows unrelated nodes to run concurrently, while limiting shared things
    such as io slots, cpu workers, accelerator slots, vram budget, etc.
    """

    limits: dict[str, float]
    _used: dict[str, float] = field(default_factory=dict)
    _exclusive_in_use: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._condition = asyncio.Condition()

    def _can_acquire(self, policy: ResourcePolicy) -> bool:
        exclusive_key = policy.exclusive_key()
        if exclusive_key and exclusive_key in self._exclusive_in_use:
            return False

        for key, amount in policy.requirements().items():
            limit = self.limits.get(key)
            if limit is None:
                # Unknown resources are treated as unlimited labels.
                continue
            if self._used.get(key, 0.0) + amount > limit:
                return False

        return True

    async def acquire(self, policy: ResourcePolicy) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._can_acquire(policy))

            exclusive_key = policy.exclusive_key()
            if exclusive_key:
                self._exclusive_in_use.add(exclusive_key)

            for key, amount in policy.requirements().items():
                self._used[key] = self._used.get(key, 0.0) + amount

    async def release(self, policy: ResourcePolicy) -> None:
        async with self._condition:
            exclusive_key = policy.exclusive_key()
            if exclusive_key:
                self._exclusive_in_use.discard(exclusive_key)

            for key, amount in policy.requirements().items():
                self._used[key] = max(0.0, self._used.get(key, 0.0) - amount)

            self._condition.notify_all()

    @asynccontextmanager
    async def lease(self, policy: ResourcePolicy):
        await self.acquire(policy)
        try:
            yield
        finally:
            await self.release(policy)
