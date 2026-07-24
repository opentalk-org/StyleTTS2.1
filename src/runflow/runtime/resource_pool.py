import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from runflow.policies import ResourcePolicy


@dataclass
class ResourceWaiter:
    ticket: int
    priority: int
    groups: set[str]
    policy: ResourcePolicy


@dataclass
class ResourcePool:
    """Coordinate concurrent access to declared resource capacities."""

    limits: dict[str, float]
    _used: dict[str, float] = field(default_factory=dict)
    _exclusive_in_use: set[str] = field(default_factory=set)
    _waiters: dict[int, ResourceWaiter] = field(default_factory=dict)
    _next_ticket: int = 0

    def __post_init__(self) -> None:
        self._condition = asyncio.Condition()

    def _can_acquire(self, policy: ResourcePolicy) -> bool:
        exclusive_key = policy.exclusive_key()
        if exclusive_key and exclusive_key in self._exclusive_in_use:
            return False

        for key, amount in policy.requirements().items():
            limit = self.limits[key]
            if self._used.get(key, 0.0) + amount > limit:
                return False

        return True

    def _groups(self, policy: ResourcePolicy) -> set[str]:
        groups = set(policy.requirements())
        exclusive_key = policy.exclusive_key()
        if exclusive_key:
            groups.add(exclusive_key)
        return groups

    def _has_prior_waiter(self, waiter: ResourceWaiter) -> bool:
        for other in self._waiters.values():
            if other.ticket == waiter.ticket or waiter.groups.isdisjoint(other.groups):
                continue
            if (other.priority, other.ticket) < (waiter.priority, waiter.ticket) and self._can_acquire(other.policy):
                return True
        return False

    async def acquire(self, policy: ResourcePolicy, priority: int = 0) -> None:
        async with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            waiter = ResourceWaiter(ticket=ticket, priority=priority, groups=self._groups(policy), policy=policy)
            self._waiters[ticket] = waiter
            try:
                await self._condition.wait_for(lambda: self._can_acquire(policy) and not self._has_prior_waiter(waiter))
            finally:
                del self._waiters[ticket]

            exclusive_key = policy.exclusive_key()
            if exclusive_key:
                self._exclusive_in_use.add(exclusive_key)

            for key, amount in policy.requirements().items():
                self._used[key] = self._used.get(key, 0.0) + amount

    async def release(self, policy: ResourcePolicy) -> None:
        async with self._condition:
            exclusive_key = policy.exclusive_key()
            if exclusive_key:
                self._exclusive_in_use.remove(exclusive_key)

            for key, amount in policy.requirements().items():
                used = self._used[key]
                if amount > used:
                    raise RuntimeError(f"released {key} exceeds leased capacity")
                self._used[key] = used - amount

            self._condition.notify_all()

    @asynccontextmanager
    async def lease(self, policy: ResourcePolicy, priority: int = 0):
        await self.acquire(policy, priority)
        try:
            yield
        finally:
            await self.release(policy)
