from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WindowPolicy:
    """Generic runtime windowing policy.

    The runtime does not know whether an item is audio, an image, a document, a
    URL, or a custom object. A window is just a bounded group of source items.

    Limits are optional and can be combined:
      * max_items: maximum number of source items in one window
      * max_cost: maximum accumulated item cost in one window

    `cost_fn` may be supplied by caller/runtime integration. By default every
    item has cost 1.0.
    """

    max_items: int = 10
    max_cost: float | None = None

    def normalized(self) -> "WindowPolicy":
        return WindowPolicy(
            max_items=max(1, int(self.max_items)),
            max_cost=None if self.max_cost is None else max(0.0, float(self.max_cost)),
        )


class WindowManager:
    """Split arbitrary source items into bounded windows.

    This class is intentionally domain-agnostic. It never imports or references
    audio-specific types. Domain-specific nodes can still estimate item cost by
    passing a custom `cost_fn`.
    """

    def __init__(
        self,
        items: Iterable[Any],
        policy: WindowPolicy | None = None,
        cost_fn: Callable[[Any], float] | None = None,
    ):
        self.items = list(items)
        self.policy = (policy or WindowPolicy()).normalized()
        self.cost_fn = cost_fn or (lambda _item: 1.0)

    @classmethod
    def from_config(
        cls,
        items: Iterable[Any],
        config: dict[str, Any] | None,
        cost_fn: Callable[[Any], float] | None = None,
    ) -> "WindowManager":
        cfg = dict(config or {})
        # Runtime config is generic; no domain-specific source-item names here.
        policy = WindowPolicy(
            max_items=int(cfg.get("max_items", 10)),
            max_cost=cfg.get("max_cost"),
        )
        return cls(items=items, policy=policy, cost_fn=cost_fn)

    def iter_windows(self) -> Iterator[list[Any]]:
        window: list[Any] = []
        cost = 0.0

        for item in self.items:
            item_cost = max(0.0, float(self.cost_fn(item)))
            would_exceed_count = len(window) >= self.policy.max_items
            would_exceed_cost = (
                self.policy.max_cost is not None
                and window
                and cost + item_cost > self.policy.max_cost
            )

            if would_exceed_count or would_exceed_cost:
                yield window
                window = []
                cost = 0.0

            window.append(item)
            cost += item_cost

        if window:
            yield window
