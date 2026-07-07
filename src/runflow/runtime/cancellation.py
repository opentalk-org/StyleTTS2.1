from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Iterable, Iterator


_current_cancel_token: ContextVar[CancelToken | None] = ContextVar("runflow_cancel_token", default=None)


class CancelToken:
    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def check(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError()


def check_cancel() -> None:
    token = _current_cancel_token.get()
    if token is not None:
        token.check()


def cancellable[T](items: Iterable[T]) -> Iterator[T]:
    for item in items:
        check_cancel()
        yield item


class cancellation_scope:
    def __init__(self, token: CancelToken) -> None:
        self._token = token
        self._reset_token: Token[CancelToken | None] | None = None

    def __enter__(self) -> None:
        self._reset_token = _current_cancel_token.set(self._token)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._reset_token is not None:
            _current_cancel_token.reset(self._reset_token)
