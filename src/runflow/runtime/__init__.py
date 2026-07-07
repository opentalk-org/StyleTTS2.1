"""Async runtime components for executing node graphs."""

__all__ = ["WindowedScheduler"]


def __getattr__(name: str):
    if name == "WindowedScheduler":
        from runflow.runtime.scheduler import WindowedScheduler

        return WindowedScheduler
    raise AttributeError(name)
