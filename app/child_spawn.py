"""Hold startup termination signals until a spawned child enters its owner registry.

The startup signal handler calls defer_termination. The context records the signal.
It replays the signal after registration, so cleanup sees the child.
This module changes no signal masks or handlers.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Generator
from contextlib import contextmanager

__all__ = ["defer_termination", "registering_child"]

_SPAWN_PENDING: list[int] | None = None


@contextmanager
def registering_child() -> Generator[None]:
    """Defer signals until the owner registers the child and its exit handler."""
    global _SPAWN_PENDING
    # Nested contexts reuse the outer list. Signal handlers run only on the main thread,
    # so other threads leave the shared state unchanged.
    if _SPAWN_PENDING is not None or threading.current_thread() is not threading.main_thread():
        yield
        return
    pending: list[int] = []
    _SPAWN_PENDING = pending
    try:
        yield
    finally:
        _SPAWN_PENDING = None
        for signum in pending:
            signal.raise_signal(signum)


def defer_termination(signum: int) -> bool:
    """Record signum for replay when a registering_child context is active."""
    pending = _SPAWN_PENDING
    if pending is None:
        return False
    pending.append(signum)
    return True
