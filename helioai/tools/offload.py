"""Run synchronous library calls off the event loop.

Every data tool is declared `async def` because the registry always awaits, yet speasy,
ChromaDB, the sentence-transformer encoder and `numpy.savez_compressed` are all
synchronous. Called inline, a 60-second speasy download froze the event loop — and with
it every other user's stream on the web and MCP servers — for the whole download. The
CLI never noticed: it runs one question per `asyncio.run`.

`run_blocking` is `asyncio.to_thread` with the decision written down: `to_thread` copies
the calling context (PEP 567), so `workspace.get_session_dir()` and the structlog
bindings resolve to the same session inside the worker thread as outside it. A future
`ThreadPoolExecutor.submit` would *not* copy it, and would write the download into the
wrong session — that is why nothing here reaches for an executor directly.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

# speasy's caches and the archives behind them are not built for a dozen simultaneous
# downloads; four is what a data_analyst's first turn typically batches. A threading
# semaphore rather than an asyncio one because it is taken inside the worker thread,
# around the network call itself, and binds to no event loop.
speasy_gate = threading.BoundedSemaphore(4)


async def run_blocking[T](fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Await a synchronous function in the default executor, context preserved.

    Args:
        fn: The synchronous callable — a tool body, a library call.
        *args: Positional arguments for `fn`.
        **kwargs: Keyword arguments for `fn`.

    Returns:
        Whatever `fn` returns.
    """
    return await asyncio.to_thread(fn, *args, **kwargs)
