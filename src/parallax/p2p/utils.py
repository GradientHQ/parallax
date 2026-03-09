"""
Utility functions for P2P server.

This module contains utility functions for the P2P server.
"""

import asyncio
import os
from concurrent.futures import Future
from threading import Thread
from typing import Awaitable

import uvloop


def switch_to_uvloop() -> asyncio.AbstractEventLoop:
    """stop any running event loops; install uvloop; then create, set and return a new event loop"""
    try:
        # if we're in jupyter, get rid of its built-in event loop
        asyncio.get_event_loop().stop()
    except RuntimeError:
        pass  # this allows running DHT from background threads with no event loop
    uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


class AsyncWorker:
    """
    Async worker class for Parallax.

    This class is used to run coroutines in a separate thread.
    """

    def __init__(self) -> None:
        self._event_thread = None
        self._event_loop_fut = None
        self._pid = None

    def _run_event_loop(self):
        try:
            loop = switch_to_uvloop()
            self._event_loop_fut.set_result(loop)
        except Exception as e:
            self._event_loop_fut.set_exception(e)
        loop.run_forever()

    def run_coroutine(self, coro: Awaitable, return_future: bool = False):
        """Run a coroutine in a separate thread."""
        if self._event_thread is None or self._pid != os.getpid():
            self._pid = os.getpid()
            self._event_loop_fut = Future()
            self._event_thread = Thread(target=self._run_event_loop, daemon=True)
            self._event_thread.start()

        loop = self._event_loop_fut.result()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future if return_future else future.result()
