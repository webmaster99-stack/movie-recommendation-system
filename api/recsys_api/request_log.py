"""One row in `request_log` per answered request: the source of the operational dashboards
(request volume, error rate, latency) and of the alerts built on them.

A request only puts its row on a queue. One background thread writes the rows, taking
whatever has queued up in a single insert, so logging costs a request nothing and the
database one statement, however busy the API is. Writing the row in the request's own task,
even after the response had gone out, held the connection until the insert was done: under
a load test the median latency went from 110 ms to 290 ms. A failure to write is logged and
otherwise ignored: monitoring must not break the API.

Left out on purpose: `/health`, which is polled while the host wakes up and promises not to
touch the database, and CORS preflights, which no endpoint ever sees. The route is logged as
its template (`/movies/{movie_id}`), never the address that was called, and no user is
recorded: the table says how the API behaves, not what anyone did.
"""

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from recsys_api.db import RequestLog, now

logger = logging.getLogger(__name__)

UNMATCHED = "(unmatched)"  # no endpoint has this address: a 404 from the router
_SKIPPED_PATHS = frozenset({"/health"})
_PURGE_EVERY = timedelta(hours=6)
_MAX_BATCH = 500
# Rows waiting while the database is unreachable; beyond this, new ones are dropped rather
# than kept in the little memory the host has.
_MAX_WAITING = 10_000


class RequestLogWriter:
    """Writes queued rows from a thread of its own and deletes those past `keep_days`."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        keep_days: int = 90,
        clock: Callable[[], datetime] = now,
    ) -> None:
        self.sessions = sessions
        self.keep = timedelta(days=keep_days)
        self.clock = clock
        self._last_purge: datetime | None = None
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=_MAX_WAITING)
        self._thread = threading.Thread(target=self._run, name="request-log", daemon=True)
        self._thread.start()

    def add(self, method: str, route: str, status: int, latency_ms: float) -> None:
        """Queue one row. Never blocks and never raises."""
        row = {
            "method": method,
            "route": route,
            "status": status,
            "latency_ms": latency_ms,
            "created_at": self.clock(),
        }
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(row)

    def flush(self) -> None:
        """Wait until every row queued so far has been written (or has failed to be)."""
        self._queue.join()

    def close(self) -> None:
        """Write what is queued and stop the thread: called when the API shuts down."""
        self._queue.put(None)
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            while len(batch) < _MAX_BATCH:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self.write([row for row in batch if row is not None])
            for _ in batch:
                self._queue.task_done()
            if None in batch:
                return

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        try:
            with self.sessions() as session:
                current = self.clock()
                if self._last_purge is None or current - self._last_purge > _PURGE_EVERY:
                    session.execute(
                        delete(RequestLog).where(RequestLog.created_at < current - self.keep)
                    )
                    self._last_purge = current
                session.execute(insert(RequestLog), rows)
                session.commit()
        except SQLAlchemyError as error:
            # Only the type: a driver's message can contain the connection string.
            logger.warning("request log not written: %s", type(error))


class RequestLogMiddleware:
    def __init__(self, app: ASGIApp, writer: RequestLogWriter) -> None:
        self.app = app
        self.writer = writer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        skipped = (
            scope["type"] != "http"
            or scope["method"] == "OPTIONS"
            or scope["path"] in _SKIPPED_PATHS
        )
        if skipped:
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status = 500  # what the caller gets if the endpoint raises before answering

        async def send_and_note(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_and_note)
        except Exception:
            status = 500
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            # The router leaves the matched route in the scope on its way through.
            route = getattr(scope.get("route"), "path", UNMATCHED)
            self.writer.add(scope["method"], route, status, latency_ms)
