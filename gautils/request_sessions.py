"""Serial, caller-owned HTTP connection batches."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
from urllib.parse import urlsplit

import requests


def _owner() -> tuple[int, object]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), task


def _close_all(sessions, *, suppress: bool) -> None:
    first_error = None
    for session in sessions:
        try:
            session.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None and not suppress:
        raise first_error


@dataclass
class _Batch:
    owner: tuple[int, object]
    policy: object
    sessions: dict = field(default_factory=dict)
    closed: bool = False
    borrowed: bool = False

    def check(self, policy: object) -> None:
        if self.closed:
            raise RuntimeError('connection batch is closed')
        if self.owner != _owner():
            raise RuntimeError('connection batch belongs to another task or thread')
        if self.policy != policy:
            raise RuntimeError('connection batch transport configuration changed')


class RequestSessions:
    """One instance per Web; inherited child tasks cannot borrow its batch."""

    def __init__(self, session_factory: Callable | None = None) -> None:
        self._factory = session_factory
        self._current: ContextVar[_Batch | None] = ContextVar('request_sessions', default=None)

    @contextmanager
    def batch(self, *, policy: object = None) -> Iterator[None]:
        current = self._current.get()
        if current is not None:
            current.check(policy)
            yield
            return
        batch = _Batch(_owner(), policy)
        token = self._current.set(batch)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            batch.closed = True
            try:
                _close_all(batch.sessions.values(), suppress=failed)
            finally:
                self._current.reset(token)

    @contextmanager
    def borrow(self, url: str, *, policy: object = None, reuse: bool = True) -> Iterator[requests.Session]:
        batch = self._current.get()
        if batch is not None:
            batch.check(policy)
            if batch.borrowed:
                raise RuntimeError('connection batch only supports serial requests')
            batch.borrowed = True
        try:
            pooled = batch is not None and reuse
            if pooled:
                origin = self._origin(url)
                session = batch.sessions.get(origin)
                if session is None:
                    session = self._new_session()
                    batch.sessions[origin] = session
            else:
                session = self._new_session()
            failed = False
            try:
                yield session
            except BaseException:
                failed = True
                raise
            finally:
                if not pooled:
                    _close_all((session,), suppress=failed)
        finally:
            if batch is not None:
                batch.borrowed = False

    def _new_session(self) -> requests.Session:
        return requests.Session() if self._factory is None else self._factory()

    @staticmethod
    def _origin(url: str) -> tuple[str, str, int]:
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname:
                raise ValueError
            port = parsed.port
            if port is None:
                port = 443 if parsed.scheme == 'https' else 80
            return parsed.scheme, parsed.hostname.lower(), port
        except ValueError:
            raise ValueError('connection batch requires a valid HTTP origin') from None
