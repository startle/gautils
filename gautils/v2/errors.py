from __future__ import annotations

from typing import Any


class GautilsError(Exception):
    """Base exception for the v2 API."""


class HttpRequestError(GautilsError):
    def __init__(self, method: str, url: str, *, response=None):
        self.method = method.upper()
        self.url = url
        self.response = response
        status = getattr(response, 'status_code', None)
        suffix = f' status={status}' if status is not None else ''
        super().__init__(f'{self.method} {url} failed{suffix}')


class DatabaseOperationError(GautilsError):
    def __init__(self, operation: str, *, statement: str | None = None):
        self.operation = operation
        self.statement = statement
        super().__init__(f'database {operation} failed')


class ApiError(GautilsError):
    def __init__(
        self,
        service: str,
        operation: str,
        *,
        code: int | str | None = None,
        message: str | None = None,
        details: Any = None,
    ):
        self.service = service
        self.operation = operation
        self.code = code
        self.message = message
        self.details = details
        suffix = f' code={code}' if code is not None else ''
        if message:
            suffix += f' message={message}'
        super().__init__(f'{service}.{operation} failed{suffix}')


class ResourceNotFoundError(GautilsError):
    def __init__(self, resource: str, identifier: str):
        self.resource = resource
        self.identifier = identifier
        super().__init__(f'{resource} not found: {identifier}')


class UnsupportedOperationError(GautilsError):
    """Raised instead of returning a fake successful/empty value."""
