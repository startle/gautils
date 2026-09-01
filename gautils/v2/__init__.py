"""Versioned, strict APIs that coexist with the legacy gautils surface."""

from .database import Database
from .errors import (
    ApiError,
    DatabaseOperationError,
    GautilsError,
    HttpRequestError,
    ResourceNotFoundError,
    UnsupportedOperationError,
)
from .http import HttpClient, RetryPolicy, Timeout
from .models import BatchError, BatchResult
from .feishu import Bitable, Feishu, Sheet, Spreadsheet, Table, Wiki, WikiNode, WikiSpace

__all__ = [
    'ApiError',
    'BatchError',
    'BatchResult',
    'Database',
    'DatabaseOperationError',
    'Bitable',
    'Feishu',
    'GautilsError',
    'HttpClient',
    'HttpRequestError',
    'ResourceNotFoundError',
    'RetryPolicy',
    'Timeout',
    'Sheet',
    'Spreadsheet',
    'Table',
    'UnsupportedOperationError',
    'Wiki',
    'WikiNode',
    'WikiSpace',
]
