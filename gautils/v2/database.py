from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
from sqlalchemy import URL, Engine, create_engine, text

from .errors import DatabaseOperationError
from .models import BatchResult


_IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f'invalid SQL identifier: {value!r}')
    return value


def _batches(values: list[Mapping[str, Any]], size: int):
    if size <= 0:
        raise ValueError('batch_size must be positive')
    for start in range(0, len(values), size):
        yield values[start:start + size]


class Database:
    """SQLAlchemy Core based database facade with named parameters only."""

    def __init__(self, engine: Engine, *, owns_engine: bool = False):
        self.engine = engine
        self._owns_engine = owns_engine
        self._closed = False

    @classmethod
    def from_url(cls, url: str | URL, **engine_options) -> 'Database':
        return cls(create_engine(url, **engine_options), owns_engine=True)

    @classmethod
    def mysql(
        cls,
        *,
        user: str,
        password: str,
        host: str,
        database: str,
        port: int = 3306,
        driver: str = 'mysqlconnector',
        query: Mapping[str, str] | None = None,
        **engine_options,
    ) -> 'Database':
        url = URL.create(
            f'mysql+{driver}', username=user, password=password,
            host=host, port=port, database=database, query=dict(query or {}))
        defaults = {
            'pool_size': 5,
            'max_overflow': 10,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
        }
        defaults.update(engine_options)
        return cls.from_url(url, **defaults)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError('Database is closed')

    def fetch_all(
        self,
        statement: str,
        params: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        self._ensure_open()
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(statement), dict(params or {}))
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as exc:
            raise DatabaseOperationError('fetch_all', statement=statement) from exc

    def execute(
        self,
        statement: str,
        params: Mapping[str, Any] | None = None,
    ) -> int:
        self._ensure_open()
        try:
            with self.engine.begin() as connection:
                result = connection.execute(text(statement), dict(params or {}))
                return result.rowcount
        except Exception as exc:
            raise DatabaseOperationError('execute', statement=statement) from exc

    def execute_many(
        self,
        statement: str,
        params: Iterable[Mapping[str, Any]],
        *,
        batch_size: int = 1000,
    ) -> BatchResult:
        self._ensure_open()
        records = [dict(item) for item in params]
        if not records:
            return BatchResult(attempted=0, succeeded=0)
        try:
            affected = 0
            with self.engine.begin() as connection:
                sql = text(statement)
                for batch in _batches(records, batch_size):
                    result = connection.execute(sql, batch)
                    if result.rowcount and result.rowcount > 0:
                        affected += result.rowcount
            return BatchResult(attempted=len(records), succeeded=len(records), updated=affected)
        except Exception as exc:
            raise DatabaseOperationError('execute_many', statement=statement) from exc

    def upsert_dataframe(
        self,
        table: str,
        frame: pd.DataFrame,
        *,
        key_columns: Iterable[str],
        update_columns: Iterable[str] | None = None,
        batch_size: int = 1000,
    ) -> BatchResult:
        self._ensure_open()
        if frame is None or frame.empty:
            return BatchResult(attempted=0, succeeded=0)
        table = _identifier(table)
        keys = [_identifier(value) for value in key_columns]
        if not keys:
            raise ValueError('key_columns cannot be empty')
        missing = set(keys) - set(frame.columns)
        if missing:
            raise ValueError(f'missing key columns: {sorted(missing)}')
        updates = (
            [_identifier(value) for value in update_columns]
            if update_columns is not None
            else [_identifier(value) for value in frame.columns if value not in keys]
        )
        if not updates:
            raise ValueError('update_columns cannot be empty')
        missing = set(updates) - set(frame.columns)
        if missing:
            raise ValueError(f'missing update columns: {sorted(missing)}')

        columns = keys + [value for value in updates if value not in keys]
        quoted = ','.join(f'`{value}`' for value in columns)
        placeholders = ','.join(f':{value}' for value in columns)
        assignments = ','.join(
            f'`{value}`=VALUES(`{value}`)' for value in updates)
        statement = (
            f'INSERT INTO `{table}` ({quoted}) VALUES ({placeholders}) '
            f'ON DUPLICATE KEY UPDATE {assignments}')
        data = frame.loc[:, columns].astype(object)
        data = data.where(data.notna(), None)
        records = data.to_dict(orient='records')
        return self.execute_many(statement, records, batch_size=batch_size)

    def close(self) -> None:
        if not self._closed and self._owns_engine:
            self.engine.dispose()
        self._closed = True

    def __enter__(self) -> 'Database':
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
