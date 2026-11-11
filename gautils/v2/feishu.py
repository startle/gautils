from __future__ import annotations

from typing import Any

import pandas as pd

from gautils.feishu.core import (
    BiTable as LegacyBiTable,
    Feishu as LegacyFeishu,
    Sheet as LegacySheet,
    Spreadsheet as LegacySpreadsheet,
    Table as LegacyTable,
    Wiki as LegacyWiki,
    WikiNode as LegacyWikiNode,
    WikiSpace as LegacyWikiSpace,
)

from .errors import ApiError, ResourceNotFoundError, UnsupportedOperationError
from .models import BatchError, BatchResult
from gautils.feishu.core._errors import _FeishuApiError


def _frame(value: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame() if value is None else value


def _strict_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except _FeishuApiError as exc:
        raise ApiError('feishu', exc.operation, code=exc.code,
                       message=exc.message, details=exc.details) from exc


class Table:
    def __init__(self, legacy: LegacyTable):
        self.legacy = legacy

    @property
    def id(self) -> str:
        return self.legacy.id

    @property
    def name(self) -> str:
        return self.legacy.name

    def fields(self) -> pd.DataFrame:
        return _frame(self.legacy.query_fields())

    def records(self, **search_options) -> pd.DataFrame:
        return _frame(self.legacy.search_records(**search_options))

    def upsert(self, frame: pd.DataFrame, *, insert_only: bool = False) -> BatchResult:
        attempted = 0 if frame is None else len(frame)
        if attempted == 0:
            return BatchResult(attempted=0, succeeded=0)
        result = _strict_call(self.legacy._insert_records_result, frame,
                              upsert=not insert_only, strict=True)
        failed = result.attempted - result.skipped - result.succeeded
        errors = tuple(BatchError(index=None, message=error.message,
                                  code=error.code, details=error.details)
                       for error in result.errors)
        return BatchResult(
            attempted=result.attempted, succeeded=result.succeeded, failed=failed,
            created=result.created, updated=result.updated, skipped=result.skipped,
            errors=errors)

    def delete(self, *, filter=None) -> int:
        return self.legacy.del_rows_by_filter(filter=filter)


class Bitable:
    def __init__(self, legacy: LegacyBiTable):
        self.legacy = legacy

    @property
    def token(self) -> str:
        return self.legacy.app_token

    def tables(self) -> pd.DataFrame:
        return _frame(self.legacy.query_tables())

    def table(self, *, name: str | None = None, table_id: str | None = None) -> Table:
        legacy = self.legacy.get_table(table_name=name, table_id=table_id)
        if legacy is None:
            identifier = table_id or name or '<unspecified>'
            raise ResourceNotFoundError('Feishu table', identifier)
        return Table(legacy)

    def delete_table(self, *, name: str | None = None, table_id: str | None = None) -> None:
        if not self.legacy.delete_table(table_name=name, table_id=table_id):
            raise ApiError('feishu.bitable', 'delete_table')

    def add_collaborator(self, member_id: str, **options) -> None:
        if not self.legacy.add_collaborator(member_id, **options):
            raise ApiError('feishu.bitable', 'add_collaborator')


class Sheet:
    def __init__(self, legacy: LegacySheet):
        self.legacy = legacy

    @property
    def id(self) -> str:
        return self.legacy.id

    @property
    def title(self) -> str:
        return self.legacy.title

    def read(self, *args, **kwargs):
        raise UnsupportedOperationError(
            'Sheet cell reads are unavailable in lark-oapi 1.4.24')

    def write(self, *args, **kwargs):
        raise UnsupportedOperationError(
            'Sheet cell writes are unavailable in lark-oapi 1.4.24')

    append = write
    clear = write


class Spreadsheet:
    def __init__(self, legacy: LegacySpreadsheet):
        self.legacy = legacy

    @property
    def token(self) -> str:
        return self.legacy.spreadsheet_token

    def sheets(self) -> list[dict[str, Any]]:
        return list(self.legacy.query_sheets())

    def metadata(self) -> dict[str, Any]:
        return dict(self.legacy.get_metadata())

    def sheet(self, *, sheet_id: str | None = None, title: str | None = None) -> Sheet:
        legacy = self.legacy.get_sheet(sheet_id=sheet_id, sheet_title=title)
        if legacy is None:
            identifier = sheet_id or title or '<unspecified>'
            raise ResourceNotFoundError('Feishu sheet', identifier)
        return Sheet(legacy)

    def add_collaborator(self, member_id: str, **options) -> None:
        if not self.legacy.add_collaborator(member_id, **options):
            raise ApiError('feishu.spreadsheet', 'add_collaborator')


class WikiNode:
    def __init__(self, legacy: LegacyWikiNode):
        self.legacy = legacy

    @property
    def id(self) -> str:
        return self.legacy.id

    @property
    def name(self) -> str:
        return self.legacy.name


class WikiSpace:
    def __init__(self, legacy: LegacyWikiSpace):
        self.legacy = legacy

    @property
    def id(self) -> str:
        return self.legacy.space_id

    def info(self) -> dict[str, Any]:
        return dict(self.legacy.get_info())

    def nodes(self, *, parent_token: str | None = None) -> pd.DataFrame:
        return _frame(_strict_call(self.legacy._list_nodes, parent_node_token=parent_token))

    def node(self, token: str) -> WikiNode:
        legacy = _strict_call(self.legacy.get_node, token, strict=True)
        if legacy is None:
            raise ResourceNotFoundError('Feishu wiki node', token)
        return WikiNode(legacy)


class Wiki:
    def __init__(self, legacy: LegacyWiki):
        self.legacy = legacy

    def spaces(self) -> pd.DataFrame:
        return _frame(self.legacy.list_spaces())

    def space(self, space_id: str) -> WikiSpace:
        return WikiSpace(self.legacy.get_space(space_id))


class Feishu:
    """Strict v2 facade backed by the existing Feishu SDK adapter."""

    def __init__(self, legacy: LegacyFeishu | None = None, **credentials):
        self.legacy = legacy or LegacyFeishu(**credentials)

    def bitable(self, app_token: str) -> Bitable:
        return Bitable(self.legacy.get_bitable(app_token))

    def spreadsheet(self, spreadsheet_token: str) -> Spreadsheet:
        return Spreadsheet(self.legacy.get_spreadsheet(spreadsheet_token))

    def wiki(self) -> Wiki:
        return Wiki(self.legacy.get_wiki())
