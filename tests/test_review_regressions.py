"""Regression checks using real SDK response models and offline transports."""
from unittest.mock import MagicMock
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pandas as pd
import pytest
from lark_oapi.api.bitable.v1 import (
    BatchCreateAppTableRecordResponse, BatchUpdateAppTableRecordResponse,
    SearchAppTableRecordResponse,
)
from lark_oapi.api.sheets.v3 import GetSpreadsheetResponse
from lark_oapi.api.wiki.v2 import (
    CreateSpaceNodeResponse, GetNodeSpaceResponse, GetSpaceResponse,
    ListSpaceNodeResponse,
)

from gautils.feishu.core.bitable import Table
from gautils.feishu.core.spreadsheets import Spreadsheet
from gautils.feishu.core.wiki import WikiSpace
from gautils.mysqldb import connect_mysql
from gautils.v2 import ApiError, BatchResult
from gautils.v2.feishu import Table as V2Table, WikiSpace as V2WikiSpace


def test_cache_separates_parameters_types_and_databases(tmp_path, monkeypatch):
    backends = []

    def backend(*args, **kwargs):
        db = MagicMock()
        db.query.side_effect = lambda *a, **k: pd.DataFrame({'value': [str((args, a, k))]})
        backends.append(db.query)
        return db

    monkeypatch.setattr('gautils.mysqldb.MysqlDbImpl', backend)
    first = connect_mysql('host', 3306, 'user', 'secret', 'one',
                          is_use_file_cache=True, cache_dir=tmp_path)
    sql = 'SELECT * FROM t WHERE a=%s AND b=%s'
    a = first.query(sql, 'a_b', 'c')
    b = first.query(sql, 'a', 'b_c')
    assert not a.equals(b)
    assert first.query(sql, 'a_b', 'c').equals(a)
    assert backends[0].call_count == 2
    assert not first.query('SELECT %s', 1).equals(first.query('SELECT %s', '1'))
    second = connect_mysql('host', 3306, 'user', 'secret', 'two',
                           is_use_file_cache=True, cache_dir=tmp_path)
    assert not second.query(sql, 'a_b', 'c').equals(a)
    assert backends[1].call_count == 1


def test_wiki_uses_real_response_shapes():
    client = MagicMock()
    space = WikiSpace(client, '123')
    client.wiki.v2.space_node.create.return_value = CreateSpaceNodeResponse({
        'code': 0, 'data': {'node': {'node_token': 'new', 'title': 'Doc'}},
    })
    assert space.create_node('Doc', 'docx').id == 'new'
    client.wiki.v2.space.get_node.return_value = GetNodeSpaceResponse({
        'code': 0, 'data': {'node': {'node_token': 'new', 'space_id': '456'}},
    })
    client.wiki.v2.space.get.return_value = GetSpaceResponse({
        'code': 0, 'data': {'space': {'space_id': '456', 'name': 'Space'}},
    })
    assert space.get_info_by_node_token('new') == {
        'space_id': '456', 'name': 'Space', 'node_token': 'new',
    }


def test_metadata_uses_real_response_shape():
    client = MagicMock()
    client.sheets.v3.spreadsheet.get.return_value = GetSpreadsheetResponse({
        'code': 0, 'data': {'spreadsheet': {'title': 'Book', 'url': 'https://example.com'}},
    })
    assert Spreadsheet(client, 'token').get_metadata() == {
        'title': 'Book', 'url': 'https://example.com', 'token': 'token',
    }


def test_unsupported_sheet_mutations_are_explicit():
    book = Spreadsheet(MagicMock(), 'token')
    with pytest.raises(NotImplementedError):
        book.create_sheet('New')
    with pytest.raises(NotImplementedError):
        book.delete_sheet(sheet_id='s1')


def make_table(fields):
    bitable = MagicMock(app_token='token')
    table = Table(bitable, pd.Series({'id': 't1', 'name': 'Data'}))
    table._fields = pd.DataFrame(fields)
    return table, bitable.client.bitable.v1.app_table_record


def test_date_read_write_roundtrip_and_mixed_timezones():
    table, api = make_table([{'name': 'date', 'type': 5, 'description': None}])
    api.search.return_value = SearchAppTableRecordResponse({
        'code': 0, 'data': {'items': [
            {'record_id': 'r1', 'fields': {'date': 1704067200123}},
        ], 'has_more': False},
    })
    assert table.format_type_df_before_CU(table.search_records()).iloc[0]['date'] == 1704067200123
    values = pd.Series([
        '2024-01-01 08:00:00.123', pd.Timestamp('2024-01-01T00:00:00.123Z'),
        pd.Timestamp('2024-01-01T09:00:00.123+09:00'), None, pd.NaT,
        pd.Timestamp('1970-01-01T00:00:00Z'), 'NaT', pd.NA,
    ], dtype=object)
    converted = table.format_type_df_before_CU(pd.DataFrame({'date': values}))
    assert converted['date'].tolist() == [1704067200123] * 3 + [None, None, 0, None, None]


def test_failed_wiki_page_is_not_cached_and_v2_raises():
    client = MagicMock()
    space = WikiSpace(client, '123')
    page = ListSpaceNodeResponse({'code': 0, 'data': {
        'items': [{'node_token': 'root', 'title': 'Root'}],
        'has_more': True, 'page_token': 'next',
    }})
    failed = ListSpaceNodeResponse({'code': 999, 'msg': 'temporary error'})
    final = ListSpaceNodeResponse({'code': 0, 'data': {
        'items': [{'node_token': 'last', 'title': 'Last'}], 'has_more': False,
    }})
    api = client.wiki.v2.space_node.list
    api.side_effect = [page, failed, page, failed, page, final]
    assert space.list_nodes().empty
    assert space._nodes is None
    with pytest.raises(ApiError) as error:
        V2WikiSpace(space).nodes()
    assert error.value.code == 999
    assert error.value.message == 'temporary error'
    assert space._nodes is None
    assert V2WikiSpace(space).nodes()['node_token'].tolist() == ['root', 'last']
    assert len(V2WikiSpace(space).nodes()) == 2
    assert api.call_count == 6


def test_batch_reports_deduplication_and_real_api_failures():
    table, api = make_table([{'name': 'name', 'type': 1, 'description': 'Primary1'}])
    api.batch_create.return_value = BatchCreateAppTableRecordResponse({
        'code': 0, 'data': {'records': [{'record_id': 'r1', 'fields': {'name': 'same'}}]},
    })
    frame = pd.DataFrame({'name': ['same', 'same']})
    result = V2Table(table).upsert(frame, insert_only=True)
    assert (result.attempted, result.succeeded, result.skipped, result.failed) == (2, 1, 1, 0)
    assert result.created == 1 and result.ok and result.success_rate == 1.0
    result.raise_for_errors()
    assert table.insert_records(frame, upsert=False) == 1
    api.batch_create.return_value = BatchCreateAppTableRecordResponse({
        'code': 1254060, 'msg': 'invalid value',
    })
    failed = V2Table(table).upsert(frame, insert_only=True)
    assert (failed.failed, failed.skipped, failed.succeeded) == (1, 1, 0)
    assert failed.errors[0].code == 1254060
    assert failed.errors[0].message == 'invalid value'
    with pytest.raises(RuntimeError):
        failed.raise_for_errors()


def test_all_skipped_batch_is_successful():
    result = BatchResult(attempted=2, succeeded=0, skipped=2)
    assert result.ok and result.success_rate == 1.0
    with pytest.raises(ValueError):
        BatchResult(attempted=1, succeeded=1, skipped=1)


def test_cache_typed_values_ordering_and_unsupported_bypass(tmp_path, monkeypatch):
    backend = MagicMock()
    query = backend.query
    query.return_value = pd.DataFrame({'value': [1]})
    monkeypatch.setattr('gautils.mysqldb.MysqlDbImpl', lambda *a, **k: backend)
    db = connect_mysql('h', 3306, 'u', 'p', 'db', is_use_file_cache=True, cache_dir=tmp_path)
    values = [b'\x00_', Decimal('1.20'), date(2026, 1, 1), datetime(2026, 1, 1, 1, 2),
              time(1, 2, 3), timedelta(days=2, microseconds=3), None, True, 1.5,
              ['one', {'nested': 2}], ('one', 'two')]
    for value in values:
        calls = query.call_count
        db.query('SELECT %s', value)
        db.query('SELECT %s', value)
        assert query.call_count == calls + 1
    calls = query.call_count
    db.query('SELECT :a, :b', a=1, b=2)
    db.query('SELECT :a, :b', b=2, a=1)
    assert query.call_count == calls + 1
    recursive = []
    recursive.append(recursive)
    for value in [object(), float('nan'), {1: 'non-string-key'}, recursive]:
        calls = query.call_count
        db.query('SELECT %s', value)
        db.query('SELECT %s', value)
        assert query.call_count == calls + 2


def test_wiki_missing_data_and_bad_tokens_raise_without_caching():
    client = MagicMock()
    api = client.wiki.v2.space_node.list
    for payload in [None, {'items': [], 'has_more': True},
                    {'items': [], 'has_more': True, 'page_token': 'repeated'}]:
        space = WikiSpace(client, '123')
        api.return_value = ListSpaceNodeResponse({'code': 0, 'data': payload})
        with pytest.raises(ApiError):
            V2WikiSpace(space).nodes()
        assert space._nodes is None
    api.return_value = ListSpaceNodeResponse({'code': 999, 'msg': 'offline'})
    with pytest.raises(ApiError):
        V2WikiSpace(WikiSpace(client, '123')).node('missing')
    api.return_value = ListSpaceNodeResponse({'code': 0, 'data': {'items': [], 'has_more': False}})
    assert V2WikiSpace(WikiSpace(client, '123')).nodes().empty


def test_wiki_space_lookup_failure_does_not_fake_complete_info():
    client = MagicMock()
    client.wiki.v2.space.get_node.return_value = GetNodeSpaceResponse({
        'code': 0, 'data': {'node': {'space_id': '123'}},
    })
    client.wiki.v2.space.get.return_value = GetSpaceResponse({'code': 999, 'msg': 'denied'})
    assert WikiSpace(client, '123').get_info_by_node_token('node') == {}


def test_batch_partial_failure_preserves_previous_successes():
    table, api = make_table([{'name': 'name', 'type': 1, 'description': 'Primary1'}])
    api.batch_create.side_effect = [
        BatchCreateAppTableRecordResponse({'code': 0, 'data': {
            'records': [{'record_id': f'r{i}'} for i in range(1000)],
        }}),
        BatchCreateAppTableRecordResponse({'code': 1254060, 'msg': 'invalid value'}),
    ]
    frame = pd.DataFrame({'name': [f'n{i}' for i in range(1001)] + ['n0']})
    result = V2Table(table).upsert(frame, insert_only=True)
    assert (result.attempted, result.created, result.failed, result.skipped) == (1002, 1000, 1, 1)
    assert result.success_rate == 1000 / 1001
    assert result.errors[0].details['indexes'] == [1000]
    assert api.batch_create.call_count == 2


@pytest.mark.parametrize('records', [None, [], [{'record_id': 'r1'}],
                                     [{'record_id': f'r{i}'} for i in range(3)]])
def test_malformed_batch_response_does_not_report_full_success(records):
    table, api = make_table([{'name': 'name', 'type': 1, 'description': None}])
    api.batch_create.return_value = BatchCreateAppTableRecordResponse({
        'code': 0, 'data': {'records': records},
    })
    result = V2Table(table).upsert(pd.DataFrame({'name': ['a', 'b']}), insert_only=True)
    assert not result.ok and result.failed > 0
    assert result.errors[0].message == 'incomplete batch response'


def test_upsert_counts_creates_and_updates_separately():
    table, api = make_table([{'name': 'name', 'type': 1, 'description': 'Primary1'}])
    api.search.return_value = SearchAppTableRecordResponse({'code': 0, 'data': {
        'items': [{'record_id': 'r1', 'fields': {'name': 'a'}}], 'has_more': False,
    }})
    api.batch_update.return_value = BatchUpdateAppTableRecordResponse({
        'code': 0, 'data': {'records': [{'record_id': 'r1'}]},
    })
    api.batch_create.return_value = BatchCreateAppTableRecordResponse({
        'code': 0, 'data': {'records': [{'record_id': 'r2'}]},
    })
    result = V2Table(table).upsert(pd.DataFrame({'name': ['a', 'b', 'a']}))
    assert (result.created, result.updated, result.skipped, result.failed) == (1, 1, 1, 0)
    assert result.succeeded == 2
    api.batch_update.return_value = BatchUpdateAppTableRecordResponse({'code': 999, 'msg': 'denied'})
    result = V2Table(table).upsert(pd.DataFrame({'name': ['a', 'b']}))
    assert (result.created, result.updated, result.failed) == (1, 0, 1)
    assert result.errors[0].code == 999


def test_ambiguous_remote_primary_keys_fail_before_writing():
    table, api = make_table([{'name': 'name', 'type': 1, 'description': 'Primary1'}])
    api.search.return_value = SearchAppTableRecordResponse({'code': 0, 'data': {
        'items': [{'record_id': 'r1', 'fields': {'name': 'a'}},
                  {'record_id': 'r2', 'fields': {'name': 'a'}}], 'has_more': False,
    }})
    with pytest.raises(ApiError, match='multiple records'):
        V2Table(table).upsert(pd.DataFrame({'name': ['a']}))
    api.batch_update.assert_not_called()
    api.batch_create.assert_not_called()
