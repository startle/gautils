"""Regression checks for typed, connection-scoped query caches."""
from unittest.mock import MagicMock
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pandas as pd

from gautils.mysqldb import connect_mysql


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
