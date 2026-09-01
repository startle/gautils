import gc
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from mysql.connector import FieldType

from gautils.mysqldb import (
    And, ConnType, DbAlchemy, MysqlDb, MysqlDbImpl, MysqlQuery,
    _db_field_type2dtype, connect_mysql,
)


class TestMysqlQueryExtended(unittest.TestCase):
    def test_empty_and_query_builder_paths(self):
        where = And()
        self.assertTrue(where.empty())
        self.assertEqual(where.to_sql(), '')

        db = MagicMock()
        db.query.return_value = pd.DataFrame({'id': [1]})
        query = MysqlQuery(db, 'users', ['id'])
        self.assertIs(query.and_cond('`active`=%s', 1), query)
        self.assertIs(query.and_in('id', [2, 1]), query)
        self.assertIs(query.and_eq('name', 'gau'), query)
        self.assertEqual(query.query()['id'].tolist(), [1])
        sql, *params = db.query.call_args.args
        self.assertIn('SELECT `id` FROM users WHERE', sql)
        self.assertEqual(params, [1, 1, 2, 'gau'])

        db.reset_mock()
        MysqlQuery(db, 'users').query()
        db.query.assert_called_once_with('SELECT * FROM users')

    def test_query_logs_and_reraises(self):
        db = MagicMock()
        db.query.side_effect = RuntimeError('db failed')
        with patch('gautils.mysqldb.logging.error') as log:
            with self.assertRaisesRegex(RuntimeError, 'db failed'):
                MysqlQuery(db, 'users').query()
        log.assert_called_once()


class TestMysqlDbBase(unittest.TestCase):
    def test_abstract_operations_and_helpers(self):
        db = MysqlDb()
        for call in (
            lambda: db.s_query('t'), lambda: db.update('t', pd.DataFrame()),
            lambda: db.query('select 1'), lambda: db.execute('select 1'),
        ):
            with self.assertRaises(NotImplementedError):
                call()

        db.query = MagicMock(return_value=pd.DataFrame([
            {'Field': 'id', 'Key': 'PRI'}, {'Field': 'name', 'Key': ''},
        ]))
        self.assertEqual(db.keys_cols('users'), (['id'], ['name']))
        self.assertEqual(db.describe('users').iloc[0]['Field'], 'id')
        db.execute = MagicMock(return_value=1)
        self.assertEqual(db.delete_all('users'), 1)
        db.execute.assert_called_with('TRUNCATE TABLE users')
        self.assertIsNone(db.close())

    def test_field_type_mapping(self):
        self.assertEqual(_db_field_type2dtype(FieldType.VARCHAR), 'object')
        self.assertEqual(str(_db_field_type2dtype(FieldType.LONG)), 'Int64')
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            _db_field_type2dtype(999999)


class TestMysqlDbImpl(unittest.TestCase):
    def setUp(self):
        self.db = MysqlDbImpl('host', 3306, 'user', 'pwd', 'db', charset='utf8')

    def _connection(self):
        cm = MagicMock()
        conn = cm.__enter__.return_value
        self.db._get_conn = MagicMock(return_value=cm)
        return conn

    @patch('gautils.mysqldb.mysql.connector.connect')
    def test_connection_success_failure_and_identity(self, connect):
        connect.return_value = 'connection'
        self.assertEqual(self.db._get_conn(), 'connection')
        connect.assert_called_once_with(
            host='host', port=3306, user='user', passwd='pwd', database='db', charset='utf8')
        self.assertEqual(str(self.db), 'db[host:3306]')
        self.assertIsInstance(self.db.s_query('users'), MysqlQuery)

        connect.side_effect = OSError('offline')
        with patch('gautils.mysqldb.logging.error'):
            with self.assertRaisesRegex(OSError, 'offline'):
                self.db._get_conn()

    def test_update_empty_validation_success_and_failure(self):
        self.assertEqual(self.db._update_each_batch('users', pd.DataFrame()), 0)
        self.db.keys_cols = MagicMock(return_value=(['id'], ['name']))
        with self.assertRaisesRegex(ValueError, 'update cols'):
            self.db._update_each_batch('users', pd.DataFrame({'id': [1], 'other': ['x']}))
        self.db.keys_cols.return_value = ([], ['name'])
        with self.assertRaisesRegex(ValueError, 'index cols'):
            self.db._update_each_batch('users', pd.DataFrame({'name': ['x']}))

        self.db.keys_cols.return_value = (['id'], ['name'])
        conn = self._connection()
        cursor = conn.cursor.return_value
        cursor.rowcount = 2
        frame = pd.DataFrame({'id': [1, 2], 'name': ['a', None]})
        self.assertEqual(self.db._update_each_batch('users', frame), 2)
        cursor.executemany.assert_called_once()
        conn.commit.assert_called_once()
        cursor.close.assert_called_once()

        self.db._get_conn.side_effect = RuntimeError('write failed')
        with patch('gautils.mysqldb.logging.error'):
            with self.assertRaisesRegex(RuntimeError, 'write failed'):
                self.db._update_each_batch('users', frame)

    def test_update_batches(self):
        self.db._update_each_batch = MagicMock(side_effect=lambda _table, part: len(part))
        frame = pd.DataFrame({'id': range(10001)}).set_index('id')
        self.assertEqual(self.db.update('users', frame), 10001)
        self.assertEqual(self.db._update_each_batch.call_count, 2)

    def test_execute(self):
        conn = self._connection()
        cursor = conn.cursor.return_value
        cursor.rowcount = 3
        self.db._after_execute = MagicMock()
        self.assertEqual(self.db.execute('DELETE FROM users WHERE id=%s', 1), 3)
        cursor.execute.assert_called_once_with('DELETE FROM users WHERE id=%s', (1,))
        conn.commit.assert_called_once()
        self.db._after_execute.assert_called_once()

    def test_query_decodes_bytes_and_applies_types(self):
        conn = self._connection()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [(1, bytearray(b'gau'))]
        cursor.description = [('id', FieldType.LONG), ('name', FieldType.VARCHAR)]
        self.db._after_execute = MagicMock()
        result = self.db.query('SELECT id,name FROM users WHERE id=%s', 1)
        self.assertEqual(result.iloc[0].to_dict(), {'id': 1, 'name': 'gau'})
        cursor.execute.assert_called_once_with('SELECT id,name FROM users WHERE id=%s', (1,))
        self.db._after_execute.assert_called_once()
        self.assertIsNone(self.db.close())

    def test_batch_section_paths(self):
        small = pd.DataFrame({'x': [1]})
        self.assertEqual(len(list(MysqlDbImpl._section_batch(small, 10))), 1)
        large = pd.DataFrame({'x': range(5)})
        self.assertEqual([len(x) for x in MysqlDbImpl._section_batch(large, 2)], [2, 2, 1])


class TestDbAlchemyExtended(unittest.TestCase):
    def _db(self):
        with patch('gautils.mysqldb.create_engine') as create:
            db = DbAlchemy('user', 'pwd', 'host', 3306, 'db', compress=True, charset='utf8')
        return db

    def _connection(self, db):
        cm = MagicMock()
        conn = cm.__enter__.return_value
        db.engine.connect.return_value = cm
        return conn

    @patch('gautils.mysqldb.create_engine')
    def test_constructor_query_and_unsupported_builder(self, create):
        db = DbAlchemy('user', 'p@ss', 'host', 3306, 'db', compress=True, charset='utf8')
        self.assertIn('compress=true', create.call_args.args[0])
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            db.s_query('users')

        conn = self._connection(db)
        result = conn.execute.return_value.__enter__.return_value
        result.fetchall.return_value = [(1, 'gau')]
        result.keys.return_value = ['id', 'name']
        frame = db.query('SELECT * FROM users WHERE id=:id', id=1)
        self.assertEqual(frame.iloc[0]['name'], 'gau')

    def test_update_paths_and_execute(self):
        db = self._db()
        self.assertEqual(db.update('users', pd.DataFrame()), 0)
        db.keys_cols = MagicMock(return_value=(['id'], ['name']))
        conn = self._connection(db)
        result = conn.execute.return_value.__enter__.return_value
        result.rowcount = 2
        frame = pd.DataFrame({'id': [1, 2], 'name': ['a', None], 'ignored': [1, 2]})
        self.assertEqual(db.update('users', frame), 2)
        conn.commit.assert_called_once()

        conn = self._connection(db)
        result = conn.execute.return_value.__enter__.return_value
        result.rowcount = 4
        self.assertEqual(db.execute('UPDATE users SET name=:name', name='x'), 4)
        conn.commit.assert_called_once()


class TestConnectMysql(unittest.TestCase):
    @patch('gautils.mysqldb.MysqlDbImpl')
    def test_common_and_invalid_type(self, impl):
        impl.return_value = MagicMock()
        self.assertIs(connect_mysql('h', 1, 'u', 'p', 'd'), impl.return_value)
        with self.assertRaisesRegex(ValueError, 'conn_type'):
            connect_mysql('h', 1, 'u', 'p', 'd', conn_type='bad')

    @patch('gautils.mysqldb.DbAlchemy')
    def test_alchemy(self, alchemy):
        alchemy.return_value = MagicMock()
        result = connect_mysql('h', 1, 'u', 'p', 'd', conn_type=ConnType.ALCHEMY)
        self.assertIs(result, alchemy.return_value)

    @patch('gautils.mysqldb.MysqlDbImpl')
    def test_file_cache_miss_hit_empty_and_expiry(self, impl):
        with tempfile.TemporaryDirectory() as directory:
            db = MagicMock()
            db.query.side_effect = [pd.DataFrame({'x': [1]}), pd.DataFrame()]
            original_query = db.query
            impl.return_value = db
            cached = connect_mysql('h', 1, 'u', 'p', 'd', is_use_file_cache=True,
                                   cache_dir=directory, cache_ttl=60)
            first = cached.query('SELECT :x', x=1)
            second = cached.query('SELECT :x', x=1)
            self.assertEqual(first.to_dict(), second.to_dict())
            self.assertEqual(original_query.call_count, 1)

            files = os.listdir(directory)
            self.assertEqual(len(files), 1)
            path = os.path.join(directory, files[0])
            old = time.time() - 120
            os.utime(path, (old, old))
            self.assertTrue(cached.query('SELECT :x', x=1).empty)
            self.assertEqual(original_query.call_count, 2)


if __name__ == '__main__':
    unittest.main()
