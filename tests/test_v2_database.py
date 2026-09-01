import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from gautils.v2.database import Database, _batches, _identifier
from gautils.v2.errors import DatabaseOperationError


class TestDatabaseHelpers(unittest.TestCase):
    def test_identifier_and_batches(self):
        self.assertEqual(_identifier('valid_name1'), 'valid_name1')
        for value in ('bad-name', '1bad', 'bad name'):
            with self.assertRaises(ValueError):
                _identifier(value)
        self.assertEqual(
            list(_batches([{'x': 1}, {'x': 2}, {'x': 3}], 2)),
            [[{'x': 1}, {'x': 2}], [{'x': 3}]],
        )
        with self.assertRaises(ValueError):
            list(_batches([], 0))


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.db = Database(self.engine)

    def test_factories(self):
        engine = MagicMock()
        with patch('gautils.v2.database.create_engine', return_value=engine) as create:
            database = Database.from_url('sqlite://', echo=True)
        self.assertIs(database.engine, engine)
        self.assertTrue(database._owns_engine)
        create.assert_called_once_with('sqlite://', echo=True)

        with patch.object(Database, 'from_url', return_value='db') as from_url:
            result = Database.mysql(
                user='u', password='p@ss', host='h', database='d',
                query={'charset': 'utf8'}, pool_size=2)
        self.assertEqual(result, 'db')
        url = from_url.call_args.args[0]
        self.assertEqual(url.drivername, 'mysql+mysqlconnector')
        self.assertEqual(url.password, 'p@ss')
        self.assertEqual(from_url.call_args.kwargs['pool_size'], 2)
        self.assertTrue(from_url.call_args.kwargs['pool_pre_ping'])

    def test_fetch_all_success_and_failure(self):
        connection = self.engine.connect.return_value.__enter__.return_value
        result = connection.execute.return_value
        result.fetchall.return_value = [(1, 'gau')]
        result.keys.return_value = ['id', 'name']
        frame = self.db.fetch_all('SELECT * FROM users WHERE id=:id', {'id': 1})
        self.assertEqual(frame.to_dict('records'), [{'id': 1, 'name': 'gau'}])

        self.engine.connect.side_effect = RuntimeError('offline')
        with self.assertRaises(DatabaseOperationError) as caught:
            self.db.fetch_all('SELECT 1')
        self.assertEqual(caught.exception.operation, 'fetch_all')
        self.assertEqual(caught.exception.statement, 'SELECT 1')

    def test_execute_success_and_failure(self):
        connection = self.engine.begin.return_value.__enter__.return_value
        connection.execute.return_value.rowcount = 3
        self.assertEqual(self.db.execute('UPDATE users SET x=:x', {'x': 1}), 3)
        self.engine.begin.side_effect = RuntimeError('offline')
        with self.assertRaises(DatabaseOperationError) as caught:
            self.db.execute('DELETE FROM users')
        self.assertEqual(caught.exception.operation, 'execute')

    def test_execute_many_empty_batches_and_failure(self):
        empty = self.db.execute_many('UPDATE t SET x=:x', [])
        self.assertTrue(empty.ok)
        self.assertEqual(empty.attempted, 0)

        connection = self.engine.begin.return_value.__enter__.return_value
        connection.execute.side_effect = [
            MagicMock(rowcount=2), MagicMock(rowcount=-1),
        ]
        result = self.db.execute_many(
            'UPDATE t SET x=:x', ({'x': value} for value in range(3)), batch_size=2)
        self.assertEqual(result.attempted, 3)
        self.assertEqual(result.succeeded, 3)
        self.assertEqual(result.updated, 2)
        self.assertEqual(connection.execute.call_count, 2)

        self.engine.begin.side_effect = RuntimeError('offline')
        with self.assertRaises(DatabaseOperationError) as caught:
            self.db.execute_many('UPDATE t SET x=:x', [{'x': 1}])
        self.assertEqual(caught.exception.operation, 'execute_many')

    def test_upsert_dataframe_and_validation(self):
        self.assertTrue(self.db.upsert_dataframe(
            'users', pd.DataFrame(), key_columns=['id']).ok)
        frame = pd.DataFrame({'id': [1, 2], 'name': ['a', None]})
        self.db.execute_many = MagicMock(return_value='result')
        self.assertEqual(
            self.db.upsert_dataframe('users', frame, key_columns=['id'], batch_size=1),
            'result')
        statement, records = self.db.execute_many.call_args.args
        self.assertIn('INSERT INTO `users`', statement)
        self.assertIn('ON DUPLICATE KEY UPDATE', statement)
        self.assertEqual(records, [{'id': 1, 'name': 'a'}, {'id': 2, 'name': None}])

        cases = [
            dict(table='bad-table', key_columns=['id']),
            dict(table='users', key_columns=[]),
            dict(table='users', key_columns=['missing']),
            dict(table='users', key_columns=['id'], update_columns=[]),
            dict(table='users', key_columns=['id'], update_columns=['missing']),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.db.upsert_dataframe(frame=frame, **kwargs)

    def test_context_close_and_closed_guard(self):
        owned = Database(self.engine, owns_engine=True)
        with owned as entered:
            self.assertIs(entered, owned)
        self.engine.dispose.assert_called_once()
        owned.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            owned.fetch_all('SELECT 1')
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            owned.upsert_dataframe(
                'users', pd.DataFrame(), key_columns=['id'])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            owned.__enter__()

        external_engine = MagicMock()
        external = Database(external_engine)
        external.close()
        external_engine.dispose.assert_not_called()


if __name__ == '__main__':
    unittest.main()
