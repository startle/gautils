import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

from gautils.table import KVTable, KEY_ENCODE_MD5, KEY_ENCODE_JSON


class TestKeyEncode(unittest.TestCase):
    def test_key_encode_md5(self):
        sr = pd.Series({'col1': 'value1', 'col2': 123})
        result = KEY_ENCODE_MD5(sr)
        self.assertEqual(len(result), 32)

    def test_key_encode_json(self):
        sr = pd.Series({'col1': 'value1', 'col2': 123})
        result = KEY_ENCODE_JSON(sr)
        expected = sr.to_json()
        self.assertEqual(result, expected)

    def test_key_encode_md5_consistency(self):
        sr1 = pd.Series({'col1': 'value1', 'col2': 123})
        sr2 = pd.Series({'col1': 'value1', 'col2': 123})
        self.assertEqual(KEY_ENCODE_MD5(sr1), KEY_ENCODE_MD5(sr2))

    def test_key_encode_md5_different_values(self):
        sr1 = pd.Series({'col1': 'value1', 'col2': 123})
        sr2 = pd.Series({'col1': 'value2', 'col2': 123})
        self.assertNotEqual(KEY_ENCODE_MD5(sr1), KEY_ENCODE_MD5(sr2))


class TestKVTable(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.kv_table = KVTable(
            db=self.mock_db,
            table='test_table',
            name='test_name',
            keys=['key1', 'key2'],
            key_encode=KEY_ENCODE_MD5
        )

    def test_init(self):
        self.assertEqual(self.kv_table._db, self.mock_db)
        self.assertEqual(self.kv_table._table, 'test_table')
        self.assertEqual(self.kv_table._name, 'test_name')
        self.assertEqual(self.kv_table._keys, ['key1', 'key2'])
        self.assertEqual(self.kv_table._key_encode, KEY_ENCODE_MD5)

    def test_init_default_key_encode(self):
        kv_table = KVTable(
            db=self.mock_db,
            table='test_table',
            name='test_name',
            keys=['key1']
        )
        self.assertEqual(kv_table._key_encode, KEY_ENCODE_MD5)

    def test_insert_empty_df(self):
        result = self.kv_table.insert(pd.DataFrame())
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    def test_insert_none_df(self):
        result = self.kv_table.insert(None)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    def test_insert_with_new_data(self):
        self.mock_db.query.return_value = pd.DataFrame(columns=['keys'])
        self.mock_db.update.return_value = 2

        df = pd.DataFrame({
            'key1': ['a', 'b'],
            'key2': [1, 2],
            'value': ['x', 'y']
        })

        df_insert, df_update = self.kv_table.insert(df)

        self.mock_db.update.assert_called_once()

    def test_insert_with_existing_data(self):
        existing_key = KEY_ENCODE_MD5(pd.Series({'key1': 'a', 'key2': 1}))
        self.mock_db.query.return_value = pd.DataFrame({'keys': [existing_key]})
        self.mock_db.update.return_value = 2

        df = pd.DataFrame({
            'key1': ['a', 'b'],
            'key2': [1, 2],
            'value': ['new_x', 'y']
        })

        df_insert, df_update = self.kv_table.insert(df)

        self.mock_db.update.assert_called_once()


if __name__ == '__main__':
    unittest.main()
