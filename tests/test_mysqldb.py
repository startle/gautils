import unittest
from unittest.mock import patch

import numpy as np

from gautils.mysqldb import And, DbAlchemy


class TestAnd(unittest.TestCase):

    def test_vin_int_values_are_native_python_types(self):
        """vin 整型参数必须是 Python 原生 int，numpy 标量 mysql-connector 无法转换"""
        a = And()
        a.vin('id', [3, 1, 2, 2])
        self.assertEqual(a.to_sql().strip(), '`id` IN(%s,%s,%s)')
        params = a.params()
        self.assertEqual(params, [1, 2, 3])
        for p in params:
            self.assertIs(type(p), int, f'param {p!r} should be native int, got {type(p)}')

    def test_vin_str_values(self):
        a = And()
        a.vin('vin_code', ['b', 'a'])
        self.assertEqual(a.to_sql().strip(), '`vin_code` IN(%s,%s)')
        params = a.params()
        self.assertEqual(params, ['a', 'b'])
        for p in params:
            self.assertIs(type(p), str)

    def test_vin_numpy_array_input(self):
        a = And()
        a.vin('id', np.array([5, 4]))
        params = a.params()
        self.assertEqual(params, [4, 5])
        for p in params:
            self.assertIs(type(p), int)

    def test_vin_empty_raises(self):
        with self.assertRaises(ValueError):
            And().vin('id', [])

    def test_cond_params_accumulate(self):
        a = And()
        a.cond('`a`=%s', 1).cond('`b`=%s', 2)
        self.assertEqual(a.params(), [1, 2])
        self.assertEqual(a.to_sql().strip(), '`a`=%s AND `b`=%s')

    def test_eq_and_between(self):
        a = And()
        a.eq('x', 7).between('d', '2024-01-01', '2024-12-31')
        self.assertEqual(a.params(), [7, '2024-01-01', '2024-12-31'])


class TestDbAlchemy(unittest.TestCase):

    @patch('gautils.mysqldb.create_engine')
    def test_default_driver_remains_pymysql(self, mock_create_engine):
        DbAlchemy('u', 'p', 'h', 3306, 'd', charset='utf8')

        args, kwargs = mock_create_engine.call_args
        self.assertTrue(args[0].startswith('mysql+pymysql://'))
        self.assertIn('charset=utf8', args[0])
        self.assertEqual(kwargs['connect_args'], {})

    @patch('gautils.mysqldb.create_engine')
    def test_mysqldb_compress_uses_connect_args(self, mock_create_engine):
        DbAlchemy('u', 'p', 'h', 3306, 'd', driver='mysqldb', compress=True, charset='utf8')

        args, kwargs = mock_create_engine.call_args
        self.assertTrue(args[0].startswith('mysql+mysqldb://'))
        self.assertIn('charset=utf8', args[0])
        self.assertEqual(kwargs['connect_args'], {'compress': True})


if __name__ == '__main__':
    unittest.main()
