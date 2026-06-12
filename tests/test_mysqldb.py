import unittest

import numpy as np

from gautils.mysqldb import And


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


if __name__ == '__main__':
    unittest.main()
