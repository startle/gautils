import os
import tempfile
import unittest
import math
import time

import pandas as pd
import numpy as np

from gautils.utils import (
    convert_url_to_windows_filename,
    batch_split,
    read_dicts,
    url_parse_unquote,
    md5,
    floor,
    ceil,
    binsearch,
    list_files,
    read_lines,
    write_lines,
    singleton,
)


class TestConvertUrlToWindowsFilename(unittest.TestCase):
    def test_basic_replacement(self):
        url = 'https://example.com/path/to/file?name=value'
        result = convert_url_to_windows_filename(url)
        self.assertNotIn('<', result)
        self.assertNotIn('>', result)
        self.assertNotIn(':', result)
        self.assertNotIn('?', result)

    def test_all_invalid_chars(self):
        url = 'a<b>c:d/e\\f|g?h*i"j'
        result = convert_url_to_windows_filename(url)
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            self.assertNotIn(char, result)


class TestBatchSplit(unittest.TestCase):
    def test_list_split(self):
        data = list(range(10))
        batches = list(batch_split(data, 3))
        self.assertEqual(len(batches), 4)
        self.assertEqual(list(batches[0]), [0, 1, 2])
        self.assertEqual(list(batches[1]), [3, 4, 5])
        self.assertEqual(list(batches[2]), [6, 7, 8])
        self.assertEqual(list(batches[3]), [9])

    def test_dataframe_split(self):
        df = pd.DataFrame({'a': range(10)})
        batches = list(batch_split(df, 3))
        self.assertEqual(len(batches), 4)
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[3]), 1)


class TestReadDicts(unittest.TestCase):
    def test_basic_parse(self):
        s = 'key1=value1&key2=value2'
        result = read_dicts(s)
        self.assertEqual(result['key1'], 'value1')
        self.assertEqual(result['key2'], 'value2')

    def test_url_encoded(self):
        s = 'name=hello%20world&key=test%26value'
        result = read_dicts(s)
        self.assertEqual(result['name'], 'hello world')
        self.assertEqual(result['key'], 'test&value')

    def test_empty_string(self):
        result = read_dicts('')
        self.assertEqual(result, {})

    def test_custom_split(self):
        s = 'key1=value1;key2=value2'
        result = read_dicts(s, split=';')
        self.assertEqual(result['key1'], 'value1')
        self.assertEqual(result['key2'], 'value2')

    def test_no_parse_unquote(self):
        s = 'key=hello%20world'
        result = read_dicts(s, parse_unquote=None)
        self.assertEqual(result['key'], 'hello%20world')


class TestUrlParseUnquote(unittest.TestCase):
    def test_basic_unquote(self):
        self.assertEqual(url_parse_unquote('hello%20world'), 'hello world')
        self.assertEqual(url_parse_unquote('test%26value'), 'test&value')

    def test_empty_string(self):
        self.assertEqual(url_parse_unquote(''), '')

    def test_none_input(self):
        self.assertEqual(url_parse_unquote(None), '')


class TestMd5(unittest.TestCase):
    def test_single_value(self):
        result = md5('test')
        self.assertEqual(len(result), 16)

    def test_multiple_values(self):
        result1 = md5('a', 'b', 'c')
        result2 = md5('a', 'b', 'c')
        self.assertEqual(result1, result2)

    def test_consistency(self):
        result1 = md5('test_value')
        result2 = md5('test_value')
        self.assertEqual(result1, result2)


class TestFloor(unittest.TestCase):
    def test_positive_number(self):
        self.assertEqual(floor(123.456, 1), 123.4)
        self.assertEqual(floor(123.456, 0), 123.0)
        self.assertEqual(floor(123.456, -1), 120.0)

    def test_negative_number(self):
        self.assertEqual(floor(-123.456, 1), -123.5)


class TestCeil(unittest.TestCase):
    def test_positive_number(self):
        self.assertEqual(ceil(123.456, 1), 123.5)
        self.assertEqual(ceil(123.456, 0), 124.0)
        self.assertEqual(ceil(123.456, -1), 130.0)

    def test_negative_number(self):
        self.assertEqual(ceil(-123.456, 1), -123.4)


class TestBinsearch(unittest.TestCase):
    def test_found(self):
        data = [1, 3, 5, 7, 9, 11, 13]
        self.assertEqual(binsearch(data, 5), 2)
        self.assertEqual(binsearch(data, 1), 0)
        self.assertEqual(binsearch(data, 13), 6)

    def test_not_found(self):
        data = [1, 3, 5, 7, 9]
        self.assertEqual(binsearch(data, 0), -1)
        self.assertEqual(binsearch(data, 100), -1)

    def test_with_key_function(self):
        data = [{'val': 1}, {'val': 3}, {'val': 5}]
        result = binsearch(data, 3, key_f=lambda x: x['val'])
        self.assertEqual(result, 1)

    def test_two_elements(self):
        data = [10, 20]
        self.assertEqual(binsearch(data, 10), 0)
        self.assertEqual(binsearch(data, 20), 1)


class TestListFiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.file1 = os.path.join(self.temp_dir, 'file1.txt')
        self.file2 = os.path.join(self.temp_dir, 'file2.txt')
        open(self.file1, 'w').close()
        open(self.file2, 'w').close()

    def tearDown(self):
        for f in [self.file1, self.file2]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.temp_dir)

    def test_list_files(self):
        files = list_files(self.temp_dir)
        self.assertEqual(len(files), 2)
        self.assertIn(self.file1, files)
        self.assertIn(self.file2, files)

    def test_not_recursion(self):
        files = list_files(self.temp_dir, recursion=False)
        self.assertEqual(len(files), 2)

    def test_invalid_dir(self):
        with self.assertRaises(Exception):
            list_files('/nonexistent/path')


class TestReadWriteLines(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.mktemp()

    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def test_write_and_read_lines(self):
        lines = ['line1', 'line2', 'line3']
        write_lines(self.temp_file, lines, mode='w')
        read = list(read_lines(self.temp_file))
        self.assertEqual(read, lines)

    def test_append_lines(self):
        write_lines(self.temp_file, ['line1'], mode='w')
        write_lines(self.temp_file, ['line2'], mode='a')
        read = list(read_lines(self.temp_file))
        self.assertEqual(read, ['line1', 'line2'])

    def test_single_string(self):
        write_lines(self.temp_file, 'single_line', mode='w')
        read = list(read_lines(self.temp_file))
        self.assertEqual(read, ['single_line'])


class TestSingleton(unittest.TestCase):
    def test_singleton(self):
        @singleton
        class TestClass:
            def __init__(self, value):
                self.value = value

        obj1 = TestClass(1)
        obj2 = TestClass(2)
        self.assertIs(obj1, obj2)
        self.assertEqual(obj1.value, 1)


if __name__ == '__main__':
    unittest.main()
