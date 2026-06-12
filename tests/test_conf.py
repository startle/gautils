import os
import tempfile
import unittest

from gautils.conf import Conf


class TestConf(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.conf_path = os.path.join(self.temp_dir, 'test_conf.yml')

    def tearDown(self):
        if os.path.exists(self.conf_path):
            os.remove(self.conf_path)
        os.rmdir(self.temp_dir)

    def _create_conf_file(self, content):
        with open(self.conf_path, 'w', encoding='utf8') as f:
            f.write(content)

    def test_basic_get(self):
        self._create_conf_file('''
db:
  host: localhost
  port: 3306
''')
        conf = Conf(self.conf_path)
        self.assertEqual(conf.get(['db', 'host']), 'localhost')
        self.assertEqual(conf.get(['db', 'port']), 3306)

    def test_get_with_default(self):
        self._create_conf_file('''
db:
  host: localhost
''')
        conf = Conf(self.conf_path)
        self.assertEqual(conf.get(['db', 'host']), 'localhost')
        self.assertEqual(conf.get(['db', 'port'], default='3306'), '3306')

    def test_get_not_found_raise(self):
        self._create_conf_file('''
db:
  host: localhost
''')
        conf = Conf(self.conf_path)
        with self.assertRaises(ValueError):
            conf.get(['db', 'not_exist'])

    def test_get_int(self):
        self._create_conf_file('''
db:
  port: 3306
  timeout: 30.5
''')
        conf = Conf(self.conf_path)
        self.assertEqual(conf.get_int(['db', 'port']), 3306)
        self.assertEqual(conf.get_int(['db', 'timeout']), 30)

    def test_get_float(self):
        self._create_conf_file('''
db:
  timeout: 30.5
''')
        conf = Conf(self.conf_path)
        self.assertEqual(conf.get_float(['db', 'timeout']), 30.5)

    def test_get_bool(self):
        self._create_conf_file('''
db:
  ssl: true
  debug: false
''')
        conf = Conf(self.conf_path)
        self.assertTrue(conf.get_bool(['db', 'ssl']))
        self.assertFalse(conf.get_bool(['db', 'debug']))

    def test_get_bool_string_falsy(self):
        self._create_conf_file('''
flags:
  a: "false"
  b: "False"
  c: "0"
  d: "no"
  e: "off"
  f: ""
  g: "true"
  h: "1"
  i: "yes"
''')
        conf = Conf(self.conf_path)
        for key in ('a', 'b', 'c', 'd', 'e', 'f'):
            self.assertFalse(conf.get_bool(['flags', key]), f'flags.{key} should be False')
        for key in ('g', 'h', 'i'):
            self.assertTrue(conf.get_bool(['flags', key]), f'flags.{key} should be True')

    def test_get_dict(self):
        self._create_conf_file('''
db:
  host: localhost
  port: 3306
  settings:
    charset: utf8
    timeout: 30
''')
        conf = Conf(self.conf_path)
        db_dict = conf.get_dict(['db'])
        self.assertEqual(db_dict['host'], 'localhost')
        self.assertEqual(db_dict['port'], 3306)

    def test_get_dict_not_dict_raise(self):
        self._create_conf_file('''
db: localhost
''')
        conf = Conf(self.conf_path)
        with self.assertRaises(ValueError):
            conf.get_dict(['db'])

    def test_string_trim(self):
        self._create_conf_file('''
db:
  host: "  localhost  "
''')
        conf = Conf(self.conf_path)
        self.assertEqual(conf.get(['db', 'host']), 'localhost')

    def test_nested_path(self):
        self._create_conf_file('''
level1:
  level2:
    level3:
      value: deep_value
''')
        conf = Conf(self.conf_path)
        self.assertEqual(conf.get(['level1', 'level2', 'level3', 'value']), 'deep_value')


if __name__ == '__main__':
    unittest.main()
