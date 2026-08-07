import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import yaml
import requests

from gautils.web import (
    get_host,
    default_pc_headers,
    default_phone_headers,
    retry_run,
    CookieManager,
    Web,
)


class TestGetHost(unittest.TestCase):
    def test_http_url(self):
        self.assertEqual(get_host('http://www.example.com/path'), 'www.example.com')

    def test_https_url(self):
        self.assertEqual(get_host('https://www.example.com/path'), 'www.example.com')

    def test_url_with_port(self):
        self.assertEqual(get_host('http://www.example.com:8080/path'), 'www.example.com')

    def test_url_without_scheme(self):
        self.assertEqual(get_host('www.example.com/path'), 'www.example.com')

    def test_invalid_url(self):
        self.assertEqual(get_host('not_a_url'), 'not_a_url')


class TestDefaultHeaders(unittest.TestCase):
    def test_pc_headers(self):
        headers = default_pc_headers()
        self.assertIn('user-agent', headers)
        self.assertIn('accept', headers)
        self.assertIn('Chrome', headers['user-agent'])

    def test_phone_headers(self):
        headers = default_phone_headers()
        self.assertIn('User-Agent', headers)
        self.assertIn('Android', headers['User-Agent'])
        self.assertIn('sec-ch-ua-platform', headers)


class TestRetryRun(unittest.TestCase):
    @patch('gautils.web.logging.error')
    def test_success_on_first_try(self, mock_log_error):
        mock_func = MagicMock(return_value='success')
        result = retry_run(mock_func, retry_times=3, sleep_s=0.01)
        self.assertEqual(result, 'success')
        self.assertEqual(mock_func.call_count, 1)
        mock_log_error.assert_not_called()

    @patch('gautils.web.logging.error')
    def test_success_after_retry(self, mock_log_error):
        mock_func = MagicMock(side_effect=[Exception('error'), 'success'])
        result = retry_run(mock_func, retry_times=3, sleep_s=0.01)
        self.assertEqual(result, 'success')
        self.assertEqual(mock_func.call_count, 2)
        mock_log_error.assert_called_once()

    @patch('gautils.web.logging.error')
    def test_all_retries_failed(self, mock_log_error):
        mock_func = MagicMock(side_effect=Exception('error'))
        with self.assertRaises(RuntimeError) as ctx:
            retry_run(mock_func, retry_times=3, sleep_s=0.01)
        self.assertIn('exhausted 3 retries', str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, Exception)
        self.assertEqual(mock_func.call_count, 3)
        self.assertEqual(mock_log_error.call_count, 3)

    def test_with_params(self):
        mock_func = MagicMock(return_value='success')
        result = retry_run(mock_func, 'arg1', 'arg2', key='value', retry_times=3, sleep_s=0.01)
        mock_func.assert_called_once_with('arg1', 'arg2', key='value')


class TestCookieManager(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.mktemp(suffix='.yml')

    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def test_load_empty_cookies(self):
        manager = CookieManager(self.temp_file)
        self.assertEqual(manager.cookies, {})

    def test_load_existing_cookies(self):
        cookies = {'example.com': {'session': 'abc123'}}
        with open(self.temp_file, 'w', encoding='utf8') as f:
            yaml.dump(cookies, f)

        manager = CookieManager(self.temp_file)
        self.assertEqual(manager.cookies, cookies)

    def test_load_blank_or_null_file_returns_empty_dict(self):
        # 文件存在但内容为空/null/纯注释：yaml.safe_load 返回 None，
        # 必须兜底成 {}，否则后续 .get 会 AttributeError（生产事故根因）
        for body in ('', 'null\n', '# only comment\n'):
            with open(self.temp_file, 'w', encoding='utf8') as f:
                f.write(body)
            self.assertEqual(
                CookieManager(self.temp_file).cookies, {},
                f'blank/null cookie file should degrade to {{}}, body={body!r}',
            )

    def test_update_and_save_cookies(self):
        manager = CookieManager(self.temp_file)
        manager.update_cookies({'newsite.com': {'token': 'xyz'}})

        manager2 = CookieManager(self.temp_file)
        self.assertIn('newsite.com', manager2.cookies)

    def test_save_cookies_writes_valid_dict(self):
        # 原子写落盘的应是可被 safe_load 还原为 dict 的有效 YAML，
        # 不能是 null/半成品（否则下次 load 又锁死）
        manager = CookieManager(self.temp_file)
        manager.cookies = {'host.com': {'k': 'v'}}
        manager.save_cookies()
        with open(self.temp_file, 'r', encoding='utf8') as f:
            self.assertEqual(yaml.safe_load(f), {'host.com': {'k': 'v'}})


class TestWeb(unittest.TestCase):
    def setUp(self):
        self.temp_cookie_file = tempfile.mktemp(suffix='.yml')

    def tearDown(self):
        if os.path.exists(self.temp_cookie_file):
            os.remove(self.temp_cookie_file)

    @patch('gautils.web.requests.Session')
    @patch('gautils.web.retry_run2')
    def test_get_request(self, mock_retry, mock_session_class):
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.text = 'response content'
        mock_response.apparent_encoding = 'utf-8'
        mock_retry.return_value = mock_response

        web = Web(cookies_filepath=self.temp_cookie_file)
        result = web.get('http://example.com')

        self.assertEqual(result, 'response content')
        mock_retry.assert_called_once()
        mock_response.raise_for_status.assert_not_called()

    @patch('gautils.web.requests.Session')
    @patch('gautils.web.retry_run2')
    def test_post_request(self, mock_retry, mock_session_class):
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.text = 'post response'
        mock_response.apparent_encoding = 'utf-8'
        mock_retry.return_value = mock_response

        web = Web(cookies_filepath=self.temp_cookie_file)
        result = web.post('http://example.com', data={'key': 'value'})

        self.assertEqual(result, 'post response')
        mock_retry.assert_called_once()

    @patch('gautils.web.requests.Session')
    @patch('gautils.web.retry_run2')
    def test_get_keeps_legacy_http_error_body_by_default(self, mock_retry, mock_session_class):
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.text = 'not found'
        mock_response.apparent_encoding = 'utf-8'
        mock_response.raise_for_status.side_effect = requests.HTTPError('404')
        mock_retry.return_value = mock_response

        web = Web(cookies_filepath=self.temp_cookie_file)
        result = web.get('http://example.com/missing')

        self.assertEqual(result, 'not found')
        mock_response.raise_for_status.assert_not_called()

    @patch('gautils.web.requests.Session')
    @patch('gautils.web.retry_run2')
    def test_get_can_raise_for_http_error(self, mock_retry, mock_session_class):
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.text = 'server error'
        mock_response.apparent_encoding = 'utf-8'
        mock_response.raise_for_status.side_effect = requests.HTTPError('500')
        mock_retry.return_value = mock_response

        web = Web(cookies_filepath=self.temp_cookie_file)
        with self.assertRaises(requests.HTTPError):
            web.get('http://example.com/error', raise_for_status=True)

        mock_response.raise_for_status.assert_called_once()

    @patch('gautils.web.requests.Session')
    @patch('gautils.web.retry_run2')
    def test_post_can_raise_for_http_error(self, mock_retry, mock_session_class):
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.text = 'server error'
        mock_response.apparent_encoding = 'utf-8'
        mock_response.raise_for_status.side_effect = requests.HTTPError('500')
        mock_retry.return_value = mock_response

        web = Web(cookies_filepath=self.temp_cookie_file)
        with self.assertRaises(requests.HTTPError):
            web.post('http://example.com/error', data={'key': 'value'}, raise_for_status=True)

        mock_response.raise_for_status.assert_called_once()

    def test_parse_url(self):
        web = Web(cookies_filepath=self.temp_cookie_file)
        domain, path, params = web.parse_url('https://example.com/path?key1=value1&key2=value2')

        self.assertEqual(domain, 'example.com')
        self.assertEqual(path, '/path')
        self.assertEqual(params['key1'], 'value1')
        self.assertEqual(params['key2'], 'value2')

    def test_parse_url_single_value(self):
        web = Web(cookies_filepath=self.temp_cookie_file)
        domain, path, params = web.parse_url('http://test.com/api?name=john')

        self.assertEqual(domain, 'test.com')
        self.assertEqual(path, '/api')
        self.assertEqual(params['name'], 'john')


if __name__ == '__main__':
    unittest.main()
