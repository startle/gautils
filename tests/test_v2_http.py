import unittest
from unittest.mock import MagicMock, patch

import requests

from gautils.v2.errors import HttpRequestError
from gautils.v2.http import HttpClient, RetryPolicy, Timeout


class TestTimeoutAndRetry(unittest.TestCase):
    def test_timeout_validation(self):
        self.assertEqual(Timeout(1, 2).as_requests(), (1, 2))
        for timeout in (Timeout(0, 1), Timeout(1, 0)):
            with self.assertRaises(ValueError):
                timeout.as_requests()

    def test_retry_policy(self):
        retry = RetryPolicy(total=2).build()
        self.assertEqual(retry.total, 2)
        self.assertIn('GET', retry.allowed_methods)
        self.assertNotIn('POST', retry.allowed_methods)
        with self.assertRaises(ValueError):
            RetryPolicy(total=-1).build()


class TestHttpClient(unittest.TestCase):
    def setUp(self):
        self.session = MagicMock(spec=requests.Session)
        self.session.headers = {}

    def test_request_defaults_proxy_and_verbs(self):
        response = MagicMock()
        response.text = 'body'
        response.json.return_value = {'ok': True}
        self.session.request.return_value = response
        client = HttpClient(
            session=self.session,
            timeout=Timeout(1, 2),
            headers={'X-Test': '1'},
            proxies_provider=lambda: {'https': 'proxy'},
        )
        self.assertEqual(self.session.headers['X-Test'], '1')
        self.assertIs(client.session, self.session)

        self.assertIs(client.get('https://example.com'), response)
        self.session.request.assert_called_with(
            'GET', 'https://example.com', timeout=(1, 2), verify=True,
            proxies={'https': 'proxy'})
        response.raise_for_status.assert_called_once()
        self.assertIs(client.post('u'), response)
        self.assertIs(client.put('u'), response)
        self.assertIs(client.patch('u'), response)
        self.assertIs(client.delete('u'), response)
        self.assertEqual(client.get_text('u'), 'body')
        self.assertEqual(client.get_json('u'), {'ok': True})

    def test_explicit_options_and_no_raise(self):
        response = MagicMock()
        self.session.request.return_value = response
        client = HttpClient(session=self.session, retry=None, raise_for_status=False)
        self.assertIs(
            client.request('get', 'u', timeout=1, verify=False, proxies={}), response)
        self.session.request.assert_called_once_with(
            'GET', 'u', timeout=1, verify=False, proxies={})
        response.raise_for_status.assert_not_called()

    def test_request_error_is_wrapped(self):
        response = MagicMock(status_code=503)
        error = requests.HTTPError('bad', response=response)
        self.session.request.side_effect = error
        client = HttpClient(session=self.session, retry=None)
        with self.assertRaises(HttpRequestError) as caught:
            client.get('https://example.com')
        self.assertEqual(caught.exception.method, 'GET')
        self.assertEqual(caught.exception.url, 'https://example.com')
        self.assertIs(caught.exception.response, response)
        self.assertIs(caught.exception.__cause__, error)

    @patch('gautils.v2.http.requests.Session')
    def test_owned_session_context_and_closed_guard(self, session_class):
        session = session_class.return_value
        session.headers = {}
        with HttpClient(retry=None) as client:
            self.assertIs(client.session, session)
        session.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            client.get('u')
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            client.__enter__()
        client.close()

    def test_injected_session_is_not_closed(self):
        client = HttpClient(session=self.session, retry=None)
        client.close()
        self.session.close.assert_not_called()


if __name__ == '__main__':
    unittest.main()
