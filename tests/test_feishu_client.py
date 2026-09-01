import unittest
from unittest.mock import MagicMock, patch

from gautils.feishu.core.client import FeishuClient
from tests.feishu_fakes import obj, response


class TestFeishuClient(unittest.TestCase):
    def _client(self, **kwargs):
        sdk_client = MagicMock()
        builder = MagicMock()
        builder.app_id.return_value = builder
        builder.app_secret.return_value = builder
        builder.log_level.return_value = builder
        builder.build.return_value = sdk_client
        with patch('gautils.feishu.core.client.lark.Client.builder', return_value=builder):
            client = FeishuClient(app_id='app-id', app_secret='secret', **kwargs)
        self.assertIs(client.client, sdk_client)
        return client

    @patch('gautils.feishu.core.client.time.time', return_value=1000)
    def test_tenant_token_success_and_cache(self, _mock_time):
        client = self._client()
        api = client.client.auth.v3.tenant_access_token.internal
        api.return_value = response(content={'tenant_access_token': 'token', 'expire': 3600})

        self.assertEqual(client.get_tenant_access_token(), 'token')
        self.assertEqual(client.get_tenant_access_token(), 'token')
        self.assertEqual(api.call_count, 1)
        self.assertEqual(client._token_expire_at, 4300)

    def test_tenant_token_failure_and_invalid_content(self):
        client = self._client()
        api = client.client.auth.v3.tenant_access_token.internal
        api.return_value = response(success=False, code=1, msg='denied')
        self.assertIsNone(client.get_tenant_access_token())

        api.return_value = response(content='{not-json')
        self.assertIsNone(client.get_tenant_access_token())

    def test_app_info_success_and_cache(self):
        client = self._client(open_id='explicit-open-id')
        api = client.client.application.v6.application.list
        api.return_value = response(content={'data': {'items': [
            {'app_id': 'other'},
            {'app_id': 'app-id', 'open_id': 'bot-id', 'name': 'bot',
             'description': 'desc', 'avatar_url': 'avatar'},
        ]}})

        info = client.get_app_info()
        self.assertEqual(info['open_id'], 'bot-id')
        self.assertEqual(client.get_app_info(), info)
        self.assertEqual(api.call_count, 1)
        self.assertEqual(client.open_id, 'explicit-open-id')

    def test_app_info_failure_invalid_content_and_not_found(self):
        client = self._client()
        api = client.client.application.v6.application.list
        api.return_value = response(success=False)
        self.assertEqual(client.get_app_info(), {})

        api.return_value = response(content='{bad-json')
        self.assertEqual(client.get_app_info(), {})

        api.return_value = response(content={'data': {'items': [{'app_id': 'other'}]}})
        self.assertEqual(client.get_app_info(), {})


if __name__ == '__main__':
    unittest.main()
