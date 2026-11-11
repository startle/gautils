import sys
import unittest
from unittest.mock import MagicMock, patch
from lark_oapi.api.wiki.v2 import CreateSpaceNodeResponse

from gautils.feishu.core.feishu import Feishu
from gautils.feishu.core.wiki import Wiki, WikiSpace
from tests.feishu_fakes import obj, response


class TestPinnedLarkSdkCompatibility(unittest.TestCase):
    def test_wiki_builders_match_pinned_sdk(self):
        client = MagicMock()
        space = WikiSpace(client, '123')
        client.wiki.v2.space_node.create.return_value = CreateSpaceNodeResponse({
            'code': 0, 'data': {'node': {'node_token': 'node'}},
        })
        self.assertEqual(space.create_node('Doc', 'docx').id, 'node')

        client.wiki.v2.space_node.move.return_value = response()
        self.assertTrue(space.move_node('node', 'parent'))
        client.wiki.v2.space_node.update_title.return_value = response()
        self.assertTrue(space.update_node_title('node', 'New'))
        client.wiki.v2.space_member.create.return_value = response()
        self.assertTrue(space.add_member('user'))

        wiki = Wiki(client)
        client.wiki.v2.space.create.return_value = response(
            data=obj(space=obj(space_id='456')))
        self.assertEqual(wiki.create_space('Space').space_id, '456')

    def test_bot_open_id_does_not_require_missing_bot_module(self):
        client = Feishu.__new__(Feishu)
        client._open_id = 'explicit'
        self.assertEqual(client.get_bot_open_id(), 'explicit')

        client._open_id = None
        client.get_app_info = MagicMock(return_value={'open_id': 'from-app'})
        self.assertEqual(client.get_bot_open_id(), 'from-app')

        client.get_app_info.return_value = {}
        with patch.dict(sys.modules, {'lark_oapi.api.bot.v3': None}):
            self.assertIsNone(client.get_bot_open_id())


if __name__ == '__main__':
    unittest.main()
