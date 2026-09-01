import unittest
import sys
from types import ModuleType
from unittest.mock import MagicMock

from gautils.feishu.core.bitable import BiTable
from gautils.feishu.core.feishu import Feishu
from gautils.feishu.core.spreadsheets import Spreadsheet
from gautils.feishu.core.wiki import Wiki
from tests.feishu_fakes import obj, response


class TestFeishuFacade(unittest.TestCase):
    def setUp(self):
        self.feishu = Feishu.__new__(Feishu)
        self.feishu.client = MagicMock()
        self.feishu._open_id = None
        self.feishu.get_app_info = MagicMock(return_value={})

    def test_resource_factories(self):
        bitable = self.feishu.get_bitable('app-token')
        spreadsheet = self.feishu.get_spreadsheet('sheet-token')
        wiki = self.feishu.get_wiki()
        self.assertIsInstance(bitable, BiTable)
        self.assertEqual(bitable.app_token, 'app-token')
        self.assertIsInstance(spreadsheet, Spreadsheet)
        self.assertEqual(spreadsheet.spreadsheet_token, 'sheet-token')
        self.assertIsInstance(wiki, Wiki)

    def test_create_bitable_success_and_failure(self):
        api = self.feishu.client.bitable.v1.app.create
        api.return_value = response(data=obj(app=obj(app_token='new-token')))
        created = self.feishu.create_bitable('name', folder_token='folder')
        self.assertIsInstance(created, BiTable)
        self.assertEqual(created.app_token, 'new-token')

        api.return_value = response(success=False, content={'error': 'bad'})
        self.assertIsNone(self.feishu.create_bitable('name'))

    def test_get_bot_open_id_success_empty_and_failure(self):
        fake_module = ModuleType('lark_oapi.api.bot.v3')
        fake_module.GetBotInfoRequest = MagicMock()
        fake_module.GetBotInfoResponse = MagicMock()
        with unittest.mock.patch.dict(sys.modules, {'lark_oapi.api.bot.v3': fake_module}):
            api = self.feishu.client.bot.v3.info.get
            api.return_value = response(data=obj(bot=obj(open_id='bot-id')))
            self.assertEqual(self.feishu.get_bot_open_id(), 'bot-id')
            api.return_value = response(data=obj(bot=None))
            self.assertIsNone(self.feishu.get_bot_open_id())
            api.return_value = response(success=False)
            self.assertIsNone(self.feishu.get_bot_open_id())

    def test_query_from_space_preserves_legacy_none_contract(self):
        api = self.feishu.client.wiki.v2.space.get_node
        api.return_value = response(data=obj(space=obj(space_id='1')))
        self.assertIsNone(self.feishu.query_from_space('node', 'docx'))
        api.return_value = response(success=False, content={'error': 'bad'})
        self.assertIsNone(self.feishu.query_from_space('node', 'docx'))


if __name__ == '__main__':
    unittest.main()
