import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from gautils.v2.errors import ApiError, ResourceNotFoundError, UnsupportedOperationError
from gautils.feishu.core.bitable import _WriteResult
from gautils.feishu.core._errors import _FeishuApiError
from gautils.v2.feishu import (
    Bitable, Feishu, Sheet, Spreadsheet, Table, Wiki, WikiNode, WikiSpace,
)


class TestV2Table(unittest.TestCase):
    def setUp(self):
        self.legacy = MagicMock()
        self.legacy.id = 't1'
        self.legacy.name = 'Table'
        self.table = Table(self.legacy)

    def test_properties_frames_and_delete(self):
        self.assertEqual(self.table.id, 't1')
        self.assertEqual(self.table.name, 'Table')
        self.legacy.query_fields.return_value = None
        self.legacy.search_records.return_value = None
        self.assertTrue(self.table.fields().empty)
        self.assertTrue(self.table.records(filter='x').empty)
        self.legacy.search_records.assert_called_once_with(filter='x')
        self.legacy.del_rows_by_filter.return_value = 2
        self.assertEqual(self.table.delete(filter='x'), 2)

    def test_upsert_empty_success_and_partial(self):
        self.assertTrue(self.table.upsert(pd.DataFrame()).ok)
        frame = pd.DataFrame({'x': [1, 2]})
        self.legacy._insert_records_result.return_value = _WriteResult(attempted=2, created=2)
        result = self.table.upsert(frame, insert_only=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.created, 2)
        self.legacy._insert_records_result.assert_called_with(frame, upsert=False, strict=True)

        self.legacy._insert_records_result.return_value = _WriteResult(
            attempted=2, updated=1, errors=[_FeishuApiError('update', message='bad')])
        result = self.table.upsert(frame)
        self.assertFalse(result.ok)
        self.assertEqual(result.failed, 1)
        self.assertEqual(len(result.errors), 1)


class TestV2Bitable(unittest.TestCase):
    def setUp(self):
        self.legacy = MagicMock(app_token='token')
        self.bitable = Bitable(self.legacy)

    def test_tables_and_table_lookup(self):
        self.assertEqual(self.bitable.token, 'token')
        self.legacy.query_tables.return_value = None
        self.assertTrue(self.bitable.tables().empty)
        legacy_table = MagicMock()
        self.legacy.get_table.return_value = legacy_table
        self.assertIs(self.bitable.table(name='name').legacy, legacy_table)
        self.legacy.get_table.return_value = None
        with self.assertRaises(ResourceNotFoundError):
            self.bitable.table(table_id='missing')

    def test_mutations_raise_on_false(self):
        self.legacy.delete_table.return_value = True
        self.assertIsNone(self.bitable.delete_table(name='name'))
        self.legacy.delete_table.return_value = False
        with self.assertRaises(ApiError):
            self.bitable.delete_table(table_id='t1')

        self.legacy.add_collaborator.return_value = True
        self.assertIsNone(self.bitable.add_collaborator('u1', perm='edit'))
        self.legacy.add_collaborator.return_value = False
        with self.assertRaises(ApiError):
            self.bitable.add_collaborator('u1')


class TestV2Spreadsheet(unittest.TestCase):
    def setUp(self):
        self.legacy = MagicMock(spreadsheet_token='token')
        self.book = Spreadsheet(self.legacy)

    def test_sheet_contract(self):
        legacy_sheet = MagicMock(id='s1', title='Data')
        sheet = Sheet(legacy_sheet)
        self.assertEqual(sheet.id, 's1')
        self.assertEqual(sheet.title, 'Data')
        for call in (sheet.read, sheet.write, sheet.append, sheet.clear):
            with self.assertRaises(UnsupportedOperationError):
                call()

    def test_book_queries_lookup_and_collaborator(self):
        self.assertEqual(self.book.token, 'token')
        self.legacy.query_sheets.return_value = [{'sheet_id': 's1'}]
        self.legacy.get_metadata.return_value = {'title': 'Book'}
        self.assertEqual(self.book.sheets(), [{'sheet_id': 's1'}])
        self.assertEqual(self.book.metadata(), {'title': 'Book'})

        self.legacy.get_sheet.return_value = MagicMock()
        self.assertIsInstance(self.book.sheet(title='Data'), Sheet)
        self.legacy.get_sheet.return_value = None
        with self.assertRaises(ResourceNotFoundError):
            self.book.sheet(sheet_id='missing')

        self.legacy.add_collaborator.return_value = True
        self.assertIsNone(self.book.add_collaborator('u1'))
        self.legacy.add_collaborator.return_value = False
        with self.assertRaises(ApiError):
            self.book.add_collaborator('u1')


class TestV2Wiki(unittest.TestCase):
    def test_node_space_and_wiki(self):
        legacy_node = MagicMock()
        legacy_node.id = 'n1'
        legacy_node.name = 'Doc'
        node = WikiNode(legacy_node)
        self.assertEqual(node.id, 'n1')
        self.assertEqual(node.name, 'Doc')

        legacy_space = MagicMock(space_id='sp1')
        legacy_space.get_info.return_value = {'name': 'Space'}
        legacy_space._list_nodes.return_value = None
        legacy_space.get_node.return_value = legacy_node
        space = WikiSpace(legacy_space)
        self.assertEqual(space.id, 'sp1')
        self.assertEqual(space.info(), {'name': 'Space'})
        self.assertTrue(space.nodes(parent_token='p1').empty)
        self.assertIsInstance(space.node('n1'), WikiNode)
        legacy_space.get_node.return_value = None
        with self.assertRaises(ResourceNotFoundError):
            space.node('missing')

        legacy_wiki = MagicMock()
        legacy_wiki.list_spaces.return_value = None
        legacy_wiki.get_space.return_value = legacy_space
        wiki = Wiki(legacy_wiki)
        self.assertTrue(wiki.spaces().empty)
        self.assertIsInstance(wiki.space('sp1'), WikiSpace)


class TestV2FeishuFacade(unittest.TestCase):
    def test_legacy_facade_factories(self):
        legacy = MagicMock()
        legacy.get_bitable.return_value = MagicMock()
        legacy.get_spreadsheet.return_value = MagicMock()
        legacy.get_wiki.return_value = MagicMock()
        api = Feishu(legacy)
        self.assertIsInstance(api.bitable('token'), Bitable)
        self.assertIsInstance(api.spreadsheet('token'), Spreadsheet)
        self.assertIsInstance(api.wiki(), Wiki)

    @patch('gautils.v2.feishu.LegacyFeishu')
    def test_credentials_construct_legacy_client(self, legacy_class):
        api = Feishu(app_id='id', app_secret='secret')
        self.assertIs(api.legacy, legacy_class.return_value)
        legacy_class.assert_called_once_with(app_id='id', app_secret='secret')


if __name__ == '__main__':
    unittest.main()
