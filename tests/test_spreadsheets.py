import unittest
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock

import pandas as pd

from gautils.feishu.core.spreadsheets import Sheet, Spreadsheet
from tests.feishu_fakes import obj, response


@contextmanager
def fake_sheet_sdk(*names):
    module = ModuleType('lark_oapi.api.sheets.v3')
    for name in names:
        setattr(module, name, MagicMock())
    with unittest.mock.patch.dict(sys.modules, {'lark_oapi.api.sheets.v3': module}):
        yield


class TestSheetLegacyContract(unittest.TestCase):
    def setUp(self):
        self.spreadsheet = MagicMock()
        self.sheet = Sheet(self.spreadsheet, {
            'sheet_id': 's1', 'title': 'Data', 'index': 2,
            'row_count': 100, 'column_count': 20,
        })

    def test_properties_and_ranges(self):
        self.assertEqual(self.sheet.id, 's1')
        self.assertEqual(self.sheet.title, 'Data')
        self.assertEqual(self.sheet.index, 2)
        self.assertEqual(self.sheet.row_count, 100)
        self.assertEqual(self.sheet.col_count, 20)
        self.assertEqual(self.sheet._build_range(), 's1')
        self.assertEqual(self.sheet._build_range('A1'), 's1!A1')
        self.assertEqual(self.sheet._build_range('A1', 'C2'), 's1!A1:C2')

    def test_unimplemented_methods_keep_legacy_results(self):
        frame = pd.DataFrame({'a': [1]})
        self.assertTrue(self.sheet.read().empty)
        self.assertFalse(self.sheet.write(frame))
        self.assertFalse(self.sheet.append(frame))
        self.assertFalse(self.sheet.clear())

    def test_column_helpers(self):
        self.assertEqual(Sheet._col_index_to_letter(0), 'A')
        self.assertEqual(Sheet._col_index_to_letter(26), 'AA')
        self.assertEqual(Sheet._col_letter_add('Z', 2), 'AB')


class TestSpreadsheet(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.book = Spreadsheet(self.client, 'book-token')

    def test_query_sheets_success_cache_and_lookup(self):
        sheet = obj(sheet_id='s1', title='Data', index=0,
                    grid_properties=obj(row_count=10, column_count=3))
        sheet_without_grid = obj(sheet_id='s2', title='Raw', index=1, grid_properties=None)
        api = self.client.sheets.v3.spreadsheet_sheet.query
        api.return_value = response(data=obj(sheets=[sheet, sheet_without_grid]))

        sheets = self.book.query_sheets()
        self.assertEqual(len(sheets), 2)
        self.assertEqual(self.book.query_sheets(), sheets)
        self.assertEqual(api.call_count, 1)
        self.assertEqual(self.book.get_sheet(sheet_id='s1').title, 'Data')
        self.assertEqual(self.book.get_sheet(sheet_title='Raw').id, 's2')
        self.assertIsNone(self.book.get_sheet(sheet_id='missing'))

    def test_query_sheets_failure_and_empty(self):
        api = self.client.sheets.v3.spreadsheet_sheet.query
        api.return_value = response(success=False)
        self.assertEqual(self.book.query_sheets(), [])
        api.return_value = response(data=None)
        self.assertEqual(self.book.query_sheets(), [])
        self.book._sheets = []
        self.assertIsNone(self.book.get_sheet(sheet_id='missing'))

    def test_create_sheet_success_and_failure(self):
        names = ('AddSheetRequest', 'AddSheetRequestBody', 'AddSheetResponse',
                 'SheetProperties', 'GridProperties')
        with fake_sheet_sdk(*names):
            self.book._sheets = []
            api = self.client.sheets.v3.spreadsheet_sheet.add
            api.return_value = response(data=obj())
            self.book.get_sheet = MagicMock(return_value='created-sheet')
            self.assertEqual(self.book.create_sheet('New', 20, 4), 'created-sheet')
            self.assertIsNone(self.book._sheets)

            api.return_value = response(success=False)
            self.assertIsNone(self.book.create_sheet('Bad'))

    def test_delete_sheet_contracts(self):
        with self.assertRaises(ValueError):
            self.book.delete_sheet()
        self.book._query_sheet = MagicMock(return_value=None)
        self.assertTrue(self.book.delete_sheet(sheet_title='missing'))

        with fake_sheet_sdk('DeleteSheetRequest', 'DeleteSheetRequestBody', 'DeleteSheetResponse'):
            api = self.client.sheets.v3.spreadsheet_sheet.delete
            api.return_value = response(success=False)
            self.assertFalse(self.book.delete_sheet(sheet_id='s1'))
            api.return_value = response()
            self.book._sheets = [{'sheet_id': 's1'}]
            self.assertTrue(self.book.delete_sheet(sheet_id='s1'))
            self.assertIsNone(self.book._sheets)

        self.book._query_sheet = MagicMock(return_value={'sheet_id': 's2'})
        with fake_sheet_sdk('DeleteSheetRequest', 'DeleteSheetRequestBody', 'DeleteSheetResponse'):
            self.client.sheets.v3.spreadsheet_sheet.delete.return_value = response()
            self.assertTrue(self.book.delete_sheet(sheet_title='Data'))

    def test_metadata_success_cache_and_failure(self):
        api = self.client.sheets.v3.spreadsheet.get
        api.return_value = response(data=obj(title='Book', url='https://sheet'))
        expected = {'title': 'Book', 'token': 'book-token', 'url': 'https://sheet'}
        self.assertEqual(self.book.get_metadata(), expected)
        self.assertEqual(self.book.get_metadata(), expected)
        self.assertEqual(api.call_count, 1)

        failed = Spreadsheet(self.client, 'failed')
        api.return_value = response(success=False)
        self.assertEqual(failed.get_metadata(), {})
        empty = Spreadsheet(self.client, 'empty')
        api.return_value = response(data=None)
        self.assertEqual(empty.get_metadata(), {'title': None, 'token': 'empty', 'url': None})

    def test_add_collaborator_success_and_failure(self):
        api = self.client.drive.v1.permission_member.create
        api.return_value = response()
        self.assertTrue(self.book.add_collaborator('user'))
        api.return_value = response(success=False)
        self.assertFalse(self.book.add_collaborator('user'))


if __name__ == '__main__':
    unittest.main()
