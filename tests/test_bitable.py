import unittest
import warnings
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd

from gautils.feishu.core.bitable import BiTable, Table, TableField, _FS, _query_has_more_list_by_page_token


def _mock_table_row(table_id='tbl_001', name='TestTable', revision='1'):
    return pd.Series({
        _FS.BITABLE.TABLE.ID: table_id,
        _FS.BITABLE.TABLE.NAME: name,
        _FS.BITABLE.TABLE.REVISION: revision,
    })


def _mock_response(success=True, data=None, code=0, msg='ok'):
    response = MagicMock()
    response.success.return_value = success
    response.code = code
    response.msg = msg
    response.get_log_id.return_value = 'log_id'
    response.raw = MagicMock()
    response.raw.content = b'{}'
    response.data = data
    return response


def _mock_field_item(field_id='fld_001', field_name='name', field_type=1, desc=None, is_primary=False):
    item = MagicMock()
    item.field_id = field_id
    item.field_name = field_name
    item.type = field_type
    item.description = desc
    item.is_primary = is_primary
    return item


def _mock_record_item(record_id='rec_001', fields=None):
    item = MagicMock()
    item.record_id = record_id
    item.fields = fields if fields is not None else {}
    return item


def _mock_table_item(table_id='tbl_001', name='TestTable', revision='1'):
    item = MagicMock()
    item.table_id = table_id
    item.name = name
    item.revision = revision
    return item


class TestQueryHasMoreListByPageToken(unittest.TestCase):
    def test_concat_multiple_pages(self):
        calls = []

        def query_f(page_token=None):
            calls.append(page_token)
            if page_token is None:
                return True, 'next', pd.DataFrame({'a': [1]})
            return False, None, pd.DataFrame({'a': [2]})

        result = _query_has_more_list_by_page_token(query_f)

        self.assertEqual(calls, [None, 'next'])
        self.assertEqual(result['a'].tolist(), [1, 2])

    def test_returns_none_when_all_pages_empty(self):
        result = _query_has_more_list_by_page_token(lambda page_token=None: (False, None, None))

        self.assertIsNone(result)


class TestTableField(unittest.TestCase):
    def test_field_values_keep_legacy_enum_mapping(self):
        self.assertEqual(TableField.FieldType.Text.value, 1)
        self.assertEqual(TableField.FieldType.Number.value, 2)
        self.assertEqual(TableField.FieldType.FuXuan.value, 7)
        self.assertEqual(TableField.FieldType.AutoId.value, 1005)

    def test_field_keeps_name_and_type(self):
        field = TableField('name', TableField.FieldType.Text)

        self.assertEqual(field.name, 'name')
        self.assertEqual(field.fieldtype, TableField.FieldType.Text)


class TestTableQueryAndFormat(unittest.TestCase):
    def setUp(self):
        self.bitable = MagicMock()
        self.bitable.app_token = 'app_token'
        self.table = Table(self.bitable, _mock_table_row())

    def _mock_fields(self, fields):
        data = MagicMock()
        data.items = [
            _mock_field_item(field_id=f'fld_{i}', field_name=name, field_type=field_type, desc=desc)
            for i, (name, field_type, desc) in enumerate(fields)
        ]
        self.bitable.client.bitable.v1.app_table_field.list.return_value = _mock_response(data=data)

    def test_query_fields_success_and_cache(self):
        self._mock_fields([
            ('name', _FS.BITABLE.TABLE.FIELD.V_TEXT, 'Primary1'),
            ('value', _FS.BITABLE.TABLE.FIELD.V_NUMBER, None),
        ])

        first = self.table.query_fields()
        second = self.table.query_fields()

        self.assertIs(first, second)
        self.assertEqual(first[_FS.BITABLE.TABLE.FIELD.NAME].tolist(), ['name', 'value'])
        self.bitable.client.bitable.v1.app_table_field.list.assert_called_once()

    def test_query_fields_failure_keeps_legacy_none(self):
        self.bitable.client.bitable.v1.app_table_field.list.return_value = _mock_response(success=False, code=1)

        self.assertIsNone(self.table.query_fields())

    def test_primary_and_modifiable_fields(self):
        self._mock_fields([
            ('id', _FS.BITABLE.TABLE.FIELD.V_TEXT, 'Primary2'),
            ('name', _FS.BITABLE.TABLE.FIELD.V_TEXT, 'Primary1'),
            ('formula', _FS.BITABLE.TABLE.FIELD.V_FORMULA, None),
            ('auto', _FS.BITABLE.TABLE.FIELD.V_TYPE_AUTO_START_ID, None),
        ])

        self.assertEqual(self.table.primary_fields, ['name', 'id'])
        self.assertEqual(self.table.modifiable_fields, ['id', 'name'])

    def test_clean_df_keeps_only_modifiable_fields_and_resets_index(self):
        with patch.object(Table, 'modifiable_fields', new_callable=PropertyMock, return_value=['name', 'value']):
            df = pd.DataFrame({'name': ['a'], 'value': [np.inf], 'drop_me': ['x']}, index=['idx'])

            result = self.table.clean_df(df)

        self.assertEqual(result.columns.tolist(), ['name', 'value'])
        self.assertEqual(result.index.tolist(), [0])
        self.assertTrue(pd.isna(result.loc[0, 'value']))

    def test_format_type_df_before_cu_keeps_local_conversions(self):
        over_text = 'x' * (Table._TEXT_CELL_MAX + 3)
        fields = pd.DataFrame([
            {_FS.BITABLE.TABLE.FIELD.NAME: 'text', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_TEXT},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'number', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_NUMBER},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'checkbox', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_FUXUAN},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'multi', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_MSELECT},
        ])
        df = pd.DataFrame({
            'text': [over_text, None],
            'number': [1.234567, None],
            'checkbox': ['yes', '0'],
            'multi': ['tag', np.array(['a', 'b'], dtype=object)],
        })

        with patch.object(self.table, 'query_fields', return_value=fields):
            result = self.table.format_type_df_before_CU(df)

        self.assertEqual(len(result.loc[0, 'text']), Table._TEXT_CELL_MAX)
        self.assertEqual(result.loc[1, 'text'], '')
        self.assertEqual(result['number'].tolist(), [1.23457, 0.0])
        self.assertEqual(result['checkbox'].tolist(), [True, False])
        self.assertEqual(result['multi'].tolist(), [['tag'], ['a', 'b']])

    def test_search_records_normalizes_text_and_datetime(self):
        record_data = MagicMock()
        record_data.has_more = False
        record_data.page_token = None
        record_data.items = [
            _mock_record_item('rec_1', {
                'text': [{'text': 'hello'}],
                'date': 0,
            })
        ]
        self.bitable.client.bitable.v1.app_table_record.search.return_value = _mock_response(data=record_data)
        fields = pd.DataFrame([
            {_FS.BITABLE.TABLE.FIELD.NAME: 'text', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_TEXT},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'date', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_DATETIME},
        ])

        with patch.object(self.table, 'query_fields', return_value=fields):
            result = self.table.search_records()

        self.assertEqual(result.loc['rec_1', 'text'], 'hello')
        self.assertEqual(str(result.loc['rec_1', 'date'].tz), 'Asia/Shanghai')

    def test_search_records_raises_on_api_failure(self):
        self.bitable.client.bitable.v1.app_table_record.search.return_value = _mock_response(success=False, code=1)

        with self.assertRaises(ValueError):
            self.table.search_records()

    def test_search_records_normalizes_existing_field_types(self):
        record_data = MagicMock()
        record_data.has_more = False
        record_data.page_token = None
        record_data.items = [
            _mock_record_item('rec_1', {
                'person': [{'name': '张三'}, {'name': '李四'}],
                'phone': {'phone_number': '13800000000'},
                'link': {'text': '官网', 'link': 'https://example.com'},
                'formula_number': {'type': _FS.BITABLE.TABLE.FIELD.V_NUMBER, 'value': [3.14]},
                'formula_text': {'type': _FS.BITABLE.TABLE.FIELD.V_TEXT, 'value': [{'text': 'ok'}]},
                'formula_select': {'type': _FS.BITABLE.TABLE.FIELD.V_SSELECT, 'value': ['A']},
                'file': [{'name': 'a.txt'}],
                'auto': 'AUTO-1',
            })
        ]
        self.bitable.client.bitable.v1.app_table_record.search.return_value = _mock_response(data=record_data)
        fields = pd.DataFrame([
            {_FS.BITABLE.TABLE.FIELD.NAME: 'person', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_RENYUAN},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'phone', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_DIANHUA},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'link', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_CHAOLIANJIE},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'formula_number', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_FORMULA},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'formula_text', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_FORMULA},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'formula_select', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_FORMULA},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'file', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_FUJIAN},
            {_FS.BITABLE.TABLE.FIELD.NAME: 'auto', _FS.BITABLE.TABLE.FIELD.TYPE: _FS.BITABLE.TABLE.FIELD.V_TYPE_AUTO_START_ID},
        ])

        with patch.object(self.table, 'query_fields', return_value=fields):
            result = self.table.search_records()

        self.assertEqual(result.loc['rec_1', 'person'], ['张三', '李四'])
        self.assertEqual(result.loc['rec_1', 'phone'], '13800000000')
        self.assertEqual(result.loc['rec_1', 'link'], {'text': '官网', 'link': 'https://example.com'})
        self.assertEqual(result.loc['rec_1', 'formula_number'], 3.14)
        self.assertEqual(result.loc['rec_1', 'formula_text'], 'ok')
        self.assertEqual(result.loc['rec_1', 'formula_select'], ['A'])
        self.assertEqual(result.loc['rec_1', 'file'], 'a.txt')
        self.assertEqual(result.loc['rec_1', 'auto'], 'AUTO-1')

    def test_search_records_raises_on_unsupported_field_type(self):
        record_data = MagicMock()
        record_data.has_more = False
        record_data.page_token = None
        record_data.items = [_mock_record_item('rec_1', {'bad': 'value'})]
        self.bitable.client.bitable.v1.app_table_record.search.return_value = _mock_response(data=record_data)
        fields = pd.DataFrame([
            {_FS.BITABLE.TABLE.FIELD.NAME: 'bad', _FS.BITABLE.TABLE.FIELD.TYPE: 999},
        ])

        with patch.object(self.table, 'query_fields', return_value=fields):
            with self.assertRaises(ValueError):
                self.table.search_records()

    def test_deprecated_list_and_insert_rows_delegate(self):
        with patch.object(self.table, 'search_records', return_value='records') as mock_search:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = self.table.list_records(filter='filter', field_names=['name'])

        self.assertEqual(result, 'records')
        mock_search.assert_called_once_with(field_names=['name'], filter='filter')
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in caught))

        df = pd.DataFrame({'name': ['a']})
        with patch.object(self.table, 'insert_records', return_value=1) as mock_insert:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                self.table.insert_rows(df)

        mock_insert.assert_called_once_with(df)
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in caught))

    def test_dump_oversized_cells_never_raises(self):
        df = pd.DataFrame({'name': ['a' * 10], 'obj': [object()]})

        self.table._dump_oversized_cells(df, op='test')


class TestTableInsertRecords(unittest.TestCase):
    def setUp(self):
        self.bitable = MagicMock()
        self.bitable.app_token = 'app_token'
        self.table = Table(self.bitable, _mock_table_row())

    def test_insert_records_queries_existing_primary_rows_in_batches(self):
        df = pd.DataFrame({
            'name': [f'name_{i}' for i in range(251)],
            'value': list(range(251)),
        })
        existed = pd.DataFrame({'name': ['name_0', 'name_150']}, index=['rec_0', 'rec_150'])

        with patch.object(Table, 'primary_fields', new_callable=PropertyMock, return_value=['name']), \
                patch.object(self.table, 'clean_df', return_value=df), \
                patch.object(self.table, 'format_type_df_before_CU', side_effect=lambda x: x), \
                patch.object(self.table, 'search_records', side_effect=[
                    pd.DataFrame({'name': ['name_0']}, index=['rec_0']),
                    None,
                    pd.DataFrame({'name': ['name_150']}, index=['rec_150']),
                ]) as mock_search, \
                patch.object(self.table, '_update_rows', return_value=2) as mock_update, \
                patch.object(self.table, '_insert_records', return_value=249) as mock_insert:
            count = self.table.insert_records(df)

        self.assertEqual(count, 251)
        self.assertEqual(mock_search.call_count, 3)
        search_sizes = [len(call.kwargs['filter'].conditions) for call in mock_search.call_args_list]
        self.assertEqual(search_sizes, [100, 100, 51])
        mock_update.assert_called_once()
        mock_insert.assert_called_once()
        self.assertEqual(set(mock_update.call_args.args[0].index), set(existed.index))
        self.assertEqual(len(mock_insert.call_args.args[0]), 249)

    def test_insert_records_without_primary_inserts_directly(self):
        df = pd.DataFrame({'name': ['a']})

        with patch.object(Table, 'primary_fields', new_callable=PropertyMock, return_value=None), \
                patch.object(self.table, 'clean_df', return_value=df), \
                patch.object(self.table, 'format_type_df_before_CU', side_effect=lambda x: x), \
                patch.object(self.table, '_insert_records', return_value=1) as mock_insert:
            count = self.table.insert_records(df)

        self.assertEqual(count, 1)
        mock_insert.assert_called_once_with(df)

    def test_insert_records_deduplicates_primary_rows(self):
        df = pd.DataFrame({'name': ['a', 'a', 'b'], 'value': [1, 2, 3]})

        with patch.object(Table, 'primary_fields', new_callable=PropertyMock, return_value=['name']), \
                patch.object(self.table, 'clean_df', return_value=df), \
                patch.object(self.table, 'format_type_df_before_CU', side_effect=lambda x: x), \
                patch.object(self.table, '_search_existing_primary_records', return_value=None), \
                patch.object(self.table, '_insert_records', return_value=2) as mock_insert:
            count = self.table.insert_records(df)

        self.assertEqual(count, 2)
        self.assertEqual(mock_insert.call_args.args[0]['name'].tolist(), ['a', 'b'])

    def test_batch_create_failure_returns_zero_and_dumps_too_large_cell(self):
        df = pd.DataFrame({'name': ['a']})
        self.bitable.client.bitable.v1.app_table_record.batch_create.return_value = _mock_response(
            success=False,
            code=1254130,
        )

        with patch.object(self.table, '_dump_oversized_cells') as mock_dump:
            result = self.table._insert_records(df)

        self.assertEqual(result, 0)
        mock_dump.assert_called_once()

    def test_batch_create_success_returns_input_count(self):
        df = pd.DataFrame({'name': ['a', 'b']})
        self.bitable.client.bitable.v1.app_table_record.batch_create.return_value = _mock_response()

        result = self.table._insert_records(df)

        self.assertEqual(result, 2)

    def test_batch_update_failure_returns_zero_and_dumps_too_large_cell(self):
        df = pd.DataFrame({'name': ['a']}, index=['rec_1'])
        self.bitable.client.bitable.v1.app_table_record.batch_update.return_value = _mock_response(
            success=False,
            code=1254130,
        )

        with patch.object(self.table, '_dump_oversized_cells') as mock_dump:
            result = self.table._update_rows(df)

        self.assertEqual(result, 0)
        mock_dump.assert_called_once()

    def test_batch_update_success_returns_response_record_count(self):
        df = pd.DataFrame({'name': ['a', 'b']}, index=['rec_1', 'rec_2'])
        data = MagicMock()
        data.records = [MagicMock(), MagicMock()]
        self.bitable.client.bitable.v1.app_table_record.batch_update.return_value = _mock_response(data=data)

        result = self.table._update_rows(df)

        self.assertEqual(result, 2)


class TestTableDeleteRows(unittest.TestCase):
    def setUp(self):
        self.bitable = MagicMock()
        self.bitable.app_token = 'app_token'
        self.table = Table(self.bitable, _mock_table_row())

    def test_del_rows_delegates_to_filter_delete(self):
        with patch.object(self.table, 'del_rows_by_filter', return_value=3) as mock_delete:
            self.assertEqual(self.table.del_rows(), 3)

        mock_delete.assert_called_once_with()

    def test_del_rows_by_filter_deletes_in_500_record_batches(self):
        df = pd.DataFrame({'name': [f'n{i}' for i in range(501)]}, index=[f'rec_{i}' for i in range(501)])
        data1 = MagicMock()
        data1.records = [MagicMock() for _ in range(500)]
        data2 = MagicMock()
        data2.records = [MagicMock()]
        self.bitable.client.bitable.v1.app_table_record.batch_delete.side_effect = [
            _mock_response(data=data1),
            _mock_response(data=data2),
        ]

        with patch.object(Table, 'primary_fields', new_callable=PropertyMock, return_value=['name']), \
                patch.object(self.table, 'search_records', return_value=df):
            result = self.table.del_rows_by_filter()

        self.assertEqual(result, 501)
        self.assertEqual(self.bitable.client.bitable.v1.app_table_record.batch_delete.call_count, 2)

    def test_del_rows_by_filter_returns_zero_when_response_data_missing(self):
        df = pd.DataFrame({'name': ['a']}, index=['rec_1'])
        self.bitable.client.bitable.v1.app_table_record.batch_delete.return_value = _mock_response(data=None)

        with patch.object(Table, 'primary_fields', new_callable=PropertyMock, return_value=['name']), \
                patch.object(self.table, 'search_records', return_value=df):
            result = self.table.del_rows_by_filter()

        self.assertEqual(result, 0)

    def test_del_rows_by_filter_returns_zero_when_no_records(self):
        with patch.object(Table, 'primary_fields', new_callable=PropertyMock, return_value=['name']), \
                patch.object(self.table, 'search_records', return_value=None):
            result = self.table.del_rows_by_filter()

        self.assertEqual(result, 0)


class TestBiTable(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.bitable = BiTable(self.client, 'app_token')

    def test_query_tables_paginates_and_caches(self):
        data1 = MagicMock()
        data1.has_more = True
        data1.page_token = 'next'
        data1.items = [_mock_table_item('tbl_1', '表1')]
        data2 = MagicMock()
        data2.has_more = False
        data2.page_token = None
        data2.items = [_mock_table_item('tbl_2', '表2')]
        self.client.bitable.v1.app_table.list.side_effect = [
            _mock_response(data=data1),
            _mock_response(data=data2),
        ]

        first = self.bitable.query_tables()
        second = self.bitable.query_tables()

        self.assertIs(first, second)
        self.assertEqual(first[_FS.BITABLE.TABLE.ID].tolist(), ['tbl_1', 'tbl_2'])
        self.assertEqual(self.client.bitable.v1.app_table.list.call_count, 2)

    def test_query_tables_failure_raises(self):
        self.client.bitable.v1.app_table.list.return_value = _mock_response(success=False, code=1)

        with self.assertRaises(ValueError):
            self.bitable.query_tables()

    def test_get_table_by_name_and_id(self):
        df = pd.DataFrame([
            {_FS.BITABLE.TABLE.ID: 'tbl_1', _FS.BITABLE.TABLE.NAME: '表1', _FS.BITABLE.TABLE.REVISION: '1'},
            {_FS.BITABLE.TABLE.ID: 'tbl_2', _FS.BITABLE.TABLE.NAME: '表2', _FS.BITABLE.TABLE.REVISION: '1'},
        ])

        with patch.object(self.bitable, 'query_tables', return_value=df):
            self.assertEqual(self.bitable.get_table(table_id='tbl_1').id, 'tbl_1')
            self.assertEqual(self.bitable.get_table(table_name='表2').name, '表2')
            self.assertIsNone(self.bitable.get_table(table_id='missing'))

    def test_create_table_success_clears_cache_and_returns_table(self):
        self.bitable._tables = pd.DataFrame()
        self.client.bitable.v1.app_table.create.return_value = _mock_response()

        with patch.object(self.bitable, 'get_table', return_value='table') as mock_get:
            result = self.bitable.create_table('表', [TableField('name', TableField.FieldType.Text)])

        self.assertEqual(result, 'table')
        self.assertIsNone(self.bitable._tables)
        mock_get.assert_called_once_with(table_name='表')

    def test_create_table_failure_returns_none(self):
        self.client.bitable.v1.app_table.create.return_value = _mock_response(success=False, code=1)

        result = self.bitable.create_table('表', [TableField('name', TableField.FieldType.Text)])

        self.assertIsNone(result)

    def test_delete_table_by_name_handles_missing_as_success(self):
        with patch.object(self.bitable, 'query_table', return_value=None):
            self.assertTrue(self.bitable.delete_table(table_name='missing'))

    def test_delete_table_success_clears_cache(self):
        self.bitable._tables = pd.DataFrame()
        self.client.bitable.v1.app_table.delete.return_value = _mock_response()

        self.assertTrue(self.bitable.delete_table(table_id='tbl_1'))
        self.assertIsNone(self.bitable._tables)

    def test_delete_table_failure_returns_false(self):
        self.client.bitable.v1.app_table.delete.return_value = _mock_response(success=False, code=1)

        self.assertFalse(self.bitable.delete_table(table_id='tbl_1'))

    def test_delete_table_requires_name_or_id(self):
        with self.assertRaises(ValueError):
            self.bitable.delete_table()

    def test_add_collaborator_returns_false_on_failure(self):
        self.client.drive.v1.permission_member.create.return_value = _mock_response(success=False, code=1)

        self.assertFalse(self.bitable.add_collaborator('open_id'))

    def test_add_collaborator_success(self):
        self.client.drive.v1.permission_member.create.return_value = _mock_response()

        self.assertTrue(self.bitable.add_collaborator('open_id', member_id_type='userid', perm='edit'))

    def test_query_table_returns_none_for_empty_tables(self):
        with patch.object(self.bitable, 'query_tables', return_value=None):
            self.assertIsNone(self.bitable.query_table(table_id='tbl_1'))


if __name__ == '__main__':
    unittest.main()
