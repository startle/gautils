import requests
import json
import pandas as pd
from .utils import *
from abc import ABC, abstractmethod
import numpy as np

FEISHU_API_ADDRESS = 'https://open.feishu.cn/open-apis/'

class _FS:
    class SHEET:
        ID = 'sheet_id'
        TITLE = 'sheet_title'
        COL_COUNT = 'sheet_cols'
        ROW_COUNT = 'sheet_rows'
        RES_TYPE = 'res_type'
    class SPACE:
        ID = 'space_id'
        NAME = 'name'
        class NODE:
            NODE_TOKEN = 'node_token'
            OBJ_TOKEN = 'obj_token'
            TITLE = 'title'
            HAS_CHILD = 'has_child'
            OBJ_TYPE = 'obj_type'
            PARENT_TOKEN = 'parent_node_token'
    class BITABLE:
        class TABLE:
            ID = 'id'
            NAME = 'name'
            class ROW:
                ID = 'record_id'
            class FIELD:
                ID = 'id'
                NAME = 'name'
                TYPE = 'type'
                IS_PRIMARY = 'is_primary'
                V_FORMULA = 20
                V_TEXT = 1
                V_NUMBER = 2
                V_SINGLE_SELECT = 3
                V_MULTI_SELECT = 4
                V_DATETIME = 5
                V_CONN = 18  # 单向关联
                V_REFERENCE = 19  # 查找引用
                V_TYPE_AUTO_START_ID = 1000

class FeishuAccessor(ABC):
    @abstractmethod
    def request(self, uri, http_method='GET', params=None, json=None, **kws):
        pass
class BaseAccessor(FeishuAccessor):
    def __init__(self, access_token, is_debug=True):
        self._access_token = access_token
        self._is_debug = is_debug
    def _debug_out(self, uri, response):
        if self._is_debug:
            debug_file = f'{DEBUG_OUT_DIR}/{convert_url_to_windows_filename(uri)}.json'
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(response.json(), file, indent=4, ensure_ascii=False)
    def request(self, uri, http_method='GET', params=None, json=None, **kws):
        headers = {'Authorization': f'Bearer {self._access_token}'}
        url = FEISHU_API_ADDRESS + uri
        for k, v in kws.items():
            url = url.replace(f':{k}', str(v))
        from ..web import retry_run
        res = retry_run(lambda : requests.request(http_method, url, headers=headers, params=params, json=json))
        if res.status_code != 200:
            raise Exception(f'[request failed] url[{url}] code[{res.status_code}]\n{res.text}')
        self._debug_out(uri, res)
        json = res.json()
        if json['code'] != 0:
            raise Exception(f'[request code!=0] url[{url}] code[{json["code"]}] {res.text}')
        return json
class AccessorWrapper(BaseAccessor):
    def __init__(self, acs: FeishuAccessor, keys=None):
        self._acs = acs
        self.keys = keys
    def request(self, uri, http_method='GET', **kws):
        if self.keys is not None:
            kws = dict(kws)
            kws.update(self.keys)
        return self._acs.request(uri, http_method, **kws)
class Spreadsheet:
    def __init__(self, acs: FeishuAccessor, spreadsheet_token):
        self._acs = acs
        self._spreadsheet_token = spreadsheet_token
    def query_rows(self, sheet_id, range) -> pd.DataFrame:
        url = f'/sheets/v2/spreadsheets/:spreadsheet_token/values/:range'
        range = f'{sheet_id}!{range}' if range is not None else sheet_id
        res = self._acs.request(url, range=range)
        datas = res['data']['valueRange']['values']
        df = pd.DataFrame(datas[1:], columns=datas[0])
        return df
    def query_info(self) -> pd.DataFrame:
        url = f'/sheets/v3/spreadsheets/:spreadsheet_token/sheets/query'
        res = self._acs.request(url)
        df = pd.DataFrame(pd.Series({
            _FS.SHEET.ID: x['sheet_id'],
            _FS.SHEET.TITLE: x['title'].strip(),
            _FS.SHEET.COL_COUNT: x['grid_properties']['column_count'],
            _FS.SHEET.ROW_COUNT: x['grid_properties']['row_count'],
            _FS.SHEET.RES_TYPE: x['resource_type'],
        }) for x in res['data']['sheets'])
        return df
    def query_sheet_info(self, sheet_name) -> pd.Series:
        df_info = self.query_info()
        infos = df_info.set_index(_FS.SHEET.TITLE).T.to_dict()
        if sheet_name not in infos:
            raise Exception(f'sheet not found.[{sheet_name}]')
        return infos[sheet_name]
    def read(self, sheet_name, range=None) -> pd.DataFrame:
        sheet_info = self.query_sheet_info(sheet_name)
        sheet_id = sheet_info[_FS.SHEET.ID]
        return self.query_rows(sheet_id, range)
class Space:
    def __init__(self, acs: FeishuAccessor, space_series: pd.Series):
        self._acs = acs
        self.space_series = space_series
    @property
    def space_id(self):
        return self.space_series[_FS.SPACE.ID]
    @property
    def space_name(self):
        return self.space_series[_FS.SPACE.NAME]
    def request(self, uri: str, http_method='GET', params=None, **kws):
        kws = {'space_id': self.space_id, **kws}
        return self._acs.request(uri, http_method=http_method, params=params, **kws)
    def _build_node_series(self, d):
        return pd.Series({
            _FS.SPACE.NODE.TITLE: d['title'],
            _FS.SPACE.NODE.NODE_TOKEN: d['node_token'],
            _FS.SPACE.NODE.OBJ_TOKEN: d['obj_token'],
            _FS.SPACE.NODE.OBJ_TYPE: d['obj_type'],
            _FS.SPACE.NODE.PARENT_TOKEN: d['parent_node_token'],
            _FS.SPACE.NODE.HAS_CHILD: d['has_child'],
        })
    def query_nodes(self, parent_node_token=None):
        url = '/wiki/v2/spaces/:space_id/nodes'
        params = {'page_size': 50, 'parent_node_token': parent_node_token}
        res = self.request(url, space_id=self.space_id, params=params)
        df = pd.DataFrame([self._build_node_series(x) for x in res['data']['items']])
        return df
    def query_node(self, path=None, parent_node_token=None) -> pd.Series:
        if path is None or len(path) == 0:
            uri = '/wiki/v2/spaces/get_node'
            res = self.request(uri, params={'token': parent_node_token})
            return self._build_node_series(res['data']['node'])
        if isinstance(path, str):
            path = [path]

        def inner(path, parent_node_token):
            df = self.query_nodes(parent_node_token=parent_node_token)
            nodes = df.set_index(_FS.SPACE.NODE.TITLE).T.to_dict()
            if path[0] not in nodes:
                return None
            rs = nodes[path[0]]
            if len(path) == 1:
                return rs
            else:
                return inner(path[1:], rs[_FS.SPACE.NODE.NODE_TOKEN])
        return inner(path, parent_node_token=parent_node_token)
    def get_node(self, path=None, parent_node_token=None):
        sr = self.query_node(path, parent_node_token)
        if sr is None:
            return None
        return Space.Node(self, sr)
    class Node:
        def __init__(self, space, series: pd.Series):
            self._space: Space = space
            self._series = series
        def get_node(self, path):
            self._space.get_node(path, parent_node_token=self.node_token)
        @property
        def node_token(self):
            return self._series[_FS.SPACE.NODE.NODE_TOKEN]
        @property
        def obj_token(self):
            return self._series[_FS.SPACE.NODE.OBJ_TOKEN]
        @property
        def title(self):
            return self._series[_FS.SPACE.NODE.TITLE]
        @property
        def obj_type(self):
            return self._series[_FS.SPACE.NODE.OBJ_TYPE]
class BiTable:
    def __init__(self, acs: FeishuAccessor, app_token):
        self._acs = acs
        self._app_token = app_token
    class Table:
        def __init__(self, acs: FeishuAccessor, table_row):
            self._acs = acs
            self._table_row = table_row
            self._df_fields = None
        def query_fields(self):
            if self._df_fields is not None:
                return self._df_fields
            uri = '/bitable/v1/apps/:app_token/tables/:table_id/fields'
            res = self._acs.request(uri)
            df = pd.DataFrame([pd.Series({
                _FS.BITABLE.TABLE.FIELD.ID: x['field_id'],
                _FS.BITABLE.TABLE.FIELD.NAME: x['field_name'],
                _FS.BITABLE.TABLE.FIELD.TYPE: x['type'],
                _FS.BITABLE.TABLE.FIELD.IS_PRIMARY: x['is_primary'],
            }) for x in res['data']['items']])
            self._df_fields = df
            return df
        def list_records(self, filter=None, field_names=None):
            if field_names is not None:
                field_names_str = ','.join([f'"{field}"' for field in field_names])
                field_names_str = f'[{field_names_str}]'
            else:
                field_names_str = None
            uri = '/bitable/v1/apps/:app_token/tables/:table_id/records'
            page_token = None
            all_rows = []
            while True:
                res = self._acs.request(uri, table_id=self.table_id, params={'page_size': 500, 'filter': filter, 'page_token': page_token, 'field_names': field_names_str})['data']
                if 'items' not in res:
                    break
                all_rows.extend(res['items'])
                if res['has_more']:
                    page_token = res['page_token']
                else:
                    break
            if len(all_rows) == 0:
                return None

            def _row_to_dict(item):
                d = {_FS.BITABLE.TABLE.ROW.ID : item['record_id']}
                for field_name, value in item['fields'].items():
                    d[field_name] = value
                return d
            df = pd.DataFrame(pd.Series(_row_to_dict(x)) for x in all_rows)
            dict_field = self.query_fields().set_index(_FS.BITABLE.TABLE.FIELD.NAME).to_dict('index')

            def _format_row(row: pd.Series):
                if row.name == _FS.BITABLE.TABLE.ROW.ID:
                    return row
                field_name = row.name
                field_info = dict_field[field_name]
                field_type = field_info[_FS.BITABLE.TABLE.FIELD.TYPE]
                if field_type in [_FS.BITABLE.TABLE.FIELD.V_TEXT, _FS.BITABLE.TABLE.FIELD.V_SINGLE_SELECT]:
                    row = row.transform(str)
                elif field_type in [_FS.BITABLE.TABLE.FIELD.V_NUMBER]:
                    row = row.transform(lambda value: float(value) if value is not None else None)
                elif field_type in [_FS.BITABLE.TABLE.FIELD.V_DATETIME]:
                    def trans_datetime(value):
                        if value is not None:
                            dt = pd.to_datetime(value, unit='ms', utc=True)
                            dt = dt.tz_convert('Asia/Shanghai')
                            value = dt
                        else:
                            value = None
                        return value
                    row = row.transform(trans_datetime)
                elif field_type in [_FS.BITABLE.TABLE.FIELD.V_MULTI_SELECT]:
                    row = row.transform(lambda value: value if isinstance(value, list) else [value] if (value not in [None, np.nan]) else [])
                return row
            df = df.apply(_format_row)
            df.set_index(_FS.BITABLE.TABLE.ROW.ID, inplace=True)
            return df

        def _format_before_update(self, df: pd.DataFrame):
            df_fields = self.query_fields()
            for _, row_field in df_fields.iterrows():
                col = row_field[_FS.BITABLE.TABLE.FIELD.NAME]
                _type = row_field[_FS.BITABLE.TABLE.FIELD.TYPE]
                if col not in df.columns:
                    continue
                if _type in [_FS.BITABLE.TABLE.FIELD.V_TEXT, _FS.BITABLE.TABLE.FIELD.V_SINGLE_SELECT]:
                    df[col] = df[col].astype(str).fillna('')
                elif _type in [_FS.BITABLE.TABLE.FIELD.V_NUMBER]:
                    df[col] = df[col].astype(float).fillna(0).round(5)
                elif _type in [_FS.BITABLE.TABLE.FIELD.V_DATETIME]:
                    df[col] = pd.to_datetime(df[col])
                    df[col] = df[col].fillna(pd.Timestamp(0))
                    has_tz = df[col].dt.tz is not None
                    if has_tz:
                        df[col] = df[col].dt.tz_convert('Asia/Shanghai')
                    else:
                        df[col] = df[col].dt.tz_localize('Asia/Shanghai')
                    df[col] = df[col].astype(np.int64) // 10**6
                    df[col] = df[col].replace(-28800000, None)  # 硬编码，将None的col转回来
                elif _type in [_FS.BITABLE.TABLE.FIELD.V_MULTI_SELECT]:
                    def ensure_list(obj):
                        if obj is None:
                            return []
                        elif isinstance(obj, list):
                            return obj
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        else:
                            return [obj]
                    df[col] = df[col].apply(ensure_list)
                else:
                    raise ValueError(f'unsupported feishu type.{_type}')
            return df
        def insert_rows(self, df: pd.DataFrame):
            if df is None or len(df) == 0:
                return 0
            uri = '/bitable/v1/apps/:app_token/tables/:table_id/records/batch_create'
            fields_modifiable = np.intersect1d(self.field_names_modifiable, df.columns.to_list())
            df = df.reset_index(drop=False)
            df = df.drop(columns=list(set(df.columns) - set(fields_modifiable)))
            df = self._format_before_update(df)

            def inner(df: pd.DataFrame):
                data = {'records': [{'fields': {k: v for k, v in x.items() if k in df.columns}} for x in df.to_dict(orient='records')]}
                res = self._acs.request(uri, http_method='POST', json=data)
                return len(res['data']['records'])
            return sum([inner(x) for x in batch_split(df, 500)])
        def insert_on_duplicate_update_rows(self, df: pd.DataFrame):
            if df is None or len(df) == 0:
                return 0
            df = df.reset_index(drop=False)
            fields_modifiable = np.intersect1d(self.field_names_modifiable, df.columns.to_list())
            primary_col = self.field_names_primary

            def split_update_insert(df: pd.DataFrame):
                if _FS.BITABLE.TABLE.ROW.ID in df.columns:
                    return df[df[_FS.BITABLE.TABLE.ROW.ID].notnull()], df[df[_FS.BITABLE.TABLE.ROW.ID].isnull()]
                else:
                    if primary_col not in df.columns:
                        return None, df
                    else:
                        df_existed = self.list_records(field_names=[primary_col])
                        if df_existed is not None:
                            df_existed.drop_duplicates(subset=primary_col, inplace=True, keep='first')
                            existed_cols = df_existed[primary_col].unique()
                            df_insert = df[~df[primary_col].isin(existed_cols)]
                            df_update = df[df[primary_col].isin(existed_cols)]
                            df_update = df.merge(df_existed.reset_index(drop=False), on=primary_col, how='inner')
                            return df_update, df_insert
                        else:
                            return None, df
            df.drop_duplicates(subset=primary_col, inplace=True, keep='first')
            df_update, df_insert = split_update_insert(df)

            cnt = 0
            if df_update is not None:
                cnt += self.update_rows(df_update)
            if df_insert is not None:
                cnt += self.insert_rows(df_insert.loc[:, fields_modifiable])
            return cnt

        def update_rows(self, df: pd.DataFrame):
            if df is None or len(df) == 0:
                return 0
            uri = '/bitable/v1/apps/:app_token/tables/:table_id/records/batch_update'
            fields_modifiable = np.intersect1d(self.field_names_modifiable, df.columns.to_list()).tolist()
            df = df.loc[:, fields_modifiable + ['record_id']]
            df = self._format_before_update(df)

            def inner(df: pd.DataFrame):
                data = {'records': [{'record_id': x['record_id'], 'fields': {k: v for k, v in x.items() if k in fields_modifiable}} for x in df.to_dict(orient='records')]}
                res = self._acs.request(uri, http_method='POST', json=data)
                return len(res['data']['records'])
            return sum([inner(x) for x in batch_split(df, 500)])
        def del_rows(self, row_tokens=None, filter=None):
            if row_tokens is None:
                df_rows = self.list_records(filter=filter, field_names=[self.field_names[0]])
                if df_rows is None or len(df_rows) == 0:
                    return 0
                row_tokens = df_rows.index.to_list()
            if len(row_tokens) == 0:
                return 0
            uri = '/bitable/v1/apps/:app_token/tables/:table_id/records/batch_delete'
            ret = True
            for x in batch_split(row_tokens, 500):
                res = self._acs.request(uri, http_method='POST', table_id=self.table_id, json={'records': x})
                ret &= (res['code'] == 0)
            return ret
        @property
        def field_names(self) -> pd.DataFrame:
            return self.query_fields()[_FS.BITABLE.TABLE.FIELD.NAME].to_list()
        @property
        def field_names_modifiable(self):
            df_fields = self.query_fields()
            modifiable_cond = (df_fields[_FS.BITABLE.TABLE.FIELD.TYPE] != _FS.BITABLE.TABLE.FIELD.V_FORMULA) \
                & (df_fields[_FS.BITABLE.TABLE.FIELD.TYPE] < _FS.BITABLE.TABLE.FIELD.V_TYPE_AUTO_START_ID) \
                & (~df_fields[_FS.BITABLE.TABLE.FIELD.TYPE].isin([_FS.BITABLE.TABLE.FIELD.V_REFERENCE]))
            df_fields = df_fields.loc[modifiable_cond]
            return df_fields[_FS.BITABLE.TABLE.FIELD.NAME].tolist()
        @property
        def field_names_primary(self):
            df_fields = self.query_fields()
            return df_fields[df_fields[_FS.BITABLE.TABLE.FIELD.IS_PRIMARY]].iloc[0][_FS.BITABLE.TABLE.FIELD.NAME]
        @property
        def table_id(self):
            return self._table_row[_FS.BITABLE.TABLE.ID]
        @property
        def table_name(self):
            return self._table_row[_FS.BITABLE.TABLE.NAME]
    def get_table(self, table_name=None, table_id=None) -> Table:
        table_row = self.query_table(table_id=table_id, table_name=table_name)
        if table_row is None:
            return None
        return BiTable.Table(AccessorWrapper(self._acs, keys={'table_id': table_row[_FS.BITABLE.TABLE.ID]}), table_row)
    def query_tables(self) -> pd.DataFrame:
        uri = '/bitable/v1/apps/:app_token/tables'
        res = self._acs.request(uri)
        df = pd.DataFrame(pd.Series({
            _FS.BITABLE.TABLE.ID: x['table_id'],
            _FS.BITABLE.TABLE.NAME: x['name'],
        }) for x in res['data']['items'])
        return df
    def query_table(self, table_name=None, table_id=None) -> str:
        df_table = self.query_tables()
        if df_table is None or len(df_table) == 0:
            return None
        if table_id is not None:
            if table_id in df_table[_FS.BITABLE.TABLE.ID].to_list():
                return df_table[df_table[_FS.BITABLE.TABLE.ID] == table_id].iloc[0]
        elif table_name is not None:
            if table_name in df_table[_FS.BITABLE.TABLE.NAME].to_list():
                return df_table[df_table[_FS.BITABLE.TABLE.NAME] == table_name].iloc[0]
        return None
    @property
    def obj_token(self):
        return self._app_token
class Feishu:
    ''' 飞书多维表格操作 '''
    def __init__(self, app_id=None, app_secret=None, access_token=None, is_debug=True):
        if access_token is not None:
            self._access_token = access_token
        else:
            self._app_id = app_id
            self._app_secret = app_secret
            self.init_tenant_access_token()
        self._accessor = BaseAccessor(self._access_token, is_debug=is_debug)
    def init_tenant_access_token(self):
        res = requests.post('https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal', params={'app_id': self._app_id, 'app_secret': self._app_secret})
        self._access_token = res.json()['tenant_access_token']

    def query_open_id(self) -> str:
        data = self._accessor.request('/bot/v3/info')
        return data['bot']['open_id']
    def query_spaces(self):
        uri = '/wiki/v2/spaces'
        res = self._accessor.request(uri, params={'page_size': 50})
        df = pd.DataFrame(pd.Series({
            _FS.SPACE.ID: x['space_id'],
            _FS.SPACE.NAME: x['name'],
        }) for x in res['data']['items'])
        return df

    def get_spreadsheet(self, spreadsheet_token) -> Spreadsheet:
        return Spreadsheet(AccessorWrapper(self._accessor, keys={'spreadsheet_token': spreadsheet_token}), spreadsheet_token)
    def get_space(self, space_id=None, space_name=None) -> Space:
        df = self.query_spaces()
        if space_id is not None:
            df.set_index(_FS.SPACE.ID, inplace=True)
            if space_id not in df.index.to_list():
                return None
            return Space(self._accessor, df.loc[space_id])
        elif space_name is not None:
            df.set_index(_FS.SPACE.NAME, inplace=True)
            if space_name not in df.index.to_list():
                return None
            return Space(self._accessor, df.loc[space_name])
    def get_bitable(self, app_token) -> BiTable:
        return BiTable(AccessorWrapper(self._accessor, keys={'app_token': app_token}), app_token)
    def get_bitable_from_space(self, space_name, path):
        return self.get_bitable(self.get_space(space_name=space_name).get_node(path).obj_token)
