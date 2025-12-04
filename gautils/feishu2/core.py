import sys
import warnings
import json
import pandas as pd
import numpy as np
from enum import Enum
import lark_oapi as lark
from lark_oapi.api.bitable.v1 import *
from ..utils import batch_split

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="lark_oapi.ws.pb.google.__init__",
    message="pkg_resources is deprecated as an API"
)

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
            REVISION = 'revision'
            class RECORD:
                ID = 'record_id'
            class FIELD:
                ID = 'id'
                NAME = 'name'
                TYPE = 'type'
                IS_PRIMARY = 'is_primary'
                DESC = 'description'
                V_TEXT = 1
                V_NUMBER = 2
                V_SSELECT = 3
                V_MSELECT = 4
                V_DATETIME = 5
                V_FUJIAN = 17
                V_CONN = 18  # 单向关联
                V_REF = 19  # 查找引用
                V_FORMULA = 20
                V_TYPE_AUTO = 1000
                V_TYPE_AUTO_START_ID = 1005


class TableField:
    class FieldType(Enum):
        Text = 1
        Number = 2
        DanXuan = 3
        DuoXuan = 4
        Date = 5
        FuXuan = 7
        RenYuan = 11
        DianHuaHaoMa = 13
        ChaoLianJie = 15
        AutoId = 1005

    def __init__(self, name, fieldtype: FieldType):
        self.name = name
        self.fieldtype = fieldtype
def _query_has_more_list_by_page_token(query_f: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
    dfs = []
    has_more = True
    page_token = None
    while has_more:
        has_more, page_token, df = query_f(page_token=page_token)
        if df is not None and (len(df) > 0):
            dfs.append(df)
    return pd.concat(dfs) if len(dfs) > 0 else None

class Table:
    def __init__(self, bitable, table_row: pd.Series):
        self._bitable: BiTable = bitable
        self._table_row = table_row
        self._fields = None
    def query_fields(self):
        def inner():
            request = ListAppTableFieldRequest.builder() \
                .app_token(self._bitable.app_token) \
                .table_id(self.id).build()
            response: ListAppTableFieldResponse = self._bitable.client.bitable.v1.app_table_field.list(request)
            if not response.success():
                lark.logger.error(f"client.bitable.v1.app_table_field.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
                return
            indexes = [x.field_id for x in response.data.items]
            df = pd.DataFrame([pd.Series({
                _FS.BITABLE.TABLE.FIELD.NAME: x.field_name,
                _FS.BITABLE.TABLE.FIELD.IS_PRIMARY: x.is_primary,
                _FS.BITABLE.TABLE.FIELD.DESC: x.description,
                _FS.BITABLE.TABLE.FIELD.TYPE: x.type,
            }) for x in response.data.items], index=indexes)
            return df
        if self._fields is None:
            self._fields = inner()
        return self._fields
    def clean_df(self, df: pd.DataFrame):
        fields_modifiable = np.intersect1d(self.modifiable_fields, df.columns.to_list())
        df = df.reset_index(drop=False)
        df = df.where(~df.isin([np.inf, -np.inf, np.nan]), None)
        df = df.drop(columns=list(set(df.columns) - set(fields_modifiable)))
        return df
    def search_records(self, field_names: list[str] = None, sorts: list[Sort] = None, filter: FilterInfo = None) -> Optional[pd.DataFrame]:
        ''' filter = FilterInfo.builder().conjunction("and").conditions([Condition.builder()
        .field_name("ts_code").operator("isNot").value([]).build(),]
        ).build()'''
        def inner(page_token=None) -> pd.DataFrame:
            request: SearchAppTableRecordRequest = (SearchAppTableRecordRequest.builder()
                                                    .app_token(self._bitable.app_token)
                                                    .table_id(self.id)
                                                    .page_token(page_token if (page_token is not None) else '')
                                                    .page_size(500)
                                                    .request_body(SearchAppTableRecordRequestBody.builder().field_names(field_names).sort(sorts).filter(filter).build()
                                                                  )).build()
            response: SearchAppTableRecordResponse = self._bitable.client.bitable.v1.app_table_record.search(request)
            if not response.success():
                raise ValueError(f"client.bitable.v1.app_table.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            # lark.logger.info(lark.JSON.marshal(response.data, indent=4))
            data = response.data
            indexes = [x.record_id for x in data.items]
            df = pd.DataFrame(data=[pd.Series({k: v for k, v in x.fields.items()}) for x in data.items], index=indexes)
            return data.has_more, data.page_token, df
        df = _query_has_more_list_by_page_token(inner)

        def normalize_datas_from_fs(df: pd.DataFrame, df_fields: pd.DataFrame):
            field_dict = df_fields.set_index(_FS.BITABLE.TABLE.FIELD.NAME).to_dict('index')

            def read_text(v):
                if isinstance(v, list):
                    v0 = v[0]
                    if v0['type'] in {'text', 'url'}:
                        return v0['text']
                    else:
                        return v0['text']
                return v

            def read_datetime(v):
                if v is not None:
                    return pd.to_datetime(v, unit='ms', utc=True).tz_convert('Asia/Shanghai') if v is not None else None
                else:
                    return None

            def normalize_by_col(sr: pd.Series):
                FT = _FS.BITABLE.TABLE.FIELD
                field_info = field_dict[sr.name]
                field_type = field_info[FT.TYPE]
                try:
                    if field_type in [FT.V_NUMBER, FT.V_SSELECT, FT.V_MSELECT, FT.V_TYPE_AUTO]:
                        return sr
                    elif field_type in [FT.V_TEXT]:
                        if isinstance(sr.iloc[0], str):
                            return sr
                        else:
                            return sr.apply(read_text)  # handle url
                    elif field_type in [FT.V_DATETIME]:
                        return sr.apply(read_datetime)
                    elif field_type in {FT.V_REF, FT.V_FORMULA}:
                        sr_valid = sr.dropna()  # dropna会同时过滤NaN和None
                        type_v0 = sr_valid.iloc[0]['type'] if not sr_valid.empty else None
                        if type_v0 in {FT.V_NUMBER}:
                            return sr.apply(lambda x: x['value'][0] if isinstance(x, dict) else x)
                        if type_v0 in {FT.V_TEXT}:
                            return sr.apply(lambda x: read_text(x['value'] if isinstance(x, dict) else x))
                        elif type_v0 in {FT.V_SSELECT, FT.V_MSELECT}:
                            return sr.apply(lambda x: x['value'] if isinstance(x, dict) else x)
                    elif field_type in [FT.V_FUJIAN]:
                        return sr.apply(lambda x: x[0]['name'] if isinstance(x, list) else None)
                    elif field_type > FT.V_TYPE_AUTO:
                        return sr
                except Exception as e:
                    raise ValueError(f"[readdata error]: field[{sr.name}] type[{field_type}] n_data[{len(sr)}] data:\n{sr.iloc[:5]}") from e
                raise ValueError(f"[unsupported type]: {field_type}")
            if df is None or len(df) == 0:
                return None
            df = df.apply(normalize_by_col)
            return df
        return normalize_datas_from_fs(df, self.query_fields())
    def insert_rows(self, df: pd.DataFrame):
        warnings.warn(
            "insert_rows()已废弃，请使用insert_records()替代",
            DeprecationWarning,
            stacklevel=2  # 显示调用者的代码位置，更易定位
        )
        self.insert_records(df)
    def format_type_df_before_CU(self, df: pd.DataFrame):
        FTF = _FS.BITABLE.TABLE.FIELD
        for index, row in self.query_fields().iterrows():
            _type = row[FTF.TYPE]
            col = row[FTF.NAME]
            if col not in df.columns:
                continue
            if _type in [FTF.V_TEXT, FTF.V_SSELECT]:
                df[col] = df[col].astype(str).fillna('').replace('<NA>', '')
            elif _type in [FTF.V_NUMBER]:
                df[col] = df[col].astype('Float64').fillna(0).round(5)
            elif _type in [FTF.V_DATETIME]:
                # df[col] = pd.to_datetime(df[col])
                # df[col] = df[col].fillna(pd.Timestamp(0))
                # has_tz = df[col].dt.tz is not None
                # if has_tz:
                #     df[col] = df[col].dt.tz_convert('Asia/Shanghai')
                # else:
                #     df[col] = df[col].dt.tz_localize('Asia/Shanghai')
                # df[col] = df[col].astype('Int64') // 10**6
                # df[col] = df[col].replace(-28800000, None)  # 硬编码，将None的col转回来
                df[col] = pd.to_datetime(df[col])
                epoch = pd.Timestamp(0, tz='Asia/Shanghai')
                df[col] = df[col].fillna(epoch)
                df[col] = df[col].dt.tz_convert('Asia/Shanghai') if df[col].dt.tz is not None else df[col].dt.tz_localize('Asia/Shanghai')
                df[col] = (df[col].astype('int64') // 10**6).astype('Int64')
                df[col] = df[col].replace(epoch, None)
            elif _type in [FTF.V_MSELECT]:
                def ensure_list(obj):
                    if obj is None:
                        return []
                    elif isinstance(obj, list):
                        return obj
                    elif isinstance(obj, set):
                        return list(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return [obj]
                df[col] = df[col].apply(ensure_list)
            # else:
            #     raise ValueError(f'unsupported feishu type.{_type}')
        df: pd.DataFrame = df.replace([np.inf, -np.inf, np.nan], None)
        return df
    def insert_records(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return 0

        df = self.clean_df(df)
        df = self.format_type_df_before_CU(df)
        if self.primary_fields is None or len(self.primary_fields) == 0:
            self._insert_records(df)
        else:
            df = df.drop_duplicates(subset=self.primary_fields, keep='first')

            def build_primary_filter() -> FilterInfo:
                FilterInfo().builder().conjunction('or').conditions([
                    FilterInfo().builder().conjunction('and').conditions([
                        Condition().builder().field_name(k).operator("is").value(v).build()
                        for k, v in row.items()]).build()
                    for _, row in df[self.primary_fields].iterrows()]).build()
            finfo: FilterInfo = build_primary_filter()
            df_existed = self.search_records(field_names=self.primary_fields, filter=finfo)
            count = 0
            if df_existed is not None and len(df_existed) > 0:
                field_record_id = _FS.BITABLE.TABLE.RECORD.ID
                df_existed.reset_index(names=field_record_id, inplace=True)
                df = df.merge(df_existed, on=self.primary_fields, how='left')
                df_update = df[df[field_record_id].notnull()]
                df_update = df_update.set_index(field_record_id).sort_index()
                count += self._update_rows(df_update)

                df_insert = df[df[field_record_id].isnull()].drop(columns=[field_record_id])
            else:
                df_insert = df
            count += self._insert_records(df_insert)
            return count
    def _insert_records(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return 0

        def inner(df: pd.DataFrame):
            request: BatchCreateAppTableRecordRequest = BatchCreateAppTableRecordRequest.builder() \
                .app_token(self._bitable.app_token) \
                .table_id(self.id) \
                .request_body(BatchCreateAppTableRecordRequestBody.builder()
                              .records([AppTableRecord.builder()
                                        .fields({k: v for k, v in row.items() if v is not None}).build()
                                        for _, row in df.iterrows()]).build()
                              ).build()
            response: BatchCreateAppTableRecordResponse = self._bitable.client.bitable.v1.app_table_record.batch_create(request)
            if not response.success():
                lark.logger.error(f"client.bitable.v1.app_table_record.batch_create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            # lark.logger.info(f"client.bitable.v1.app_table_record.batch_create success, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return len(df)
        return sum([inner(x) for x in batch_split(df, 1000)])
    def _update_rows(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return 0

        def inner(df: pd.DataFrame):
            request: BatchUpdateAppTableRecordRequest = (BatchUpdateAppTableRecordRequest.builder()
                                                         .app_token(self._bitable.app_token)
                                                         .table_id(self.id)
                                                         .request_body(BatchUpdateAppTableRecordRequestBody.builder()
                                                                       .records([
                                                                           AppTableRecord.builder()
                                                                           .fields({k: v for k, v in row.items() if v is not None}).record_id(index).build()
                                                                           for index, row in df.iterrows()])
                                                                       .build())
                                                         ).build()
            response: BatchUpdateAppTableRecordResponse = self._bitable.client.bitable.v1.app_table_record.batch_update(request)
            if not response.success():
                lark.logger.error(f"client.bitable.v1.app_table_record.batch_update failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            # lark.logger.info(f"client.bitable.v1.app_table_record.batch_update success, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return len(response.data.records)
        return sum([inner(x) for x in batch_split(df, 1000)])
    def del_rows(self):
        return self.del_rows_by_filter()
    def del_rows_by_filter(self, filter: FilterInfo = None):
        fields = [self.modifiable_fields[0]] if self.primary_fields is None else self.primary_fields
        df: pd.DataFrame = self.search_records(field_names=fields, filter=filter)
        if df is None or len(df) == 0:
            return 0
        record_ids = df.index.to_list()

        def inner(record_ids: list):
            request: BatchDeleteAppTableRecordRequest = (BatchDeleteAppTableRecordRequest.builder()
                                                         .app_token(self._bitable.app_token).table_id(self.id)
                                                         .request_body(BatchDeleteAppTableRecordRequestBody.builder()
                                                                       .records(record_ids).build())).build()
            response: BatchDeleteAppTableRecordResponse = self._bitable.client.bitable.v1.app_table_record.batch_delete(request)
            if not response.success():
                lark.logger.error(f"client.bitable.v1.app_table_record.batch_delete failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            # lark.logger.info(f"client.bitable.v1.app_table_record.batch_delete success, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return len(response.data.records)
        return sum([inner(x) for x in batch_split(record_ids, 500)])
    @property
    def id(self):
        return self._table_row[_FS.BITABLE.TABLE.ID]
    @property
    def name(self):
        return self._table_row[_FS.BITABLE.TABLE.NAME]
    @property
    def primary_fields(self):
        '''在字段注释中备注"Primary{n}"即可标记为主键,n的大小决定优先级 '''
        FT = _FS.BITABLE.TABLE.FIELD
        df = self.query_fields()
        df_primary = df[df[FT.DESC].notnull() & df[FT.DESC].str.startswith('Primary')]
        if df_primary is None or (len(df_primary) == 0):
            return None
        # return [df[df[FT.IS_PRIMARY]].iloc[0][FT.NAME]]
        else:
            return df_primary.sort_values([FT.DESC])[FT.NAME].to_list()

    @property
    def modifiable_fields(self):
        df_fields = self.query_fields()
        FT = _FS.BITABLE.TABLE.FIELD
        modifiable_cond = (~df_fields[FT.TYPE].isin({FT.V_FORMULA, FT.V_REF})) \
            & (df_fields[FT.TYPE] < FT.V_TYPE_AUTO)
        df_fields = df_fields.loc[modifiable_cond]
        return df_fields[FT.NAME].tolist()

class BiTable:
    def __init__(self, client, app_token: str):
        self.client: lark.client.Client = client
        self.app_token = app_token
        self._tables = None
    def create_table(self, table_name: str, table_fields: list[TableField]):
        fields = [AppTableCreateHeader.builder().field_name(x.name).type(x.fieldtype.value).build() for x in table_fields]
        request: CreateAppTableRequest = CreateAppTableRequest.builder() \
            .app_token(self.app_token) \
            .request_body(CreateAppTableRequestBody.builder()
                          .table(ReqTable.builder().name(table_name).default_view_name("默认视图").fields(fields).build())
                          .build()) \
            .build()
        response: CreateAppTableResponse = self.client.bitable.v1.app_table.create(request)
        if not response.success():
            lark.logger.error(f"client.bitable.v1.app_table.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return
        # lark.logger.info(lark.JSON.marshal(response.data, indent=4))
    def get_table(self, table_name=None, table_id=None) -> Table:
        table_row = self.query_table(table_id=table_id, table_name=table_name)
        if table_row is None:
            return None
        return Table(self, table_row)
    def query_table(self, table_id=None, table_name=None) -> str:
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
    def query_tables(self) -> pd.DataFrame:
        def inner():
            ''' fixme: 要处理has_more '''
            request: ListAppTableRequest = ListAppTableRequest.builder() \
                .app_token(self.app_token) \
                .page_size(100).build()
            response: ListAppTableResponse = self.client.bitable.v1.app_table.list(request)
            if not response.success():
                lark.logger.error(f"client.bitable.v1.app_table.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
                return
            j = response.data
            return pd.DataFrame(pd.Series({_FS.BITABLE.TABLE.NAME: x.name, _FS.BITABLE.TABLE.ID: x.table_id, _FS.BITABLE.TABLE.REVISION: x.revision}) for x in j.items)
        if self._tables is None:
            self._tables = inner()
        return self._tables

class Feishu:
    def __init__(self, app_id: str = None, app_secret: str = None, log_level=lark.LogLevel.INFO):
        self.client = lark.Client.builder() \
            .app_id(app_id=app_id) \
            .app_secret(app_secret=app_secret) \
            .log_level(log_level) \
            .build()
    def get_bitable(self, app_token):
        return BiTable(self.client, app_token)
