import sys
import pandas as pd
import lark_oapi as lark
from lark_oapi.api.bitable.v1 import *


class FS:
    def __init__(self, app_id, app_secret):
        self.client: lark.Client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()
    def get_bitable(self, app_token):
        return BiTable(self, app_token)

class BiTable:
    def __init__(self, fs: FS, token):
        self.fs = fs
        self.app_token = token
        self.client = self.fs.client
    def create_table(self, table_name):
        request: BatchCreateAppTableRequest = BatchCreateAppTableRequest.builder() \
            .app_token("appbcbWCzen6D8dezhoCH2RpMAh") \
            .user_id_type("user_id") \
            .request_body(BatchCreateAppTableRequestBody.builder()
                          .tables([])
                          .build()) \
            .build()

        # 发起请求
        response: BatchCreateAppTableResponse = self.client.bitable.v1.app_table.batch_create(request)
        if not response.success():
            lark.logger.error(f"client.bitable.v1.app_table.batch_create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return
    def get_table(self, table_name=None, table_id=None):
        table_row = self.query_table(table_id=table_id, table_name=table_name)
        if table_row is None:
            return None
        return Table(self, table_id=table_row['id'])
    def query_tables(self) -> pd.DataFrame:
        request: ListAppTableRequest = ListAppTableRequest.builder() \
            .app_token(self.app_token) \
            .build()
        response: ListAppTableResponse = self.client.bitable.v1.app_table.list(request)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.bitable.v1.app_table.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return
        if response.data.items is None:
            return None
        df = pd.DataFrame([pd.Series({'id': x.table_id, 'name': x.name}) for x in response.data.items])
        return df

    def query_table(self, table_name=None, table_id=None) -> str:
        df_table = self.query_tables()
        if df_table is None or len(df_table) == 0:
            return None
        if table_id is not None:
            if table_id in df_table['id'].to_list():
                return df_table[df_table['id'] == table_id].iloc[0]
        elif table_name is not None:
            if table_name in df_table['name'].to_list():
                return df_table[df_table['name'] == table_name].iloc[0]
        return None
class Table:
    def __init__(self, bitable: BiTable, table_id: str):
        self.bitable = bitable
        self.app_token = bitable.app_token
        self.client = bitable.client
        self.table_id = table_id
    def list_fields(self):
        request: ListAppTableFieldRequest = ListAppTableFieldRequest.builder() \
            .app_token(self.app_token) \
            .table_id(self.table_id) \
            .text_field_as_array(True) \
            .build()
        response: ListAppTableFieldResponse = self.client.bitable.v1.app_table_field.list(request)
        if not response.success():
            lark.logger.error(f"client.bitable.v1.app_table_field.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return None
        return pd.DataFrame([pd.Series({'name': x.field_name, 'type': x.type}) for x in response.data.items])
    def list_records(self):
        request: ListAppTableRecordRequest = ListAppTableRecordRequest.builder() \
            .app_token(self.bitable.app_token) \
            .table_id(self.table_id) \
            .build()
        response: ListAppTableRecordResponse = self.bitable.client.bitable.v1.app_table_record.get(request)
        lark.logger.info(lark.JSON.marshal(response.data, indent=4))
        # response.data.
        fields = self.list_fields()
        print(fields)
        # [x.fields[] for x in response.data.items]
    def insert_rows(self, df: pd.DataFrame):
        pass
    def delete_rows(self):
        pass
