from typing import Literal
import json
import pandas as pd
import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.bitable.v1 import CreateAppRequest, CreateAppResponse
from .bitable import BiTable, Table, TableField

__all__ = ['Feishu', 'BiTable', 'Table', 'TableField']


class Feishu:
    def __init__(self, app_id: str = None, app_secret: str = None, log_level=lark.LogLevel.INFO):
        self.client = lark.Client.builder() \
            .app_id(app_id=app_id) \
            .app_secret(app_secret=app_secret) \
            .log_level(log_level) \
            .build()

    def get_bitable(self, app_token):
        return BiTable(self.client, app_token)

    def create_bitable(self, name: str, folder_token: str = None) -> BiTable:
        '''创建多维表格
        name: 多维表格名称
        folder_token: 文件夹token（可选，不传则创建在"我的多维表格"）
        '''
        request: CreateAppRequest = CreateAppRequest.builder() \
            .request_body({'name': name, 'folder_token': folder_token}) \
            .build()
        response: CreateAppResponse = self.client.bitable.v1.app.create(request)
        if not response.success():
            lark.logger.error(f"client.bitable.v1.app.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return None
        app_token = response.data.app.app_token
        return BiTable(self.client, app_token)

    def query_from_space(self, wiki_token, obj_type: Literal['docx', 'sheet', 'bitable']) -> pd.DataFrame:
        request: GetNodeSpaceRequest = GetNodeSpaceRequest.builder().token(wiki_token).obj_type(obj_type).build()
        response: GetNodeSpaceResponse = self.client.wiki.v2.space.get_node(request)
        if not response.success():
            lark.logger.error(
                f"client.wiki.v2.space.get_node failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return
        lark.logger.info(lark.JSON.marshal(response.data, indent=4))
        return None
