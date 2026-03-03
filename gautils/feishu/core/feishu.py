from typing import Literal
import json
import pandas as pd
import lark_oapi as lark
from lark_oapi.api.wiki.v2 import GetNodeSpaceRequest, GetNodeSpaceResponse
from lark_oapi.api.bitable.v1 import CreateAppRequest, CreateAppResponse

from .client import FeishuClient
from .bitable import BiTable, Table, TableField
from .spreadsheets import Spreadsheet, Sheet
from .wiki import Wiki


class Feishu(FeishuClient):
    """飞书 API 主入口

    提供对飞书各种功能的访问：
    - 多维表格 (Bitable)
    - 电子表格 (Spreadsheet)
    - 知识库 (Wiki)
    """

    def get_bitable(self, app_token) -> BiTable:
        """获取多维表格

        Args:
            app_token: 多维表格token

        Returns:
            BiTable对象
        """
        return BiTable(self.client, app_token)

    def get_spreadsheet(self, spreadsheet_token: str) -> Spreadsheet:
        """获取电子表格

        Args:
            spreadsheet_token: 电子表格token（从URL中获取）

        Returns:
            Spreadsheet对象
        """
        return Spreadsheet(self.client, spreadsheet_token)

    def get_wiki(self) -> Wiki:
        """获取知识库管理对象

        Returns:
            Wiki对象
        """
        return Wiki(self.client)

    def create_bitable(self, name: str, folder_token: str = None) -> BiTable:
        """创建多维表格

        Args:
            name: 多维表格名称
            folder_token: 文件夹token（可选，不传则创建在"我的多维表格"）

        Returns:
            BiTable对象
        """
        request: CreateAppRequest = CreateAppRequest.builder() \
            .request_body({'name': name, 'folder_token': folder_token}) \
            .build()
        response: CreateAppResponse = self.client.bitable.v1.app.create(request)
        if not response.success():
            lark.logger.error(f"client.bitable.v1.app.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return None
        app_token = response.data.app.app_token
        return BiTable(self.client, app_token)

    def get_bot_open_id(self) -> str:
        """获取应用机器人的 open_id

        Returns:
            机器人的 open_id
        """
        from lark_oapi.api.bot.v3 import GetBotInfoRequest, GetBotInfoResponse

        request: GetBotInfoRequest = GetBotInfoRequest.builder().build()
        response: GetBotInfoResponse = self.client.bot.v3.info.get(request)

        if not response.success():
            lark.logger.error(f"client.bot.v3.info.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return None

        return response.data.bot.open_id if response.data and response.data.bot else None

    def query_from_space(self, wiki_token, obj_type: Literal['docx', 'sheet', 'bitable']) -> pd.DataFrame:
        """从知识空间查询节点信息

        Args:
            wiki_token: 知识节点token
            obj_type: 对象类型

        Returns:
            DataFrame
        """
        request: GetNodeSpaceRequest = GetNodeSpaceRequest.builder().token(wiki_token).obj_type(obj_type).build()
        response: GetNodeSpaceResponse = self.client.wiki.v2.space.get_node(request)
        if not response.success():
            lark.logger.error(
                f"client.wiki.v2.space.get_node failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return
        lark.logger.info(lark.JSON.marshal(response.data, indent=4))
        return None
