import json
from enum import Enum
import lark_oapi as lark
from lark_oapi.api.bitable.v1 import *

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
class BiTable:
    def __init__(self, client, app_token: str):
        self.client = client
        self.app_token = app_token
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
        # 处理业务结果
        lark.logger.info(lark.JSON.marshal(response.data, indent=4))
class Feishu:
    def __init__(self, app_id: str = None, app_secret: str = None):
        self.client = lark.Client.builder() \
            .app_id(app_id=app_id) \
            .app_secret(app_secret=app_secret) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()
    def get_bitable(self, app_token):
        return BiTable(self.client, app_token)
