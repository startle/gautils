import warnings
import json
import pandas as pd
import numpy as np
from typing import Optional, List
import lark_oapi as lark
from lark_oapi.api.sheets.v3 import *
from ...utils import batch_split

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="lark_oapi.ws.pb.google.__init__",
    message="pkg_resources is deprecated as an API"
)


class _FS:
    class SHEET:
        ID = 'sheet_id'
        TITLE = 'title'
        INDEX = 'index'
        ROW_COUNT = 'row_count'
        COL_COUNT = 'column_count'
        class CELL:
            VALUE = 'value'
            FORMULA = 'formula'
            TEXT = 'text'


class Sheet:
    """飞书电子表格中的工作表"""

    def __init__(self, spreadsheet: 'Spreadsheet', sheet_info: dict):
        self._spreadsheet = spreadsheet
        self._sheet_info = sheet_info
        self._fields = None

    @property
    def id(self) -> str:
        return self._sheet_info.get(_FS.SHEET.ID)

    @property
    def title(self) -> str:
        return self._sheet_info.get(_FS.SHEET.TITLE)

    @property
    def index(self) -> int:
        return self._sheet_info.get(_FS.SHEET.INDEX, 0)

    @property
    def row_count(self) -> int:
        return self._sheet_info.get(_FS.SHEET.ROW_COUNT, 0)

    @property
    def col_count(self) -> int:
        return self._sheet_info.get(_FS.SHEET.COL_COUNT, 0)

    def _build_range(self, start_cell: str = None, end_cell: str = None) -> str:
        """构建范围字符串，如 sheetId!A1:C10"""
        if start_cell:
            if end_cell:
                return f"{self.id}!{start_cell}:{end_cell}"
            return f"{self.id}!{start_cell}"
        return f"{self.id}"

    def read(self, range_str: str = None, start_cell: str = None, end_cell: str = None,
             value_render: str = 'FormattedValue') -> Optional[pd.DataFrame]:
        """读取工作表数据（当前版本的lark_oapi SDK暂不支持直接读取单元格值）

        Args:
            range_str: 完整的范围字符串，如 "sheetId!A1:C10"，如果提供则忽略start_cell/end_cell
            start_cell: 起始单元格，如 "A1"
            end_cell: 结束单元格，如 "C10"
            value_render: 值渲染选项，可选 'FormattedValue'(格式化值)/'ToString'(转为字符串)/'Formula'(公式)

        Returns:
            DataFrame，列名为A, B, C...，索引为行号
        """
        # TODO: 当前版本的 lark_oapi SDK 没有提供直接的 spreadsheet_values API
        # 需要使用 FindSpreadsheetSheetRequest 或其他方式实现
        lark.logger.warning("Sheet.read() 方法在当前版本的SDK中暂未实现，需要使用 Find API 或其他方式读取单元格数据")
        return pd.DataFrame()

    def write(self, df: pd.DataFrame, start_cell: str = 'A1', value_input: str = 'RAW') -> bool:
        """写入数据到工作表（当前版本的lark_oapi SDK暂不支持直接写入单元格值）

        Args:
            df: 要写入的DataFrame
            start_cell: 起始单元格，如 "A1"
            value_input: 值输入选项，'RAW'(原始值)/'USER_ENTERED'(用户输入)

        Returns:
            是否成功
        """
        # TODO: 当前版本的 lark_oapi SDK 没有提供直接的 spreadsheet_values API
        lark.logger.warning("Sheet.write() 方法在当前版本的SDK中暂未实现")
        return False

    def append(self, df: pd.DataFrame, value_input: str = 'RAW') -> bool:
        """追加数据到工作表末尾（当前版本的lark_oapi SDK暂不支持）

        Args:
            df: 要追加的DataFrame
            value_input: 值输入选项，'RAW'(原始值)/'USER_ENTERED'(用户输入)

        Returns:
            是否成功
        """
        # TODO: 当前版本的 lark_oapi SDK 没有提供直接的 spreadsheet_values API
        lark.logger.warning("Sheet.append() 方法在当前版本的SDK中暂未实现")
        return False

    def clear(self, range_str: str = None, start_cell: str = None, end_cell: str = None) -> bool:
        """清空指定范围的数据（当前版本的lark_oapi SDK暂不支持）

        Args:
            range_str: 完整的范围字符串
            start_cell: 起始单元格
            end_cell: 结束单元格
        """
        # TODO: 当前版本的 lark_oapi SDK 没有提供直接的 spreadsheet_values API
        lark.logger.warning("Sheet.clear() 方法在当前版本的SDK中暂未实现")
        return False

    @staticmethod
    def _col_index_to_letter(index: int) -> str:
        """将列索引转换为字母（0 -> A, 1 -> B...）"""
        result = ""
        index = index + 1  # 1-based
        while index > 0:
            index, remainder = divmod(index - 1, 26)
            result = chr(65 + remainder) + result
        return result

    @staticmethod
    def _col_letter_add(letter: str, add: int) -> str:
        """列字母相加（A + 2 = C）"""
        index = 0
        for char in letter:
            index = index * 26 + (ord(char) - ord('A') + 1)
        index += add
        return Sheet._col_index_to_letter(index - 1)


class Spreadsheet:
    """飞书电子表格"""

    def __init__(self, client, spreadsheet_token: str):
        self.client: lark.client.Client = client
        self.spreadsheet_token = spreadsheet_token
        self._sheets = None
        self._metadata = None

    def get_sheet(self, sheet_id: str = None, sheet_title: str = None) -> Optional[Sheet]:
        """获取工作表

        Args:
            sheet_id: 工作表ID
            sheet_title: 工作表标题（名称）

        Returns:
            Sheet对象，未找到返回None
        """
        sheet_info = self._query_sheet(sheet_id=sheet_id, sheet_title=sheet_title)
        if sheet_info is None:
            return None
        return Sheet(self, sheet_info)

    def _query_sheet(self, sheet_id: str = None, sheet_title: str = None) -> Optional[dict]:
        """查询工作表信息"""
        sheets = self.query_sheets()
        if not sheets:
            return None

        if sheet_id:
            for sheet in sheets:
                if sheet.get(_FS.SHEET.ID) == sheet_id:
                    return sheet
        elif sheet_title:
            for sheet in sheets:
                if sheet.get(_FS.SHEET.TITLE) == sheet_title:
                    return sheet
        return None

    def query_sheets(self) -> List[dict]:
        """查询所有工作表信息"""
        if self._sheets is not None:
            return self._sheets

        request: QuerySpreadsheetSheetRequest = QuerySpreadsheetSheetRequest.builder() \
            .spreadsheet_token(self.spreadsheet_token) \
            .build()

        response: QuerySpreadsheetSheetResponse = self.client.sheets.v3.spreadsheet_sheet.query(request)

        if not response.success():
            lark.logger.error(f"sheets.v3.spreadsheet_sheet.query failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return []

        sheets = []
        if response.data and response.data.sheets:
            for s in response.data.sheets:
                # row_count 和 column_count 在 grid_properties 中
                row_count = None
                col_count = None
                if s.grid_properties:
                    row_count = s.grid_properties.row_count
                    col_count = s.grid_properties.column_count
                sheets.append({
                    _FS.SHEET.ID: s.sheet_id,
                    _FS.SHEET.TITLE: s.title,
                    _FS.SHEET.INDEX: s.index,
                    _FS.SHEET.ROW_COUNT: row_count,
                    _FS.SHEET.COL_COUNT: col_count,
                })

        self._sheets = sheets
        return sheets

    def create_sheet(self, title: str, row_count: int = 1000, column_count: int = 20) -> Optional[Sheet]:
        """创建工作表

        Args:
            title: 工作表标题
            row_count: 行数
            column_count: 列数

        Returns:
            新创建的Sheet对象
        """
        from lark_oapi.api.sheets.v3 import AddSheetRequest, AddSheetRequestBody, AddSheetResponse, SheetProperties, GridProperties

        request: AddSheetRequest = AddSheetRequest.builder() \
            .spreadsheet_token(self.spreadsheet_token) \
            .request_body(AddSheetRequestBody.builder()
                          .sheets([SheetProperties.builder()
                                   .title(title)
                                   .index(str(len(self.query_sheets())))
                                   .grid_properties(GridProperties.builder()
                                                    .row_count(row_count)
                                                    .column_count(column_count)
                                                    .build())
                                   .build()])
                          .build()) \
            .build()

        response: AddSheetResponse = self.client.sheets.v3.spreadsheet_sheet.add(request)

        if not response.success():
            lark.logger.error(f"sheets.v3.spreadsheet_sheet.add failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return None

        self._sheets = None  # 清除缓存
        return self.get_sheet(sheet_title=title)

    def delete_sheet(self, sheet_id: str = None, sheet_title: str = None) -> bool:
        """删除工作表

        Args:
            sheet_id: 工作表ID
            sheet_title: 工作表标题
        """
        if sheet_id is None:
            if sheet_title is None:
                raise ValueError('sheet_id or sheet_title is required')
            sheet_info = self._query_sheet(sheet_title=sheet_title)
            if sheet_info is None:
                return True
            sheet_id = sheet_info[_FS.SHEET.ID]

        from lark_oapi.api.sheets.v3 import DeleteSheetRequest, DeleteSheetRequestBody, DeleteSheetResponse

        request: DeleteSheetRequest = DeleteSheetRequest.builder() \
            .spreadsheet_token(self.spreadsheet_token) \
            .request_body(DeleteSheetRequestBody.builder()
                          .sheets([sheet_id])
                          .build()) \
            .build()

        response: DeleteSheetResponse = self.client.sheets.v3.spreadsheet_sheet.delete(request)

        if not response.success():
            lark.logger.error(f"sheets.v3.spreadsheet_sheet.delete failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return False

        self._sheets = None  # 清除缓存
        return True

    def get_metadata(self) -> dict:
        """获取电子表格元数据"""
        if self._metadata is not None:
            return self._metadata

        request: GetSpreadsheetRequest = GetSpreadsheetRequest.builder() \
            .spreadsheet_token(self.spreadsheet_token) \
            .build()

        response: GetSpreadsheetResponse = self.client.sheets.v3.spreadsheet.get(request)

        if not response.success():
            lark.logger.error(f"sheets.v3.spreadsheet.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return {}

        metadata = {
            'title': response.data.title if response.data else None,
            'token': self.spreadsheet_token,
            'url': response.data.url if response.data else None,
        }
        self._metadata = metadata
        return metadata

    def add_collaborator(self, member_id: str, member_id_type: str = 'openid', perm: str = 'full_access') -> bool:
        """添加协作者

        Args:
            member_id: 用户标识
            member_id_type: 标识类型，openid/userid/unionid/email
            perm: 权限，full_access(完全权限)/edit(可编辑)/view(可查看)
        """
        from lark_oapi.api.drive.v1 import CreatePermissionMemberRequest, CreatePermissionMemberResponse

        request = CreatePermissionMemberRequest.builder() \
            .token(self.spreadsheet_token) \
            .type('sheet') \
            .need_notification(False) \
            .request_body({
                'member_type': member_id_type,
                'member_id': member_id,
                'perm': perm
            }) \
            .build()

        response: CreatePermissionMemberResponse = self.client.drive.v1.permission_member.create(request)

        if not response.success():
            lark.logger.error(f"drive.v1.permission_member.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return False
        return True
