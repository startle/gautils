import pandas as pd
import unittest
import numpy as np
import json
import sys
import time
from datetime import datetime

from gautils.feishu.core import Spreadsheet, Feishu, Sheet
from test_gautils.env import require_env


class TestSpreadsheet(unittest.TestCase):
    """飞书电子表格单元测试"""

    @classmethod
    def setUpClass(cls):
        app_id, app_secret, spreadsheet_token = require_env(
            'FEISHU_APP_ID',
            'FEISHU_APP_SECRET',
            'FEISHU_SPREADSHEET_TOKEN',
        )
        cls.fs = Feishu(app_id, app_secret)
        # 使用指定的电子表格token
        cls.spreadsheet_token = spreadsheet_token
        cls.spreadsheet: Spreadsheet = cls.fs.get_spreadsheet(cls.spreadsheet_token)
        cls.test_marker = f"unittest_{int(time.time())}"

    @classmethod
    def tearDownClass(cls):
        # 清理：删除测试中创建的工作表（保留原始工作表）
        pass

    def test_query_sheets(self):
        """测试查询所有工作表"""
        sheets = self.spreadsheet.query_sheets()
        print(f'工作表列表: {len(sheets)} 个')
        for sheet in sheets:
            print(f"  - {sheet['title']} (id={sheet['sheet_id']}, rows={sheet['row_count']}, cols={sheet['column_count']})")
        self.assertIsInstance(sheets, list)
        self.assertGreater(len(sheets), 0)

    def test_get_sheet_by_title(self):
        """测试通过标题获取工作表"""
        sheets = self.spreadsheet.query_sheets()
        if sheets:
            first_sheet_title = sheets[0]['title']
            sheet = self.spreadsheet.get_sheet(sheet_title=first_sheet_title)
            print(f'通过标题获取工作表: {sheet.title if sheet else None}')
            self.assertIsNotNone(sheet)
            self.assertEqual(sheet.title, first_sheet_title)

    def test_get_sheet_by_id(self):
        """测试通过ID获取工作表"""
        sheets = self.spreadsheet.query_sheets()
        if sheets:
            first_sheet_id = sheets[0]['sheet_id']
            sheet = self.spreadsheet.get_sheet(sheet_id=first_sheet_id)
            print(f'通过ID获取工作表: {sheet.id if sheet else None}')
            self.assertIsNotNone(sheet)
            self.assertEqual(sheet.id, first_sheet_id)

    # def test_create_and_delete_sheet(self):
    #     """测试创建和删除工作表（需要写权限，暂时注释）"""
    #     sheet_title = f'测试表_{self.test_marker}'
    #
    #     # 创建工作表
    #     new_sheet = self.spreadsheet.create_sheet(
    #         title=sheet_title,
    #         row_count=100,
    #         column_count=10
    #     )
    #     print(f'创建工作表: {new_sheet.title if new_sheet else None}')
    #     self.assertIsNotNone(new_sheet)
    #     self.assertEqual(new_sheet.title, sheet_title)
    #     self.assertEqual(new_sheet.row_count, 100)
    #     self.assertEqual(new_sheet.col_count, 10)
    #
    #     # 删除工作表
    #     result = self.spreadsheet.delete_sheet(sheet_title=sheet_title)
    #     print(f'删除工作表结果: {result}')
    #     self.assertTrue(result)
    #
    #     # 验证已删除
    #     deleted_sheet = self.spreadsheet.get_sheet(sheet_title=sheet_title)
    #     self.assertIsNone(deleted_sheet)

    # def test_sheet_read_write(self):
    #     """测试工作表读写操作（需要写权限，暂时注释）"""
    #     sheet_title = f'读写测试_{self.test_marker}'
    #
    #     # 创建工作表
    #     sheet = self.spreadsheet.create_sheet(title=sheet_title, row_count=50, column_count=10)
    #     self.assertIsNotNone(sheet)
    #
    #     try:
    #         # 准备测试数据
    #         df_write = pd.DataFrame({
    #             '姓名': ['张三', '李四', '王五'],
    #             '年龄': [25, 30, 35],
    #             '分数': [85.5, 92.0, 78.5],
    #             '日期': [datetime.now().strftime('%Y-%m-%d')] * 3,
    #         })
    #
    #         # 写入数据
    #         write_result = sheet.write(df_write, start_cell='A1')
    #         print(f'写入数据结果: {write_result}')
    #         self.assertTrue(write_result)
    #
    #         # 读取数据
    #         df_read = sheet.read(start_cell='A1', end_cell='D4')
    #         print(f'读取数据:\n{df_read}')
    #         self.assertIsNotNone(df_read)
    #         self.assertGreater(len(df_read), 0)
    #
    #         # 验证数据（考虑表头行）
    #         self.assertEqual(len(df_read), 4)  # 表头 + 3行数据
    #
    #     finally:
    #         # 清理
    #         self.spreadsheet.delete_sheet(sheet_title=sheet_title)

    # def test_sheet_append(self):
    #     """测试工作表追加数据（需要写权限，暂时注释）"""
    #     sheet_title = f'追加测试_{self.test_marker}'
    #
    #     # 创建工作表
    #     sheet = self.spreadsheet.create_sheet(title=sheet_title, row_count=50, column_count=5)
    #     self.assertIsNotNone(sheet)
    #
    #     try:
    #         # 先写入表头
    #         df_header = pd.DataFrame(columns=['产品', '销量', '价格'])
    #         sheet.write(df_header, start_cell='A1')
    #
    #         # 追加数据
    #         df_append1 = pd.DataFrame({
    #             '产品': ['产品A', '产品B'],
    #             '销量': [100, 200],
    #             '价格': [50.0, 80.0],
    #         })
    #         result1 = sheet.append(df_append1)
    #         print(f'第一次追加结果: {result1}')
    #         self.assertTrue(result1)
    #
    #         # 再次追加
    #         df_append2 = pd.DataFrame({
    #             '产品': ['产品C'],
    #             '销量': [150],
    #             '价格': [60.0],
    #         })
    #         result2 = sheet.append(df_append2)
    #         print(f'第二次追加结果: {result2}')
    #         self.assertTrue(result2)
    #
    #         # 验证总数据量
    #         df_read = sheet.read(start_cell='A1')
    #         print(f'追加后数据:\n{df_read}')
    #         # 表头1行 + 第一次追加2行 + 第二次追加1行 = 4行
    #         self.assertGreaterEqual(len(df_read), 3)
    #
    #     finally:
    #         # 清理
    #         self.spreadsheet.delete_sheet(sheet_title=sheet_title)

    # def test_sheet_clear(self):
    #     """测试清空工作表数据（需要写权限，暂时注释）"""
    #     sheet_title = f'清空测试_{self.test_marker}'
    #
    #     # 创建工作表
    #     sheet = self.spreadsheet.create_sheet(title=sheet_title, row_count=30, column_count=5)
    #     self.assertIsNotNone(sheet)
    #
    #     try:
    #         # 写入数据
    #         df = pd.DataFrame({
    #             'A': [1, 2, 3],
    #             'B': ['x', 'y', 'z'],
    #         })
    #         sheet.write(df, start_cell='A1')
    #
    #         # 验证数据已写入
    #         df_before = sheet.read(start_cell='A1')
    #         print(f'清空前数据行数: {len(df_before)}')
    #         self.assertGreater(len(df_before), 0)
    #
    #         # 清空数据
    #         clear_result = sheet.clear(start_cell='A1', end_cell='B4')
    #         print(f'清空结果: {clear_result}')
    #         self.assertTrue(clear_result)
    #
    #     finally:
    #         # 清理
    #         self.spreadsheet.delete_sheet(sheet_title=sheet_title)

    def test_sheet_properties(self):
        """测试工作表属性"""
        sheets = self.spreadsheet.query_sheets()
        if sheets:
            sheet_info = sheets[0]
            sheet = self.spreadsheet.get_sheet(sheet_id=sheet_info['sheet_id'])

            print(f'工作表属性:')
            print(f'  id: {sheet.id}')
            print(f'  title: {sheet.title}')
            print(f'  index: {sheet.index}')
            print(f'  row_count: {sheet.row_count}')
            print(f'  col_count: {sheet.col_count}')

            self.assertIsNotNone(sheet.id)
            self.assertIsNotNone(sheet.title)
            self.assertIsInstance(sheet.index, int)
            self.assertIsInstance(sheet.row_count, int)
            self.assertIsInstance(sheet.col_count, int)

    # def test_read_with_value_render(self):
    #     """测试不同值渲染选项的读取（需要写权限，暂时注释）"""
    #     sheet_title = f'渲染测试_{self.test_marker}'
    #
    #     sheet = self.spreadsheet.create_sheet(title=sheet_title, row_count=20, column_count=3)
    #     self.assertIsNotNone(sheet)
    #
    #     try:
    #         # 写入测试数据
    #         df = pd.DataFrame({
    #             '文本': ['Hello', 'World'],
    #             '数字': [123.456, 789.012],
    #         })
    #         sheet.write(df, start_cell='A1')
    #
    #         # 使用不同渲染选项读取
    #         df_formatted = sheet.read(start_cell='A1', end_cell='B3', value_render='FormattedValue')
    #         print(f'格式化值:\n{df_formatted}')
    #
    #         df_string = sheet.read(start_cell='A1', end_cell='B3', value_render='ToString')
    #         print(f'字符串值:\n{df_string}')
    #
    #         self.assertIsNotNone(df_formatted)
    #         self.assertIsNotNone(df_string)
    #
    #     finally:
    #         self.spreadsheet.delete_sheet(sheet_title=sheet_title)

    # def test_range_string_read(self):
    #     """测试使用范围字符串读取（API暂不支持，注释掉）"""
    #     sheets = self.spreadsheet.query_sheets()
    #     if sheets:
    #         sheet_info = sheets[0]
    #         sheet = self.spreadsheet.get_sheet(sheet_id=sheet_info['sheet_id'])
    #
    #         # 使用范围字符串读取
    #         range_str = f"{sheet.id}!A1:C5"
    #         df = sheet.read(range_str=range_str)
    #         print(f'使用范围字符串读取 {range_str}:')
    #         print(df)
    #         self.assertIsNotNone(df)

    # def test_large_data_write(self):
    #     """测试大数据量写入（需要写权限，暂时注释）"""
    #     sheet_title = f'大数据测试_{self.test_marker}'
    #
    #     sheet = self.spreadsheet.create_sheet(title=sheet_title, row_count=200, column_count=10)
    #     self.assertIsNotNone(sheet)
    #
    #     try:
    #         # 生成100行测试数据
    #         import random
    #         df_large = pd.DataFrame({
    #             '序号': range(1, 101),
    #             '数值A': [random.uniform(0, 1000) for _ in range(100)],
    #             '数值B': [random.uniform(-500, 500) for _ in range(100)],
    #             '类别': [random.choice(['A', 'B', 'C']) for _ in range(100)],
    #         })
    #
    #         write_result = sheet.write(df_large, start_cell='A1')
    #         print(f'大数据写入结果: {write_result}')
    #         self.assertTrue(write_result)
    #
    #         # 读取验证
    #         df_read = sheet.read(start_cell='A1', end_cell='D102')
    #         print(f'大数据读取行数: {len(df_read)}')
    #         self.assertGreaterEqual(len(df_read), 100)
    #
    #     finally:
    #         self.spreadsheet.delete_sheet(sheet_title=sheet_title)


if __name__ == '__main__':
    unittest.main()
