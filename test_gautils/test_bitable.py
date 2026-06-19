import pandas as pd
import unittest
import numpy as np
import json
import sys
import time

from gautils.feishu.core import BiTable, Feishu, TableField
from lark_oapi.api.bitable.v1 import *
from test_gautils.env import require_env

class TestBiTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_id, app_secret, app_token, table_id = require_env(
            'FEISHU_APP_ID',
            'FEISHU_APP_SECRET',
            'FEISHU_BITABLE_APP_TOKEN',
            'FEISHU_BITABLE_TABLE_ID',
        )
        cls.fs = Feishu(app_id, app_secret)
        # 使用固定的bitable
        cls.bitable_name = f'unittest_temp_{int(time.time())}'
        cls.bitable: BiTable = cls.fs.get_bitable(app_token)
        cls.table = cls.bitable.get_table(table_id=table_id)

    @classmethod
    def tearDownClass(cls):
        # 注意：飞书暂无删除多维表格API，只能手动清理
        pass

    def test_querytables(self):
        print(self.bitable.query_tables())
        print(self.table.id, self.table.name)

    def test_queryfields(self):
        print('字段信息:\n', self.table.query_fields())
        print('主键字段:', self.table.primary_fields)
        print('可修改字段:', self.table.modifiable_fields)

    def test_all_types_CRUD(self):
        """测试所有字段类型的增删改查，创建>5条数据，删除3条保留2条"""
        from datetime import datetime
        import time

        # 生成唯一标记
        test_marker = f"test_{int(time.time())}"

        # 插入5条常规数据（带标记便于后续删除）
        df_normal = pd.DataFrame([
            {
                '文本': f'常规数据1-{test_marker}-待删除',
                '数字': 100.0,
                '单选': '选项A',
                '多选': ['标签1'],
                '日期': datetime.now(),
                '复选框': True,
                '人员': [],
            },
            {
                '文本': f'常规数据2-{test_marker}-待删除',
                '数字': 200.0,
                '单选': '选项B',
                '多选': ['标签1', '标签2'],
                '日期': datetime.now(),
                '复选框': False,
                '人员': [],
            },
            {
                '文本': f'常规数据3-{test_marker}-待删除',
                '数字': 300.0,
                '单选': '选项A',
                '多选': ['标签3'],
                '日期': datetime.now(),
                '复选框': True,
                '人员': [],
            },
            {
                '文本': f'常规数据4-{test_marker}-保留',
                '数字': 400.0,
                '单选': '选项C',
                '多选': ['标签1', '标签3'],
                '日期': datetime.now(),
                '复选框': False,
                '人员': [],
            },
            {
                '文本': f'常规数据5-{test_marker}-保留',
                '数字': 500.0,
                '单选': '选项B',
                '多选': ['标签2'],
                '日期': datetime.now(),
                '复选框': True,
                '人员': [],
            },
        ])
        print('插入5条常规记录:', self.table.insert_records(df_normal))

        # 插入边界数据（保留供人工检查）
        df_boundary = pd.DataFrame([
            {
                '文本': f'边界-空值-{test_marker}-保留',
                '数字': 0,
                '单选': '',
                '多选': [],
                '日期': datetime(1970, 1, 1),
                '复选框': False,
                '人员': [],
            },
            {
                '文本': f'边界-超长-{test_marker}-保留',
                '数字': 999999.99,
                '单选': '超长' + 'A' * 50,
                '多选': [f'选项{i}' for i in range(20)],
                '日期': datetime(2099, 12, 31),
                '复选框': True,
                '人员': [],
            },
        ])
        print('插入2条边界记录:', self.table.insert_records(df_boundary))

        # 查询所有记录
        records = self.table.search_records()
        print(f'查询到 {len(records) if records is not None else 0} 条记录')

        # 删除标记为"待删除"的3条记录
        f0 = FilterInfo.builder().conjunction('and').conditions([
            Condition.builder().field_name('文本').operator('contains').value([f'{test_marker}-待删除']).build(),
        ]).build()
        deleted_count = self.table.del_rows_by_filter(filter=f0)
        print(f'已删除 {deleted_count} 条标记记录')

        # 验证剩余记录数
        remaining = self.table.search_records()
        keep_count = len(remaining[remaining['文本'].str.contains(f'{test_marker}-保留', na=False)]) if remaining is not None else 0
        print(f'保留 {keep_count} 条记录供人工检查')
        print(f'表名: {self.table.name}, 标记: {test_marker}')

    def test_filter_operations(self):
        """测试各种过滤操作"""
        from datetime import datetime

        # 插入测试数据
        df_test = pd.DataFrame([
            {'文本': 'filter_test_A', '数字': 100, '单选': '类型A'},
            {'文本': 'filter_test_B', '数字': 200, '单选': '类型B'},
            {'文本': 'filter_test_C', '数字': 300, '单选': '类型A'},
        ])
        self.table.insert_records(df_test)

        # 测试数字过滤
        f_number = FilterInfo.builder().conjunction('and').conditions([
            Condition.builder().field_name('数字').operator('isGreater').value([150]).build(),
        ]).build()
        result = self.table.search_records(filter=f_number)
        print(f'数字>150: {len(result) if result is not None else 0} 条')

        # 测试文本包含
        f_text = FilterInfo.builder().conjunction('and').conditions([
            Condition.builder().field_name('文本').operator('contains').value(['filter_test']).build(),
        ]).build()
        result = self.table.search_records(filter=f_text)
        print(f'文本包含filter_test: {len(result) if result is not None else 0} 条')

        # 测试单选过滤
        f_select = FilterInfo.builder().conjunction('and').conditions([
            Condition.builder().field_name('单选').operator('is').value(['类型A']).build(),
        ]).build()
        result = self.table.search_records(filter=f_select)
        print(f'单选=类型A: {len(result) if result is not None else 0} 条')

        print(f'过滤测试数据保留在表: {self.table.name}')
