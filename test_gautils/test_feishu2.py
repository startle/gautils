import pandas as pd
import unittest
import numpy as np
import json
import sys

from gautils.feishu.core import BiTable, Feishu
from lark_oapi.api.bitable.v1 import *

class TestFeishu2(unittest.TestCase):
    def setUp(self):
        self.fs = Feishu('cli_a5e0366e397cd00e', 'abi2ikgvkTDr2V3K8aykphJfVC0YUrGR')
        self.bitable: BiTable = self.fs.get_bitable('Hr1rbmwpna7HOPs7x6rcCdAZnXf')
        self.table = self.bitable.get_table(table_name='test')
    def test_querytables(self):
        print(self.bitable.query_tables())
        print(self.table.id, self.table.name)
    def test_listrecords(self):
        print('query_fields:\n', self.table.query_fields())
        print('search_records():\n', self.table.search_records())
        print(self.table.primary_fields, self.table.modifiable_fields)
        f0 = (FilterInfo.builder()
              .conjunction("and")
              .conditions([Condition.builder()
                           .field_name("ts_code")
                           .operator("is")
                           .value(["000001.SZ"])
                           .build(),
                           ])
              ).build()
        print('search_records(filter):\n', self.table.search_records(field_names=['文本', 'ts_code'], filter=f0))
    def test_records_CRUD(self):
        df0 = pd.DataFrame([pd.Series({'ts_code': '000001.SZ', '数字': 178, '多选': ['矮', '胖'], '日期': 1780000202})])
        print(self.table.insert_records(df0))
        df1 = df0.copy().assign(**{'数字': 201})
        self.table.insert_records(df1)
        f0 = FilterInfo.builder().conjunction('and').conditions([
            Condition.builder().field_name('ts_code').operator('is').value([df0.iloc[0]['ts_code']]).build(),
        ]).build()
        self.table.del_rows_by_filter(filter=f0)
