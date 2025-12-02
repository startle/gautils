import unittest
import numpy as np

from gautils.feishu2.core import BiTable, Feishu

class TestFeishu2(unittest.TestCase):
    #       app_id: "cli_a5e0366e397cd00e"
    #   app_secret: "abi2ikgvkTDr2V3K8aykphJfVC0YUrGR"
    def setUp(self):
        self.fs2 = Feishu('cli_a5e0366e397cd00e', 'abi2ikgvkTDr2V3K8aykphJfVC0YUrGR')

    def test_listrecords(self):
        print('xxxxx')
        bitable: BiTable = self.fs2.get_bitable('Hr1rbmwpna7HOPs7x6rcCdAZnXf')
        df = bitable.query_tables()
        print(df)
        table = bitable.get_table(table_name='stock')
