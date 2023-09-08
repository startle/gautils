import unittest

import pandas as pd

from gautils import mysqldb

def conn() -> mysqldb.DbAlchemy:
    host = '124.222.96.156'
    port = 3306
    account = 'startle'
    pwd = 'wuidkcm_67!&ks9'
    db_name = 'test'
    ret:mysqldb.DbAlchemy = mysqldb.connect_mysql(host, port, account, pwd, db_name, conn_type=mysqldb.ConnType.ALCHEMY)
    return ret
class TestDbAlchemy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = conn()
        sql = '''CREATE TABLE IF NOT EXISTS `test` (
                    `Id` INT(11) NOT NULL,
                    `Name` VARCHAR(50) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
                    `Description` VARCHAR(200) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
                    `Owner` VARCHAR(50) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
                    `Score` DOUBLE NULL DEFAULT NULL,
                    `CreateTime` DATETIME NULL DEFAULT NULL,
                    PRIMARY KEY (`Id`) USING BTREE
                )
                COLLATE='utf8mb4_general_ci'
                ENGINE=InnoDB
        '''
        cls.db.execute(sql)
        
    @classmethod
    def tearDownClass(cls):
        sql = 'DROP TABLE IF EXISTS `test`'
        cnt = cls.db.execute(sql)
        print(cnt)

    def test_keys_cols(self):
        keys, cols = self.db.keys_cols('test')
        self.assertEqual(keys, ['Id']) 
        self.assertEqual(cols, ['Name', 'Description', 'Owner', 'Score', 'CreateTime'])

    def test_create_update_sql(self):
        # 省略
        pass

    def test_1_update(self):
        db = self.db
        data = [{'Id': 1, 'Name': '张三', 'Score': 88.5, 'Description': '张三的描述', 'Owner':'李四', 'CreateTime':'2023-09-07 08:00:00'}]
        count = db.update('test', pd.DataFrame(data))
        self.assertEqual(count, 1)
        
        df = db.query('SELECT Id, Name, Score, Description, Owner, CreateTime FROM test')
        self.assertEqual(df.loc[0].tolist(), [1, '张三', 88.5, '张三的描述', '李四', pd.to_datetime('2023-09-07 08:00:00')])
        
        df = db.query('SELECT * FROM test')
        self.assertEqual(df.shape, (1, 6))

if __name__ == '__main__':
    import logging
    logging.basicConfig() 
    logging.root.setLevel(logging.INFO)
    unittest.main()