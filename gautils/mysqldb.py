#mysql-connector
import mysql.connector
from mysql.connector import FieldType
import pandas as pd
import os
import hashlib
import numpy as np
import logging
import math
import types

class And:
    def __init__(self):
        self.__conds = list()
        self.__params = list()
    def cond(self, cond, *params):
        self.__conds.append(cond)
        if len(params) > 0:
            self.__params += list(*params)
        return self
    def vin(self, col, values):
        if values is not None and len(values) > 0:
            values = np.unique(values)
            if type(values[0]) in [int]: vstrs = values
            else: vstrs = ['\''+str(x)+'\'' for x in values]
            cond = '`{0}`IN({1})'.format(col, ','.join(vstrs))
            return self.cond(cond)
        else: raise ValueError('values is empty.')
    def between(self, col, fr, to):
        cond = '`{0}` BETWEEN %s AND %s'.format(col)
        self.cond(cond, fr, to)
        return self
    def eq(self, col, value):
        cond = '`{0}`=%s'.format(col)
        return self.cond(cond, value)
    def empty(self):
        return len(self.__conds) <= 0
    def to_sql(self):
        if not self.empty():
            return ' ' + ' AND '.join(self.__conds)
        else: return ''
    def params(self):
        return self.__params
class MysqlQuery:
    def __init__(self, db, table, cols = None):
        self.__table = table
        self.__cols = cols
        self.__where = And()
        self.__db = db
    def and_cond(self, cond, *params):
        self.__where.cond(cond, params)
        return self
    ''' values must be a list or array '''
    def and_in(self, col, values):
        self.__where.vin(col, values)
        return self
    def and_eq(self, col, value):
        self.__where.eq(col, [value])
        return self
    def query(self) -> pd.DataFrame:
        if self.__cols == None: colSql = '*'
        else: colSql = ','.join(['`'+x+'`' for x in self.__cols])
        sql = 'SELECT {0} FROM {1}'.format(colSql, self.__table)
        try :
            if not self.__where.empty():
                sql = sql + ' WHERE ' + self.__where.to_sql()
                return self.__db.query(sql, *(self.__where.params()))
            else :return self.__db.query(sql)
        except Exception as ex:
            logging.error('error sql: %s' % sql, exc_info=ex)
            raise ex
class MysqlDb:
    def s_query(self, table, cols=None) -> MysqlQuery: pass
    def update(self, table, df:pd.DataFrame) -> int: pass
    def query(self, sql, *params): pass 
    def execute(self, sql, *params): pass
    def describe(self, table): pass
    def keys_cols(self, table): pass
    def delete_all(self, table): pass
    def close(self):pass
    pass
_db_field_type2dtype_dict = {
    FieldType.TINY: 'int64',
    FieldType.SHORT: 'int64',
    FieldType.LONG: 'int64',
    FieldType.DECIMAL: 'float64',
    FieldType.FLOAT: 'float64',
    FieldType.DOUBLE: 'float64',
    # FieldType.NULL:
    FieldType.LONGLONG: 'int64',
    FieldType.INT24: 'int64',
    FieldType.DATE: 'U13',
    FieldType.TIME: 'U13',
    # FieldType.TIMESTAMP: 'datetime64',
    FieldType.DATETIME: 'datetime64',
    FieldType.VARCHAR: 'U13',
    # FieldType.BIT: ,
    FieldType.NEWDECIMAL: 'float64',
    FieldType.ENUM: 'U13',
    # FieldType.SET: ,
    # FieldType.TINY_BLOB: ,
    # FieldType.MEDIUM_BLOB: ,
    # FieldType.LONG_BLOB: ,
    FieldType.BLOB: 'U13',
    FieldType.VAR_STRING: 'U13',
    FieldType.STRING: 'U13',
    # FieldType.GEOMETRY: ,
}
def _db_field_type2dtype(db_field_type:int):
    if db_field_type in _db_field_type2dtype_dict:
        return _db_field_type2dtype_dict[db_field_type]
    else:
        raise Exception("db_field_type[%s] unsupported." % (FieldType.get_info(db_field_type)))
class MysqlDbImpl(MysqlDb):
    def __init__(self, host, port, account, pwd, db, **kws):
        self._host = host
        self._port = port
        self._account = account
        self._pwd = pwd
        self._db = db
        self._conn_kws = kws
    def _get_conn(self):
        try:
            return mysql.connector.connect(host=self._host, port=self._port, user=self._account, passwd=self._pwd, database=self._db, **self._conn_kws)
        except Exception as e:
            logging.error('conn failed.h[%s] p[%s] u[%s] db[%s]' %(self._host, self._port, self._account, self._db), exc_info=e)
            raise e
    def s_query(self, table, cols=None) -> MysqlQuery:
        return MysqlQuery(self, table, cols)
    def update(self, table, df:pd.DataFrame) -> int:
        batch=10000
        df = df.reset_index()
        cnt = 0
        for idf in MysqlDbImpl._section_batch(df, batch):
            cnt += self._update_each_batch(table, idf)
        return cnt
    def _update_each_batch(self, table, df:pd.DataFrame):
        if df.empty: return 0
        UPDATE_SQL = 'INSERT INTO {0}({1}) VALUES({2}) ON DUPLICATE KEY UPDATE {3}'
        idx_cols, upd_cols= self.keys_cols(table)
        upd_cols = np.intersect1d(upd_cols, df.columns.values).tolist()
        if len(upd_cols) <= 0:raise ValueError('update cols cannot be empty.')
        if len(idx_cols) <= 0:raise ValueError('index cols cannot be empty.')
        col_sql = '`' + '`,`'.join(idx_cols + upd_cols) + '`'
        val_sql = ','.join([' %s' for x in range(len(idx_cols) + len(upd_cols))])
        upd_sql = ','.join(['`{0}`=VALUES(`{0}`)'.format(x) for x in upd_cols])
        sql = UPDATE_SQL.format(table, col_sql, val_sql, upd_sql)
        cnt = 0
        def format(x): return tuple(None if isinstance(v, float) and math.isnan(v) else v for v in x.values)
        datas = [format(x) for _, x in df.loc[:, idx_cols + upd_cols].iterrows()]
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.executemany(sql, datas)
            cnt = cursor.rowcount
            conn.commit()
            cursor.close()
            return cnt
        except BaseException as e:
            logging.error(f'execute failed. sql[{sql}]')
            raise e
        finally:
            self.close_conn(conn)
    def execute(self, sql, *params):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cnt = cursor.execute(sql, params)
            conn.commit()
            cursor.close()
            self._after_execute()
            return cnt
        finally:
            self.close_conn(conn)
    def query(self, sql, *params):
        def read_row(row):
            def trans(x):
                if type(x) in [bytearray]:
                    return x.decode('utf8')
                else: return x
            return [trans(x) for x in row]
        def query_from_db():
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                cols = [x[0] for x in cursor.description]
                field_types = [x[1] for x in cursor.description]
                dtypes = {cols[i]:_db_field_type2dtype(x) for i,x in enumerate(field_types)}
                df = pd.DataFrame(data = [read_row(row) for row in rows], columns = cols)
                df = df.astype(dtypes)
                cursor.close()
                self._after_execute()
                return df
            finally:
                self.close_conn(conn)
        return query_from_db()
    def _after_execute(self):
        pass
    def close(self):
        pass
    def close_conn(self, conn):
        try:
            conn.close()
        except Exception as e:
            logging.error(f'close conn failed.', exc_info=e)
    def describe(self, table):
        sql = 'describe {0}'.format(table)
        df = self.query(sql)
        return df
    def keys_cols(self, table):
        df:pd.DataFrame = self.describe(table)
        keys = (df[df['Key']=='PRI'].loc[:,'Field'].tolist())
        cols = (df[df['Key']!='PRI'].loc[:,'Field'].tolist())
        return keys, cols
    def delete_all(self, table):
        sql = 'TRUNCATE TABLE {0}'.format(table)
        return self.execute(sql)
    def __del__(self):
        try:
            self.close()
            logging.log(f'{self} closed.')
        except Exception as e:
            pass
    def __str__(self) -> str:
        return f'db[{self._host}:{self._port}]'

    @staticmethod
    def _section_batch(df, b):
        if len(df) > b:
            for i in range(0, len(df), b):
                yield df.iloc[i:(i+b)]
        else: yield df
    
def connect_mysql(h, p, u, pwd, db, is_use_file_cache = False, cache_dir=None, charset='utf8', **kws) -> MysqlDb:
    kws.update({'charset':charset})
    db = MysqlDbImpl(h, p, u, pwd, db, **kws)
    if cache_dir is None: cache_dir = './temp/cache'
    def build_cache_path(sql:str, *params):
        sql = sql + '_' + '_'.join([str(x) for x in params])
        sign = '%d_%s' % (len(sql) ,hashlib.md5(sql.encode(encoding='utf8')).hexdigest())
        return cache_dir + sign
    def file_cache_query(table:str, sql:str, *params):
        cache_path = build_cache_path(sql, params)
        if is_use_file_cache and os.path.isfile(cache_path):
            return pd.read_json(cache_path)
        else:
            df = MysqlDbImpl.query(db, sql, *params)
            if is_use_file_cache and (len(df) > 0):
                df.to_json(cache_path)
            return df
    if is_use_file_cache:
        db.query = file_cache_query
        pass
    return db
