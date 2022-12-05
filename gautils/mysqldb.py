#mysql-connector
import mysql.connector
from mysql.connector import FieldType
import pandas as pd
import os
import hashlib
import numpy as np
import logging

log = logging.getLogger(__name__)
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
            log.error('error sql: %s' % sql, exc_info=ex)
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
    # FieldType.LONGLONG: ,
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
    # FieldType.BLOB: ,
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
    def __init__(self, host, port, account, pwd,db, **kws):
        try:
            # self.__conn = mysql.connector.connect(host=h, port=p, user=u, passwd=pwd, database=db, **kws)
            self.__conn = mysql.connector.connect(host=host, port=port, user=account, passwd=pwd, database=db, charset='utf8')
        except Exception as e:
            log.error('conn failed.h[%s] p[%s] u[%s] db[%s]' %(host, port, account, db), exc_info=e)
            raise e
    def s_query(self, table, cols=None) -> MysqlQuery:
        return MysqlQuery(self, table, cols)
    def update(self, table, df:pd.DataFrame, is_close_conn=True) -> int:
        batch=10000
        df = df.reset_index()
        cnt = 0
        for idf in MysqlDbImpl._section_batch(df, batch):
            cnt += self._update(table, idf, is_close_conn=False)
        return cnt
    def execute(self, sql, *params):
        return self._execute(sql, *params)
    def query(self, sql, *params):
        return self._query(sql, *params)
    def close(self):
        self.__conn.close()
    def describe(self, table):
        return self._describe(table)
    def keys_cols(self, table):
        return self._keys_cols(table)
    def delete_all(self, table):
        self._delete_all(table)
    
    def _execute(self, sql, is_close_conn=True, *params):
        cursor = self.__conn.cursor()
        cnt = cursor.execute(sql, params)
        self.__conn.commit()
        cursor.close()
        if is_close_conn : self.close()
        return cnt
    def _dbtype2dtype(self, dbtype:str):
        index = dbtype.find('(')
        if index > 0 : dbtype = dbtype[:index]
        index = dbtype.find(' ')
        if index > 0 : dbtype = dbtype[:index]
        if dbtype in ['varchar','char','datetime','date','time']: return 'U13'
        elif dbtype in ['float','double','decimal']: return 'float64'
        elif dbtype in ['tinyint','smallint','int','bigint'] : return 'int'
        raise KeyError('dbtype[%s] not found.' % dbtype)
    def _query(self, sql:str, *params, is_close_conn=True):
        def read_row(row):
            def trans(x):
                if type(x) in [bytearray]:
                    return x.decode('utf8')
                else: return x
            return [trans(x) for x in row]
        def query_from_db():
            cursor = self.__conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            cols = [x[0] for x in cursor.description]
            field_types = [x[1] for x in cursor.description]
            dtypes = {cols[i]:_db_field_type2dtype(x) for i,x in enumerate(field_types)}
            df = pd.DataFrame(data = [read_row(row) for row in rows], columns = cols)
            df = df.astype(dtypes)
            cursor.close()
            if is_close_conn : self.close()
            return df
        return query_from_db()
    def _describe(self, table, is_close_conn = True):
        cursor = self.__conn.cursor()
        sql = 'describe {0}'.format(table)
        cursor.execute(sql)
        l=list()
        rows = cursor.fetchall()
        for row in rows:
            l.append(row)
        df = pd.DataFrame(data=l, columns=['Field', 'Type','Null','Key','Default','Extra'])
        df['Type'] = df['Type'].apply(lambda x: x.decode('UTF-8','strict') if type(x) != str else x)
        df['Key'] = df['Key'].apply(lambda x: x.decode('UTF-8','strict') if type(x) != str else x)
        cursor.close()
        if is_close_conn: self.close()
        return df
    def _keys_cols(self, table, is_close_conn = True):
        df:pd.DataFrame = self._describe(table, is_close_conn = is_close_conn)
        keys = (df[df['Key']=='PRI'].loc[:,'Field'].tolist())
        cols = (df[df['Key']!='PRI'].loc[:,'Field'].tolist())
        return keys, cols
    @staticmethod
    def _section_batch(df, b):
        if len(df) > b:
            for i in range(0, len(df), b):
                yield df.iloc[i:(i+b)]
        else: yield df
    def _update(self, table, df:pd.DataFrame, is_close_conn=True):
        if df.empty: return 0
        UPDATE_SQL = 'INSERT INTO {0}({1}) VALUES({2}) ON DUPLICATE KEY UPDATE {3}'
        idx_cols, upd_cols= self._keys_cols(table, is_close_conn=False)
        upd_cols = np.intersect1d(upd_cols, df.columns.values).tolist()
        if len(upd_cols) <= 0:raise ValueError('update cols cannot be empty.')
        if len(idx_cols) <= 0:raise ValueError('index cols cannot be empty.')
        # col_sql = '`' + '`,`'.join(idx_cols + upd_cols) + '`'
        col_sql = '`' + '`,`'.join(idx_cols + upd_cols) + '`'
        val_sql = ','.join([' %s' for x in range(len(idx_cols) + len(upd_cols))])
        upd_sql = ','.join(['`{0}`=VALUES(`{0}`)'.format(x) for x in upd_cols])
        sql = UPDATE_SQL.format(table, col_sql, val_sql, upd_sql)
        cnt = 0
        try:
            datas = [tuple(x.values) for i, x in df.loc[:, idx_cols + upd_cols].replace({np.nan: None}).iterrows()]
            cursor = self.__conn.cursor()
            cursor.executemany(sql, datas)
            cnt = cursor.rowcount
            self.__conn.commit()
            cursor.close()
        except Exception as e:
            log.error('sql update failed. sql:%s'%sql, exc_info = e)
            raise e
        finally:
            if is_close_conn : self.close()
            return cnt
    def _delete_all(self, table, is_close_conn = True):
        sql = 'DELETE FROM {0}'.format(table)
        return self._execute(sql, is_close_conn = is_close_conn)
def connect_mysql(h, p, u, pwd, db, is_use_file_cache = False, cache_dir=None, charset='utf8') -> MysqlDb:
    db = MysqlDbImpl(h, p, u, pwd, db, charset=charset)
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

if __name__ == '__main__':
    import conf
    cf = conf.Conf('conf.yml')
    host = cf.get(['db','host'])
    port = cf.get(['db','port'])
    account = cf.get(['db','account'])
    pwd = cf.get(['db','pwd'])
    db = cf.get(['db','db'])
    def dbm(): return connect_mysql(host, port, account, pwd, db)
    df = pd.DataFrame(data={'id':[1,2],'name':['gau','startle'],'score':[3.1415, 1.0086],'log_time':['2022-12-01 10:24', '2019-12-17 00:25']})
    cnt = dbm().update('for_test', df)
    print(cnt)
    query = dbm().s_query('for_test')
    print(query.query())
    df = dbm().query('SELECT * from for_test')
    print(df)
    df = dbm().query('select vpn, grp, `name` from vpn_record where log_time >= date_add(now(), interval %s MINUTE) AND (speed_down + speed_up) >= %s group by vpn, grp, `name` HAVING COUNT(`name`)>=%s', -60, 100, 1)
    print(df)
    df = dbm().query('select vpn, max(log_time) as max_time from vpn_record where vpn != "" group by vpn')
    print(df)
    pass