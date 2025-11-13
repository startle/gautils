import sys
import pandas as pd

from .mysqldb import MysqlDb

def KEY_ENCODE_JSON(sr: pd.Series):
    return sr.to_json()
def KEY_ENCODE_MD5(sr: pd.Series):
    import hashlib
    content = sr.to_json()
    md5_hash = hashlib.md5(content.encode("utf-8"))
    return md5_hash.hexdigest()
class KVTable:
    def __init__(self, db: MysqlDb, table: str, name: str, keys: list[str], key_encode=None):
        self._db = db
        self._table = table
        self._name = name
        self._keys = keys
        self._key_encode = KEY_ENCODE_MD5 if key_encode is None else key_encode
    def insert(self, sdf: pd.DataFrame):
        def data_encode(sr: pd.Series):
            return sr.to_json()
        if sdf is None or len(sdf) == 0:
            return None, None
        sdf = sdf.astype(str)
        sdf = sdf.sort_values(self._keys)

        key_col = 'keys'
        name_col = 'name'
        datas_col = 'datas'

        sdf[datas_col] = sdf.apply(data_encode, axis=1)
        sdf[key_col] = sdf[self._keys].apply(self._key_encode, axis=1)
        sdf[name_col] = self._name

        keys = sdf[key_col].unique().tolist()
        print('---------------')
        print('tables', self._table, self._name)
        print('keys', len(keys), keys)
        df_existed: pd.DataFrame = self._db.query(f'SELECT * FROM {self._table} WHERE `name`=:name AND `keys` in :keys', name=self._name, keys=keys)
        print('existed', len(df_existed), df_existed[key_col].to_list())
        df_existed = sdf[sdf[key_col].isin(df_existed[key_col].unique())]

        if not df_existed.empty:
            existed_keys = df_existed[key_col].unique()
            df_insert = sdf[~sdf[key_col].isin(existed_keys)]
            df_update = sdf[sdf[key_col].isin(existed_keys)]
            if len(df_update) > 0:
                df_update = df_update[df_update[datas_col] != df_existed[datas_col]]
        else:
            df_insert = sdf
            df_update = None

        if (df_insert is not None) and (not df_insert.empty):
            df_insert = df_insert.drop(columns=[key_col, datas_col, name_col])
        if (df_update is not None) and (not df_update.empty):
            df_update = df_update.drop(columns=[key_col, datas_col, name_col])

        df_dbdata = sdf[[name_col, key_col, datas_col]]
        cnt = self._db.update(self._table, df_dbdata)
        # print('fs update cnt:', cnt)
        # print(f'-------{self._keys}')
        # print('[df]:', sdf.index[-10:], '\n', len(sdf))
        # print('[df_insert]:', df_insert.index[-10:], '\n', len(df_insert))
        # print('[df_update]:', df_update.index[-10:], '\n', len(df_update))

        return df_insert, df_update
