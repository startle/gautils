import json

import pandas as pd

from .mysqldb import MysqlDb

class KVTable:
    def __init__(self, db: MysqlDb, table: str, name: str, keys: list[str]):
        self._db = db
        self._table = table
        self._name = name
        self._keys = keys
    def insert(self, sdf: pd.DataFrame):
        def key_encode(sr: pd.Series):
            return sr.to_json()
        def data_encode(sr: pd.Series):
            return sr.to_json()
        def data_decode(sr: pd.Series):
            data_dict = json.loads(sr[datas_col])
            return pd.Series({k: str(v) for k, v in data_dict.items()})

        if sdf is None or len(sdf) == 0:
            return None, None
        sdf = sdf.astype(str)
        sdf = sdf.sort_values(self._keys)
        df = sdf.copy()

        key_col = 'keys'
        name_col = 'name'
        datas_col = 'datas'
        keys = [x for x in df[self._keys].apply(key_encode, axis=1).tolist()]

        df_existed: pd.DataFrame = self._db.query(f'SELECT {datas_col} FROM {self._table} WHERE `name`=:name AND `keys` in :keys', name=self._name, keys=keys)
        df_existed = df_existed.apply(data_decode, axis=1)

        if not df_existed.empty:
            df_insert = df[~df.set_index(self._keys).index.isin(df_existed.set_index(self._keys).index)]
            df_update = df[df.set_index(self._keys).index.isin(df_existed.set_index(self._keys).index)]
            if len(df_update) > 0:
                if not any(x not in df_existed.columns for x in sdf.columns.tolist()):
                    df_update = df_update[~df_update.set_index(sdf.columns.tolist()).index.isin(df_existed.set_index(sdf.columns.tolist()).index)]
        else:
            df_insert = df
            df_update = None

        df = pd.concat([df_insert, df_update])
        if not df.empty:
            df[datas_col] = df.apply(data_encode, axis=1)
            df[name_col] = self._name
            df[key_col] = df[self._keys].apply(key_encode, axis=1)
            df = df[[name_col, key_col, datas_col]]
            self._db.update(self._table, df)

        return df_insert, df_update
