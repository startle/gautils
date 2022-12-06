from numpy import isin
import yaml
import os
import math

class Conf:
    def __init__(self, conf_path):
        with open(conf_path, 'r') as f:
            yml = yaml.safe_load(f.read())
            self.datas = yml
    def get(self, path, default=None, ignore_none = False):
        d = self._get(path, default=default, ignore_none=ignore_none)
        if isinstance(d, str): d = d.strip()
        return d
    def _get(self, path, default=None, ignore_none = False):
        if default is not None: ignore_none = True
        d = self.datas
        for k in path:
            if k not in d:
                if not ignore_none: raise ValueError('conf path not found. [%s]' % self._path_to_str(path))
                else: return default
            d = d[k]
        return d
    def get_int(self, path, default=None, ignore_none = False): return int(self.get(path, default=default,ignore_none=ignore_none))
    def get_float(self, path, default=None, ignore_none = False): return float(self.get(path, default=default,ignore_none=ignore_none))
    def get_bool(self, path, default=None, ignore_none = False): return bool(self.get(path, default=default,ignore_none=ignore_none))
    def get_dict(self, path, default=None, ignore_none = False): 
        data = self._get(path, default=default,ignore_none=ignore_none)
        if isinstance(data, dict): return data
        else: raise ValueError('path not dict. [%s]', self._path_to_str(path))
    def _path_to_str(self, path): return ('.'.join(path))

if __name__ == '__main__':
    from collections import namedtuple
    DbConf = namedtuple('DbConf', ['host','port','account','pwd','db'])

    conf = Conf('conf.yml')
    db = conf.get_dict(['db'])
    db = DbConf(**db)
    print(db)
    pass