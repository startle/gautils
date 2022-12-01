from numpy import isin
import yaml
import os
import math

class Conf:
    def __init__(self, conf_path):
        with open(conf_path, 'r') as f:
            yml = yaml.safe_load(f.read())
            self.datas = yml
    def get(self, path, default=None):
        d = self.datas
        for k in path:
            if k not in d:
                if default is None: raise ValueError('conf not found.[%s]' % ('.'.join(path)))
                else: return default
            d = d[k]
        if isinstance(d, str): d = d.strip()
        return d
    def get_int(self, path, default=None): return int(self.get(path, default=default))
    def get_float(self, path, default=None): return float(self.get(path, default=default))
    def get_bool(self, path, default=None): return bool(self.get(path, default=default))

if __name__ == '__main__':
    pass