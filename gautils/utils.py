import logging
import datetime
import os
import math

def conf_logging_by_yml(yml_conf_path='./log.yml'):
    import yaml
    import os
    from logging import config
    # log_path = os.path.abspath(os.path.dirname(__file__)) + '/log.yml'
    with open(yml_conf_path, 'r') as f_conf:
        dict_conf = yaml.safe_load(f_conf)
    config.dictConfig(dict_conf)

def url_parse_unquote(s):
    if s is None or len(s) <= 0: return ''
    from urllib.parse import unquote
    return unquote(s)
def read_dicts(s:str, kv_split='=', split='&', parse_unquote=url_parse_unquote):
    d = {}
    if parse_unquote is None: parse_unquote = lambda x: x
    for l in s.split(split):
        l = l.strip()
        if len(l) <= 0: continue
        index = l.find(kv_split)
        if index <= 0: continue
        d[l[:index]] = parse_unquote(l[index+1:].strip())
    return d

def md5(*objs):
    import hashlib
    s = ','.join([str(x) for x in objs])
    hexs = hashlib.md5(s.encode(encoding='utf8')).hexdigest()
    return hexs[:16]
def floor(num:float, ratio):
    ratio = math.pow(10, -ratio)
    return math.floor(num / ratio) * ratio
def ceil(num, ratio):
    ratio = math.pow(10, -ratio)
    return math.ceil(num / ratio) * ratio
def binsearch(l:list, e, key_f=None):
    if not key_f:
        key_f = lambda x:x
    n = len(l)
    bi,ei,mi = 0, n-1, 0
    if e>key_f(l[ei]) or e<key_f(l[bi]):
        return -1
    if e == l[ei]:
        return ei
    if e == l[bi]:
        return bi
    while 1:
        mi = math.floor((bi+ei)/2)
        if mi == bi or mi == ei:
            return mi
        if e == key_f(l[mi]):
            return mi
        elif e > key_f(l[mi]):
            bi = mi
        else:
            ei = mi
###### files ################
def list_files(_dir:str = None, recursion=True):
    def recursion_list_file(path:str):
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for cpath in os.listdir(path):
                for x in recursion_list_file(os.path.join(path, cpath)):
                    yield x
    if _dir is None : raise Exception('_dir cannot be None.')
    if not os.path.exists(_dir) or not os.path.isdir(_dir): raise Exception('_dir is not a dir. [%s]' % _dir)
    
    if not recursion: return os.listdir(_dir)
    else: return list(recursion_list_file(_dir))
def read_lines(file:str, encoding='utf8'):
    with open(file, 'r', encoding=encoding, buffering=2<<16) as f:
        while True:
            line=f.readline()
            if line:
                yield line.strip()
            else:
                return 'done'
def write_lines(file:str, lines, encoding='utf8', mode='a'):
    if isinstance(lines, str): lines = [lines]
    with open(file, mode, encoding=encoding, buffering=2<<16) as f:
        for line in lines:
            f.write(str(line) + '\n')
###### comments ##########
def singleton(cls):
    _instance = {}
    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]
    return _singleton
def watch_process(exclude_ks=[], exclude_arg_indexes=[], kw_formatter={}, arg_index_formatter={}, threshold=1, logger_name = 'perf'):
    ''' output to perf.logging '''
    def str_format(x):
        x_str = str(x).replace('\n',' ')
        if len(x_str) < 150: return x_str
        else: return x_str[:100] + '...' + x_str[-20:]
    def df_format(x):
        return 'df[%d][%d]' % (len(x), len(x.columns))
    def series_format(x):
        return 'series[%d]' % len(x.index)
    def default_str_format(x):
        import pandas as pd
        if isinstance(x, pd.DataFrame): return df_format(x)
        elif isinstance(x, pd.Series): return series_format(x)
        else: return str_format(x)
    def decorator(func):
        def arg_str_format(index, arg):
            if index in exclude_arg_indexes: return '_'
            elif index in arg_index_formatter: return arg_index_formatter[index](arg)
            else: return default_str_format(arg)
        def kw_str_format(k, v):
            if k in exclude_ks: return '%s:_'%k
            elif k in kw_formatter: return '%s:%s'%(k, kw_formatter(k))
            else: return '%s:%s'%(k, default_str_format(v))
        def wrapper(*args, **kw):
            def call_str_format():
                str_args = [arg_str_format(id,arg) for id, arg in enumerate(args)]
                str_args += [kw_str_format(k,v) for k, v in kw.items()]
                text = '%s.%s(%s)' % (func.__module__, func.__name__, ', '.join(str_args))
                return text
            b = datetime.datetime.now()
            obj = func(*args, **kw)
            e = datetime.datetime.now()
            time_s = (e-b).total_seconds()
            if time_s >= threshold:
                logging.getLogger(logger_name).info('(%.3fs)call %s' % (time_s, call_str_format()))
            return obj
        return wrapper
    return decorator
benchmark = watch_process

if __name__ == '__main__':
    conf_logging_by_yml()
    
    import time
    @benchmark()
    def f2():
        print('start f2')
        time.sleep(1.2)
    @benchmark()
    def f0_8():
        print('start f0_8')
        time.sleep(0.8)
    f2()
    f0_8()
    
    d = read_dicts('controler=State& action=GetOnlineUsers_Local &token=ujiqjwehxnacnkkjheqoijksjaldo')
    print(d)
    
    files = list_files('gautils')
    for f in files: print(f)