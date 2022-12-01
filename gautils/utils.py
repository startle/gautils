import logging
import datetime

def benchmark(exclude_kw=[], exclude_arg=[], kw_format={}, arg_format={}, threshold=1):
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
            if index in exclude_arg: return '_'
            elif index in arg_format: return arg_format[index](arg)
            else: return default_str_format(arg)
        def kw_str_format(k, v):
            if k in exclude_kw: return '%s:_'%k
            elif k in kw_format: return '%s:%s'%(k, kw_format(k))
            else: return '%s:%s'%(k, default_str_format(v))
        def wrapper(*args, **kw):
            def call_str_format():
                str_args = [arg_str_format(id,arg) for id, arg in enumerate(args)]
                str_args += [kw_str_format(k,v) for k, v in kw]
                text = '%s.%s(%s)' % (func.__module__, func.__name__, ', '.join(str_args))
                return text
            b = datetime.datetime.now()
            obj = func(*args, **kw)
            e = datetime.datetime.now()
            time_s = (e-b).total_seconds()
            if time_s >= threshold:
                logging.getLogger('perf').info('(%.3fs)call %s' % (time_s, call_str_format()))
            return obj
        return wrapper
    return decorator

def conf_logging_by_yml(yml_conf_path='./log.yml'):
    import yaml
    import os
    from logging import config
    # log_path = os.path.abspath(os.path.dirname(__file__)) + '/log.yml'
    with open(yml_conf_path, 'r') as f_conf:
        dict_conf = yaml.safe_load(f_conf)
    config.dictConfig(dict_conf)

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