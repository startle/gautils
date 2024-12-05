import pandas as pd
import datetime
import logging
from openpyxl import Workbook
from openpyxl.utils.cell import get_column_letter
import os

DEBUG_OUT_DIR = 'temp/debug_out'
if not os.path.exists(DEBUG_OUT_DIR):
    os.makedirs(DEBUG_OUT_DIR)

def convert_url_to_windows_filename(url: str):
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        url = url.replace(char, '_')
    return url

def batch_split(l, n):
    for i in range(0, len(l), n):
        if isinstance(l, pd.DataFrame):
            yield l.iloc[i:i + n, :]
        else:
            yield l[i:i + n]

def watch_process(exclude_ks=[], exclude_arg_indexes=[], kw_formatter={}, arg_index_formatter={}, threshold=1):
    def str_format(x):
        x_str = str(x).replace('\n', ' ')
        if len(x_str) < 150:
            return x_str
        else:
            return x_str[:100] + '...' + x_str[-20:]

    def df_format(x: pd.DataFrame):
        return f'df[{len(x)}][{len(x.columns)}]'

    def series_format(x: pd.Series):
        return f'series[{len(x.index)}]'

    def default_str_format(x):
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            return df_format(x)
        elif isinstance(x, pd.Series):
            return series_format(x)
        else:
            return str_format(x)

    def decorator(func):
        def arg_str_format(index, arg):
            if index in exclude_arg_indexes:
                return '_'
            elif index in arg_index_formatter:
                return arg_index_formatter[index](arg)
            else:
                return default_str_format(arg)

        def kw_str_format(k, v):
            if k in exclude_ks:
                return '%s:_' % k
            elif k in kw_formatter:
                return '%s:%s' % (k, kw_formatter(k))
            else:
                return '%s:%s' % (k, default_str_format(v))

        def wrapper(*args, **kw):
            def call_str_format():
                str_args = [arg_str_format(id, arg) for id, arg in enumerate(args)]
                str_args += [kw_str_format(k, v) for k, v in kw.items()]
                text = f'{func.__module__}.{func.__name__}({", ".join(str_args)})'
                return text
            b = datetime.datetime.now()
            obj = func(*args, **kw)
            e = datetime.datetime.now()
            time_s = (e - b).total_seconds()
            if time_s >= threshold:
                logging.getLogger(__name__).info('(%.3fs)call %s' % (time_s, call_str_format()))
            return obj
        return wrapper
    return decorator

def format_workbook(wb: Workbook, max_col_width=80):
    def auto_col_width(worksheet):
        def compute_text_width(text):
            return sum(2 if '\u4e00' <= ch <= '\u9fff' else 1 for ch in text)
        for worksheet in wb.worksheets:
            for col in worksheet.columns:
                max_length = max([compute_text_width(str(cell.value)) for cell in col])
                adjusted_width = (min(max_length, max_col_width) + 2)
                worksheet.column_dimensions[get_column_letter(col[0].column)].width = adjusted_width
    for worksheet in wb.worksheets:
        auto_col_width(worksheet)
    pass

def write_excel(book_name, dfs: dict[str, pd.DataFrame]):
    with pd.ExcelWriter(f'{DEBUG_OUT_DIR}/{book_name}.xlsx', engine='openpyxl') as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=True)
        format_workbook(writer.book)

def time2timestr(dtime: datetime):
    return dtime.strftime('%Y-%m-%d %H:%M:%S')
def time2datestr(dtime: datetime):
    return dtime.strftime('%Y-%m-%d')
