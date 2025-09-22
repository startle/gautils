import pandas as pd
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
