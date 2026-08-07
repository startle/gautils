import sys
import os
import time
import logging
import tempfile
import yaml
import requests

log = logging.getLogger(__name__)

# (连接超时, 读取超时) 秒。connect 短，连不上即弃；read 宽，容忍慢响应接口
DEFAULT_TIMEOUT = (5, 30)

##################  http  #######################
def request_regular_fetch(pattern, url, params=None, headers=None):
    text = None
    try:
        text = request(url, params=params, headers=headers)

        m = pattern.findall(text)
        if (m is not None) and (len(m) > 0):
            return m
        else:
            raise ValueError("request_regular_fetch failed. url[%s] pattern[%s]" % (url, pattern))
    except Exception as e:
        logging.getLogger("dbupdate").error("url fetch failed.url[%s] text:\n%s" , url, text[:50] if text else '<unavailable>', exc_info=e)
        raise
def request(url, params=None, headers=None, _type="get") -> str:
    if _type == "get":
        return Web().get(url, params=params, headers=headers)
    elif _type == "post":
        return Web().post(url, params=params, headers=headers)
def default_pc_headers():
    return {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    }
def default_phone_headers():
    return {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "sec-ch-ua-platform": "Android",
        "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Mobile Safari/537.36",
    }
def retry_run(f, *params, retry_times=3, sleep_s=5, **kw):
    last_err = None
    for i in range(retry_times):
        try:
            return f(*params, **kw)
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            last_err = e
            logging.error("retry %d/%d failed", i + 1, retry_times, exc_info=e)
            if i < retry_times - 1:
                time.sleep(sleep_s * (i + 1))
    raise RuntimeError(f"retry_run exhausted {retry_times} retries") from last_err
def retry_run2(f, retry_times=3, sleep_s=5) -> requests.Response:
    last_err = None
    for i in range(retry_times):
        try:
            return f()
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            last_err = e
            logging.error("retry %d/%d failed", i + 1, retry_times, exc_info=e)
            if i < retry_times - 1:
                time.sleep(sleep_s * (i + 1))
    raise RuntimeError(f"retry_run2 exhausted {retry_times} retries") from last_err
def get_host(url):
    import re
    match = re.search(r'^(?:http[s]?://)?([^:/?#]+)', url)
    if match:
        return match.group(1)  # 输出：www.example.com
    else:
        return None

class CookieManager:
    def __init__(self, filename: str = 'cookies.yml'):
        self.filename = filename
        self.cookies = self.load_cookies()
    def load_cookies(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf8') as f:
                    data = yaml.safe_load(f)
                    # 空文件/纯注释/null 会让 safe_load 返回 None，兜底成 {} 防止 cookies 锁死
                    return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"Error loading cookies: {e}")
        return {}
    def save_cookies(self):
        try:
            # 原子写：先写临时文件再 os.replace，避免并发读时读到截断的半成品（cookies 锁死触发源）
            dir_ = os.path.dirname(os.path.abspath(self.filename)) or '.'
            fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf8') as f:
                    yaml.dump(self.cookies, f)
                os.replace(tmp_path, self.filename)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        except Exception as e:
            print(f"Error saving cookies: {e}")
    def update_cookies(self, new_cookies):
        self.cookies.update(new_cookies)
        self.save_cookies()
class Web:
    def __init__(self, cookies_filepath: str = 'cookies.yml', headers=None, verify=True, build_proxies_f=None):
        self.cookie_manager = CookieManager(cookies_filepath)
        self.headers = headers
        self.verify = verify
        self.build_proxies_f = (lambda : None) if build_proxies_f is None else build_proxies_f
    def get(self, url, params=None, retry_times=3, encoding=None, timeout=DEFAULT_TIMEOUT, raise_for_status=False):
        def request(session: requests.Session, url: str):
            proxies = self.build_proxies_f()
            return session.get(url, headers=self.headers, params=params, verify=self.verify, proxies=proxies, timeout=timeout)
        return self.request(url, request_f=request, retry_times=retry_times, encoding=encoding, raise_for_status=raise_for_status)
    def post(self, url, params=None, json=None, data=None, retry_times=3, encoding=None, timeout=DEFAULT_TIMEOUT, raise_for_status=False):
        def request(session: requests.Session, url: str):
            return session.post(url, headers=self.headers, params=params, json=json, data=data, verify=self.verify, proxies=self.build_proxies_f(), timeout=timeout)
        return self.request(url, request_f=request, retry_times=retry_times, encoding=encoding, raise_for_status=raise_for_status)
    def request(self, url, request_f, retry_times=3, encoding=None, raise_for_status=False):
        session = requests.Session()
        host = get_host(url)
        cookies = self.cookie_manager.cookies.get(host, {})
        session.cookies.update(cookies)
        response = retry_run2(lambda : request_f(session, url), retry_times=retry_times, sleep_s=5)
        if raise_for_status:
            response.raise_for_status()
        # 从 session 提取服务端返回的 cookies 并持久化
        updated = requests.utils.dict_from_cookiejar(session.cookies)
        if updated:
            self.cookie_manager.cookies[host] = updated
            self.cookie_manager.save_cookies()
        response.encoding = response.apparent_encoding if encoding is None else encoding
        try :
            return response.text
        except Exception as e:
            logging.error(f'[response.text failed.] response:\n {response}', exc_info=e)
            return response
    def parse_url(self, url):
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)
        cleaned_params = {}
        for key, value in query_params.items():
            if isinstance(value, list) and len(value) == 1:
                cleaned_params[key] = value[0]
            else:
                cleaned_params[key] = value
        return domain, path, cleaned_params
