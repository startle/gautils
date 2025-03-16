import sys
import os
import time
import logging
import yaml
import requests

log = logging.getLogger(__name__)

##################  http  #######################
def request_regular_fetch(pattern, url, params=None, headers=None):
    try:
        text = request(url, params=params, headers=headers)

        m = pattern.findall(text)
        if (m is not None) and (len(m) > 0):
            return m
        else:
            raise ValueError("request_regular_fetch failed. url[%s] pattern[%s]" % (url, pattern))
    except Exception as e:
        logging.getLogger("dbupdate").error("url fetch failed.url[%s] text:\n%s" , url, text[:50], exc_info=e)
        raise e
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
    for i in range(retry_times):
        try:
            result = f(*params, **kw)
            return result
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            logging.error(f"failed on {i} times...", exc_info=e)
            time.sleep(sleep_s * i)
    return None
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
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading cookies: {e}")
        return {}
    def save_cookies(self):
        try:
            with open(self.filename, 'w', encoding='utf8') as f:
                yaml.dump(self.cookies, f)
        except Exception as e:
            print(f"Error saving cookies: {e}")
    def update_cookies(self, new_cookies):
        self.cookies.update(new_cookies)
        self.save_cookies()
class Web:
    def __init__(self, cookies_filepath: str = 'cookies.yml', headers=None):
        self.cookie_manager = CookieManager(cookies_filepath)
        self.headers = headers
    def get(self, url, params=None, retry_times=3, encoding=None):
        return self.request(url, request_f=requests.Session.get, params=params, retry_times=retry_times, encoding=encoding)
    def post(self, url, params=None, retry_times=3, encoding=None):
        return self.request(url, request_f=requests.Session.post, params=params, retry_times=retry_times, encoding=encoding)
    def request(self, url, request_f, params=None, retry_times=3, encoding=None):
        session = requests.Session()
        host = get_host(url)
        if host in self.cookie_manager.cookies:
            cookies = self.cookie_manager.cookies.pop(host)
        else:
            cookies = {}
        session.cookies.update(cookies)
        response = retry_run(request_f, session, url, headers=self.headers, params=params, retry_times=retry_times)
        self.cookie_manager.cookies[host] = cookies
        if encoding is not None:
            text = response.content.decode(encoding)
            return text
        else:
            return response.text
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
