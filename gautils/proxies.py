class XiaoXiangProxy:
    def __init__(self, user, pwd):
        self.user = user
        self.pwd = pwd

    def gen_short_proxies_builder_f(self):
        ''' 隧道短期代理 '''
        proxies = None
        host = 'http-short.xiaoxiangdaili.com'
        port = 10010
        user = self.user
        pwd = self.pwd
        proxyMeta = f"http://{user}:{pwd}@{host}:{port}"
        proxies = {
            'http': proxyMeta,
            'https': proxyMeta,
        }

        def build_proxies():
            return proxies
        return build_proxies
