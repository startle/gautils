class XiaoXiangProxy:
    def __init__(self, user, pwd):
        self.user = user
        self.pwd = pwd

    def gen_short_proxies_builder_f(self):
        ''' 隧道短期代理 '''
        return self._gen_tunnel_proxies_build_f('http-short.xiaoxiangdaili.com', 10010)
    def gen_dynamic_proxies_builder_f(self):
        ''' 隧道动态代理 '''
        return self._gen_tunnel_proxies_build_f('http-dynamic.xiaoxiangdaili.com', 10030)
    def _gen_tunnel_proxies_build_f(self, host, port):
        proxies = None
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
