class WXWorkRobot:
    def __init__(self, url):
        self._url = url
    def send_md(self, msg, mentioned_list=None):
        import requests
        data = '''
        {
                "msgtype": "markdown",
                "markdown": {
                    "content": "%s"
                }
        }''' % msg.encode("utf-8").decode("latin1")
        headers = {'user-agent': 'my-app/0.0.1'}
        requests.post(self._url, headers=headers, data=data, verify=False)
if __name__ == '__main__':
    import conf
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    urllib3.disable_warnings()
    cf = conf.Conf('conf.yml')
    url = cf.get(['qwx', 'robot'])
    qwx = WXWorkRobot(url)
    qwx.send_md('test')
    
