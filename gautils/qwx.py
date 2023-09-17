class WXWorkRobot:
    def __init__(self, url='conf.yml'):
        self._url = url
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        urllib3.disable_warnings()
    def send_md(self, msg, mentioned_list=None):
        import requests
        j = {
            "msgtype": "markdown",
            "markdown": {
                "content": f'{msg}'
            }
        }
        import json
        data = json.dumps(j)
        # data = '''
        # {
        #         "msgtype": "markdown",
        #         "markdown": {
        #             "content": "%s"
        #         }
        # }''' % msg.encode('utf-8').decode('unicode_escape')
        headers = {'user-agent': 'my-app/0.0.1'}
        requests.post(self._url, headers=headers, data=data, verify=False)
def send_qwx_md_msg(url, msg, mentioned_list=None):
    qwx = WXWorkRobot(url)
    qwx.send_md(msg, mentioned_list=mentioned_list)
if __name__ == '__main__':
    import conf
    cf = conf.Conf('conf.yml')
    url = cf.get(['qwx', 'robot'])
    qwx = WXWorkRobot(url)
    qwx.send_md('test')
    
