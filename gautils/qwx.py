class WXWorkRobot:
    def __init__(self, url, verify=True):
        self._url = url
        self._verify = verify
    def send_md(self, msg, mentioned_list=None):
        """发送 markdown 消息。mentioned_list 在 markdown 类型中不生效，保留参数仅为接口兼容。"""
        import requests
        j = {
            "msgtype": "markdown",
            "markdown": {
                "content": f'{msg}'
            }
        }
        import json
        data = json.dumps(j)
        headers = {
            'Content-Type': 'application/json',
            'user-agent': 'my-app/0.0.1',
        }
        requests.post(self._url, headers=headers, data=data, verify=self._verify)
def send_qwx_md_msg(url, msg, mentioned_list=None):
    qwx = WXWorkRobot(url)
    qwx.send_md(msg, mentioned_list=mentioned_list)
if __name__ == '__main__':
    import conf
    cf = conf.Conf('conf.yml')
    url = cf.get(['qwx', 'robot'])
    qwx = WXWorkRobot(url)
    qwx.send_md('test')
    
