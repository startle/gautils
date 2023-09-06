import requests
    
def send_fs_robot_msg(url, msgs=None, json=None):
    if msgs is not None and isinstance(msgs, str):
        msgs = [msgs]
    if json is None:
        if msgs is None: raise Exception('empty msgs&json')
        msg_content = '\n'.join(msgs)
        json = {
            "msg_type": "text",
            "content": {
                "text": f'{msg_content}'
            }
        }
    requests.post(url, json=json)