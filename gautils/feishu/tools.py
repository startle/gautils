import requests

def send_fs_robot_msg(url, msgs: str | list[str] = None, json=None):
    if json is None and not msgs:
        if msgs is None:
            raise Exception('empty msgs&json')
        return None
    if msgs is not None and isinstance(msgs, str):
        msgs = [msgs]
    if json is None:
        msg_content = '\n'.join(msgs)
        json = {
            "msg_type": "text",
            "content": {
                "text": f'{msg_content}'
            }
        }
    requests.post(url, json=json)
