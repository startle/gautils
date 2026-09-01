import json
from types import SimpleNamespace
from unittest.mock import MagicMock


def obj(**kwargs):
    return SimpleNamespace(**kwargs)


def response(*, success=True, data=None, content=None, code=0, msg="ok"):
    result = MagicMock()
    result.success.return_value = success
    result.data = data
    result.code = code
    result.msg = msg
    result.get_log_id.return_value = "log-id"
    if content is None:
        content = {}
    if not isinstance(content, str):
        content = json.dumps(content)
    result.raw = obj(content=content)
    return result
