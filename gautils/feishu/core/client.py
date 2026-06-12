import json
import time
import warnings
from typing import Optional

import lark_oapi as lark
from lark_oapi.api.auth.v3 import (
    InternalTenantAccessTokenRequest,
    InternalTenantAccessTokenRequestBody,
    InternalTenantAccessTokenResponse,
)
from lark_oapi.api.application.v6 import (
    GetApplicationRequest,
    GetApplicationResponse,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="lark_oapi.ws.pb.google.__init__",
    message="pkg_resources is deprecated as an API"
)


class FeishuClient:
    """飞书客户端基类"""

    def __init__(self, app_id: str = None, app_secret: str = None, open_id: str = None, log_level=lark.LogLevel.INFO):
        self.app_id = app_id
        self.app_secret = app_secret
        self._open_id = open_id  # 应用的 open_id
        self.client = lark.Client.builder() \
            .app_id(app_id=app_id) \
            .app_secret(app_secret=app_secret) \
            .log_level(log_level) \
            .build()
        self._app_info = None
        self._tenant_access_token = None
        self._token_expire_at = 0  # token 过期时间戳（秒）

    def get_tenant_access_token(self) -> Optional[str]:
        """获取 tenant_access_token，过期自动刷新"""
        if self._tenant_access_token and time.time() < self._token_expire_at:
            return self._tenant_access_token

        request: InternalTenantAccessTokenRequest = InternalTenantAccessTokenRequest.builder() \
            .request_body(InternalTenantAccessTokenRequestBody.builder()
                          .app_id(self.app_id)
                          .app_secret(self.app_secret)
                          .build()) \
            .build()

        response: InternalTenantAccessTokenResponse = self.client.auth.v3.tenant_access_token.internal(request)

        if not response.success():
            lark.logger.error(f"auth.v3.tenant_access_token.internal failed, code: {response.code}, msg: {response.msg}")
            return None

        # 解析响应获取 token
        try:
            content = json.loads(response.raw.content)
            token = content.get('tenant_access_token')
            expire = content.get('expire', 0)
            self._tenant_access_token = token
            # 提前 5 分钟过期，避免临界态请求失败
            self._token_expire_at = time.time() + expire - 300
            return token
        except (json.JSONDecodeError, AttributeError):
            lark.logger.error("Failed to parse tenant_access_token response")
            return None

    def get_app_info(self) -> dict:
        """获取应用信息

        Returns:
            包含应用信息的字典，包括 open_id、name 等
        """
        if self._app_info is not None:
            return self._app_info

        # 尝试从 ListApplication API 获取应用信息
        from lark_oapi.api.application.v6 import ListApplicationRequest, ListApplicationResponse

        request: ListApplicationRequest = ListApplicationRequest.builder() \
            .page_size(100) \
            .build()

        response: ListApplicationResponse = self.client.application.v6.application.list(request)

        if not response.success():
            lark.logger.error(f"application.v6.application.list failed, code: {response.code}, msg: {response.msg}")
            return {}

        try:
            # 从原始响应解析
            content = json.loads(response.raw.content) if response.raw else {}
            data = content.get('data', {})
            items = data.get('items', [])

            # 查找当前应用
            for app in items:
                if app.get('app_id') == self.app_id:
                    self._app_info = {
                        'app_id': self.app_id,
                        'open_id': app.get('open_id'),
                        'name': app.get('name'),
                        'description': app.get('description'),
                        'avatar_url': app.get('avatar_url'),
                    }
                    return self._app_info
        except (json.JSONDecodeError, AttributeError) as e:
            lark.logger.error(f"Failed to parse app info: {e}")

        return {}

    @property
    def open_id(self) -> Optional[str]:
        """应用 open_id

        初始化时传入的 open_id，可直接用于 wiki 成员管理等场景
        """
        return self._open_id
