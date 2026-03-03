import warnings
import json
import pandas as pd
from typing import Optional, List
import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="lark_oapi.ws.pb.google.__init__",
    message="pkg_resources is deprecated as an API"
)


class _FS:
    class WIKI:
        class SPACE:
            ID = 'space_id'
            NAME = 'name'
            DESCRIPTION = 'description'
        class NODE:
            ID = 'node_token'
            NAME = 'title'
            TYPE = 'obj_type'
            PARENT_ID = 'parent_node_token'


class WikiNode:
    """知识空间节点"""

    def __init__(self, wiki: 'WikiSpace', node_info: dict):
        self._wiki = wiki
        self._node_info = node_info

    @property
    def id(self) -> str:
        return self._node_info.get(_FS.WIKI.NODE.ID)

    @property
    def name(self) -> str:
        return self._node_info.get(_FS.WIKI.NODE.NAME)

    @property
    def type(self) -> str:
        return self._node_info.get(_FS.WIKI.NODE.TYPE)

    @property
    def parent_id(self) -> str:
        return self._node_info.get(_FS.WIKI.NODE.PARENT_ID)

class WikiSpace:
    """飞书知识空间"""

    def __init__(self, client, space_id: str):
        self.client: lark.client.Client = client
        self.space_id = space_id
        self._nodes = None
        self._info = None

    def get_info(self) -> dict:
        """获取知识空间信息

        注意：如果 space_id 是字符串类型的节点token，
        需要使用 get_info_by_node_token 方法
        """
        if self._info is not None:
            return self._info

        # 尝试将 space_id 转为整数
        try:
            space_id_int = int(self.space_id)
            request: GetSpaceRequest = GetSpaceRequest.builder() \
                .space_id(space_id_int) \
                .build()

            response: GetSpaceResponse = self.client.wiki.v2.space.get(request)

            if not response.success():
                lark.logger.error(f"wiki.v2.space.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
                return {}

            info = {
                'space_id': self.space_id,
                'name': response.data.space.name if response.data and response.data.space else None,
                'description': response.data.space.description if response.data and response.data.space else None,
            }
            self._info = info
            return info
        except ValueError:
            lark.logger.warning(f"space_id '{self.space_id}' 不是整数，无法使用 space.get API")
            return {'space_id': self.space_id, 'name': None, 'description': None}

    def get_info_by_node_token(self, node_token: str, obj_type: str = 'docx') -> dict:
        """通过节点token获取空间信息

        Args:
            node_token: 节点token
            obj_type: 对象类型，如 'docx', 'sheet', 'bitable'

        Returns:
            空间信息字典
        """
        request: GetNodeSpaceRequest = GetNodeSpaceRequest.builder() \
            .token(node_token) \
            .obj_type(obj_type) \
            .build()

        response: GetNodeSpaceResponse = self.client.wiki.v2.space.get_node(request)

        if not response.success():
            lark.logger.error(f"wiki.v2.space.get_node failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return {}

        info = {
            'space_id': response.data.space.space_id if response.data and response.data.space else None,
            'name': response.data.space.name if response.data and response.data.space else None,
            'node_token': node_token,
        }
        return info

    def list_nodes(self, parent_node_token: str = None) -> pd.DataFrame:
        """列出知识空间节点

        Args:
            parent_node_token: 父节点token，不传则获取根节点

        Returns:
            DataFrame包含节点信息
        """
        if self._nodes is not None and parent_node_token is None:
            return self._nodes

        def inner(page_token: str = None):
            request: ListSpaceNodeRequest = ListSpaceNodeRequest.builder() \
                .space_id(self.space_id) \
                .parent_node_token(parent_node_token) \
                .page_token(page_token if page_token else '') \
                .page_size(50) \
                .build()

            response: ListSpaceNodeResponse = self.client.wiki.v2.space_node.list(request)

            if not response.success():
                lark.logger.error(f"wiki.v2.space_node.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
                return False, None, pd.DataFrame()

            items = response.data.items if response.data else []
            nodes = []
            for item in items:
                nodes.append({
                    _FS.WIKI.NODE.ID: item.node_token,
                    _FS.WIKI.NODE.NAME: item.title,
                    _FS.WIKI.NODE.TYPE: item.obj_type,
                    _FS.WIKI.NODE.PARENT_ID: item.parent_node_token,
                })

            has_more = response.data.has_more if response.data else False
            next_page_token = response.data.page_token if response.data else None

            return has_more, next_page_token, pd.DataFrame(nodes)

        # 处理分页
        all_nodes = []
        has_more = True
        page_token = None
        while has_more:
            has_more, page_token, df_page = inner(page_token)
            if not df_page.empty:
                all_nodes.append(df_page)

        df = pd.concat(all_nodes, ignore_index=True) if all_nodes else pd.DataFrame()

        if parent_node_token is None:
            self._nodes = df

        return df

    def get_node(self, node_token: str) -> Optional[WikiNode]:
        """获取指定节点"""
        nodes = self.list_nodes()
        if nodes.empty:
            return None

        node_row = nodes[nodes[_FS.WIKI.NODE.ID] == node_token]
        if node_row.empty:
            return None

        return WikiNode(self, node_row.iloc[0].to_dict())

    def create_node(self, title: str, node_type: str, parent_node_token: str = None) -> Optional[WikiNode]:
        """创建节点

        Args:
            title: 节点标题
            node_type: 节点类型，如 'docx', 'sheet', 'bitable'
            parent_node_token: 父节点token

        Returns:
            新创建的WikiNode对象
        """
        request: CreateSpaceNodeRequest = CreateSpaceNodeRequest.builder() \
            .space_id(self.space_id) \
            .request_body(CreateSpaceNodeRequestBody.builder()
                          .title(title)
                          .obj_type(node_type)
                          .parent_node_token(parent_node_token)
                          .build()) \
            .build()

        response: CreateSpaceNodeResponse = self.client.wiki.v2.space_node.create(request)

        if not response.success():
            lark.logger.error(f"wiki.v2.space_node.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return None

        self._nodes = None  # 清除缓存

        node_info = {
            _FS.WIKI.NODE.ID: response.data.node_token if response.data else None,
            _FS.WIKI.NODE.NAME: title,
            _FS.WIKI.NODE.TYPE: node_type,
            _FS.WIKI.NODE.PARENT_ID: parent_node_token,
        }
        return WikiNode(self, node_info)

    def move_node(self, node_token: str, target_parent_token: str = None) -> bool:
        """移动节点

        Args:
            node_token: 要移动的节点token
            target_parent_token: 目标父节点token
        """
        request: MoveSpaceNodeRequest = MoveSpaceNodeRequest.builder() \
            .space_id(self.space_id) \
            .node_token(node_token) \
            .request_body(MoveSpaceNodeRequestBody.builder()
                          .target_parent_node_token(target_parent_token)
                          .build()) \
            .build()

        response: MoveSpaceNodeResponse = self.client.wiki.v2.space_node.move(request)

        if not response.success():
            lark.logger.error(f"wiki.v2.space_node.move failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return False

        self._nodes = None  # 清除缓存
        return True

    def update_node_title(self, node_token: str, title: str) -> bool:
        """更新节点标题

        Args:
            node_token: 节点token
            title: 新标题
        """
        request: UpdateTitleSpaceNodeRequest = UpdateTitleSpaceNodeRequest.builder() \
            .space_id(self.space_id) \
            .node_token(node_token) \
            .request_body(UpdateTitleSpaceNodeRequestBody.builder()
                          .title(title)
                          .build()) \
            .build()

        response: UpdateTitleSpaceNodeResponse = self.client.wiki.v2.space_node.update_title(request)

        if not response.success():
            lark.logger.error(f"wiki.v2.space_node.update_title failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return False

        self._nodes = None  # 清除缓存
        return True

    def list_members(self) -> pd.DataFrame:
        """列出知识空间成员

        Returns:
            DataFrame包含成员信息
        """
        import json

        def inner(page_token: str = None):
            request: ListSpaceMemberRequest = ListSpaceMemberRequest.builder() \
                .space_id(self.space_id) \
                .page_token(page_token if page_token else '') \
                .page_size(50) \
                .build()

            response: ListSpaceMemberResponse = self.client.wiki.v2.space_member.list(request)

            if not response.success():
                lark.logger.error(f"wiki.v2.space_member.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
                return False, None, pd.DataFrame()

            # 从原始响应解析数据
            try:
                content = json.loads(response.raw.content) if response.raw else {}
                data = content.get('data', {})
                items = data.get('items', [])
                members = []
                for item in items:
                    member_info = {
                        'member_id': item.get('member_id'),
                        'member_type': item.get('member_type'),
                        'member_role': item.get('member_role'),
                    }
                    members.append(member_info)

                has_more = data.get('has_more', False)
                next_page_token = data.get('page_token')

                return has_more, next_page_token, pd.DataFrame(members)
            except (json.JSONDecodeError, AttributeError) as e:
                lark.logger.error(f"Failed to parse members response: {e}")
                return False, None, pd.DataFrame()

        # 处理分页
        all_members = []
        has_more = True
        page_token = None
        while has_more:
            has_more, page_token, df_page = inner(page_token)
            if not df_page.empty:
                all_members.append(df_page)

        return pd.concat(all_members, ignore_index=True) if all_members else pd.DataFrame()

    def add_member(self, member_id: str, member_type: str = 'openid', member_role: str = 'member') -> bool:
        """添加成员到知识空间

        Args:
            member_id: 成员ID（如open_id、user_id等）
            member_type: 成员类型，如 'openid', 'userid', 'groupid'
            member_role: 成员角色，如 'member', 'admin'

        Returns:
            是否添加成功
        """
        request: CreateSpaceMemberRequest = CreateSpaceMemberRequest.builder() \
            .space_id(self.space_id) \
            .request_body({
                'member_id': member_id,
                'member_type': member_type,
                'member_role': member_role,
            }) \
            .build()

        response: CreateSpaceMemberResponse = self.client.wiki.v2.space_member.create(request)

        if not response.success():
            lark.logger.error(f"wiki.v2.space_member.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return False

        return True

    def remove_member(self, member_id: str) -> bool:
        """从知识空间移除成员

        Args:
            member_id: 成员ID

        Returns:
            是否移除成功
        """
        request: DeleteSpaceMemberRequest = DeleteSpaceMemberRequest.builder() \
            .space_id(self.space_id) \
            .member_id(member_id) \
            .build()

        response: DeleteSpaceMemberResponse = self.client.wiki.v2.space_member.delete(request)

        if not response.success():
            lark.logger.error(f"wiki.v2.space_member.delete failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return False

        return True

    def add_bot(self, bot_open_id: str, member_role: str = 'admin') -> bool:
        """添加机器人到知识空间

        Args:
            bot_open_id: 机器人的 open_id
            member_role: 成员角色，如 'admin'（飞书Wiki API目前只支持admin）

        Returns:
            是否添加成功
        """
        return self.add_member(bot_open_id, member_type='openid', member_role=member_role)

class Wiki:
    """飞书知识库管理"""

    def __init__(self, client):
        self.client: lark.client.Client = client

    def list_spaces(self) -> pd.DataFrame:
        """列出所有知识空间"""
        def inner(page_token: str = None):
            request: ListSpaceRequest = ListSpaceRequest.builder() \
                .page_token(page_token if page_token else '') \
                .page_size(50) \
                .build()

            response: ListSpaceResponse = self.client.wiki.v2.space.list(request)

            if not response.success():
                lark.logger.error(f"wiki.v2.space.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
                return False, None, pd.DataFrame()

            items = response.data.items if response.data else []
            spaces = []
            for item in items:
                spaces.append({
                    _FS.WIKI.SPACE.ID: item.space_id,
                    _FS.WIKI.SPACE.NAME: item.name,
                    _FS.WIKI.SPACE.DESCRIPTION: item.description,
                })

            has_more = response.data.has_more if response.data else False
            next_page_token = response.data.page_token if response.data else None

            return has_more, next_page_token, pd.DataFrame(spaces)

        # 处理分页
        all_spaces = []
        has_more = True
        page_token = None
        while has_more:
            has_more, page_token, df_page = inner(page_token)
            if not df_page.empty:
                all_spaces.append(df_page)

        return pd.concat(all_spaces, ignore_index=True) if all_spaces else pd.DataFrame()

    def get_space(self, space_id: str) -> Optional[WikiSpace]:
        """获取知识空间"""
        return WikiSpace(self.client, space_id)

    def create_space(self, name: str, description: str = None) -> Optional[WikiSpace]:
        """创建知识空间

        Args:
            name: 空间名称
            description: 空间描述

        Returns:
            新创建的WikiSpace对象
        """
        request: CreateSpaceRequest = CreateSpaceRequest.builder() \
            .request_body(CreateSpaceRequestBody.builder()
                          .name(name)
                          .description(description)
                          .build()) \
            .build()

        response: CreateSpaceResponse = self.client.wiki.v2.space.create(request)

        if not response.success():
            lark.logger.error(f"wiki.v2.space.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
            return None

        space_id = response.data.space.space_id if response.data and response.data.space else None
        if space_id:
            return WikiSpace(self.client, space_id)
        return None
