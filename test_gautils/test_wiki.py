import pandas as pd
import unittest
import time

from gautils.feishu.core import Wiki, WikiSpace, Feishu
from test_gautils.env import require_env


class TestWiki(unittest.TestCase):
    """飞书知识库单元测试"""

    @classmethod
    def setUpClass(cls):
        app_id, app_secret, space_id, node_token = require_env(
            'FEISHU_APP_ID',
            'FEISHU_APP_SECRET',
            'FEISHU_WIKI_SPACE_ID',
            'FEISHU_WIKI_NODE_TOKEN',
        )
        cls.fs = Feishu(app_id, app_secret)
        cls.wiki: Wiki = cls.fs.get_wiki()
        # 使用指定的知识空间token（注意：wiki space_id 需要是整数类型）
        # 如果提供的token不是纯数字，可能需要使用其他API或方式
        cls.space_id = space_id
        cls.node_token = node_token
        cls.space: WikiSpace = cls.wiki.get_space(cls.space_id)
        cls.test_marker = f"unittest_{int(time.time())}"

    @classmethod
    def tearDownClass(cls):
        pass

    def test_list_spaces(self):
        """测试列出所有知识空间"""
        spaces = self.wiki.list_spaces()
        print(f'知识空间列表: {len(spaces)} 个')
        if not spaces.empty:
            for _, space in spaces.iterrows():
                print(f"  - {space['name']} (id={space['space_id']})")
        self.assertIsInstance(spaces, pd.DataFrame)

    def test_get_space(self):
        """测试获取知识空间对象"""
        spaces = self.wiki.list_spaces()
        if spaces.empty:
            print('没有可用的知识空间，跳过测试')
            return

        space_id = spaces.iloc[0]['space_id']
        space = self.wiki.get_space(space_id)
        print(f'获取知识空间: {space_id}')
        self.assertIsNotNone(space)
        self.assertEqual(space.space_id, space_id)

    def test_space_get_info(self):
        """测试获取知识空间信息"""
        spaces = self.wiki.list_spaces()
        if spaces.empty:
            print('没有可用的知识空间，跳过测试')
            return

        space_id = spaces.iloc[0]['space_id']
        space = self.wiki.get_space(space_id)
        info = space.get_info()
        print(f'知识空间信息: {info}')
        self.assertIsInstance(info, dict)
        self.assertIn('space_id', info)

    def test_space_list_nodes(self):
        """测试列出知识空间节点"""
        spaces = self.wiki.list_spaces()
        if spaces.empty:
            print('没有可用的知识空间，跳过测试')
            return

        space_id = spaces.iloc[0]['space_id']
        space = self.wiki.get_space(space_id)
        nodes = space.list_nodes()
        print(f'知识空间节点: {len(nodes)} 个')
        if not nodes.empty:
            for _, node in nodes.iterrows():
                print(f"  - {node['title']} (type={node['obj_type']}, id={node['node_token']})")
        self.assertIsInstance(nodes, pd.DataFrame)

    def test_space_get_node(self):
        """测试获取指定节点"""
        spaces = self.wiki.list_spaces()
        if spaces.empty:
            print('没有可用的知识空间，跳过测试')
            return

        space_id = spaces.iloc[0]['space_id']
        space = self.wiki.get_space(space_id)
        nodes = space.list_nodes()

        if nodes.empty:
            print('知识空间没有节点，跳过测试')
            return

        node_token = nodes.iloc[0]['node_token']
        node = space.get_node(node_token)
        print(f'获取节点: {node.name if node else None}')
        self.assertIsNotNone(node)
        self.assertEqual(node.id, node_token)

    def test_specific_space_list_nodes(self):
        """测试使用指定空间token读取节点"""
        print(f'使用指定空间token: {self.space_id}')
        nodes = self.space.list_nodes()
        print(f'知识空间节点: {len(nodes)} 个')
        if not nodes.empty:
            for _, node in nodes.iterrows():
                print(f"  - {node['title']} (type={node['obj_type']}, id={node['node_token']})")
        else:
            print('  空间没有节点或没有访问权限')
        self.assertIsInstance(nodes, pd.DataFrame)

    def test_specific_space_get_info(self):
        """测试使用指定空间token获取空间信息"""
        print(f'使用指定空间token: {self.space_id}')
        info = self.space.get_info()
        print(f'知识空间信息: {info}')
        self.assertIsInstance(info, dict)
        self.assertEqual(info.get('space_id'), self.space_id)

    def test_get_info_by_node_token(self):
        """测试通过节点token获取空间信息"""
        # 使用提供的字符串token作为节点token
        node_token = self.node_token
        print(f'使用节点token: {node_token}')
        info = self.space.get_info_by_node_token(node_token, obj_type='docx')
        print(f'通过节点获取的空间信息: {info}')
        self.assertIsInstance(info, dict)
        if info.get('space_id'):
            print(f'  空间ID: {info["space_id"]}')
            print(f'  空间名称: {info.get("name")}')

    # def test_create_space(self):
    #     """测试创建知识空间（需要写权限，暂时注释）"""
    #     space_name = f'测试空间_{self.test_marker}'
    #     space = self.wiki.create_space(space_name, description='单元测试创建')
    #     print(f'创建知识空间: {space.space_id if space else None}')
    #     self.assertIsNotNone(space)

    # def test_space_create_node(self):
    #     """测试创建节点（需要写权限，暂时注释）"""
    #     spaces = self.wiki.list_spaces()
    #     if spaces.empty:
    #         print('没有可用的知识空间，跳过测试')
    #         return
    #
    #     space_id = spaces.iloc[0]['space_id']
    #     space = self.wiki.get_space(space_id)
    #
    #     node = space.create_node('测试文档', 'docx')
    #     print(f'创建节点: {node.id if node else None}')
    #     self.assertIsNotNone(node)

    # def test_space_move_node(self):
    #     """测试移动节点（需要写权限，暂时注释）"""
    #     pass

    # def test_space_update_node_title(self):
    #     """测试更新节点标题（需要写权限，暂时注释）"""
    #     pass


if __name__ == '__main__':
    unittest.main()
