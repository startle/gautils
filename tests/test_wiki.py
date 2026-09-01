import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock

import pandas as pd

import gautils.feishu.core.wiki as wiki_module
from gautils.feishu.core.wiki import Wiki, WikiNode, WikiSpace
from tests.feishu_fakes import obj, response


@contextmanager
def fake_wiki_builders(*names):
    replacements = {name: MagicMock() for name in names}
    with unittest.mock.patch.multiple(wiki_module, create=True, **replacements):
        yield


class TestWikiNode(unittest.TestCase):
    def test_properties(self):
        node = WikiNode(MagicMock(), {
            'node_token': 'n1', 'title': 'Doc', 'obj_type': 'docx',
            'parent_node_token': 'p1',
        })
        self.assertEqual(node.id, 'n1')
        self.assertEqual(node.name, 'Doc')
        self.assertEqual(node.type, 'docx')
        self.assertEqual(node.parent_id, 'p1')


class TestWikiSpace(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.space = WikiSpace(self.client, '123')

    def test_get_info_success_cache_failure_and_non_numeric(self):
        api = self.client.wiki.v2.space.get
        api.return_value = response(data=obj(space=obj(name='Space', description='Desc')))
        expected = {'space_id': '123', 'name': 'Space', 'description': 'Desc'}
        self.assertEqual(self.space.get_info(), expected)
        self.assertEqual(self.space.get_info(), expected)
        self.assertEqual(api.call_count, 1)

        failed = WikiSpace(self.client, '456')
        api.return_value = response(success=False)
        self.assertEqual(failed.get_info(), {})

        empty = WikiSpace(self.client, '789')
        api.return_value = response(data=None)
        self.assertEqual(empty.get_info(), {'space_id': '789', 'name': None, 'description': None})

        text_id = WikiSpace(self.client, 'node-token')
        self.assertEqual(text_id.get_info(), {
            'space_id': 'node-token', 'name': None, 'description': None,
        })

    def test_get_info_by_node_token(self):
        api = self.client.wiki.v2.space.get_node
        api.return_value = response(data=obj(space=obj(space_id='123', name='Space')))
        self.assertEqual(self.space.get_info_by_node_token('node'), {
            'space_id': '123', 'name': 'Space', 'node_token': 'node',
        })
        api.return_value = response(data=None)
        self.assertEqual(self.space.get_info_by_node_token('empty'), {
            'space_id': None, 'name': None, 'node_token': 'empty',
        })
        api.return_value = response(success=False)
        self.assertEqual(self.space.get_info_by_node_token('bad'), {})

    def test_list_nodes_pagination_cache_and_parent_query(self):
        api = self.client.wiki.v2.space_node.list
        first = obj(node_token='n1', title='One', obj_type='docx', parent_node_token=None)
        second = obj(node_token='n2', title='Two', obj_type='sheet', parent_node_token='n1')
        api.side_effect = [
            response(data=obj(items=[first], has_more=True, page_token='next')),
            response(data=obj(items=[second], has_more=False, page_token=None)),
        ]
        nodes = self.space.list_nodes()
        self.assertEqual(nodes['node_token'].tolist(), ['n1', 'n2'])
        self.assertIs(self.space.list_nodes(), nodes)
        self.assertEqual(api.call_count, 2)

        api.side_effect = None
        api.return_value = response(data=obj(items=[], has_more=False, page_token=None))
        self.assertTrue(self.space.list_nodes(parent_node_token='n1').empty)
        self.assertEqual(api.call_count, 3)

    def test_list_nodes_failure_and_no_data(self):
        api = self.client.wiki.v2.space_node.list
        api.return_value = response(success=False)
        self.assertTrue(self.space.list_nodes().empty)
        self.space._nodes = None
        api.return_value = response(data=None)
        self.assertTrue(self.space.list_nodes().empty)

    def test_get_node_found_missing_and_empty(self):
        self.space.list_nodes = MagicMock(return_value=pd.DataFrame([
            {'node_token': 'n1', 'title': 'One', 'obj_type': 'docx', 'parent_node_token': None},
        ]))
        found = self.space.get_node('n1')
        self.assertIsInstance(found, WikiNode)
        self.assertEqual(found.name, 'One')
        self.assertIsNone(self.space.get_node('missing'))
        self.space.list_nodes.return_value = pd.DataFrame()
        self.assertIsNone(self.space.get_node('n1'))

    def test_create_node_success_empty_data_and_failure(self):
        with fake_wiki_builders('CreateSpaceNodeRequest', 'CreateSpaceNodeRequestBody'):
            api = self.client.wiki.v2.space_node.create
            api.return_value = response(data=obj(node_token='new-node'))
            self.space._nodes = pd.DataFrame({'x': [1]})
            node = self.space.create_node('Title', 'docx', 'parent')
            self.assertEqual(node.id, 'new-node')
            self.assertEqual(node.parent_id, 'parent')
            self.assertIsNone(self.space._nodes)

            api.return_value = response(data=None)
            self.assertIsNone(self.space.create_node('Empty', 'docx').id)
            api.return_value = response(success=False)
            self.assertIsNone(self.space.create_node('Bad', 'docx'))

    def test_move_and_update_title_success_and_failure(self):
        names = ('MoveSpaceNodeRequest', 'MoveSpaceNodeRequestBody',
                 'UpdateTitleSpaceNodeRequest', 'UpdateTitleSpaceNodeRequestBody')
        with fake_wiki_builders(*names):
            move = self.client.wiki.v2.space_node.move
            move.return_value = response()
            self.space._nodes = 'cached'
            self.assertTrue(self.space.move_node('n1', 'parent'))
            self.assertIsNone(self.space._nodes)
            move.return_value = response(success=False)
            self.assertFalse(self.space.move_node('n1'))

            update = self.client.wiki.v2.space_node.update_title
            update.return_value = response()
            self.space._nodes = 'cached'
            self.assertTrue(self.space.update_node_title('n1', 'New'))
            self.assertIsNone(self.space._nodes)
            update.return_value = response(success=False)
            self.assertFalse(self.space.update_node_title('n1', 'Bad'))

    def test_list_members_pagination_failure_and_parse_error(self):
        api = self.client.wiki.v2.space_member.list
        api.side_effect = [
            response(content={'data': {'items': [
                {'member_id': 'u1', 'member_type': 'openid', 'member_role': 'admin'},
            ], 'has_more': True, 'page_token': 'next'}}),
            response(content={'data': {'items': [
                {'member_id': 'u2', 'member_type': 'userid', 'member_role': 'member'},
            ], 'has_more': False}}),
        ]
        self.assertEqual(self.space.list_members()['member_id'].tolist(), ['u1', 'u2'])

        api.side_effect = None
        api.return_value = response(success=False)
        self.assertTrue(self.space.list_members().empty)
        api.return_value = response(content='{bad-json')
        self.assertTrue(self.space.list_members().empty)
    def test_member_mutations_and_add_bot(self):
        create = self.client.wiki.v2.space_member.create
        create.return_value = response()
        self.assertTrue(self.space.add_member('u1'))
        create.return_value = response(success=False)
        self.assertFalse(self.space.add_member('u1'))

        delete = self.client.wiki.v2.space_member.delete
        delete.return_value = response()
        self.assertTrue(self.space.remove_member('u1'))
        delete.return_value = response(success=False)
        self.assertFalse(self.space.remove_member('u1'))

        self.space.add_member = MagicMock(return_value=True)
        self.assertTrue(self.space.add_bot('bot'))
        self.space.add_member.assert_called_once_with(
            'bot', member_type='openid', member_role='admin')


class TestWiki(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.wiki = Wiki(self.client)

    def test_list_spaces_pagination_failure_and_empty(self):
        api = self.client.wiki.v2.space.list
        first = obj(space_id='1', name='One', description='D1')
        second = obj(space_id='2', name='Two', description='D2')
        api.side_effect = [
            response(data=obj(items=[first], has_more=True, page_token='next')),
            response(data=obj(items=[second], has_more=False, page_token=None)),
        ]
        spaces = self.wiki.list_spaces()
        self.assertEqual(spaces['space_id'].tolist(), ['1', '2'])

        api.side_effect = None
        api.return_value = response(success=False)
        self.assertTrue(self.wiki.list_spaces().empty)
        api.return_value = response(data=None)
        self.assertTrue(self.wiki.list_spaces().empty)

    def test_get_and_create_space_contracts(self):
        self.assertIsInstance(self.wiki.get_space('123'), WikiSpace)
        with fake_wiki_builders('CreateSpaceRequest', 'CreateSpaceRequestBody'):
            api = self.client.wiki.v2.space.create
            api.return_value = response(data=obj(space=obj(space_id='456')))
            created = self.wiki.create_space('New', 'Desc')
            self.assertIsInstance(created, WikiSpace)
            self.assertEqual(created.space_id, '456')

            api.return_value = response(data=None)
            self.assertIsNone(self.wiki.create_space('Empty'))
            api.return_value = response(data=obj(space=None))
            self.assertIsNone(self.wiki.create_space('No Space'))
            api.return_value = response(success=False)
            self.assertIsNone(self.wiki.create_space('Bad'))


if __name__ == '__main__':
    unittest.main()
