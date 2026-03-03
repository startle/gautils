import unittest
from unittest.mock import patch, MagicMock

from gautils.feishu.tools import send_fs_robot_msg


class TestSendFsRobotMsg(unittest.TestCase):
    @patch('gautils.feishu.tools.requests.post')
    def test_send_single_message(self, mock_post):
        send_fs_robot_msg('https://open.feishu.cn/open-apis/bot/v2/hook/test', 'Hello World')

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], 'https://open.feishu.cn/open-apis/bot/v2/hook/test')

        import json
        sent_json = call_args[1]['json']
        self.assertEqual(sent_json['msg_type'], 'text')
        self.assertEqual(sent_json['content']['text'], 'Hello World')

    @patch('gautils.feishu.tools.requests.post')
    def test_send_multiple_messages(self, mock_post):
        send_fs_robot_msg('https://open.feishu.cn/open-apis/bot/v2/hook/test', ['Line 1', 'Line 2', 'Line 3'])

        mock_post.assert_called_once()
        call_args = mock_post.call_args

        import json
        sent_json = call_args[1]['json']
        self.assertEqual(sent_json['content']['text'], 'Line 1\nLine 2\nLine 3')

    @patch('gautils.feishu.tools.requests.post')
    def test_send_with_custom_json(self, mock_post):
        custom_json = {
            'msg_type': 'interactive',
            'card': {
                'header': {'title': 'Test'}
            }
        }
        send_fs_robot_msg('https://open.feishu.cn/open-apis/bot/v2/hook/test', json=custom_json)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['json'], custom_json)

    def test_send_empty_raises(self):
        with self.assertRaises(Exception) as context:
            send_fs_robot_msg('https://open.feishu.cn/open-apis/bot/v2/hook/test')
        self.assertIn('empty', str(context.exception).lower())

    def test_send_empty_list(self):
        result = send_fs_robot_msg('https://open.feishu.cn/open-apis/bot/v2/hook/test', [])
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
