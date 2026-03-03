import unittest
from unittest.mock import patch, MagicMock
import sys

from gautils.qwx import WXWorkRobot, send_qwx_md_msg


class TestWXWorkRobot(unittest.TestCase):
    def test_init(self):
        robot = WXWorkRobot('https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test')
        self.assertEqual(robot._url, 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test')

    @patch('requests.post')
    def test_send_md(self, mock_post):
        robot = WXWorkRobot('https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test')
        robot.send_md('test message')

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test')
        self.assertIn('headers', call_args[1])
        self.assertIn('data', call_args[1])
        self.assertEqual(call_args[1]['verify'], False)

    @patch('requests.post')
    def test_send_md_with_mentioned_list(self, mock_post):
        robot = WXWorkRobot('https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test')
        robot.send_md('test message', mentioned_list=['user1', 'user2'])

        mock_post.assert_called_once()
        # Verify the message was sent (mentioned_list is accepted but not used in current implementation)
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test')

    @patch('requests.post')
    def test_send_md_message_format(self, mock_post):
        robot = WXWorkRobot('https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test')
        robot.send_md('Hello World')

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test')
        # Verify data was sent
        self.assertIn('data', call_args[1])


class TestSendQwxMdMsg(unittest.TestCase):
    @patch('gautils.qwx.WXWorkRobot', spec=WXWorkRobot)
    def test_send_qwx_md_msg(self, mock_robot_class):
        mock_robot = MagicMock()
        mock_robot_class.return_value = mock_robot

        send_qwx_md_msg('https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test', 'test message')

        mock_robot_class.assert_called_once_with('https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test')
        mock_robot.send_md.assert_called_once_with('test message', mentioned_list=None)

    @patch('gautils.qwx.WXWorkRobot', spec=WXWorkRobot)
    def test_send_qwx_md_msg_with_mentioned(self, mock_robot_class):
        mock_robot = MagicMock()
        mock_robot_class.return_value = mock_robot

        send_qwx_md_msg('https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test', 'test message', mentioned_list=['user1'])

        mock_robot.send_md.assert_called_once_with('test message', mentioned_list=['user1'])


if __name__ == '__main__':
    unittest.main()
