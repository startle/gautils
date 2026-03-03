import unittest

from gautils.proxies import XiaoXiangProxy


class TestXiaoXiangProxy(unittest.TestCase):
    def test_init(self):
        proxy = XiaoXiangProxy('test_user', 'test_pwd')
        self.assertEqual(proxy.user, 'test_user')
        self.assertEqual(proxy.pwd, 'test_pwd')

    def test_gen_short_proxies_builder_f(self):
        proxy = XiaoXiangProxy('user123', 'pass456')
        builder_f = proxy.gen_short_proxies_builder_f()
        proxies = builder_f()

        self.assertIn('http', proxies)
        self.assertIn('https', proxies)
        self.assertEqual(proxies['http'], proxies['https'])
        self.assertIn('user123', proxies['http'])
        self.assertIn('pass456', proxies['http'])
        self.assertIn('http-short.xiaoxiangdaili.com', proxies['http'])
        self.assertIn('10010', proxies['http'])

    def test_gen_dynamic_proxies_builder_f(self):
        proxy = XiaoXiangProxy('user123', 'pass456')
        builder_f = proxy.gen_dynamic_proxies_builder_f()
        proxies = builder_f()

        self.assertIn('http', proxies)
        self.assertIn('https', proxies)
        self.assertIn('user123', proxies['http'])
        self.assertIn('pass456', proxies['http'])
        self.assertIn('http-dynamic.xiaoxiangdaili.com', proxies['http'])
        self.assertIn('10030', proxies['http'])

    def test_proxies_format(self):
        proxy = XiaoXiangProxy('myuser', 'mypassword')
        builder_f = proxy.gen_short_proxies_builder_f()
        proxies = builder_f()

        expected_format = 'http://myuser:mypassword@http-short.xiaoxiangdaili.com:10010'
        self.assertEqual(proxies['http'], expected_format)
        self.assertEqual(proxies['https'], expected_format)

    def test_multiple_calls_same_result(self):
        proxy = XiaoXiangProxy('user', 'pwd')
        builder_f = proxy.gen_short_proxies_builder_f()

        proxies1 = builder_f()
        proxies2 = builder_f()

        self.assertEqual(proxies1, proxies2)


if __name__ == '__main__':
    unittest.main()
