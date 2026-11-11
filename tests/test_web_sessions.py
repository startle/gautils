from copy import deepcopy

import pytest
import requests

import gautils.web as web_module


class Response:
    def __init__(self, text='ok', status_error=None, text_error=None):
        self.value = text
        self.status_error = status_error
        self.text_error = text_error
        self.encoding = None
        self.apparent_encoding = 'utf-8'

    @property
    def text(self):
        if self.text_error is not None:
            raise self.text_error
        return self.value

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error


class Session:
    def __init__(self, factory):
        self.factory = factory
        self.cookies = requests.cookies.RequestsCookieJar()
        self.closed = 0

    def get(self, url, **kwargs):
        return self._request('GET', url, kwargs)

    def post(self, url, **kwargs):
        return self._request('POST', url, kwargs)

    def _request(self, method, url, kwargs):
        event = {'session': self, 'method': method, 'url': url, 'kwargs': kwargs,
                 'cookies': requests.utils.dict_from_cookiejar(self.cookies)}
        self.factory.calls.append(event)
        effect = self.factory.effects.pop(0) if self.factory.effects else Response()
        if isinstance(effect, BaseException):
            raise effect
        return effect(self, event) if callable(effect) else effect

    def close(self):
        self.closed += 1
        if self.factory.close_error is not None:
            raise self.factory.close_error


class Factory:
    def __init__(self):
        self.sessions = []
        self.calls = []
        self.effects = []
        self.close_error = None

    def __call__(self):
        session = Session(self)
        self.sessions.append(session)
        return session


@pytest.fixture
def client(monkeypatch, tmp_path):
    factory = Factory()
    monkeypatch.setattr(web_module.requests, 'Session', factory)
    sleeps = []
    monkeypatch.setattr(web_module.time, 'sleep', sleeps.append)
    web = web_module.Web(cookies_filepath=str(tmp_path/'cookies.yml'),
                         headers={'Client': 'original'}, verify=False)
    saved = []
    web.cookie_manager.save_cookies = lambda: saved.append(deepcopy(web.cookie_manager.cookies))
    return web, factory, saved, sleeps


def test_default_remains_one_session_per_web_request(client):
    web, factory, _, _ = client
    assert web.get('https://a.test') == 'ok'
    assert factory.sessions[0].closed == 1
    assert web.get('https://a.test/again') == 'ok'
    assert len(factory.sessions) == 2
    assert all(session.closed == 1 for session in factory.sessions)


def test_batch_of_twenty_five_pairs_only_creates_two_sessions(client):
    web, factory, _, _ = client
    with web.connection_batch():
        for code in range(25):
            for host in ('history.test', 'current.test'):
                assert web.get(f'https://{host}/kline', params={'code': code}) == 'ok'
        assert len(factory.sessions) == 2
        assert all(session.closed == 0 for session in factory.sessions)
    assert len(factory.calls) == 50
    assert [session.closed for session in factory.sessions] == [1, 1]


def test_cookie_jar_is_authoritative_each_request_and_empty_jar_clears_session(client):
    web, factory, saved, _ = client
    web.cookie_manager.cookies['a.test'] = {'auth': 'account-a', 'removed': 'old'}

    def receive(session, event):
        session.cookies.pop('removed')
        session.cookies['server'] = 'new'
        return Response()

    factory.effects = [receive, Response()]
    with web.connection_batch():
        web.get('https://a.test')
        assert saved == [{'a.test': {'auth': 'account-a', 'server': 'new'}}]
        web.cookie_manager.cookies['a.test'] = {}
        web.get('https://a.test/again')
        assert factory.calls[0]['cookies'] == {'auth': 'account-a', 'removed': 'old'}
        assert factory.calls[1]['cookies'] == {}
    assert len(saved) == 1
    assert factory.calls[0]['session'] is factory.calls[1]['session']


def test_cookies_are_separate_by_host(client):
    web, factory, saved, _ = client
    web.cookie_manager.cookies.update({'a.test': {'auth': 'a'}, 'b.test': {'auth': 'b'}})
    with web.connection_batch():
        web.get('https://a.test')
        web.get('https://b.test')
        web.get('https://a.test/again')
    assert [event['cookies'] for event in factory.calls] == [{'auth': 'a'}, {'auth': 'b'}, {'auth': 'a'}]
    assert len(saved) == 3
    assert len(factory.sessions) == 2


def test_retry_uses_same_session_and_loads_cookies_only_once(client):
    web, factory, saved, sleeps = client
    web.cookie_manager.cookies['a.test'] = {'auth': 'a'}

    def fail_after_cookie(session, event):
        session.cookies['attempt'] = 'retained'
        raise requests.ConnectionError('fake disconnected')

    factory.effects = [fail_after_cookie, Response('recovered')]
    with web.connection_batch():
        assert web.get('https://a.test', retry_times=2) == 'recovered'
    assert len(factory.sessions) == 1
    assert factory.calls[1]['cookies'] == {'auth': 'a', 'attempt': 'retained'}
    assert sleeps == [5]
    assert saved == [{'a.test': {'auth': 'a', 'attempt': 'retained'}}]


def test_custom_proxy_builder_keeps_fresh_and_selects_proxy_per_retry(client, tmp_path):
    _, factory, _, sleeps = client
    routes = iter(({'https': 'route-1'}, {'https': 'route-2'}, {'https': 'route-3'}))
    web = web_module.Web(cookies_filepath=str(tmp_path/'proxy-cookies.yml'),
                         build_proxies_f=lambda: next(routes))
    factory.effects = [requests.exceptions.ProxyError('fake failed proxy'), Response('retry'), Response('next')]
    with web.connection_batch():
        assert web.get('https://a.test', retry_times=2) == 'retry'
        assert factory.sessions[0].closed == 1
        assert web.get('https://a.test') == 'next'
        assert factory.sessions[1].closed == 1
    assert len(factory.sessions) == 2
    assert [event['kwargs']['proxies'] for event in factory.calls] == [
        {'https': 'route-1'}, {'https': 'route-2'}, {'https': 'route-3'}]
    assert factory.calls[0]['session'] is factory.calls[1]['session']
    assert sleeps == [5]


def test_get_and_post_keep_headers_parameters_tls_timeout_and_encoding(client):
    web, factory, _, _ = client
    first, second = Response('first'), Response('second')
    factory.effects = [first, second]
    with web.connection_batch():
        assert web.get('https://a.test', params={'x': 1}, headers={'Override': 'yes'},
                       timeout=(2, 9), encoding='latin-1') == 'first'
        assert web.post('https://a.test/submit', params={'p': 2}, json={'j': 3}, data={'d': 4}) == 'second'
    assert factory.calls[0]['kwargs'] == {'headers': {'Override': 'yes'}, 'params': {'x': 1},
        'verify': False, 'proxies': None, 'timeout': (2, 9)}
    assert factory.calls[1]['kwargs'] == {'headers': {'Client': 'original'}, 'params': {'p': 2},
        'json': {'j': 3}, 'data': {'d': 4}, 'verify': False, 'proxies': None, 'timeout': (5, 30)}
    assert first.encoding == 'latin-1' and second.encoding == 'utf-8'
    assert web.headers == {'Client': 'original'}
    assert len(factory.sessions) == 1


def test_raise_for_status_keeps_cookie_save_after_success_only(client):
    web, factory, saved, _ = client
    web.cookie_manager.cookies['a.test'] = {'auth': 'a'}
    factory.effects = [Response(status_error=requests.HTTPError('fake status'))]
    with pytest.raises(requests.HTTPError), web.connection_batch():
        web.get('https://a.test', raise_for_status=True)
    assert len(factory.calls) == 1 and saved == []
    assert factory.sessions[0].closed == 1


def test_request_error_closes_fresh_and_preserves_original_retry_chain(client):
    web, factory, _, sleeps = client
    failure = requests.Timeout('fake timeout')
    factory.effects = [failure, failure]
    with pytest.raises(RuntimeError, match='exhausted 2') as caught:
        web.get('https://a.test', retry_times=2)
    assert caught.value.__cause__ is failure
    assert len(factory.sessions) == 1 and factory.sessions[0].closed == 1
    assert sleeps == [5]


def test_failed_request_caught_inside_batch_does_not_poison_next_borrow(client):
    web, factory, _, _ = client
    factory.effects = [requests.Timeout('fake timeout'), Response('next')]
    with web.connection_batch():
        with pytest.raises(RuntimeError):
            web.get('https://a.test', retry_times=1)
        assert web.get('https://a.test') == 'next'
        assert factory.sessions[0].closed == 0
    assert len(factory.sessions) == 1 and factory.sessions[0].closed == 1


def test_cookie_save_error_closes_and_does_not_get_replaced_by_close_error(client):
    web, factory, _, _ = client
    web.cookie_manager.cookies['a.test'] = {'auth': 'a'}
    factory.close_error = RuntimeError('close failed')

    def save():
        raise ValueError('cookie save failed')

    web.cookie_manager.save_cookies = save
    with pytest.raises(ValueError, match='cookie save failed'), web.connection_batch():
        web.get('https://a.test')
    assert factory.sessions[0].closed == 1


def test_text_decode_fallback_still_returns_response_and_releases(client):
    web, factory, _, _ = client
    response = Response(text_error=ValueError('fake text failure'))
    factory.effects = [response]
    assert web.get('https://a.test') is response
    assert factory.sessions[0].closed == 1


@pytest.mark.parametrize('change', ['verify', 'cookie_manager', 'build_proxies_f'])
def test_active_batch_rejects_transport_or_identity_container_changes(client, change):
    web, factory, _, _ = client
    with web.connection_batch():
        setattr(web, change, {'verify': True, 'cookie_manager': object(),
                             'build_proxies_f': lambda: None}[change])
        with pytest.raises(RuntimeError, match='configuration changed'):
            web.get('https://a.test')
    assert factory.calls == []


def test_separate_webs_do_not_share_sessions_even_for_same_origin(client, tmp_path):
    first, factory, _, _ = client
    second = web_module.Web(cookies_filepath=str(tmp_path/'second-cookies.yml'))
    with first.connection_batch(), second.connection_batch():
        first.get('https://a.test')
        second.get('https://a.test')
        first.get('https://a.test/again')
    assert len(factory.sessions) == 2
    assert factory.calls[0]['session'] is factory.calls[2]['session']
    assert factory.calls[0]['session'] is not factory.calls[1]['session']
