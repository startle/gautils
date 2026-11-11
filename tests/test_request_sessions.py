import asyncio
from contextvars import copy_context
from threading import Thread

import pytest

from gautils.request_sessions import RequestSessions


class Session:
    def __init__(self, close_error=None):
        self.closed = 0
        self.close_error = close_error

    def close(self):
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def sessions():
    created = []

    def factory():
        session = Session()
        created.append(session)
        return session

    return RequestSessions(factory), created


def test_default_is_fresh_and_releases_each_request(sessions):
    pool, created = sessions
    with pool.borrow('https://a.test/one') as first:
        assert first.closed == 0
    assert first.closed == 1
    with pool.borrow('https://a.test/two') as second:
        assert second is not first
    assert [s.closed for s in created] == [1, 1]


def test_batch_reuses_origin_and_effective_port_then_releases(sessions):
    pool, created = sessions
    with pool.batch():
        with pool.borrow('https://EXAMPLE.test/one') as first:
            pass
        with pool.borrow('https://example.test:443/two?x=1') as same:
            assert same is first
        for url in ('http://example.test', 'https://example.test:8443', 'https://other.test'):
            with pool.borrow(url) as other:
                assert other is not first
        assert len(created) == 4
        assert all(s.closed == 0 for s in created)
    assert all(s.closed == 1 for s in created)
    with pool.batch():
        with pool.borrow('https://example.test') as next_batch:
            assert next_batch is not first


def test_instances_do_not_share_pool(sessions):
    first, created = sessions
    second = RequestSessions(lambda: Session())
    with first.batch(), second.batch():
        with first.borrow('https://a.test') as a:
            pass
        with second.borrow('https://a.test') as b:
            assert b is not a
        with first.borrow('https://a.test') as again:
            assert again is a
    assert len(created) == 1


def test_nested_batch_only_outer_owner_closes(sessions):
    pool, created = sessions
    with pool.batch(policy='fixed'):
        with pool.borrow('https://a.test', policy='fixed') as first:
            pass
        with pool.batch(policy='fixed'):
            with pool.borrow('https://a.test', policy='fixed') as same:
                assert same is first
        assert first.closed == 0
    assert first.closed == 1
    assert len(created) == 1


def test_nested_error_caught_inside_outer_does_not_close_pool(sessions):
    pool, _ = sessions
    with pool.batch():
        with pytest.raises(ValueError), pool.batch():
            with pool.borrow('https://a.test') as first:
                pass
            raise ValueError('inner failure')
        assert first.closed == 0
        with pool.borrow('https://a.test') as again:
            assert again is first
    assert first.closed == 1


@pytest.mark.parametrize('nested', [False, True])
def test_configuration_change_rejected_before_borrow(sessions, nested):
    pool, created = sessions
    with pool.batch(policy=False):
        with pool.borrow('https://a.test', policy=False):
            pass
        call = pool.batch(policy=True) if nested else pool.borrow('https://a.test', policy=True)
        with pytest.raises(RuntimeError, match='configuration changed'), call:
            pass
        assert len(created) == 1
        with pool.borrow('https://a.test', policy=False):
            pass


def test_custom_transport_can_use_fresh_inside_batch(sessions):
    pool, created = sessions
    with pool.batch():
        with pool.borrow('https://a.test', reuse=False) as first:
            pass
        assert first.closed == 1
        with pool.borrow('https://a.test', reuse=False) as second:
            assert second is not first
    assert [s.closed for s in created] == [1, 1]


def test_reentrant_borrow_rejected_without_releasing_outer_lease(sessions):
    pool, _ = sessions
    with pool.batch():
        with pool.borrow('https://a.test') as first:
            with pytest.raises(RuntimeError, match='serial requests'), pool.borrow('https://b.test'):
                pass
            assert first.closed == 0
        with pool.borrow('https://a.test') as again:
            assert again is first


def test_session_factory_error_does_not_poison_batch():
    session = Session()
    attempts = iter([ValueError('factory failed'), session])

    def factory():
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    pool = RequestSessions(factory)
    with pool.batch():
        with pytest.raises(ValueError, match='factory failed'), pool.borrow('https://a.test'):
            pass
        with pool.borrow('https://a.test') as value:
            assert value is session
    assert session.closed == 1


@pytest.mark.parametrize('body_error', [ValueError('request failed'), KeyboardInterrupt(), asyncio.CancelledError()])
def test_original_failure_wins_over_close_errors_and_all_sessions_close(body_error):
    created = []

    def factory():
        session = Session(RuntimeError('close failed'))
        created.append(session)
        return session

    pool = RequestSessions(factory)
    with pytest.raises(type(body_error)) as caught:
        with pool.batch():
            for host in ('a.test', 'b.test'):
                with pool.borrow('https://' + host):
                    pass
            raise body_error
    assert caught.value is body_error
    assert [s.closed for s in created] == [1, 1]
    # Reset happened even when every close failed.
    with pytest.raises(RuntimeError, match='close failed'), pool.borrow('https://a.test'):
        pass
    assert len(created) == 3 and created[-1].closed == 1


def test_close_failure_propagates_after_closing_all_and_resets_context():
    created = [Session(ValueError('first close')), Session(RuntimeError('second close')), Session()]
    pool = RequestSessions(iter(created).__next__)
    with pytest.raises(ValueError, match='first close'), pool.batch():
        for host in ('a.test', 'b.test'):
            with pool.borrow('https://' + host):
                pass
    assert [s.closed for s in created[:2]] == [1, 1]
    with pool.borrow('https://a.test') as fresh:
        assert fresh is created[2]
    assert fresh.closed == 1


def test_fresh_close_failure_does_not_mask_request_error():
    session = Session(RuntimeError('close failed'))
    pool = RequestSessions(lambda: session)
    with pytest.raises(ValueError, match='request failed'), pool.borrow('https://a.test'):
        raise ValueError('request failed')
    assert session.closed == 1


def test_copied_context_cannot_use_closed_batch(sessions):
    pool, created = sessions
    with pool.batch():
        copied = copy_context()
    with pytest.raises(RuntimeError, match='closed'):
        copied.run(lambda: pool.borrow('https://a.test').__enter__())
    assert not created


def test_inherited_thread_rejected(sessions):
    pool, created = sessions
    errors = []

    def use():
        try:
            with pool.borrow('https://a.test'):
                pass
        except RuntimeError as error:
            errors.append(str(error))

    with pool.batch():
        context = copy_context()
        thread = Thread(target=lambda: context.run(use))
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == ['connection batch belongs to another task or thread']
    assert not created


def test_inherited_child_task_rejected_even_on_same_thread(sessions):
    pool, created = sessions

    async def use():
        with pytest.raises(RuntimeError, match='another task or thread'), pool.borrow('https://a.test'):
            pass
        with pytest.raises(RuntimeError, match='another task or thread'), pool.batch():
            pass

    async def run():
        with pool.batch():
            await asyncio.create_task(use())
            with pool.borrow('https://a.test'):
                pass
    asyncio.run(run())
    assert len(created) == 1 and created[0].closed == 1


def test_independent_async_tasks_each_have_their_own_batch(sessions):
    pool, created = sessions

    async def worker(ready, proceed):
        with pool.batch():
            with pool.borrow('https://a.test') as first:
                pass
            ready.set()
            await proceed.wait()
            with pool.borrow('https://a.test') as again:
                assert again is first
            return first

    async def run():
        one, two = asyncio.Event(), asyncio.Event()
        first = asyncio.create_task(worker(one, two))
        second = asyncio.create_task(worker(two, one))
        a, b = await asyncio.gather(first, second)
        assert a is not b
    asyncio.run(run())
    assert len(created) == 2 and all(s.closed == 1 for s in created)


def test_cancellation_closes_and_resets_before_next_request(sessions):
    pool, created = sessions

    async def worker(started):
        try:
            with pool.batch():
                with pool.borrow('https://a.test') as first:
                    pass
                started.set()
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            assert first.closed == 1
            with pool.borrow('https://a.test') as fresh:
                assert fresh is not first
            raise

    async def run():
        started = asyncio.Event()
        task = asyncio.create_task(worker(started))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert len(created) == 2 and all(s.closed == 1 for s in created)


def test_invalid_scoped_origin_does_not_echo_url_or_poison_scope(sessions):
    pool, created = sessions
    with pool.batch():
        with pytest.raises(ValueError, match='valid HTTP origin') as caught:
            with pool.borrow('https://a.test:secret-material'):
                pass
        assert 'secret-material' not in str(caught.value)
        with pool.borrow('https://a.test'):
            pass
    assert len(created) == 1
