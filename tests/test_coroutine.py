import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from gautils.coroutine import CScheduler


class TestCScheduler(unittest.TestCase):
    def test_register_line(self):
        scheduler = CScheduler()
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)

        scheduler.register_line('test_line', 1.0, start_time=start_time, end_time=end_time)

        self.assertIn('test_line', scheduler.lines)
        self.assertEqual(scheduler.lines['test_line']['interval'], 1.0)
        self.assertEqual(scheduler.lines['test_line']['start_time'], start_time)
        self.assertEqual(scheduler.lines['test_line']['end_time'], end_time)

    def test_register_tasks(self):
        scheduler = CScheduler()
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)

        scheduler.register_line('test_line', 1.0, start_time=start_time, end_time=end_time)

        mock_task = MagicMock()
        scheduler.register_tasks('test_line', [mock_task])

        self.assertEqual(len(scheduler.lines['test_line']['tasks']), 1)
        self.assertEqual(scheduler.lines['test_line']['tasks'][0], mock_task)

    def test_register_tasks_invalid_line(self):
        scheduler = CScheduler()
        mock_task = MagicMock()

        scheduler.register_tasks('nonexistent_line', [mock_task])
        self.assertNotIn('nonexistent_line', scheduler.lines)

    def test_is_active_within_range(self):
        scheduler = CScheduler()
        now = datetime.now()
        start_time = now - timedelta(seconds=5)
        end_time = now + timedelta(seconds=5)

        scheduler.register_line('test_line', 1.0, start_time=start_time, end_time=end_time)

        self.assertTrue(scheduler.is_active('test_line'))

    def test_is_active_before_start(self):
        scheduler = CScheduler()
        now = datetime.now()
        start_time = now + timedelta(seconds=5)
        end_time = now + timedelta(seconds=10)

        scheduler.register_line('test_line', 1.0, start_time=start_time, end_time=end_time)

        self.assertFalse(scheduler.is_active('test_line'))

    def test_is_active_after_end(self):
        scheduler = CScheduler()
        now = datetime.now()
        start_time = now - timedelta(seconds=10)
        end_time = now - timedelta(seconds=5)

        scheduler.register_line('test_line', 1.0, start_time=start_time, end_time=end_time)

        self.assertFalse(scheduler.is_active('test_line'))

    def test_is_loop_false(self):
        scheduler = CScheduler()
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)

        scheduler.register_line('test_line', 1.0, is_loop=False, start_time=start_time, end_time=end_time)

        self.assertFalse(scheduler.lines['test_line']['is_loop'])

    def test_is_active_without_time_bounds(self):
        scheduler = CScheduler()
        scheduler.register_line('test_line', 1.0)

        self.assertTrue(scheduler.is_active('test_line'))


class TestCSchedulerAsync(unittest.TestCase):
    def test_run_line_non_loop(self):
        scheduler = CScheduler()
        start_time = datetime.now() - timedelta(seconds=1)
        end_time = start_time + timedelta(seconds=10)

        scheduler.register_line('test_line', 0.1, is_loop=False, start_time=start_time, end_time=end_time)

        mock_task = MagicMock()
        scheduler.register_tasks('test_line', [mock_task])

        async def run_test():
            await scheduler._run_line('test_line')

        asyncio.run(run_test())
        mock_task.assert_called_once()

    @patch('gautils.coroutine.logging.error')
    def test_run_line_with_exception(self, mock_log_error):
        scheduler = CScheduler()
        start_time = datetime.now() - timedelta(seconds=1)
        end_time = start_time + timedelta(seconds=10)

        scheduler.register_line('test_line', 0.1, is_loop=False, start_time=start_time, end_time=end_time)

        def failing_task():
            raise ValueError('test error')

        scheduler.register_tasks('test_line', [failing_task])

        async def run_test():
            await scheduler._run_line('test_line')

        asyncio.run(run_test())
        mock_log_error.assert_called_once()

    def test_run_multiple_lines(self):
        scheduler = CScheduler()
        start_time = datetime.now() - timedelta(seconds=1)
        end_time = start_time + timedelta(seconds=2)

        scheduler.register_line('line1', 0.1, is_loop=False, start_time=start_time, end_time=end_time)
        scheduler.register_line('line2', 0.1, is_loop=False, start_time=start_time, end_time=end_time)

        mock_task1 = MagicMock()
        mock_task2 = MagicMock()
        scheduler.register_tasks('line1', [mock_task1])
        scheduler.register_tasks('line2', [mock_task2])

        async def run_test():
            await scheduler._run()

        asyncio.run(run_test())
        mock_task1.assert_called_once()
        mock_task2.assert_called_once()

    def test_run_line_loop_without_start_time_runs_immediately(self):
        scheduler = CScheduler()
        scheduler.register_line('test_line', 0.01, is_loop=True, end_time=datetime.now() + timedelta(seconds=0.03))

        mock_task = MagicMock()
        scheduler.register_tasks('test_line', [mock_task])

        async def run_test():
            await scheduler._run_line('test_line')

        asyncio.run(run_test())
        mock_task.assert_called()


if __name__ == '__main__':
    unittest.main()
