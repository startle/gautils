import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, List, Dict

class CScheduler:
    def __init__(self):
        self.lines: Dict[str, Dict] = {}
    def register_line(self, line: str, interval_s: float, is_loop=True, start_time: datetime = None, end_time: datetime = None):
        self.lines[line] = {
            'interval': interval_s,
            'tasks': [],
            'start_time': start_time,
            'end_time': end_time,
            'is_loop': is_loop,
        }
    def register_tasks(self, line: str, tasks: List[Callable]):
        if line in self.lines:
            for task in tasks:
                self.lines[line]['tasks'].append(task)
    def is_active(self, line: str) -> bool:
        current_time = datetime.now()
        line_data = self.lines[line]
        return line_data['start_time'] <= current_time < line_data['end_time']
    async def _run_line(self, line: str):
        data = self.lines[line]

        async def run_tasks():
            for task in data['tasks']:
                try:
                    task()
                except Exception as e:
                    logging.error(f"task error", exc_info=e)
                await asyncio.sleep(data['interval'])
        if data['is_loop']:
            await asyncio.sleep(max(0, (data['start_time'] - datetime.now()).total_seconds()))
            while self.is_active(line):
                await run_tasks()
        else:
            await run_tasks()
    async def _run(self):
        tasks = [self._run_line(line) for line in self.lines]
        await asyncio.gather(*tasks)
    def run(self):
        asyncio.run(self._run())


if __name__ == '__main__':
    def task1():
        print("Task 1 executed")

    def task2():
        print("Task 2 executed")

    time1 = datetime.now()
    time2 = time1 + timedelta(seconds=5)
    time3 = time1 + timedelta(seconds=10)

    scheduler = CScheduler()
    scheduler.register_line('line1', 1.0, time1, time3)
    scheduler.register_tasks('line1', [task1], is_loop=True)
    scheduler.register_line('line2', 2.0, time2, time3)
    scheduler.register_tasks('line2', [task2], is_loop=True)

    scheduler.run()
