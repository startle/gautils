import unittest

from gautils.v2.errors import (
    ApiError, DatabaseOperationError, ResourceNotFoundError,
    UnsupportedOperationError,
)
from gautils.v2.models import BatchError, BatchResult


class TestBatchModels(unittest.TestCase):
    def test_success_empty_and_failure(self):
        empty = BatchResult(attempted=0, succeeded=0)
        self.assertTrue(empty.ok)
        self.assertEqual(empty.success_rate, 1.0)
        empty.raise_for_errors()

        error = BatchError(index=1, message='bad', code='E1', details={'x': 1})
        result = BatchResult(
            attempted=2, succeeded=1, failed=1, updated=1, errors=(error,))
        self.assertFalse(result.ok)
        self.assertEqual(result.success_rate, 0.5)
        with self.assertRaises(RuntimeError):
            result.raise_for_errors()

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):
            BatchResult(attempted=-1, succeeded=0, failed=-1)
        with self.assertRaises(ValueError):
            BatchResult(attempted=2, succeeded=1)


class TestErrors(unittest.TestCase):
    def test_error_context(self):
        api = ApiError('feishu', 'search', code=1, message='bad', details={})
        self.assertIn('code=1', str(api))
        self.assertEqual(api.service, 'feishu')
        database = DatabaseOperationError('query', statement='SELECT 1')
        self.assertEqual(database.statement, 'SELECT 1')
        missing = ResourceNotFoundError('table', 't1')
        self.assertIn('t1', str(missing))
        self.assertTrue(issubclass(UnsupportedOperationError, Exception))


if __name__ == '__main__':
    unittest.main()
