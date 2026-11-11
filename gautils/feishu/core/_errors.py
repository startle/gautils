class _FeishuApiError(RuntimeError):
    """Internal failure retained until the legacy or v2 API boundary."""

    def __init__(self, operation, response=None, *, message=None, details=None):
        self.operation = operation
        self.code = response.code if response is not None else None
        self.message = message if message is not None else response.msg
        self.details = dict(details or {})
        if response is not None:
            self.details['log_id'] = response.get_log_id()
        super().__init__(f'{operation} failed: code={self.code}, message={self.message}')
