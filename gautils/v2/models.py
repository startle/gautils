from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BatchError:
    index: int | str | None
    message: str
    code: int | str | None = None
    details: Any = None


@dataclass(frozen=True, slots=True)
class BatchResult:
    attempted: int
    succeeded: int
    failed: int = 0
    created: int = 0
    updated: int = 0
    deleted: int = 0
    errors: tuple[BatchError, ...] = field(default_factory=tuple)
    skipped: int = 0

    def __post_init__(self):
        counts = (
            self.attempted, self.succeeded, self.failed,
            self.created, self.updated, self.deleted, self.skipped,
        )
        if any(value < 0 for value in counts):
            raise ValueError('batch counts cannot be negative')
        if self.succeeded + self.failed + self.skipped != self.attempted:
            raise ValueError('succeeded + failed + skipped must equal attempted')

    @property
    def ok(self) -> bool:
        return self.failed == 0

    @property
    def success_rate(self) -> float:
        processed = self.attempted - self.skipped
        return 1.0 if processed == 0 else self.succeeded / processed

    def raise_for_errors(self) -> None:
        if self.failed:
            raise RuntimeError(
                f'batch operation failed for {self.failed}/{self.attempted} items')
