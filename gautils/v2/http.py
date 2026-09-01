from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from .errors import HttpRequestError


@dataclass(frozen=True, slots=True)
class Timeout:
    connect: float = 5.0
    read: float = 30.0

    def as_requests(self) -> tuple[float, float]:
        if self.connect <= 0 or self.read <= 0:
            raise ValueError('timeout values must be positive')
        return self.connect, self.read


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    total: int = 3
    backoff_factor: float = 0.5
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504)
    allowed_methods: frozenset[str] = frozenset(
        {'DELETE', 'GET', 'HEAD', 'OPTIONS', 'PUT', 'TRACE'})

    def build(self) -> Retry:
        if self.total < 0:
            raise ValueError('retry total cannot be negative')
        return Retry(
            total=self.total,
            connect=self.total,
            read=self.total,
            status=self.total,
            backoff_factor=self.backoff_factor,
            status_forcelist=self.status_forcelist,
            allowed_methods=self.allowed_methods,
            respect_retry_after_header=True,
            raise_on_status=False,
        )


class HttpClient:
    """Reusable synchronous HTTP client with explicit response semantics."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout: Timeout = Timeout(),
        retry: RetryPolicy | None = RetryPolicy(),
        headers: Mapping[str, str] | None = None,
        verify: bool = True,
        proxies_provider: Callable[[], Mapping[str, str] | None] | None = None,
        raise_for_status: bool = True,
    ):
        self._owns_session = session is None
        self._session = session or requests.Session()
        self._timeout = timeout
        self._verify = verify
        self._proxies_provider = proxies_provider
        self._raise_for_status = raise_for_status
        self._closed = False
        if headers:
            self._session.headers.update(headers)
        if retry is not None:
            adapter = HTTPAdapter(max_retries=retry.build())
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)

    @property
    def session(self) -> requests.Session:
        return self._session

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        if self._closed:
            raise RuntimeError('HttpClient is closed')
        kwargs.setdefault('timeout', self._timeout.as_requests())
        kwargs.setdefault('verify', self._verify)
        if 'proxies' not in kwargs and self._proxies_provider is not None:
            kwargs['proxies'] = self._proxies_provider()
        try:
            response = self._session.request(method.upper(), url, **kwargs)
            if self._raise_for_status:
                response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise HttpRequestError(
                method, url, response=getattr(exc, 'response', None)) from exc

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.request('GET', url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self.request('POST', url, **kwargs)

    def put(self, url: str, **kwargs) -> requests.Response:
        return self.request('PUT', url, **kwargs)

    def patch(self, url: str, **kwargs) -> requests.Response:
        return self.request('PATCH', url, **kwargs)

    def delete(self, url: str, **kwargs) -> requests.Response:
        return self.request('DELETE', url, **kwargs)

    def get_text(self, url: str, **kwargs) -> str:
        return self.get(url, **kwargs).text

    def get_json(self, url: str, **kwargs) -> Any:
        return self.get(url, **kwargs).json()

    def close(self) -> None:
        if not self._closed and self._owns_session:
            self._session.close()
        self._closed = True

    def __enter__(self) -> 'HttpClient':
        if self._closed:
            raise RuntimeError('HttpClient is closed')
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
