import requests
import time
from typing import Protocol

class HttpClient(Protocol):
    def get(self) -> requests.Response:
        ...

class RequestsClient():
    def __init__(self, session: requests.Session, headers: dict[str, str]):
        self.session = session
        self.headers = headers
    
    def get(self, url: str) -> requests.Response:
        return self.session.get(url, headers=self.headers)

class ProxyRoller:
    def __init__(self, session: requests.Session, proxies: list[str], headers: dict[str, str], timeout: int = 5, retry_attempts: int = 4, retry_delay: int = 1):
        self.session = session
        self.proxies = proxies
        self.timeout = timeout
        self.headers = headers
        self.proxy_idx = 0
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    def get(self, url: str):
        for _ in range(self.retry_attempts):
            try:
                proxy = self.proxies[self.proxy_idx]
                res = self.session.get(url, headers=self.headers, proxies={"http": proxy, "https": proxy}, timeout=self.timeout)
                res.raise_for_status()
                return res
            except requests.HTTPError as e:
                print(f"Request failed retrying: {e}")
                # Add delay before retry
                time.sleep(self.retry_delay)
            finally:
                self.proxy_idx = (self.proxy_idx + 1) % len(self.proxies)

        # All retry attempts failed
        return None