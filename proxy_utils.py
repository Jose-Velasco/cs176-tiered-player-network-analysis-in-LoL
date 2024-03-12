import requests
from queue import Queue
from threading import Thread
import time

def check_proxies(all_proxies: Queue, valid_proxies: Queue):
    while not all_proxies.empty():
        proxy = all_proxies.get()
        try:
            res = requests.get(
                "https://ipinfo.io/json",
                proxies= {
                    "http": proxy,
                    "https": proxy
                },
                timeout=5
            )
        except requests.Timeout:
            # Handle timeout error
            print(f"Timeout error occurred for proxy: {proxy}")
            continue
        except requests.RequestException as e:
            print(f"Request exception occurred for proxy: {proxy}. Exception: {e}")
            continue
        if res.status_code == 200:
            valid_proxies.put(proxy)

def get_valid_proxies(file_name: str, num_threads: int = 4) -> list[str]:
    all_proxies_q: Queue[str] = Queue()
    valid_proxies_q: Queue[str] = Queue()

    with open(file_name, "r") as f:
        proxies = f.read().split("\n")
        for proxy in proxies:
            all_proxies_q.put(proxy)

    threads: list[Thread] = []
    
    for _ in range(num_threads):
        thread = Thread(target=check_proxies, args=(all_proxies_q, valid_proxies_q))
        thread.start()
        threads.append(thread)

    # Wait for all validation threads to finish
    for thread in threads:
        thread.join()

    # Collect valid proxies from the queue
    valid_proxy_list = []
    while not valid_proxies_q.empty():
        valid_proxy_list.append(valid_proxies_q.get())

    return valid_proxy_list

class ProxyRoller:
    def __init__(self, session: requests.Session, proxies: list[str], headers, timeout: int = 5, retry_attempts: int = 4, retry_delay: int = 1):
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