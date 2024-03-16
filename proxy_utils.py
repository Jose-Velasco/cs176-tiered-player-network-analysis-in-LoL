import requests
from queue import Queue
from threading import Thread

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