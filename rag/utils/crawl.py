import threading, time

class SharedRateLimiter:
    def __init__(self, interval_s: float):
        self.interval_s=max(0.0, float(interval_s))
        self.lock=threading.Lock()
        self.next_allowed=0.0

    def wait(self) -> None:
        if self.interval_s<=0:
            return
        with self.lock:
            now=time.monotonic()
            delay=self.next_allowed-now
            if delay>0:
                time.sleep(delay)
            self.next_allowed=time.monotonic()+self.interval_s

def limited_get(session, url: str, *, semaphore=None, rate_limiter: SharedRateLimiter | None=None, **kwargs):
    if rate_limiter is not None:
        rate_limiter.wait()
    if semaphore is None:
        return session.get(url, **kwargs)
    with semaphore:
        return session.get(url, **kwargs)