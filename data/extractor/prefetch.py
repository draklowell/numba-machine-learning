import threading
from queue import Queue


class PrefetchSampler:
    """
    A single background worker
    """

    def __init__(self, sampler, max_prefetch: int = 2):
        self.sampler = sampler
        self.request_q = Queue()
        self.result_q = Queue(maxsize=max_prefetch)
        self.stop_event = threading.Event()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            batch_size = self.request_q.get()
            if batch_size is None:
                break
            batch = self.sampler.get_samples(batch_size)
            self.result_q.put(batch)
        self.stop_event.set()

    def get_samples(self, batch_size: int):
        self.request_q.put(batch_size)
        return self.result_q.get()

    def shutdown(self):
        self.request_q.put(None)
        self.thread.join()
        self.stop_event.wait()
