from queue import Queue
from threading import Thread

from sampler import GPUSampler, IndexShuffleSampler


class PrefetchSampler:

    def __init__(
        self,
        sampler: IndexShuffleSampler | GPUSampler,
        max_prefetch: int = 2,
        workers: int = 1,
    ):
        self.sampler = sampler
        self.request_q = Queue()
        self.result_q = Queue(maxsize=max_prefetch)
        self.workers = workers
        for _ in range(workers):
            thread = Thread(target=self._worker, deamon=True)
            thread.start()

    def _worker(self) -> None:
        while True:
            batch_size = self.request_q.get()
            if batch_size is None:
                break
            batch = self.sampler.get_samples(batch_size)
            self.result_q.put(batch)

    def get_samples(self, batch_size: int):
        try:
            self.request_q.put(batch_size, block=False)
        except:
            pass
        return self.result_q.get()

    def shutdown(self) -> None:
        for _ in range(self.workers):
            self.request_q.put(None)
