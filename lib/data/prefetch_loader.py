import threading
import queue as Queue

import torch


class PrefetchGenerator(threading.Thread):

    def __init__(self, generator, max_prefetch=1):
        super(PrefetchGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class PrefetchDataLoader(torch.utils.data.DataLoader):

    def __iter__(self):
        return PrefetchGenerator(
            super(PrefetchDataLoader, self).__iter__()
        )
