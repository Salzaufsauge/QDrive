import queue


class TeeStream:
    def __init__(self, orig, log_queue: queue.Queue):
        self.orig = orig
        self.log_queue = log_queue

    def write(self, message):
        self.orig.write(message)
        self.orig.flush()

        stripped = message.rstrip()
        if stripped:
            self.log_queue.put(stripped)

    def flush(self):
        self.orig.flush()
