import queue
from enum import Enum


class StreamType(Enum):
    STDOUT = 1
    STDERR = 2


class LogMessage:
    def __init__(self, stream_type: StreamType, message: str):
        self.stream_type = stream_type
        self.message = message

class TeeStream:
    def __init__(self, orig, stream_type: StreamType, log_queue: queue.Queue):
        self.orig = orig
        self.stream_type = stream_type
        self.log_queue = log_queue

    def write(self, message):
        self.orig.write(message)
        self.orig.flush()

        stripped = message.rstrip()
        if stripped:
            self.log_queue.put(LogMessage(self.stream_type, stripped))

        return len(message)

    def flush(self):
        self.orig.flush()

    def isatty(self):
        return self.orig.isatty()
