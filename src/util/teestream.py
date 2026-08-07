from enum import Enum

from util.LoggingBroker import LoggingBroker


class StreamType(Enum):
    STDOUT = 1
    STDERR = 2


class LogMessage:
    def __init__(self, stream_type: StreamType, message: str):
        self.stream_type = stream_type
        self.message = message


class TeeStream:
    def __init__(self, orig, stream_type: StreamType, logging_dist: LoggingBroker):
        self.orig = orig
        self.stream_type = stream_type
        self.logging_dist = logging_dist

    def write(self, message):
        self.orig.write(message)
        self.orig.flush()

        stripped = message.rstrip()
        if stripped:
            self.logging_dist.publish(LogMessage(self.stream_type, stripped))

        return len(message)

    def flush(self):
        self.orig.flush()

    def isatty(self):
        return self.orig.isatty()
