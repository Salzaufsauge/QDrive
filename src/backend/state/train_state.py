import datetime
import sys
import threading
from collections import deque


def log(level: str, message: str):
    if level == "INFO":
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {level} - {message}")
    if level == "ERROR":
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {level} - {message}", file=sys.stderr)


class TrainState:
    def __init__(self):
        self.episodes = deque(maxlen=10_000)
        self.lock = threading.Lock()

    def reward_fig(self):
        with self.lock:
            data = list(self.episodes)
            self.episodes.clear()

        return data
