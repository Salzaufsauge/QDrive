import datetime
import threading
from collections import deque


def log(level: str, message: str):
    print(f"{datetime.datetime.now().strftime("%H:%M:%S")} - {level} - {message}")


class TrainState:
    def __init__(self):
        self.episodes = deque()
        self.lock = threading.Lock()

    def reward_fig(self):
        with self.lock:
            data = list(self.episodes)
            self.episodes.clear()

        return data
