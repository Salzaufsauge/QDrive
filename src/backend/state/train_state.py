import threading
from collections import deque


class TrainState:
    def __init__(self):
        self.episodes = deque(maxlen=500)
        self.lock = threading.Lock()

    def reward_fig(self):
        with self.lock:
            data = list(self.episodes)
            self.episodes.clear()

        return data
