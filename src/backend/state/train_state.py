import threading
from collections import deque


class TrainState:
    """
    TrainState is mainly used to store training episode data.

    It retains a deque of episodes with a maximum length of 500. Each episode is a dictionary containing training data.
    """

    def __init__(self):
        self.episodes = deque(maxlen=500)
        self.sequence = 0
        self.lock = threading.Lock()

    def append_episode(self, episode):
        with self.lock:
            self.sequence += 1
            self.episodes.append(episode | {"sequence": self.sequence})

    def reward_fig(self, sequence_after: int = 0):
        with self.lock:
            return [
                episode.copy()
                for episode in self.episodes
                if episode["sequence"] > sequence_after
            ]
