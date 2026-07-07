import datetime
import threading
from collections import deque

import polars as pl
import seaborn as sb
from matplotlib import pyplot as plt


class TrainState:
    def __init__(self):
        self.logs = deque(maxlen=1000)
        self.episodes = deque(maxlen=10_000)
        self.lock = threading.Lock()

    def log(self, level: str, message: str):
        with self.lock:
            self.logs.append({
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            })

    def reward_fig(self):
        fig, ax = plt.subplots()
        with self.lock:
            if self.episodes:
                df = pl.DataFrame(self.episodes)
                sb.lineplot(
                    data=df,
                    x="timesteps",
                    y="reward",
                    ax=ax,
                )
        plt.close(fig)
        return fig

    def get_logs(self):
        with self.lock:
            return [f"{log['time']} - {log['level']} - {log['message']} " for log in self.logs]
