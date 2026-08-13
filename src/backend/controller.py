import asyncio
import sys
import threading

from backend.config.config import ExperimentConfig
from backend.evaluate import Evaluate
from backend.train import Train


class Controller:
    def __init__(self):
        self.training = Train()
        self.eval = Evaluate()
        self.thread = None

    def start_training(self, config: ExperimentConfig):
        self._start_thread(self.training.train, (config,))

    def is_alive_and_training(self):
        return (
            self.thread is not None
            and self.thread.is_alive()
            and self.training.running.is_set()
        )

    def is_alive_and_eval(self):
        return (
            self.thread is not None
            and self.thread.is_alive()
            and self.eval.running.is_set()
        )

    async def stop_training(self):
        await self._stop_thread(self.training)

    async def stop_all(self):
        try:
            await self._stop_thread(self.training)
            await self._stop_thread(self.eval)
        except Exception as e:
            print(e, file=sys.stderr)

    def start_eval(self, config: ExperimentConfig, mode: str = "rgb_array"):
        self._start_thread(self.eval.evaluate, (config, mode))

    async def stop_eval(self):
        await self._stop_thread(self.eval)

    def get_training_state(self):
        return self.training.get_state()

    def get_current_frame(self):
        return self.eval.get_current_frame()

    def _start_thread(self, target, args):  # for now allow only one of the modes to run
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("A thread is already running")
        self.thread = threading.Thread(target=target, args=args, daemon=True)
        self.thread.start()

    async def _stop_thread(self, target):
        thread = self.thread

        if thread is None:
            return

        if thread.is_alive():
            if target.running.is_set():
                target.stop()
            await asyncio.to_thread(thread.join)

            if thread == self.thread:
                self.thread = None
