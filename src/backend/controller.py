from backend.configuration import Configuration
from backend.evaluate import Evaluate
from backend.train import Train


class Controller:
    def __init__(self):
        self.training = Train()
        self.eval = Evaluate()

    def start_training(self, config: Configuration):
        return self.training.train(config)

    def stop_training(self):
        self.training.stop()

    def start_eval(self, config: Configuration, mode: str = "rgb_array"):
        yield from self.eval.evaluate(config, mode=mode)

    def stop_eval(self):
        self.eval.stop()

    def get_training_state(self):
        yield from self.training.get_state()
