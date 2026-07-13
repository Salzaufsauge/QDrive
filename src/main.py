import argparse
import queue
import sys
from pathlib import Path

from backend.config.storage import load_config
from backend.controller import Controller
from frontend import Editor
from util.teestream import TeeStream
from util.utils import get_project_root


def main(args):
    if args.config_path is not None:
        config_path = Path(args.config_path)
        configuration = load_config(config_path)
        if args.train:
            Controller().start_training(configuration)
        elif args.eval:
            for _ in Controller().start_eval(configuration, mode=args.mode):
                pass
    else:
        log_queue = queue.Queue()

        config_path = get_project_root() / "experiments"
        controller = Controller()
        Editor(controller, log_queue=log_queue, config_path=config_path).launch()

        sys.stdout = TeeStream(sys.stdout, log_queue)
        sys.stderr = TeeStream(sys.stderr, log_queue)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train", action="store_true", help="Train a new model")
    arg_parser.add_argument("--eval", action="store_true", help="Evaluate a model")
    arg_parser.add_argument("--config_path", type=str, default=None, help="Relative path to the config")
    arg_parser.add_argument("--mode", type=str, default="rgb_array", help="Observation mode, default: rgb_array")
    main(arg_parser.parse_args())
