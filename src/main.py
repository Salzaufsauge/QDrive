import argparse
import asyncio
import signal
import sys
from pathlib import Path

from backend.config.storage import load_config
from backend.controller import Controller
from frontend import Editor
from util.LoggingBroker import LoggingBroker
from util.teestream import StreamType, TeeStream
from util.utils import get_project_root


def interrupt_handler(controller):
    def handler(signum, frame):
        try:
            asyncio.create_task(controller.stop_all())
        except RuntimeError:
            print("No running loop")
            asyncio.run(controller.stop_all())

    return handler


def main(args):
    if args.config_path is not None:
        config_path = Path(args.config_path)
        configuration = load_config(config_path)
        if args.train:
            Controller().start_training(configuration)
        elif args.eval:
            Controller().start_eval(configuration, mode=args.mode)
    else:
        logging_broker = LoggingBroker()

        sys.stdout = TeeStream(sys.stdout, StreamType.STDOUT, logging_broker)
        sys.stderr = TeeStream(sys.stderr, StreamType.STDERR, logging_broker)

        config_path = get_project_root() / "experiments"
        controller = Controller()
        signal.signal(signal.SIGINT, interrupt_handler(controller))
        editor = Editor(
            controller, logging_broker=logging_broker, config_path=config_path
        )
        editor.launch()


if __name__ in {"__main__", "__mp_main__"}:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train", action="store_true", help="Train a new model")
    arg_parser.add_argument("--eval", action="store_true", help="Evaluate a model")
    arg_parser.add_argument(
        "--config_path", type=str, default=None, help="Relative path to the config"
    )
    arg_parser.add_argument(
        "--mode",
        type=str,
        default="rgb_array",
        help="Observation mode, default: rgb_array",
    )
    main(arg_parser.parse_args())
