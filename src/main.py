import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from backend.config.storage import load_config
from backend.controller import Controller
from backend.evaluate import Evaluate
from backend.train import Train
from frontend import Editor
from util.logging_broker import LoggingBroker
from util.teestream import StreamType, TeeStream
from util.utils import get_config_path, get_project_root


def interrupt_handler(controller):
    def handler(signum, frame):
        try:
            asyncio.create_task(controller.stop_all())
        except RuntimeError:
            print("No running loop")
            asyncio.run(controller.stop_all())

    return handler


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    train = sub.add_parser("train", help="Train or resume an experiment")
    train.add_argument("config", type=Path)
    train.add_argument("--offline", action="store_true", default=False)
    evaluate = sub.add_parser("evaluate", help="Evaluate a trained model")
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("--render-mode", default="rgb_array")
    gui = sub.add_parser("gui", help="Run the GUI (default)")
    gui.add_argument("--ip", type=str, default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8080)
    gui.add_argument("--offline", action="store_true", default=False)
    p.set_defaults(command="gui", offline=False, ip="127.0.0.1", port=8080)
    return p


def run_cli(args):
    config = load_config(Path(args.config).resolve())
    config.config_path = get_config_path(config.model_path)
    runner = Train() if args.command == "train" else Evaluate()
    signal.signal(signal.SIGINT, lambda *_: runner.stop())
    if isinstance(runner, Train):
        try:
            runner.train(config)
        except Exception:  # noqa: BLE001
            sys.exit(1)
    else:
        runner.evaluate(config, args.render_mode)


def main():
    args = build_parser().parse_args()
    load_dotenv()
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    if args.command in {"train", "evaluate"}:
        run_cli(args)
    elif args.command == "gui":
        logging_broker = LoggingBroker()

        sys.stdout = TeeStream(sys.stdout, StreamType.STDOUT, logging_broker)
        sys.stderr = TeeStream(sys.stderr, StreamType.STDERR, logging_broker)

        config_path = get_project_root() / "experiments"
        controller = Controller()
        signal.signal(signal.SIGINT, interrupt_handler(controller))
        editor = Editor(
            controller, logging_broker=logging_broker, config_path=config_path
        )
        editor.launch(args.ip, args.port)


if __name__ == "__main__":
    main()
