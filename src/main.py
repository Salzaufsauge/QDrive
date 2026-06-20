import argparse
from pathlib import Path

from backend.configuration import Configuration
from backend.controller import Controller
from ui import Editor
from util.utils import get_project_root


def main(args):
    if args.model_path is not None:
        model_path = Path(args.model_path)
        configuration = Configuration()
        configuration.load_model(model_path)
        if args.train:
            Controller().start_training(configuration)
        elif args.eval:
            for _ in Controller().start_eval(configuration, mode=args.mode):
                pass
    else:
        model_path = get_project_root() / "models"
        controller = Controller()
        Editor(controller, model_path=model_path).launch()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train", action="store_true", help="Train a new model")
    arg_parser.add_argument("--eval", action="store_true", help="Evaluate a model")
    arg_parser.add_argument("--model_path", type=str, default=None, help="Relative path to the model")
    arg_parser.add_argument("--mode", type=str, default="rgb_array", help="Observation mode, default: rgb_array")
    main(arg_parser.parse_args())
