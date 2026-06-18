from backend.controller import Controller
from ui import Editor
from util.utils import get_project_root


def main():
    model_path = get_project_root() / "models"
    controller = Controller()
    Editor(controller, model_path=model_path).launch()

if __name__ == "__main__":
    main()
