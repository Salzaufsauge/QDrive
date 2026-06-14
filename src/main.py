from pathlib import Path

from ui import Editor

def main():
    model_path = Path("../models")
    Editor(model_path=model_path).launch()

if __name__ == "__main__":
    main()