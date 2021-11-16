from pathlib import Path
from typing import TextIO

PATH = Path(__file__).parents[0]


def get_predictor_path(filename: str) -> Path:
    return PATH / f"{filename}.onnx"


def open_predictor(filename: str) -> TextIO:
    return open(get_predictor_path(filename))
