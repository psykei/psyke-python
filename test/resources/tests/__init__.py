from pathlib import Path
from typing import Iterable, Dict, TextIO
import csv

PATH = Path(__file__).parents[0]

# TODO: add methods for file management,
#  add method for saving file onnx from a binary string


def get_test_path(filename: str) -> Path:
    return PATH / f"{filename}.csv"


def open_test(filename: str) -> TextIO:
    return open(get_test_path(filename))


def test_cases(filename: str) -> Iterable[Dict]:
    with open_test(filename) as file:
        return [row for row in csv.DictReader(file, delimiter=';', quotechar='"')]
