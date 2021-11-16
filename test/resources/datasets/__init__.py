from pathlib import Path
from typing import TextIO

PATH = Path(__file__).parents[0]


def get_dataset_path(filename: str) -> Path:
    return PATH / f"{filename}.csv"


def open_dataset(filename: str) -> TextIO:
    return open(get_dataset_path(filename))

