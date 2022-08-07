from enum import Enum
from math import log10
from random import Random

_DEFAULT_RANDOM_SEED: int = 123

ONNX_EXTENSION: str = '.onnx'

_random_options = dict(_deterministic_mode=True, _default_random_seed=_DEFAULT_RANDOM_SEED)

_random_seed_generator: Random = Random(_DEFAULT_RANDOM_SEED)

_DEFAULT_PRECISION: float = 1e-6

_precision_options: dict = {'precision': _DEFAULT_PRECISION}


class TypeNotAllowedException(Exception):

    def __init__(self, type_name: str):
        super().__init__('Type "' + type_name + '" not allowed for discretization.')


class Range:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        self.lower = mean
        self.upper = mean

    def left_infinite(self):
        self.lower = float('-inf')

    def right_infinite(self):
        self.upper = float('inf')

    def expand_left(self):
        self.lower -= self.std

    def expand_right(self):
        self.upper += self.std


def is_deterministic_mode():
    return _random_options['_deterministic_mode']


def set_deterministic_mode(value: bool):
    _random_options['_deterministic_mode'] = value


def get_default_random_seed():
    if is_deterministic_mode():
        return _random_options['_default_random_seed']
    else:
        return _random_seed_generator.randint(0, 1 << 64)


def set_default_random_seed(value: int):
    _random_options['_default_random_seed'] = value


def get_default_precision() -> float:
    return _precision_options['precision']


def get_int_precision() -> int:
    return -1 * int(log10(get_default_precision()))


def set_default_precision(value: float):
    _precision_options['precision'] = value


class Target(Enum):
    CLASSIFICATION = 1,
    CONSTANT = 2,
    REGRESSION = 3
