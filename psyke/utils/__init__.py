from math import log10
from random import Random

_DEFAULT_RANDOM_SEED: int = 123

ONNX_EXTENSION: str = '.onnx'

_random_options = dict(_deterministic_mode=True, _default_random_seed=_DEFAULT_RANDOM_SEED)

_random_seed_generator: Random = Random(_DEFAULT_RANDOM_SEED)

_DEFAULT_PRECISION: float = 1e-4

_precision_options: dict = {'precision': _DEFAULT_PRECISION}


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
