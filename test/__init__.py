import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from psyke.utils import get_default_random_seed
from test.resources import CLASSPATH
from sklearn.datasets import fetch_california_housing
from tuprolog.core import rule, struct, logic_list, scope
from psyke.extractor import Extractor


REQUIRED_PREDICTORS: str = CLASSPATH + os.path.sep + 'required_predictors.csv'

_DEFAULT_PRECISION: float = 1e-4

_test_option: dict = {'precision': _DEFAULT_PRECISION}


def get_precision() -> float:
    return _test_option['precision']


def set_default_precision(value: float):
    _test_option['precision'] = value


def get_extractor(extractor_type: str, parameters: dict):
    if extractor_type == 'ITER':
        return Extractor.iter(**parameters)
    else:
        raise NotImplementedError(extractor_type + ' not implemented yet.')


def get_model(model_type: str, parameters: dict):
    if model_type == 'RFR':
        return RandomForestRegressor(**parameters, random_state=np.random.seed(get_default_random_seed()))
    else:
        raise NotImplementedError(model_type + ' not handled yet.')


def get_in_rule():
    local_scope = scope()
    return rule(
        struct('in', local_scope.var('X'), logic_list(local_scope.var('H'), local_scope.var('T'))),
        [
            struct('=<', local_scope.var('X'), local_scope.var('T')),
            struct('=<', local_scope.var('H'), local_scope.var('X'))
        ]
    )


def get_dataset(name: str):
    if name == 'house':
        x, y = fetch_california_housing(return_X_y=True, as_frame=True)
        normalized_x = (x - x.min()) / (x.max() - x.min())
        normalized_y = (y - y.min()) / (y.max() - y.min())
        return normalized_x.join(normalized_y)
    else:
        raise Exception('unknown dataset name.')
