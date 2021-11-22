from typing import Iterable, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from psyke.schema.discrete_feature import DiscreteFeature
from psyke.schema.value import Value
from psyke.utils import get_default_random_seed
from sklearn.datasets import fetch_california_housing, load_iris
from tuprolog.core import rule, struct, logic_list, scope
from psyke.extractor import Extractor
from test.resources.predictors import PATH
from test.resources.schemas import SCHEMAS

REQUIRED_PREDICTORS: str = PATH / '.required.csv'

_DEFAULT_PRECISION: float = 1e-4

_test_option: dict = {'precision': _DEFAULT_PRECISION}


def get_precision() -> float:
    return _test_option['precision']


def set_default_precision(value: float):
    _test_option['precision'] = value


def get_extractor(extractor_type: str, parameters: dict):
    if extractor_type.lower() == 'iter':
        return Extractor.iter(**parameters)
    elif extractor_type.lower() == 'real':
        return Extractor.real(**parameters)
    elif extractor_type.lower() == 'trepan':
        return Extractor.trepan(**parameters)
    else:
        raise NotImplementedError(extractor_type + ' not implemented yet.')


def get_model(model_type: str, parameters: dict):
    if model_type.lower() == 'rfr':
        return RandomForestRegressor(**parameters, random_state=np.random.seed(get_default_random_seed()))
    elif model_type.lower() == 'knnc':
        return KNeighborsClassifier(**parameters)  # It's deterministic, don't not have a random_state
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
    if name.lower() == 'house':
        x, y = fetch_california_housing(return_X_y=True, as_frame=True)
        normalized_x = _normalize_data(x)
        normalized_y = _normalize_data(y)
        return normalized_x.join(normalized_y)
    elif name.lower() == 'iris':
        x, y = load_iris(return_X_y=True, as_frame=True)
        y = pd.DataFrame(y).replace({"target": {0: 'setosa', 1: 'virginica', 2: 'versicolor'}})
        result = x.join(y)
        result.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'iris']
        return result
    else:
        raise Exception('unknown dataset name.')


def _normalize_data(x: pd.DataFrame) -> pd.DataFrame:
    return (x - x.min()) / (x.max() - x.min())


def get_schema(filename: str) -> Union[Iterable[DiscreteFeature], None]:
    return SCHEMAS[filename] if filename in SCHEMAS.keys() else None

    # features: list[tuple[str, dict[str, Value]]] = []
    # with open(filename) as file:
    #     for row in file:
    #         prepositions = row.split(';')
    #         features.append((prepositions[0],_get_admissible_values(prepositions[1:])))
    # return [DiscreteFeature(feature, value) for feature, value in features]


def _get_admissible_values(prepositions: Iterable[str]) -> dict[str, Value]:
    raise NotImplementedError('Automatic schema reading not implemented yet.')
