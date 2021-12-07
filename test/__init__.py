from typing import Iterable, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from psyke.schema import DiscreteFeature, Value
from psyke.utils import get_default_random_seed
from sklearn.datasets import fetch_california_housing, load_iris
from tuprolog.core import rule, struct, logic_list, scope
from psyke import Extractor
from psyke.utils.dataframe import get_discrete_features_equal_frequency
from test.resources.predictors import PATH

REQUIRED_PREDICTORS: str = PATH / '.required.csv'


def get_extractor(extractor_type: str, parameters: dict):
    if extractor_type.lower() == 'cart':
        return Extractor.cart(**parameters)
    elif extractor_type.lower() == 'iter':
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
    elif model_type.lower() == 'dtc':
        return DecisionTreeClassifier(random_state=np.random.seed(get_default_random_seed()))
    elif model_type.lower() == 'dtr':
        return DecisionTreeRegressor(max_depth=3, random_state=np.random.seed(get_default_random_seed()))
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


def get_not_in_rule():
    local_scope = scope()
    return rule(
        struct('not_in', local_scope.var('X'), logic_list(local_scope.var('H'), local_scope.var('T'))),
        [
            struct('<', local_scope.var('T'), local_scope.var('X')),
            struct('<', local_scope.var('X'), local_scope.var('H'))
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


def get_schema(dataset: pd.DataFrame, bins: int) -> Union[Iterable[DiscreteFeature], None]:
    return get_discrete_features_equal_frequency(dataset, bins)
    # return SCHEMAS[filename] if filename in SCHEMAS.keys() else None


def _get_admissible_values(prepositions: Iterable[str]) -> dict[str, Value]:
    raise NotImplementedError('Automatic schema reading not implemented yet.')
