from typing import Iterable, Union, Callable
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tuprolog.core import Clause
from tuprolog.theory.parsing import DEFAULT_CLAUSES_PARSER
from psyke.schema import DiscreteFeature, Value
from psyke.utils import get_default_random_seed
from sklearn.datasets import fetch_california_housing, load_iris
from psyke import Extractor
from psyke.utils.dataframe import get_discrete_features_supervised
from test.resources.predictors import PATH

REQUIRED_PREDICTORS: str = PATH / '.required.csv'
LE = '=<'
GE = '>='
L = '<'
G = '>'


def get_extractor(extractor_type: str, parameters: dict):
    if extractor_type.lower() == 'cart':
        return Extractor.cart(**parameters)
    elif extractor_type.lower() == 'iter':
        return Extractor.iter(**parameters)
    elif extractor_type.lower() == 'real':
        return Extractor.real(**parameters)
    elif extractor_type.lower() == 'trepan':
        return Extractor.trepan(**parameters)
    elif extractor_type.lower() == 'gridex':
        return Extractor.gridex(**parameters)
    else:
        raise NotImplementedError(extractor_type + ' not implemented yet.')


def get_model(model_type: str, parameters: dict):
    if model_type.lower() == 'rfr':
        return RandomForestRegressor(**parameters, random_state=np.random.seed(get_default_random_seed()))
    elif model_type.lower() == 'knnc':
        return KNeighborsClassifier(**parameters)  # It's deterministic, don't have a random_state
    elif model_type.lower() == 'dtc':
        return DecisionTreeClassifier(max_leaf_nodes=3, random_state=np.random.seed(get_default_random_seed()))
    elif model_type.lower() == 'dtr':
        return DecisionTreeRegressor(max_depth=3, random_state=np.random.seed(get_default_random_seed()))
    else:
        raise NotImplementedError(model_type + ' not handled yet.')


def get_in_rule(min_included: bool = True, max_included: bool = False) -> Clause:
    """
    Create the logic 'in' predicate in(X, [Min, Max]).
    The predicate is true if X is in between Min and Max.
    :param min_included: if X == Min then true
    :param max_included: if X == Max then true
    :return: the tuProlog clause for the 'in' predicate
    """
    in_textual_rule: Callable = lambda x, y: "in(X, [Min, Max]) :- !, X " + x + " Min, X " + y + " Max."
    parser = DEFAULT_CLAUSES_PARSER
    theory = parser.parse_clauses(in_textual_rule(GE if min_included else G, LE if max_included else L), operators=None)
    return theory[0]


def get_not_in_rule(min_included: bool = True, max_included: bool = False) -> Clause:
    """
    Create the logic 'not_in' predicate not_in(X, [Min, Max]).
    The predicate is true if X is outside the range between Min and Max.
    :param min_included: if X == Min then true
    :param max_included: if X == Max then true
    :return: the tuProlog clause for the 'not_in' predicate
    """
    not_in_textual_rule: Callable = lambda x, y: "not_in(X, [Min, Max]) :- X " + x + " Min; X " + y + " Max."
    parser = DEFAULT_CLAUSES_PARSER
    theory = parser.parse_clauses(not_in_textual_rule(LE if min_included else L, GE if max_included else G),
                                  operators=None)
    return theory[0]


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


def get_schema(dataset: pd.DataFrame) -> Union[Iterable[DiscreteFeature], None]:
    return get_discrete_features_supervised(dataset)
    # return SCHEMAS[filename] if filename in SCHEMAS.keys() else None


def _get_admissible_values(prepositions: Iterable[str]) -> dict[str, Value]:
    raise NotImplementedError('Automatic schema reading not implemented yet.')
