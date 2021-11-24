import ast

import pandas as pd
from sklearn.model_selection import train_test_split
from tuprolog.core import var, struct, numeric
from tuprolog.theory import Theory
from tuprolog.theory.parsing import parse_theory

from psyke.cart import CartPredictor
from psyke.predictor import Predictor
from psyke.utils import get_default_random_seed
from psyke.utils.dataframe_utils import get_discrete_dataset
from test import get_dataset, get_extractor, get_schema, get_model
from test.resources.predictors import get_predictor_path
from test.resources.tests import test_cases


def initialize(file: str) -> list[dict[str:Theory]]:
    for row in test_cases(file):
        params = dict() if row['extractor_params'] == '' else ast.literal_eval(row['extractor_params'])
        dataset = get_dataset(row['dataset'])

        # Dataset's columns are sorted.
        columns = sorted(dataset.columns[:-1]) + [dataset.columns[-1]]
        dataset = dataset.reindex(columns, axis=1)

        training_set, test_set = train_test_split(dataset, test_size=0.5, random_state=get_default_random_seed())

        if 'schema' in row.keys():
            schema = get_schema(row['schema'])
            params['discretization'] = schema
            training_set = get_discrete_dataset(training_set.iloc[:, :-1], schema)\
                .join(training_set.iloc[:, -1].reset_index(drop=True))

        # Handle cart's cases
        if row['predictor'].lower() not in ['dtc', 'dtr']:
            params['predictor'] = Predictor.load_from_onnx(str(get_predictor_path(row['predictor'])))
        else:
            tree = get_model(row['predictor'], {})
            tree.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
            params['predictor'] = CartPredictor(tree)

        extractor = get_extractor(row['extractor_type'], params)
        theory = extractor.extract(training_set)

        yield {
            'extractor': extractor,
            'extracted_theory': theory,
            'test_set': test_set,
            'expected_theory': parse_theory(row['theory'] + '.') if row['theory'] != '' else theory
        }


def data_to_struct(data: pd.Series):
    head = data.keys()[-1]
    terms = [numeric(item) for item in data.values[:-1]]
    terms.append(var('X'))
    return struct(head, terms)
