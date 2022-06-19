from __future__ import annotations
from onnxconverter_common import DataType, FloatTensorType, Int64TensorType, StringTensorType
from skl2onnx import convert_sklearn
from sklearn.model_selection import train_test_split
from psyke import get_default_random_seed
from psyke.cart import CartPredictor
from psyke.regression import Grid, FixedStrategy, FeatureRanker
from psyke.regression.strategy import AdaptiveStrategy
from psyke.utils.dataframe import get_discrete_dataset
from test import get_dataset, get_extractor, get_schema, get_model
from test.resources.predictors import get_predictor_path
from test.resources.tests import test_cases
from tuprolog.core import var, struct, numeric, Real
from tuprolog.theory import Theory
from tuprolog.theory.parsing import parse_theory
from typing import Iterable
import ast
import numpy as np
import onnxruntime as rt
import os
import pandas as pd

_DEFAULT_ACCURACY: float = 0.95

_test_options: dict = {'accuracy': _DEFAULT_ACCURACY}


def get_default_accuracy() -> float:
    return _test_options['accuracy']


def set_default_accuracy(value: float) -> None:
    _test_options['accuracy'] = value


def are_similar(a: Real, b: Real) -> bool:
    return abs(a.value - b.value) < 0.01


def are_equal(instance, expected, actual):
    if expected.is_functor_well_formed:
        instance.assertTrue(actual.is_functor_well_formed)
        instance.assertEqual(expected.functor, actual.functor)
        instance.assertTrue(expected.args[0].equals(actual.args[0], False))
        instance.assertTrue(are_similar(expected.args[1][0], actual.args[1][0]))
        instance.assertTrue(are_similar(expected.args[1][1].head, actual.args[1][1].head))
    elif expected.is_recursive:
        instance.assertTrue(actual.is_recursive)
        instance.assertEqual(expected.arity, actual.arity)
        for i in range(expected.arity):
            are_equal(instance, expected.args[i], actual.args[i])


def initialize(file: str) -> list[dict[str:Theory]]:
    for row in test_cases(file):
        params = dict() if row['extractor_params'] == '' else ast.literal_eval(row['extractor_params'])
        dataset = get_dataset(row['dataset'])

        # Dataset's columns are sorted due to alphabetically sorted extracted rules.
        columns = sorted(dataset.columns[:-1]) + [dataset.columns[-1]]
        dataset = dataset.reindex(columns, axis=1)

        training_set, test_set = train_test_split(dataset, test_size=0.5, random_state=get_default_random_seed())

        schema = None
        if 'disc' in row.keys() and bool(row['disc']):
            schema = get_schema(training_set)
            params['discretization'] = schema
            training_set = get_discrete_dataset(training_set.iloc[:, :-1], schema)\
                .join(training_set.iloc[:, -1].reset_index(drop=True))

        # Handle cart's tests.
        # Cart needs to inspect the tree of the predictor.
        # Unfortunately onnx does not provide a method to do that.
        if row['predictor'].lower() not in ['dtc', 'dtr']:
            params['predictor'] = Predictor.load_from_onnx(str(get_predictor_path(row['predictor'])))
        else:
            tree = get_model(row['predictor'], {})
            tree.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
            params['predictor'] = CartPredictor(tree)

        # Handle GridEx tests
        if 'grid' in row.keys() and bool:
            strategy, n = eval(row['strategies'])
            if strategy == "F":
                params['grid'] = Grid(int(row['grid']), FixedStrategy(n))
            else:
                ranked = FeatureRanker(training_set.columns[:-1])\
                    .fit(params['predictor'], training_set.iloc[:, :-1]).rankings()
                params['grid'] = Grid(int(row['grid']), AdaptiveStrategy(ranked, n))

        extractor = get_extractor(row['extractor_type'], params)
        theory = extractor.extract(training_set)

        yield {
            'extractor': extractor,
            'extracted_theory': theory,
            'test_set': test_set,
            'expected_theory': parse_theory(row['theory'] + '.') if row['theory'] != '' else theory,
            'discretization': schema
        }


def data_to_struct(data: pd.Series):
    head = data.keys()[-1]
    terms = [numeric(item) for item in data.values[:-1]]
    terms.append(var('X'))
    return struct(head, terms)


class Predictor:

    def __init__(self, model, from_file_onnx=False):
        self._model = model
        self._from_file_onnx = from_file_onnx

    @staticmethod
    def load_from_onnx(file: str) -> Predictor:
        return Predictor(rt.InferenceSession(file), True)

    def save_to_onnx(self, file, initial_types: list[tuple[str, DataType]]):
        file = str(file) + '.onnx'
        if not self._from_file_onnx:
            if os.path.exists(file):
                os.remove(file)
            onnx_predictor = convert_sklearn(self._model, initial_types=initial_types)
            with open(file, 'wb') as f:
                f.write(onnx_predictor.SerializeToString())

    def predict(self, dataset: pd.DataFrame | np.ndarray) -> Iterable:
        array = dataset.to_numpy() if isinstance(dataset, pd.DataFrame) else dataset
        if self._from_file_onnx:
            input_name = self._model.get_inputs()[0].name
            label_name = self._model.get_outputs()[0].name
            if array.dtype == 'float64':
                tensor_type = np.float32
            elif array.dtype == 'int64' or array.dtype == 'int32':
                tensor_type = np.int64
            else:
                tensor_type = np.str
            pred_onx = self._model.run([label_name], {input_name: array.astype(tensor_type)})[0]
            return [prediction for plist in pred_onx for prediction in plist] if isinstance(pred_onx[0], list)\
                else [prediction for prediction in pred_onx]
        else:
            return self._model.predict(dataset)

    # TODO: to be improved, make it more flexible
    @staticmethod
    def get_initial_types(dataset: pd.DataFrame | np.ndarray) -> list[tuple[str, DataType]]:
        array = dataset.to_numpy() if isinstance(dataset, pd.DataFrame) else dataset
        name = ''
        for column in dataset.columns:
            name += column + ', '
        name = name[:-2]
        shape = [None, array.shape[1]]
        if array.dtype == 'float64':
            types = FloatTensorType(shape)
        elif array.dtype == 'int64':
            types = Int64TensorType(shape)
        else:
            types = StringTensorType(shape)
        return [(name, types)]
