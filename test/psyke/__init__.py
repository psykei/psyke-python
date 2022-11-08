from __future__ import annotations
from onnxconverter_common import DataType, FloatTensorType, Int64TensorType, StringTensorType
from skl2onnx import convert_sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.python.saved_model.save import save
from tuprolog.solve.prolog import prolog_solver
from psyke.extraction.hypercubic import Grid, FeatureRanker
from psyke.utils.dataframe import get_discrete_dataset
from psyke.utils.logic import data_to_struct, get_in_rule, get_not_in_rule
from psyke.extraction.hypercubic.strategy import AdaptiveStrategy, FixedStrategy
from test import get_dataset, get_extractor, get_schema, get_model
from test.resources.predictors import get_predictor_path
from test.resources.tests import test_cases
from tuprolog.theory import Theory, mutable_theory
from tuprolog.theory.parsing import parse_theory
from typing import Iterable, Callable
import ast
import numpy as np
import onnxruntime as rt
import os
import pandas as pd
from psyke import get_default_random_seed


def initialize(file: str) -> list[dict[str:Theory]]:
    for row in test_cases(file):
        params = dict() if row['extractor_params'] == '' else ast.literal_eval(row['extractor_params'])
        dataset = get_dataset(row['dataset'])

        # Dataset's columns are sorted due to alphabetically sorted extracted rules.
        # columns = sorted(dataset.columns[:-1]) + [dataset.columns[-1]]
        # dataset = dataset.reindex(columns, axis=1)

        training_set, test_set = train_test_split(dataset, test_size=0.05 if row['dataset'].lower() == 'house' else 0.5,
                                                  random_state=get_default_random_seed())

        schema, test_set_for_predictor = None, test_set
        if 'disc' in row.keys() and bool(row['disc']):
            schema = get_schema(training_set)
            params['discretization'] = schema
            training_set = get_discrete_dataset(training_set.iloc[:, :-1], schema)\
                .join(training_set.iloc[:, -1].reset_index(drop=True))
            test_set_for_predictor = get_discrete_dataset(test_set.iloc[:, :-1], schema) \
                .join(test_set.iloc[:, -1].reset_index(drop=True))

        # Handle Cart tests.
        # Cart needs to inspect the tree of the predictor.
        # Unfortunately onnx does not provide a method to do that.
        if row['predictor'].lower() not in ['dtc', 'dtr']:
            params['predictor'] = Predictor.load_from_onnx(str(get_predictor_path(row['predictor'])))
        else:
            tree = get_model(row['predictor'], {})
            tree.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
            params['predictor'] = tree

        # Handle GridEx tests
        # TODO: this is algorithm specific therefore it should be handled inside the algorithm itself.
        if 'grid' in row.keys() and bool:
            strategy, n = eval(row['strategies'])
            if strategy == "F":
                params['grid'] = Grid(int(row['grid']), FixedStrategy(n))
            else:
                ranked = FeatureRanker(training_set.columns[:-1])\
                    .fit(params['predictor'], training_set.iloc[:, :-1]).rankings()
                params['grid'] = Grid(int(row['grid']), AdaptiveStrategy(ranked, n))

        extractor = get_extractor(row['extractor_type'], params)
        mapping = None if 'output_mapping' not in row.keys() or row['output_mapping'] == '' else ast.literal_eval(row['output_mapping'])
        theory = extractor.extract(training_set, mapping) if mapping is not None else extractor.extract(training_set)

        # Compute predictions from rules
        index = test_set.shape[1] - 1
        ordered_test_set = test_set.copy()
        ordered_test_set.iloc[:, :-1] = ordered_test_set.iloc[:, :-1].reindex(sorted(ordered_test_set.columns[:-1]), axis=1)
        is_classification = isinstance(test_set.iloc[0, -1], str)
        cast: Callable = lambda x: (str(x) if is_classification else float(x.value))
        solver = prolog_solver(static_kb=mutable_theory(theory).assertZ(get_in_rule()).assertZ(get_not_in_rule()))
        substitutions = [solver.solveOnce(data_to_struct(data)) for _, data in ordered_test_set.iterrows()]
        expected = [cast(query.solved_query.get_arg_at(index)) for query in substitutions if query.is_yes]
        if mapping is not None:
            predictions = [prediction for prediction in extractor.predict(test_set_for_predictor.iloc[:, :-1], mapping)
                          if prediction is not None]
        else:
            predictions = [prediction for prediction in extractor.predict(test_set_for_predictor.iloc[:, :-1])
                           if prediction is not None]

        yield {
            'extractor': extractor,
            'extracted_theory': theory,
            'extracted_test_y_from_theory': np.array(expected),
            'extracted_test_y_from_extractor': np.array(predictions),
            'test_set': test_set,
            'expected_theory': parse_theory(row['theory'] + '.') if row['theory'] != '' else None,
            'discretization': schema
        }


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
            if isinstance(self._model, Model):
                save(self._model, "tmp_model")
                os.system("python -m tf2onnx.convert --saved-model tmp_model --output " + file)
            else:
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
