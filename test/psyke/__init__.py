from __future__ import annotations
from sklearn.model_selection import train_test_split
from tuprolog.solve.prolog import prolog_solver
from psyke.extraction.hypercubic import Grid, FeatureRanker
from psyke.utils.dataframe import get_discrete_dataset
from psyke.utils.logic import data_to_struct, get_in_rule, get_not_in_rule
from psyke.extraction.hypercubic.strategy import AdaptiveStrategy, FixedStrategy
from test import get_dataset, get_extractor, get_schema, get_model
from test.resources.tests import test_cases
from tuprolog.theory import Theory, mutable_theory
from tuprolog.theory.parsing import parse_theory
from typing import Callable
import ast
import numpy as np
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
            training_set = get_discrete_dataset(training_set.iloc[:, :-1], schema) \
                .join(training_set.iloc[:, -1].reset_index(drop=True))
            test_set_for_predictor = get_discrete_dataset(test_set.iloc[:, :-1], schema) \
                .join(test_set.iloc[:, -1].reset_index(drop=True))

        # Handle Cart tests.
        # Cart needs to inspect the tree of the predictor.
        # Unfortunately onnx does not provide a method to do that.
        #if row['predictor'].lower() not in ['dtc', 'dtr']:
        #    params['predictor'] = Predictor.load_from_onnx(str(get_predictor_path(row['predictor'])))
        #else:
        predictor, fitted = get_model(row['predictor'], {})
        if not fitted:
            predictor.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
        params['predictor'] = predictor

        # Handle GridEx tests
        # TODO: this is algorithm specific therefore it should be handled inside the algorithm itself.
        if 'grid' in row.keys() and bool:
            strategy, n = eval(row['strategies'])
            if strategy == "F":
                params['grid'] = Grid(int(row['grid']), FixedStrategy(n))
            else:
                ranked = FeatureRanker(training_set.columns[:-1]) \
                    .fit(params['predictor'], training_set.iloc[:, :-1]).rankings()
                params['grid'] = Grid(int(row['grid']), AdaptiveStrategy(ranked, n))

        extractor = get_extractor(row['extractor_type'], params)
        theory = extractor.extract(training_set)

        # Compute predictions from rules
        index = test_set.shape[1] - 1
        ordered_test_set = test_set.copy()
        ordered_test_set.iloc[:, :-1] = ordered_test_set.iloc[:, :-1].reindex(sorted(ordered_test_set.columns[:-1]),
                                                                              axis=1)
        cast, substitutions = get_substitutions(test_set, ordered_test_set, theory)
        expected = [cast(query.solved_query.get_arg_at(index)) for query in substitutions if query.is_yes]
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


def get_substitutions(test_set, ordered_test_set, theory):
    cast: Callable = lambda x: (str(x) if isinstance(test_set.iloc[0, -1], str) else float(x.value))
    solver = prolog_solver(static_kb=mutable_theory(theory).assertZ(get_in_rule()).assertZ(get_not_in_rule()))
    substitutions = [solver.solveOnce(data_to_struct(data)) for _, data in ordered_test_set.iterrows()]
    return cast, substitutions
