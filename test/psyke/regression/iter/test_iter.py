from parameterized import parameterized_class
from sklearn.model_selection import train_test_split
from psyke.predictor import Predictor
from psyke.utils import get_default_random_seed
from test import get_dataset, get_extractor, get_precision, get_in_rule
from test.resources.predictors import get_predictor_path
from test.resources.tests import test_cases
from tuprolog.core import real, var, struct
from tuprolog.solve.prolog import prolog_solver
from tuprolog.theory import Theory
from tuprolog.theory.parsing import parse_theory
import ast
import numpy as np
import pandas as pd
import unittest


def _initialize(file: str) -> list[dict[str:Theory]]:
    for row in test_cases(file):
        dataset = get_dataset(row['dataset'])
        training_set, test_set = train_test_split(dataset, test_size=0.5, random_state=get_default_random_seed())
        params = dict() if row['extractor_params'] == '' else ast.literal_eval(row['extractor_params'])
        params['predictor'] = Predictor.load_from_onnx(str(get_predictor_path(row['predictor'])))
        extractor = get_extractor(row['extractor_type'], params)
        theory = extractor.extract(training_set)
        yield {
            'extractor': extractor,
            'extracted_theory': theory,
            'test_set': test_set,
            'expected_theory': parse_theory(row['theory'] + '.')
        }


def _data_to_struct(data: pd.Series):
    head = data.keys()[-1]
    terms = [real(item) for item in data.values[:-1]]
    terms.append(var('X'))
    return struct(head, terms)


@parameterized_class(_initialize('iter'))
class TestIter(unittest.TestCase):

    def test_extract(self):
        print(self.expected_theory)
        print(self.extracted_theory)
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))

    def test_predict(self):
        predictions = np.array(self.extractor.predict(self.test_set.iloc[:, :-1]))
        solver = prolog_solver(static_kb=self.extracted_theory.assertZ(get_in_rule()))
        substitutions = [solver.solveOnce(_data_to_struct(data)) for _, data in self.test_set.iterrows()]
        index = self.test_set.shape[1]-1
        expected = np.array([query.solved_query.get_arg_at(index).decimal_value.toDouble() for query in substitutions])
        '''
        ITER is not exhaustive so all entry's predictions that are not inside an hypercube are nan.
        All nan value are substituted with the expected one.
        '''
        predictions[np.isnan(predictions)] = expected[np.isnan(predictions)]
        results = abs(predictions - expected) <= get_precision()
        print(sum(results)/len(results))
        self.assertTrue(all(results))


if __name__ == '__main__':
    unittest.main()
