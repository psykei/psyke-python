from parameterized import parameterized_class
from psyke.extractor import Extractor
from psyke.predictor import Predictor
from psyke.utils.parsing import parse_theory
from test.resources import CLASSPATH
from tuprolog.core import scope
from tuprolog.theory import Theory
import csv
import numpy as np
import pandas as pd
import unittest

resource_dir = str(CLASSPATH) + '/'
ts = scope()


def _get_tests_from_file(file: str) -> list[dict[str:Theory]]:
    result = []
    with open(file) as f:
        rows = csv.DictReader(f, delimiter=';', quotechar='"')
        for row in rows:
            result.append({
                'training_set': row['training_set'],
                'test_set': row['test_set'],
                'predictor': row['predictor'],
                'expected_theory': parse_theory(row['theory'])
            })
    return result


@parameterized_class(_get_tests_from_file(resource_dir + 'expected_theories.csv'))
class TestIter(unittest.TestCase):

    def test_extract(self):
        training_set = pd.read_csv(resource_dir + self.training_set)
        predictor = Predictor.load_from_onnx(resource_dir + self.predictor)
        iter_extractor = Extractor.iter(predictor, min_update=1.0 / 20, threshold=0.19)
        theory = iter_extractor.extract(training_set)
        self.assertTrue(self.expected_theory.equals(theory, False))

    def test_predict(self):
        training_set = pd.read_csv(resource_dir + self.training_set)
        test_set = pd.read_csv(resource_dir + self.test_set)
        predictor = Predictor.load_from_onnx(resource_dir + self.predictor)
        iter_extractor = Extractor.iter(predictor, min_update=1.0 / 20, threshold=0.19)
        iter_extractor.extract(training_set)
        predictions = iter_extractor.predict(test_set.iloc[:, :-1])
        from_theory = []
        for _, point in test_set.iloc[:, :-1].iterrows():
            if (point['X'] >= 0.5095) & (point['Y'] < 0.4693):
                from_theory.append(0.376)
            if (point['X'] >= 0.5095) & (point['Y'] >= 0.4693):
                from_theory.append(0.0437)
            if (point['X'] < 0.5095) & (point['Y'] >= 0.4693):
                from_theory.append(0.4)
            if (point['X'] < 0.5095) & (point['Y'] < 0.4693):
                from_theory.append(0.6848)
        results = abs(np.array(predictions) - from_theory) <= 10e-4
        self.assertTrue(all(results))


if __name__ == '__main__':
    unittest.main()
