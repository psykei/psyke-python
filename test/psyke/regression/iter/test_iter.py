from psyke.extractor import Extractor
from psyke.schema.value import Between
from psyke.utils.logic_utils import create_head, create_term
from psyke.utils.predictor import Predictor
from test.resources import CLASSPATH
from tuprolog.core import scope, rule
from tuprolog.theory import theory
import numpy as np
import pandas as pd
import unittest

resource_dir = str(CLASSPATH) + '/'
ts = scope()


class TestIter(unittest.TestCase):
    dataset = pd.DataFrame(pd.read_csv(resource_dir + 'arti.csv'))
    training_set = pd.read_csv(resource_dir + 'artiTrain50.csv')
    test_set = dataset.loc[~dataset.index.isin(training_set.index)]
    predictor = Predictor.load_from_onnx(resource_dir + 'artiKNN3.onnx')
    iter = Extractor.iter(predictor, min_update=1.0 / 20, threshold=0.19)
    theory = iter.extract(training_set)

    def test_extract(self):
        expected = theory(
            rule(
                create_head('z', [ts.var('X'), ts.var('Y')], 0.4),
                [create_term(ts.var('X'), Between(0.0, 0.5095)), create_term(ts.var('Y'), Between(0.4693, 1.0))]
            ),
            rule(
                create_head('z', [ts.var('X'), ts.var('Y')], 0.6848),
                [create_term(ts.var('X'), Between(0.0, 0.5095)), create_term(ts.var('Y'), Between(0.0, 0.4693))]
            ),
            rule(
                create_head('z', [ts.var('X'), ts.var('Y')], 0.376),
                [create_term(ts.var('X'), Between(0.5095, 1.0)), create_term(ts.var('Y'), Between(0.0, 0.4693))]
            ),
            rule(
                create_head('z', [ts.var('X'), ts.var('Y')], 0.0437),
                [create_term(ts.var('X'), Between(0.5095, 1.0)), create_term(ts.var('Y'), Between(0.4693, 1.0))]
            )
        )
        self.assertTrue(expected.equals(self.theory, False))

    def test_predict(self):
        predictions = self.iter.predict(self.test_set.iloc[:, :-1])
        from_theory = []
        for _, point in self.test_set.iloc[:, :-1].iterrows():
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
