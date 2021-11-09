import pickle
import unittest
import numpy as np
import pandas as pd
from psyke.extractor import Extractor
from tuprolog.theory import *
from tuprolog.core import *
from psyke.schema.value import Between
from psyke.utils.logic_utils import create_head, create_term

resource_dir = 'test/resources/'


class TestIter(unittest.TestCase):
    dataset = pd.DataFrame(pd.read_csv(resource_dir + 'arti.csv'))
    training_set = pd.read_csv(resource_dir + 'artiTrain50.csv')
    test_set = dataset.loc[~dataset.index.isin(training_set.index)]
    predictor = pickle.load(open(resource_dir + 'artiGPR.txt', 'rb'))
    iter = Extractor.iter(predictor, min_update=1.0 / 20, threshold=0.19)
    theory = iter.extract(training_set)

    def test_extract(self):
        vars = {
            'X0': var('X'), 'Y0': var('Y'),
            'X1': var('X'), 'Y1': var('Y'),
            'X2': var('X'), 'Y2': var('Y'),
            'X3': var('X'), 'Y3': var('Y')
        }
        expected = theory(
            rule(
                create_head('z', [vars['X0'], vars['Y0']], 0.4),
                [create_term(vars['X0'], Between(0.0, 0.5095)), create_term(vars['Y0'], Between(0.5193, 1.0))]
            ),
            rule(
                create_head('z', [vars['X1'], vars['Y1']], 0.6499),
                [create_term(vars['X1'], Between(0.0, 0.5095)), create_term(vars['Y1'], Between(0.0, 0.5193))]
            ),
            rule(
                create_head('z', [vars['X2'], vars['Y2']], 0.2415),
                [create_term(vars['X2'], Between(0.5095, 1.0)), create_term(vars['Y2'], Between(0.0, 0.5193))]
            ),
            rule(
                create_head('z', [vars['X3'], vars['Y3']], -0.0481),
                [create_term(vars['X3'], Between(0.5095, 1.0)), create_term(vars['Y3'], Between(0.5193, 1.0))]
            )
        )
        self.assertTrue(expected.equals(self.theory, False))

    def test_predict(self):
        predictions = self.iter.predict(self.test_set.iloc[:, :-1])
        from_theory = []
        for _, point in self.test_set.iloc[:, :-1].iterrows():
            if (point['X'] >= 0.5095) & (point['Y'] < 0.5193):
                from_theory.append(0.2415)
            if (point['X'] >= 0.5095) & (point['Y'] >= 0.5193):
                from_theory.append(- 0.0481)
            if (point['X'] < 0.5095) & (point['Y'] >= 0.5193):
                from_theory.append(0.4)
            if (point['X'] < 0.5095) & (point['Y'] < 0.5193):
                from_theory.append(0.6499)
        results = abs(np.array(predictions) - from_theory) <= 10e-4
        self.assertTrue(all(results))


if __name__ == '__main__':
    unittest.main()
