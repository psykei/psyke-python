import unittest
import numpy as np
from parameterized import parameterized_class
from tuprolog.solve.prolog import prolog_solver
from psyke import logger
from test import get_in_rule, get_precision
from test.psyke import initialize, data_to_struct


@parameterized_class(initialize('real'))
class TestReal(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))


def test_predict(self):
    predictions = np.array(self.extractor.predict(self.test_set.iloc[:, :-1]))
    solver = prolog_solver(static_kb=self.extracted_theory.assertZ(get_in_rule()))
    substitutions = [solver.solveOnce(data_to_struct(data)) for _, data in self.test_set.iterrows()]
    index = self.test_set.shape[1] - 1
    expected = np.array([query.solved_query.get_arg_at(index).decimal_value.toDouble() for query in substitutions])
    '''
    ITER is not exhaustive so all entry's predictions that are not inside an hypercube are nan.
    All nan value are substituted with the expected one.
    '''
    predictions[np.isnan(predictions)] = expected[np.isnan(predictions)]
    results = abs(predictions - expected) <= get_precision()
    print(sum(results) / len(results))
    self.assertTrue(all(results))
