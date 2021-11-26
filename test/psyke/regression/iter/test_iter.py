from psyke import logger
from parameterized import parameterized_class
from psyke.utils import get_default_precision, get_int_precision
from test import get_in_rule
from test.psyke import initialize, data_to_struct
from tuprolog.solve.prolog import prolog_solver
import numpy as np
import unittest


@parameterized_class(initialize('iter'))
class TestIter(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        # TODO: keep un eye on it.
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))

    def test_predict(self):
        precision = get_int_precision()
        predictions = np.array(self.extractor.predict(self.test_set.iloc[:, :-1]))
        solver = prolog_solver(static_kb=self.extracted_theory.assertZ(get_in_rule()))
        substitutions = [solver.solveOnce(data_to_struct(data)) for _, data in self.test_set.iterrows()]
        index = self.test_set.shape[1] - 1
        expected = np.array([query.solved_query.get_arg_at(index).decimal_value.toDouble()
                             if query.is_yes else 0 for query in substitutions])
        '''
        ITER is not exhaustive so all entry's predictions that are not inside an hypercube are nan.
        In python nan == nan is always False so for this test we use 0 instead.
        All nan value are substituted with the expected one (which is 0).
        '''
        predictions[np.isnan(predictions)] = expected[np.isnan(predictions)]
        predictions = np.round(predictions, precision)
        results = abs(predictions - expected) <= get_default_precision()
        # logger.info(predictions[np.logical_not(results)])
        # logger.info(expected[np.logical_not(results)])
        self.assertTrue(all(results))


if __name__ == '__main__':
    unittest.main()
