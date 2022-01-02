from tuprolog.core import Var, Real, BigDecimal

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
        def to_float(n: BigDecimal) -> float:
            return float(n.value.__repr__())

        def are_equal(expected, actual):
            if expected.is_functor_well_formed:
                self.assertTrue(actual.is_functor_well_formed)
                self.assertEqual(expected.functor, actual.functor)
                self.assertTrue(expected.args[0].equals(actual.args[0], False))
                self.assertTrue(abs(to_float(expected.args[1][0]) - to_float(actual.args[1][0])) < 0.01)
                self.assertTrue(abs(to_float(expected.args[1][1].head) - to_float(actual.args[1][1].head)) < 0.01)
            elif expected.is_recursive:
                self.assertTrue(actual.is_recursive)
                self.assertEqual(expected.arity, actual.arity)
                for i in range(expected.arity):
                    are_equal(expected.args[i], actual.args[i])

        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        # TODO: keep un eye on it.
        for exp, ext in zip(self.expected_theory, self.extracted_theory):
            for v1, v2 in zip (exp.head.args, ext.head.args):
                if isinstance(v1, Var):
                    self.assertTrue(isinstance(v2, Var))
                    self.assertTrue(v1.equals(v2, False))
                elif isinstance(v1, Real):
                    self.assertTrue(isinstance(v2, Real))
                    self.assertTrue(abs(to_float(v1) - to_float(v2)) < 0.01)
            for t1, t2 in zip(exp.body, ext.body):
                are_equal(t1, t2)

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
