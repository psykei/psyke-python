from tuprolog.core import Var, Real
from psyke import logger
from parameterized import parameterized_class
from psyke.utils import get_default_precision, get_int_precision
from test import get_in_rule
from test.psyke import initialize, data_to_struct, are_similar, are_equal
from tuprolog.solve.prolog import prolog_solver
import numpy as np
import unittest


@parameterized_class(initialize('gridex'))
class TestGridEx(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)

        if isinstance(self.test_set.iloc[0, -1], str):
            self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))
        else:
            for exp, ext in zip(self.expected_theory, self.extracted_theory):
                for v1, v2 in zip(exp.head.args, ext.head.args):
                    if isinstance(v1, Var):
                        self.assertTrue(isinstance(v2, Var))
                        self.assertTrue(v1.equals(v2, False))
                    elif isinstance(v1, Real):
                        self.assertTrue(isinstance(v2, Real))
                        self.assertTrue(are_similar(v1, v2))
                for t1, t2 in zip(exp.body, ext.body):
                    are_equal(self, t1, t2)

    def test_predict(self):
        predictions = self.extractor.predict(self.test_set.iloc[:, :-1])

        # Handle both classification and regression.
        if not isinstance(predictions[0], str):
            predictions = np.array([round(x, get_int_precision()) for x in predictions])

        solver = prolog_solver(static_kb=self.extracted_theory.assertZ(get_in_rule()))
        substitutions = [solver.solveOnce(data_to_struct(data)) for _, data in self.test_set.iterrows()]
        index = self.test_set.shape[1] - 1
        expected = [query.solved_query.get_arg_at(index) if query.is_yes else '-1' for query in substitutions]
        expected = [str(x) for x in expected] if isinstance(predictions[0], str) else \
            [float(x) if isinstance(x, str) else float(x.value) for x in expected]

        '''
        GridEx is not exhaustive so all entry's predictions that are not inside an hypercube are nan.
        In python nan == nan is always False so for this test we do not consider them.
        '''
        if isinstance(predictions[0], str):

            self.assertTrue(all([pred == exp if exp != "-1" else True for (pred, exp) in zip(predictions, expected)]))
        else:
            idx = np.isnan(predictions)
            self.assertTrue(max(abs(predictions[~idx] - np.array(expected)[~idx])) < get_default_precision())


if __name__ == '__main__':
    unittest.main()
