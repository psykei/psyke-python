import numpy as np
from parameterized import parameterized_class
from psyke import logger
from psyke.utils import get_int_precision
from psyke.utils.dataframe import get_discrete_dataset
from test import get_in_rule, get_not_in_rule
from test.psyke import initialize, data_to_struct
from tuprolog.solve.prolog import prolog_solver
import unittest


@parameterized_class(initialize('cart'))
class TestCart(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))

    def test_predict(self):
        if self.discretization is not None:
            test_set = get_discrete_dataset(self.test_set.iloc[:, :-1], self.discretization)
        else:
            test_set = self.test_set.iloc[:, :-1]

        predictions = self.extractor.predict(test_set)

        # Handle both classification and regression.
        if not isinstance(predictions[0], str):
            predictions = np.array([round(x, get_int_precision()) for x in predictions])
        solver = prolog_solver(static_kb=self.extracted_theory.assertZ(get_in_rule()).assertZ(get_not_in_rule()))

        substitutions = [solver.solveOnce(data_to_struct(data)) for _, data in self.test_set.iterrows()]
        index = self.test_set.shape[1] - 1
        expected = [query.solved_query.get_arg_at(index) if query.is_yes else '-1' for query in substitutions]

        # Handle both classification and regression.
        expected = [str(x) for x in expected] if isinstance(predictions[0], str) else [float(x.value) for x in expected]

        self.assertTrue(all(predictions == expected))


if __name__ == '__main__':
    unittest.main()
