import unittest
from math import log10
from psyke import logger
from parameterized import parameterized_class
from tuprolog.solve.prolog import prolog_solver
from test import get_precision
from test.psyke import initialize, data_to_struct


@parameterized_class(initialize('cart'))
class TestCart(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))

    def test_predict(self):
        predictions = self.extractor.predict(self.test_set.iloc[:, :-1])
        if not isinstance(predictions[0], str):
            predictions = [round(x, -1 * int(log10(get_precision()))) for x in predictions]
        solver = prolog_solver(static_kb=self.extracted_theory)

        substitutions = [solver.solveOnce(data_to_struct(data)) for _, data in self.test_set.iterrows()]
        index = self.test_set.shape[1] - 1
        expected = [query.solved_query.get_arg_at(index) for query in substitutions]
        if isinstance(predictions[0], str):
            expected = [str(x) for x in expected]
        else:
            expected = [x.decimal_value.toDouble() for x in expected]

        if isinstance((predictions == expected), bool):
            accuracy = sum([v == expected[i] for i, v in enumerate(predictions)]) / len(predictions)
            # TODO: handle the concept of accuracy globally
            self.assertTrue(accuracy > 0.95)
        else:
            self.assertTrue(all(predictions == expected))


if __name__ == '__main__':
    unittest.main()
