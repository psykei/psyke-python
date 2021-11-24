from parameterized import parameterized_class
from psyke import logger
from psyke.utils import get_int_precision
from test.psyke import initialize, data_to_struct, get_default_accuracy
from tuprolog.solve.prolog import prolog_solver
import unittest


@parameterized_class(initialize('cart'))
class TestCart(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))

    def test_predict(self):
        predictions = self.extractor.predict(self.test_set.iloc[:, :-1])

        # Handle both classification and regression.
        if not isinstance(predictions[0], str):
            predictions = [round(x, get_int_precision()) for x in predictions]
        solver = prolog_solver(static_kb=self.extracted_theory)

        substitutions = [solver.solveOnce(data_to_struct(data)) for _, data in self.test_set.iterrows()]
        index = self.test_set.shape[1] - 1
        expected = [query.solved_query.get_arg_at(index) for query in substitutions]

        # Handle both classification and regression.
        if isinstance(predictions[0], str):
            expected = [str(x) for x in expected]
        else:
            expected = [x.decimal_value.toDouble() for x in expected]

        if isinstance((predictions == expected), bool):
            accuracy = sum([v == expected[i] for i, v in enumerate(predictions)]) / len(predictions)
            self.assertTrue(accuracy > get_default_accuracy())
        else:
            self.assertTrue(all(predictions == expected))


if __name__ == '__main__':
    unittest.main()
