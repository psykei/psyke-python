from tuprolog.core import Var, Real
from psyke import logger
from parameterized import parameterized_class
from psyke.utils import get_default_precision
from test.psyke import initialize, are_similar, are_equal
import numpy as np
import unittest


@parameterized_class(initialize('iter'))
class TestIter(unittest.TestCase):

    # TODO: refactor this test, it is not standard. Moreover extracted rule should be compliant with a precision value.
    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)

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
        if isinstance(self.extracted_test_y_from_theory[0], str):
            self.assertTrue(all(self.extracted_test_y_from_theory == self.extracted_test_y_from_extractor))
        else:
            from_theory = np.array([float(str(item)) for item in self.extracted_test_y_from_theory])
            from_exractor = np.array([float(str(item)) for item in self.extracted_test_y_from_extractor])
            self.assertTrue(max(abs(from_theory - from_exractor)) < get_default_precision())


if __name__ == '__main__':
    unittest.main()
