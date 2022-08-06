from psyke import logger
from parameterized import parameterized_class
from psyke.utils import get_default_precision
from test.psyke import initialize
import unittest


@parameterized_class(initialize('iter'))
class TestIter(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        #print(self.expected_theory)
        #print(self.extracted_theory)
        print()
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))

    def test_predict(self):
        self.assertTrue(max(abs(self.extracted_test_y_from_theory - self.extracted_test_y_from_extractor) < get_default_precision()))


if __name__ == '__main__':
    unittest.main()
