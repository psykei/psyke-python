import unittest
from parameterized import parameterized_class
from psyke import logger
from test.unit import initialize


@parameterized_class(initialize('real'))
class TestReal(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))

    def test_predict(self):
        self.assertTrue(all(self.extracted_test_y_from_theory == self.extracted_test_y_from_extractor))


if __name__ == '__main__':
    unittest.main()
