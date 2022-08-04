import unittest
from cmath import isclose

from parameterized import parameterized_class
from psyke import logger
from test.psyke import initialize, ACCEPTABLE_FIDELITY


@parameterized_class(initialize('real'))
class TestReal(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))

    def test_predict(self):
        if not isinstance(self.extracted_test_y_from_theory[0], str) \
                and self.extracted_test_y_from_theory[0].is_number:
            matches = sum(isclose(self.extracted_test_y_from_theory[i].value, self.extracted_test_y_from_extractor[i])
                          for i in range(len(self.extracted_test_y_from_theory)))
        else:
            matches = sum((self.extracted_test_y_from_theory[i] == self.extracted_test_y_from_extractor[i])
                          for i in range(len(self.extracted_test_y_from_theory)))
        self.assertTrue(matches / self.test_set.shape[0] > ACCEPTABLE_FIDELITY)


if __name__ == '__main__':
    unittest.main()
