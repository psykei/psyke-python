from cmath import isclose
from parameterized import parameterized_class
from psyke.utils.logic import pretty_theory

from psyke import logger
from test.psyke import initialize, ACCEPTABLE_FIDELITY
import unittest

""" 
    TODO (?): right now there is a small chance that corner data are wrongly predicted (that is fine for now).
    In other words, if we use the extracted rules (with a specific default accuracy fo float)
    and compare their result with the one obtained by the actual decision tree (thresholds do not have truncated float)
    they may be different. To avoid this, when we will refactor all extractor we will also address this issue.
"""


@parameterized_class(initialize('cart'))
class TestCart(unittest.TestCase):

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
            matches = sum(self.extracted_test_y_from_theory == self.extracted_test_y_from_extractor)
        self.assertTrue(matches / self.test_set.shape[0] > ACCEPTABLE_FIDELITY)


if __name__ == '__main__':
    unittest.main()
