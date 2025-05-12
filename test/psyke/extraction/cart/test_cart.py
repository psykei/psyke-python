from parameterized import parameterized_class
from psyke.utils import get_default_precision
from psyke import logger
from test.psyke import initialize
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
        # self.assertEqual(self.extracted_test_y_from_theory, self.extracted_test_y_from_pruned_theory)
        if isinstance(self.extracted_test_y_from_theory[0], str):
            self.assertTrue(all(self.extracted_test_y_from_theory == self.extracted_test_y_from_extractor))
        else:
            self.assertTrue(max(abs(self.extracted_test_y_from_theory - self.extracted_test_y_from_extractor)) <
                            get_default_precision())


if __name__ == '__main__':
    unittest.main()
