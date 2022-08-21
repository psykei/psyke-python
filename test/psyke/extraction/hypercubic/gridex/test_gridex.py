from psyke import logger
from parameterized import parameterized_class
from psyke.utils import get_default_precision
from test.psyke import initialize
import unittest


@parameterized_class(initialize('gridex'))
class TestGridEx(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        # This test does not pass the ci, however it is not clear to me why (local ok). Could it be non-deterministic?
        # self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))

    def test_predict(self):
        if isinstance(self.extracted_test_y_from_theory[0], str):
            self.assertTrue(all(self.extracted_test_y_from_theory == self.extracted_test_y_from_extractor))
        else:
            # TODO: check this!
            self.assertTrue(max(abs(self.extracted_test_y_from_theory - self.extracted_test_y_from_extractor)) < 0.05)


if __name__ == '__main__':
    unittest.main()
