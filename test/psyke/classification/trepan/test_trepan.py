import unittest
from parameterized import parameterized_class
from psyke import logger
from test.psyke import initialize


@parameterized_class(initialize('trepan'))
class TestReal(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))


if __name__ == '__main__':
    unittest.main()
