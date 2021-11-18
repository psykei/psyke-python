import unittest
import numpy as np
from parameterized import parameterized_class
from tuprolog.solve.prolog import prolog_solver
from psyke import logger
from test import get_in_rule, get_precision
from test.psyke import initialize, data_to_struct


@parameterized_class(initialize('real'))
class TestReal(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))
