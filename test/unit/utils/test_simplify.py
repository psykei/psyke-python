import unittest
from tuprolog.theory import mutable_theory, theory
from tuprolog.theory.parsing import parse_theory
from psyke.utils.logic import simplify


class TestSimplify(unittest.TestCase):

    def test_simplify(self):
        # TODO: if numbers are not float equals method return false (e.g., 2 instead of 2.0). @Giovanni 2ppy
        textual_theory = "p(X, Y, inside) :- ('=<'(X, 1.0), '>'(Y, 2.0), '=<'(X, 0.5))."
        textual_simplified_theory = "p(X, Y, inside) :- ('=<'(X, 0.5), '>'(Y, 2.0))."
        long_theory = mutable_theory(parse_theory(textual_theory))
        simplified_theory = theory(parse_theory(textual_simplified_theory))

        self.assertTrue(simplified_theory.equals(simplify(long_theory), False))


if __name__ == '__main__':
    unittest.main()
