import unittest
from tuprolog.theory import mutable_theory, theory
from tuprolog.theory.parsing import parse_theory
from psyke.utils.logic import prune


class TestPrune(unittest.TestCase):

    def test_prune_documentation(self):
        theory1 = "c(A, B, C, D, positive) :- ('=<'(A, 1), '>'(B, 2), '=='(C, 0)). " \
                  + "c(A, B, C, D, positive) :- ('=<'(A, 1), '>'(B, 2))."
        pruned1 = "c(A, B, C, D, positive) :- ('=<'(A, 1), '>'(B, 2))."

        theory2 = "c(A, B, C, D, positive) :- ('=<'(A, 1), '>'(B, 2), '=='(C, 0)). " \
                  + "c(A, B, C, D, positive) :- ('=<'(A, 1.3), '>'(B, 1.8))."
        pruned2 = "c(A, B, C, D, positive) :- ('=<'(A, 1.3), '>'(B, 1.8))."

        theory3 = "c(A, B, C, D, positive) :- ('=<'(A, 1.3), '>'(B, 1.8)). " \
                  + "c(A, B, C, D, positive) :- ('=<'(A, 1), '>'(B, 2), '=='(C, 0))."
        pruned3 = pruned2

        self.assertTrue(theory(parse_theory(pruned1)).equals(prune(mutable_theory(parse_theory(theory1))), False))
        self.assertTrue(theory(parse_theory(pruned2)).equals(prune(mutable_theory(parse_theory(theory2))), False))
        self.assertTrue(theory(parse_theory(pruned3)).equals(prune(mutable_theory(parse_theory(theory3))), False))

    def test_prune_success(self):
        textual_theory = "p(X, Y, inside) :- ('=<'(X, 1), '>'(Y, 2)). " \
                         + "p(X, Y, inside) :- ('=<'(X, 0.5), '>'(Y, 3))."
        textual_pruned_theory = "p(X, Y, inside) :- ('=<'(X, 1), '>'(Y, 2))."
        long_theory = mutable_theory(parse_theory(textual_theory))
        pruned_theory = theory(parse_theory(textual_pruned_theory))

        self.assertTrue(pruned_theory.equals(prune(long_theory), False))

    def test_prune_not_applied(self):
        textual_theory = "p(PL, PW, SL, SW, versicolor) :-  '=<'(SW, 3.6). " \
                         + "p(PL, PW, SL, SW, versicolor) :- ('=<'(PW, 0.35), '=<'(SL, 5.35), '=<'(SW, 3.9))."
        textual_pruned_theory = textual_theory
        long_theory = mutable_theory(parse_theory(textual_theory))
        pruned_theory = theory(parse_theory(textual_pruned_theory))

        self.assertTrue(pruned_theory.equals(prune(long_theory), False))


if __name__ == '__main__':
    unittest.main()
