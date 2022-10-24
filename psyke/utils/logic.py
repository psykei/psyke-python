from __future__ import annotations
from typing import Iterable
import pandas as pd
from tuprolog.core import Var, Struct, Real, Term, Integer, Numeric, clause
import re
from tuprolog.core import struct, real, atom, var, numeric, logic_list, Clause
from tuprolog.core.operators import DEFAULT_OPERATORS, operator, operator_set, XFX
from tuprolog.core.formatters import TermFormatter
from tuprolog.core.visitors import AbstractTermVisitor
from tuprolog.theory import mutable_theory, Theory
from psyke.schema import Value, LessThan, GreaterThan, Between, Constant, term_to_value, Outside
from psyke import DiscreteFeature
from psyke.utils import get_int_precision

PRECISION: int = get_int_precision()

OP_IN = operator('in', XFX, 700)

OP_NOT = operator('not_in', XFX, 700)

RULES_OPERATORS = DEFAULT_OPERATORS + operator_set(OP_IN, OP_NOT)

REGEX = r'(?<=\.\d\d)\d*(?=(\D|\b))'


def is_sum(term: Struct) -> bool:
    return term.getArity() == 2 and term.getFunctor() == '+'


def is_mult(term: Struct) -> bool:
    return term.getArity() == 2 and term.getFunctor() == '*'


def is_negative(term: Term) -> bool:
    if isinstance(term, Integer):
        return term < Integer.ZERO
    if isinstance(term, Real):
        return term < Real.ZERO
    if isinstance(term, Struct) and is_mult(term):
        return any(map(is_negative, term.getArgs()))
    return False


def is_zero(term: Term) -> bool:
    if isinstance(term, Integer):
        return term == Integer.ZERO
    if isinstance(term, Real):
        return term == Real.ZERO
    if isinstance(term, Struct) and is_mult(term):
        return any(map(is_zero, term.getArgs()))
    return False


def absolute(term: Term) -> bool:
    if is_negative(term):
        if isinstance(term, Numeric):
            return numeric(float(str(term.getValue().unaryMinus())))
        return struct(term.getFunctor(), map(absolute, term.getArgs()))
    return term


class Simplifier(AbstractTermVisitor):
    def defaultValue(self, term):
        return term

    def visitStruct(self, term):
        args = term.getArgs()
        if is_sum(term):
            left, right = args
            if is_mult(right):
                if is_negative(right):
                    return struct('-', map(absolute, args))
            right_left, right_right = right
            if is_sum(right) and is_negative(right_left):
                return struct(
                    '-',
                    left,
                    struct(right.getFunctor(), absolute(right_left), right_right).accept(self)
                )
        return struct(term.getFunctor(), [a.accept(self) for a in args])


def create_functor(constraint: Value, positive: bool = True) -> str:
    if isinstance(constraint, LessThan):
        return '=<' if positive else '>'
    if isinstance(constraint, GreaterThan):
        return '>' if positive else '=<'
    if isinstance(constraint, Between):
        return 'in' if positive else 'not_in'
    if isinstance(constraint, Outside):
        return 'not_in' if positive else 'in'
    if isinstance(constraint, Constant):
        return '=' if positive else '\\='


def _create_term(v: Var, constraint: Value, functor: str) -> Struct:
    if isinstance(constraint, LessThan):
        return struct(functor, v, real(round(constraint.value, PRECISION)))
    if isinstance(constraint, GreaterThan):
        return struct(functor, v, real(round(constraint.value, PRECISION)))
    if isinstance(constraint, Between):
        return struct(functor, v,
                      logic_list(real(round(constraint.lower, PRECISION)), real(round(constraint.upper, PRECISION))))
    if isinstance(constraint, Outside):
        return struct(functor, v,
                      logic_list(real(round(constraint.lower, PRECISION)), real(round(constraint.upper, PRECISION))))
    if isinstance(constraint, Constant):
        return struct(functor, v, atom(str(constraint.value)))


def create_term(v: Var, constraint: Value, positive: bool = True) -> Struct:
    if v is None:
        raise Exception('IllegalArgumentException: None variable')
    functor = create_functor(constraint, positive)
    return _create_term(v, constraint, functor)


def to_var(name: str) -> Var:
    return var(name[0].upper() + name[1:])


def create_variable_list(features: list[DiscreteFeature], dataset: pd.DataFrame = None) -> dict[str, Var]:
    values = {feature.name: to_var(feature.name) for feature in sorted(features, key=lambda x: x.name)} \
        if len(features) > 0 else {name: to_var(name) for name in sorted(dataset.columns[:-1])}
    return values


def create_head(functor: str, variables: Iterable[Var], output) -> Struct:
    if isinstance(output, Var):
        variables += [output]
    elif isinstance(output, str):
        variables += [atom(output)]
    else:
        variables += [numeric(round(float(output), PRECISION))]
    return struct(functor, variables)


def pretty_clause(clause: Clause, new_line: bool = True) -> str:
    formatter = TermFormatter.prettyExpressions(True, RULES_OPERATORS)
    if clause.is_fact:
        return str(formatter.format(clause.head))
    elif clause.is_directive:
        return ":- " + formatter.format(clause.body)
    else:
        head = str(formatter.format(clause.head))
        body = str(formatter.format(clause.body))
        return f"{re.sub(REGEX, '', head)} :-\n    {re.sub(REGEX, '', body)}"


def pretty_theory(theory: Theory, new_line: bool = True) -> str:
    if len(str(theory)) == 0:
        return ""
    else:
        clause = [pretty_clause(clause, new_line) for clause in theory]
        return ".\n".join(clause) + "."


def to_rounded_real(n: float) -> Real:
    return real(round(n, PRECISION))


def foldr(accumulator, iterable, default=None):
    items = list(iterable)
    if len(items) == 0:
        return default
    current = items[-1]
    items.pop(-1)
    while len(items) > 0:
        current = accumulator(items[-1], current)
        items.pop(-1)
    return current


def linear_function_creator(features: list[Var], weights: Iterable[Real], intercept: Real) -> Struct:
    x = zip(features[: -1], weights)
    x = filter(lambda fw: not is_zero(fw[1]), x)
    x = map(lambda fw: struct('*', fw[1], fw[0]), x)
    x = foldr(lambda a, b: struct('+', a, b), x)
    x = intercept if x is None else struct('+', intercept, x)
    return struct('is', features[-1], x)


def terms_to_intervals(terms) -> dict[str, Value]:
    return {term.args[0].name: term_to_value(term) for term in terms if term.arity > 0}


def terms_to_minimal_intervals(terms) -> dict[Var, tuple[Value, str]]:
    intervals = [(term.args[0], term_to_value(term), term.functor) for term in terms if term.arity > 0]
    intervals_per_variable: dict[Var: list[tuple[Value, str]]] = {}
    for v, value, functor in intervals:
        if v in intervals_per_variable.keys():
            intervals_per_variable[v].append((value, functor))
        else:
            intervals_per_variable[v] = [(value, functor)]

    result: dict[Var, tuple[Value, str]] = {}
    for v, values in intervals_per_variable.items():
        if len(values) == 1:
            result[v] = values[0]
        else:
            minimal_interval = None
            for value, functor in values:
                minimal_interval = value * minimal_interval
            result[v] = (minimal_interval, create_functor(minimal_interval))
    return result


def is_subset(first_sets: dict[str, Value], second_sets: dict[str, Value]) -> bool:
    def is_inside(first_value: Value, second_value: Value) -> bool:
        return first_value in second_value

    return all(is_inside(v, second_sets[k]) if k in second_sets.keys() else True for k, v in first_sets.items())


def prune(theory: Theory) -> Theory:
    """
    Prune unnecessary clauses from a logic theory T.
    This is a work in progress because it is not a trivial problem.
    Firstly, we remove redundant clauses that are easy to detect.
    Then, we will continue with more complicated ones.

    1. A clause c1 in T is removable when it is followed or preceded by a clause c2 that:
        - has the same head of c1;
        - c2 is always true if c1 is true (c1 -> c2).
    Examples:
        c1(A, B, C, D, positive) :- A =< 1, B > 2, C = 0.
        c2(A, B, C, D, positive) :- A =< 1, B > 2.
        In this case c1 is redundant.
    But also:
        c1(A, B, C, D, positive) :- A =< 1, B > 2, C = 0.
        c2(A, B, C, D, positive) :- A =< 1.3, B > 1.8.
        c1 can be removed.
    And also:
        c2(A, B, C, D, positive) :- A =< 1.3, B > 1.8.
        c1(A, B, C, D, positive) :- A =< 1, B > 2, C = 0.
        c1 can be removed.

    :param theory: the logic theory
    :return: a new simplified theory
    """

    def is_clause_included(clause, other):
        terms_clause = clause.body.unfolded if clause.body.is_recursive else [clause.body]
        terms_other = other.body.unfolded if other.body.is_recursive else [other.body]
        if clause != other and clause.head.args[-1] == other.head.args[-1]:
            set_clause = terms_to_intervals(terms_clause)
            set_other = terms_to_intervals(terms_other)
            return is_subset(set_clause, set_other) and set(set_other.keys()).issubset(set(set_clause.keys()))
        else:
            return False

    def attack(clause, clauses, index):
        after = index < len(clauses) - 1 and clause.body_size > 0 and is_clause_included(clause, clauses[index + 1])
        before = index > 0 and clause.body_size > 0 and is_clause_included(clause, clauses[index - 1])
        return after or before

    new_theory = mutable_theory()
    theory_copy = mutable_theory(theory)
    clauses_copy = theory_copy.clauses
    for i, clause in enumerate(theory.clauses):
        if not attack(clause, clauses_copy, i):
            new_theory.assertZ(clause)
    return new_theory


def simplify(theory: Theory) -> Theory:

    def simplify_clause(c: Clause) -> Clause:

        terms = c.body.unfolded if c.body.is_recursive else [c.body]
        minimal_intervals = terms_to_minimal_intervals(terms)
        minimal_terms = [_create_term(v, value[0], value[1]) for v, value in minimal_intervals.items()]
        return clause(c.head, minimal_terms) if c.body.arity > 0 else c

    new_theory = mutable_theory()
    for old_clause in theory.clauses:
        new_clause = simplify_clause(old_clause)
        new_theory.assertZ(new_clause)
    return new_theory


def data_to_struct(data: pd.Series):
    head = data.keys()[-1]
    terms = [numeric(item) for item in data.values[:-1]]
    terms.append(var('X'))
    return struct(head, terms)
