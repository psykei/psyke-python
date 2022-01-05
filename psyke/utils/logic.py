from typing import Iterable

import pandas as pd
from tuprolog.core import Var, Struct, Real, Term, Integer, Numeric
from tuprolog.core import struct, real, atom, var, numeric, logic_list, Clause
from tuprolog.core.operators import DEFAULT_OPERATORS, operator, operator_set, XFX
from tuprolog.core.formatters import TermFormatter
from tuprolog.core.visitors import AbstractTermVisitor
from tuprolog.theory import Theory
from psyke.schema import Value, LessThan, GreaterThan, Between, Constant
from psyke import DiscreteFeature
from psyke.utils import get_int_precision

PRECISION: int = get_int_precision()

OP_IN = operator('in', XFX, 700)

OP_NOT = operator('not_in', XFX, 700)

RULES_OPERATORS = DEFAULT_OPERATORS + operator_set(OP_IN, OP_NOT)


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
            return numeric(term.getValue().unaryMinus())
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


def create_functor(constraint: Value, positive: bool) -> str:
    if isinstance(constraint, LessThan):
        return '=<' if positive else '>'
    if isinstance(constraint, GreaterThan):
        return '>' if positive else '=<'
    if isinstance(constraint, Between):
        return 'in' if positive else 'not_in'
    if isinstance(constraint, Constant):
        return '=' if positive else '\\='


def create_term(v: Var, constraint: Value, positive: bool = True) -> Struct:
    if v is None:
        raise Exception('IllegalArgumentException: None variable')
    functor = create_functor(constraint, positive)
    if isinstance(constraint, LessThan):
        return struct(functor, v, real(round(constraint.value, PRECISION)))
    if isinstance(constraint, GreaterThan):
        return struct(functor, v, real(round(constraint.value, PRECISION)))
    if isinstance(constraint, Between):
        return struct(functor, v,
                      logic_list(real(round(constraint.lower, PRECISION)), real(round(constraint.upper, PRECISION))))
    if isinstance(constraint, Constant):
        return struct(functor, v, atom(str(constraint.value)))


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


def pretty_clause(clause: Clause) -> str:
    formatter = TermFormatter.prettyExpressions(True, RULES_OPERATORS)
    if clause.is_fact:
        return str(formatter.format(clause.head))
    elif clause.is_directive:
        return ":- " + formatter.format(clause.body)
    else:
        head = str(formatter.format(clause.head))
        body = str(formatter.format(clause.body))
        return f"{head} :-\n    {body}"


def pretty_theory(theory: Theory) -> str:
    if len(str(theory)) == 0:
        return ""
    else:
        clause = [pretty_clause(clause) for clause in theory]
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
