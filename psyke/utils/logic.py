import pandas as pd
from tuprolog.core import Var, Struct, struct, real, atom, var, numeric, logic_list, Clause
from tuprolog.core.operators import DEFAULT_OPERATORS, operator, operator_set, XFX
from tuprolog.core.formatters import TermFormatter
from tuprolog.theory import Theory
from psyke.schema import Value, LessThan, GreaterThan, Between, Constant
from psyke import DiscreteFeature
from psyke.utils import get_int_precision

PRECISION: int = get_int_precision()

OP_IN = operator('in', XFX, 700)

OP_NOT = operator('not_in', XFX, 700)

RULES_OPERATORS = DEFAULT_OPERATORS + operator_set(OP_IN, OP_NOT)


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
        return struct(functor, v, atom(str(Constant(constraint).value)))


def create_variable_list(features: list[DiscreteFeature], dataset: pd.DataFrame = None) -> dict[str, Var]:
    values = {feature.name: var(feature.name) for feature in sorted(features, key=lambda x: x.name)}\
        if len(features) > 0 else {name: var(name) for name in sorted(dataset.columns[:-1])}
    return values


def create_head(functor: str, variables: list[Var], output) -> Struct:
    if isinstance(output, str):
        variables.append(atom(output))
        return struct(functor, variables)
    else:
        value = round(float(output), PRECISION)
        variables.append(numeric(value))
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
