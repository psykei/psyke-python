import pandas as pd
from tuprolog.core import Var, Struct, struct, real, atom, var, numeric, logic_list
from psyke.schema.value import Value, LessThan, GreaterThan, Between
from psyke.schema.value import Constant
from psyke.schema.discrete_feature import DiscreteFeature

precision = 4


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
        return struct(functor, v, real(round(constraint.value, precision)))
    if isinstance(constraint, GreaterThan):
        return struct(functor, v, real(round(constraint.value, precision)))
    if isinstance(constraint, Between):
        return struct(functor, v,
                      logic_list(real(round(constraint.lower, precision)), real(round(constraint.upper, precision))))
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
        value = round(float(output), precision)
        variables.append(numeric(value))
        return struct(functor, variables)
