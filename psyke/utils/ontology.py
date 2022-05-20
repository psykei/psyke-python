import numbers
from owlready2 import *
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from tuprolog.core import Struct
from tuprolog.theory import Theory


def dataframe_to_ontology(dataframe: pd.DataFrame) -> owlready2.Ontology:
    name = dataframe.columns[-1]
    ontology = get_ontology(f"{name[0].lower() + name[1:]}.rdf")
    with ontology:
        for column in dataframe.columns:
            if is_numeric_dtype(dataframe[column]):
                t = float
            elif is_string_dtype(dataframe[column]):
                t = str
            else:
                raise TypeError
            type(column, (type(name.capitalize(), (Thing,), {}) >> t, FunctionalProperty), {})
    return ontology


def individuals_from_dataframe(ontology: owlready2.Ontology, dataframe: pd.DataFrame) -> list:
    ret = []
    for _, row in dataframe.iterrows():
        string_row = "list(ontology.classes())[0]("
        for name, value in zip(row.index, row):
            if isinstance(value, numbers.Number):
                string_row += f"{name}={value},"
            elif isinstance(value, str):
                string_row += f"{name}='{value}',"
            else:
                raise TypeError
        evaluated_row = eval(string_row[:-1] + ")")
        ret.append(evaluated_row)
    return ret


def rules_from_struct(struct: Struct, rules: list[str] = None) -> list[str]:
    # equal, notEqual, lessThan, lessThanOrEqual, greaterThan, greaterThanOrEqual
    if rules is None:
        rules = [""]
    if struct.is_recursive:
        return rules_from_struct(struct.args[0], rules_from_struct(struct.args[1], rules))
    if struct.functor == "=<":
        rules = [r + f"lessThanOrEqual(?{struct.args[0].name},{struct.args[1].value})," for r in rules]
    elif struct.functor == '>':
        rules = [r + f"greaterThan(?{struct.args[0].name},{struct.args[1].value})," for r in rules]
    elif struct.functor == 'in':
        rules = [r +
                 f"greaterThan(?{struct.args[0].name},{struct.args[1][0].value})," +
                 f"lessThanOrEqual(?{struct.args[0].name},{struct.args[1][1][0].value})," for r in rules]
    elif struct.functor == 'not_in':
        rules = [r + f"lessThanOrEqual(?{struct.args[0].name},{struct.args[1][0].value})," for r in rules] +\
                [r + f"greaterThan(?{struct.args[0].name},{struct.args[1][1][0].value})," for r in rules]
    else:
        raise TypeError
    return rules


def rules_from_theory(ontology: owlready2.Ontology, theory: Theory, dataframe: pd.DataFrame) -> owlready2.Ontology:
    name = ontology.name
    lower_name = name[0].lower() + name[1:]
    with ontology:
        for clause in theory:
            string_rule = list(ontology.classes())[0].name + f"(?{lower_name}),"
            for arg in clause.head.args[:-1]:
                string_rule += f"{arg.name}(?{lower_name},?{arg.name}),"
            if clause.body.arity > 0:
                body = rules_from_struct(clause.body)
                rules = [string_rule + b for b in body]
            else:
                rules = [string_rule]
            for rule in rules:
                string_rule = f"{rule[:-1]}->{lower_name}(?{lower_name},'{clause.head.args[-1]}')"
                Imp().set_as_rule(string_rule)
    return ontology
