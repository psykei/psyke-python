import re
from tuprolog.core import scope, struct, real, logic_list, rule
from tuprolog.theory import Theory, theory
from psyke.utils.logic_utils import create_head


def _get_items_between(term: str, inner_scope):
    variable = term.split(',')[0].split('_')[0].strip()
    limits = term.split('[')[1]
    x, y = limits.split(',')
    x, y = x.strip(), y.split(']')[0].strip()
    return inner_scope.var(variable), real(x), real(y)


def _get_items(term: str, inner_scope):
    items = term.split('(')[1].strip()
    variable, x = items.split(',')
    x = x.replace(')', '').strips()
    return inner_scope(variable), real(x)


def _parse_head(head: str, inner_scope):
    functor, tail = head.strip().split('(')
    variables_and_output = tail.split(',')
    variables_and_output = [item.strip().replace(')', '') for item in variables_and_output]
    variables = [inner_scope.var(variable.split('_')[0]) for variable in variables_and_output[:-1]]
    output = variables_and_output[-1]
    output = float(output) if re.match(r'^-?\d+(?:\.\d+)$', output) else output
    return create_head(functor, variables, output)


def _parse_body(body: str, inner_scope):
    structs = []
    for term in body.split('),'):
        term = term.strip()
        term = term[1:] if term[0] == '(' else term
        if 'in(' in term[0:len('in(')]:
            variable, x, y = _get_items_between(term[len('in('):], inner_scope)
            structs.append(struct('in', variable, logic_list(x, y)))
        elif 'not_in(' in term[0:len('not_in(')]:
            variable, x, y = _get_items_between(term[len('not_in('):], inner_scope)
            structs.append(struct('not_in', variable, logic_list(x, y)))
        elif "'=<'(" in term[0:len("'=<'(")]:
            variable, x = _get_items(term, inner_scope)
            structs.append(struct('=<', variable, x))
        elif "'>'(" in term[0:len("'>(")]:
            variable, x = _get_items(term, inner_scope)
            structs.append(struct('>', variable, x))
    return structs


def parse_theory(str_theory: str) -> Theory:
    inner_scope = scope()
    str_rules = str_theory.split(').')
    rules = []
    for str_rule in str_rules:
        head, body = str_rule.split(':-')
        head = _parse_head(head, inner_scope)
        structs = _parse_body(body, inner_scope)
        rules.append(rule(head, structs))
    return theory(rules)
