from tuprolog.core import rule, struct, logic_list, scope

from psyke.extractor import Extractor


def get_extractor(extractor_type: str, parameters: dict):
    if extractor_type == 'ITER':
        return Extractor.iter(**parameters)
    else:
        raise NotImplementedError(extractor_type + ' not implemented yet.')


def get_in_rule():
    local_scope = scope()
    return rule(
        struct('in', local_scope.var('X'), logic_list(local_scope.var('H'), local_scope.var('T'))),
        [
            struct('=<', local_scope.var('X'), local_scope.var('T')),
            struct('=<', local_scope.var('H'), local_scope.var('X'))
        ]
    )
