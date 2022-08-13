TASKS = ['Classification', 'Regression']

DATASETS = [
    ['Iris', [TASKS[0]]],
    ['Wine', [TASKS[0]]],
    ['Arti', [TASKS[1]]],
    ['House', [TASKS[1]]],
    ['Custom', TASKS]
]

FIXED_PREDICTOR_PARAMS = {
    'Test set': (0.5, 'float'),
    'Split seed': (0, 'int')
}

PREDICTORS = {
    'K-NN': [TASKS, {'K': (3, 'int')}],
    'DT': [TASKS, {'Max depth': (3, 'int'), 'Max leaves': (3, 'int')}],
    'RF': [TASKS, {}],
    'LR': [[TASKS[1]], {}],
    'SVM': [TASKS, {}]
}

EXTRACTORS = {
    'REAL': [[TASKS[0]], {}],
    'Trepan': [[TASKS[0]], {'Max depth': (3, 'int'), 'Min examples': (100, 'int')}],
    'CART': [TASKS, {'Max depth': (3, 'int'), 'Max leaves': (3, 'int')}],
    'Iter': [TASKS, {'Min examples': (100, 'int'), 'Threshold': (0.1, 'float'), 'Max iterations': (600, 'int'),
                     'N points': (1, 'int'), 'Min update': (0.05, 'float')}],
    'GridEx': [TASKS, {'Max depth': (3, 'int'), 'Splits': (2, 'int'),
                       'Min examples': (100, 'int'), 'Threshold': (0.1, 'float')}],
    'GridREx': [[TASKS[1]], {'Max depth': (3, 'int'), 'Splits': (2, 'int'),
                             'Min examples': (100, 'int'), 'Threshold': (0.1, 'float')}],
    'CReEPy': [TASKS, {'Max depth': (3, 'int'), 'Threshold': (0.1, 'float'),
                       'Max components': (10, 'int'), 'Feat threshold': (0.8, 'float')}],
    'ORCHiD': [TASKS, {'Max depth': (3, 'int'), 'Threshold': (0.1, 'float'),
                       'Max components': (10, 'int'), 'Feat threshold': (0.8, 'float')}]
}


def cast_param(params, key, value):
    if params[key][1] == 'int':
        return int(value)
    if params[key][1] == 'float':
        return float(value)
    if params[key][1] == 'bool':
        return bool(value)
    raise NotImplementedError
