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
    'Trepan': [[TASKS[0]], {}],
    'CART': [TASKS, {}],
    'Iter': [TASKS, {}],
    'GridEx': [TASKS, {}],
    'GridREx': [[TASKS[1]], {}],
    'CReEPy': [TASKS, {}],
    'ORCHiD': [TASKS, {}]
}
