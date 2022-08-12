TASKS = ['Classification', 'Regression']

DATASETS = [
    ['Iris', [TASKS[0]]],
    ['Wine', [TASKS[0]]],
    ['Arti', [TASKS[1]]],
    ['House', [TASKS[1]]],
    ['Custom', TASKS]
]

DATASET_MESSAGE = 'Select dataset'

INFO_DATASET_PREFIX = 'Dataset info:\n\n'

INFO_DATASET_MESSAGE = INFO_DATASET_PREFIX + 'No dataset selected\n\n'

PREDICTORS = {
    'K-NN': [TASKS, {'K': (3, 'int')}],
    'DT': [TASKS, {'Max depth': (3, 'int'), 'Max leaves': (3, 'int')}],
    'RF': [TASKS, {}],
    'LR': [[TASKS[1]], {}],
    'SVM': [TASKS, {}]
}

PREDICTOR_MESSAGE = 'Select predictor'

INFO_PREDICTOR_PREFIX = 'Predictor info:\n\n'

INFO_PREDICTOR_MESSAGE = INFO_PREDICTOR_PREFIX + 'No predictor trained\n\n\n\n\n\n\n\n\n\n\n'

FIXED_PREDICTOR_PARAMS = {
    'Test set': (0.5, 'float'),
    'Split seed': (0, 'int')
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

EXTRACTOR_MESSAGE = 'Select extractor'

INFO_EXTRACTOR_PREFIX = 'Extractor info:\n\n'

INFO_EXTRACTOR_MESSAGE = INFO_EXTRACTOR_PREFIX + 'No Extractor trained\n\n\n\n\n\n\n\n\n\n\n'
