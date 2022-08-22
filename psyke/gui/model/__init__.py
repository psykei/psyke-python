class DatasetError(Exception):
    def __init__(self, message: str = 'Dataset not found'):
        self.message = message
        super().__init__(self.message)


class PredictorError(Exception):
    def __init__(self, message: str = 'Predictor not found'):
        self.message = message
        super().__init__(self.message)


class ExtractorError(Exception):
    def __init__(self, message: str = 'Extractor not found'):
        self.message = message
        super().__init__(self.message)


class SVMError(Exception):
    def __init__(self, message: str = "Kernel must be one of ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']"):
        self.message = message
        super().__init__(self.message)


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
    'RF': [TASKS, {'N estimators': (50, 'int'), 'Max depth': (3, 'int'), 'Max leaves': (100, 'int')}],
    'LR': [[TASKS[1]], {}],
    'SVC': [TASKS[0], {'Regularization': (1.0, 'float'), 'Kernel': ('RBF', None)}],
    'SVR': [TASKS[1], {'Regularization': (1.0, 'float'), 'Kernel': ('RBF', None), 'Epsilon': (0.1, 'float')}]
}

EXTRACTORS = {
    'REAL': [[TASKS[0]], {}],
    'Trepan': [[TASKS[0]], {'Max depth': (3, 'int'), 'Min examples': (0, 'int')}],
    'CART': [TASKS, {'Max depth': (5, 'int'), 'Max leaves': (5, 'int')}],
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
    if params[key][1] is None:
        return value
    if params[key][1] == 'int':
        return int(value)
    if params[key][1] == 'float':
        return float(value)
    if params[key][1] == 'bool':
        return bool(value)
    raise NotImplementedError
