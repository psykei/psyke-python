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
    ['Bank marketing', [TASKS[1]]],
    ['Custom', TASKS]
]

FIXED_PREDICTOR_PARAMS = {
    'Test set': (0.5, 'float', None),
    'Split seed': (0, 'int', None)
}

PREDICTORS = {
    'K-NN': [TASKS, {'K': (3, 'int', None)}],
    'DT': [TASKS, {'Max depth': (3, 'int', None), 'Max leaves': (3, 'int', None)}],
    'RF': [TASKS, {'N estimators': (50, 'int', None), 'Max depth': (3, 'int', None), 'Max leaves': (100, 'int', None)}],
    'LR': [[TASKS[1]], {}],
    'SVM': [TASKS, {'Regularization': (1.0, 'float', None),
                    'Kernel': ('RBF', ['Linear', 'Poly', 'RBF', 'Sigmoid'], None),
                    'Epsilon': (0.1, 'float', TASKS[1])}]
}

MAX_DEPTH = (3, 'int', None)
THRESHOLD = (0.1, 'float', None)
MIN_EXAMPLES = (100, 'int', None)

EXTRACTORS = {
    'REAL': [[TASKS[0]], {}],
    'Trepan': [[TASKS[0]], {'Max depth': MAX_DEPTH, 'Min examples': (0, 'int', None)}],
    'CART': [TASKS, {'Max depth': MAX_DEPTH, 'Max leaves': (3, 'int', None), 'Simplify': (True, 'bool', None)}],
    'Iter': [TASKS, {'Min examples': MIN_EXAMPLES, 'Threshold': THRESHOLD, 'Max iterations': (600, 'int', None),
                     'N points': (1, 'int', None), 'Min update': (0.05, 'float', None),
                     'Fill gaps': (True, 'bool', None)}],
    'GridEx': [TASKS, {'Max depth': MAX_DEPTH, 'Splits': (2, 'int', None), 'Adaptive': (True, 'bool', None),
                       'Adaptive threshold': (0.8, 'float', None), 'Min examples': MIN_EXAMPLES,
                       'Threshold': THRESHOLD}],
    'GridREx': [[TASKS[1]], {'Max depth': MAX_DEPTH, 'Splits': (2, 'int', None), 'Adaptive': (True, 'bool', None),
                             'Adaptive threshold': (0.8, 'float', None), 'Min examples': MIN_EXAMPLES,
                             'Threshold': THRESHOLD}],
    'CReEPy': [TASKS, {'Max depth': MAX_DEPTH, 'Threshold': THRESHOLD, 'Max components': (10, 'int', None),
                       'Feat threshold': (0.8, 'float', None), 'Constant output': (True, 'bool', TASKS[1])}],
    'ORCHiD': [TASKS, {'Max depth': MAX_DEPTH, 'Threshold': THRESHOLD, 'Max components': (10, 'int', None),
                       'Feat threshold': (0.8, 'float', None), 'Constant output': (True, 'bool', TASKS[1])}]
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
    if isinstance(params[key][1], list):
        if value not in params[key][1]:
            raise ValueError
        return value
    raise NotImplementedError
