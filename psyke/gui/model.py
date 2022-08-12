TASKS = ['Classification', 'Regression']

DATASETS = [
    ['Iris', [TASKS[0]]],
    ['Wine', [TASKS[0]]],
    ['Arti', [TASKS[1]]],
    ['House', [TASKS[1]]],
    ['Custom', TASKS]
]

DATASET_MESSAGE = 'Select dataset'

INFO_DATASET_MESSAGE = 'No dataset selected'

PREDICTORS = [
    ['K-NN', TASKS],
    ['DT', TASKS],
    ['RF', TASKS],
    ['LR', [TASKS[1]]],
    ['SVM', TASKS]
]

PREDICTOR_MESSAGE = 'Select predictor'