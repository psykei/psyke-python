from psyke.gui.model import TASKS
from matplotlib import colors, pyplot as plt

DATASET_MESSAGE = 'Select dataset'

INFO_DATASET_PREFIX = 'Dataset info:\n\n'

NO_DATASET = 'No dataset selected'

INFO_DATASET_MESSAGE = INFO_DATASET_PREFIX + NO_DATASET + '\n\n'

PREDICTOR_MESSAGE = 'Select predictor'

INFO_PREDICTOR_PREFIX = 'Predictor info:\n'

INFO_PREDICTOR_MESSAGE = f'\n\n{INFO_PREDICTOR_PREFIX}\nNo predictor trained\n\n\n\n\n\n'

PREDICTOR_PERFORMANCE_PREFIX = 'Predictor performance:\n'

EXTRACTOR_MESSAGE = 'Select extractor'

INFO_EXTRACTOR_PREFIX = 'Extractor info:\n'

INFO_EXTRACTOR_MESSAGE = f'\n\n{INFO_EXTRACTOR_PREFIX}\nNo extractor trained\n\n\n\n\n\n'

THEORY_PREFIX = 'Extracted theory'

THEORY_MESSAGE = 'No extractor trained\n\n\n'

EXTRACTOR_PERFORMANCE_PREFIX = 'Extractor performance:\n'

THEORY_ERROR_MESSAGE = {
    'amount': 'Too many extracted rules\n\n\n',
    'length': 'Too long clauses\n\n\n',
}

MARKERS = [
    ['x', '*', '+', 'o', 's', '^', 'D'],
    ['o', 's', '^', 'x', '*', '+', 'D'],
    ['s', '^', 'D', 'x', '*', '+', 'o']
]

COLORS = [
    ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:olive', 'tab:pink'],
    ['tab:orange', 'tab:purple', 'tab:olive', 'tab:blue', 'tab:red', 'tab:green', 'tab:pink'],
    ['tab:purple', 'tab:olive', 'tab:pink', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
]

NAMES = {
    TASKS[0]: ['brg', 'opo', 'pop'],
    TASKS[1]: ['brg', 'rainbow', 'jet', 'turbo', 'cool', 'autumn', 'summer', 'spring', 'winter', 'plasma', 'viridis']
}

COLOR_MAPS = {
    TASKS[0]: [(name, colors.ListedColormap(cols)) for name, cols in zip(NAMES[TASKS[0]], COLORS)],
    TASKS[1]: [(name, plt.get_cmap(name)) for name in NAMES[TASKS[1]]]
}
