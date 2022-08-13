from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from psyke.gui.model import PREDICTORS, FIXED_PREDICTOR_PARAMS


class Model:

    def __init__(self):
        self.task = 'Classification'
        self.dataset = None
        self.data = None
        self.train = None
        self.test = None
        self.predictor_name = None
        self.predictor = None
        self.predictor_params = {}
        self.extractor_name = None
        self.extractor = None
        self.extractor_params = {}

    def select_task(self, task):
        self.task = task

    def select_dataset(self, dataset):
        self.dataset = dataset

    def select_predictor(self, predictor):
        self.predictor_name = predictor

    def select_extractor(self, extractor):
        self.extractor_name = extractor

    def reset_dataset(self):
        self.data = None
        self.train = None
        self.test = None

    def reset_predictor(self):
        self.predictor_name = None
        self.predictor = None
        self.predictor_params = {}

    def reset_extractor(self):
        self.extractor_name = None
        self.extractor = None
        self.extractor_params = {}

    def load_dataset(self):
        print(f'Loading {self.dataset}... ', end='')
        if self.dataset == 'Iris':
            x, y = load_iris(return_X_y=True, as_frame=True)
            self.data = (x, y.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}))
        elif self.dataset == 'Wine':
            self.data = load_wine(return_X_y=True, as_frame=True)
        elif self.dataset == "House":
            self.data = fetch_california_housing(return_X_y=True, as_frame=True)
        else:
            raise NotImplementedError
        print('Done')
        self.data = self.data[0].join(self.data[1])

    def train_predictor(self, params):
        print(f'Training {self.predictor_name}... ', end='')
        get_param = lambda param: self.get_param(params, param)
        if self.predictor_name == 'K-NN':
            self.predictor = KNeighborsClassifier() if self.task == 'Classification' else KNeighborsRegressor()
            self.predictor.n_neighbors = int(get_param('K'))
            self.predictor_params['K'] = self.predictor.n_neighbors
        elif self.predictor_name == 'DT':
            self.predictor = DecisionTreeClassifier() if self.task == 'Classification' else DecisionTreeRegressor()
            self.predictor.max_depth = int(get_param('Max depth'))
            self.predictor.max_leaf_nodes = int(get_param('Max leaves'))
            self.predictor_params['Max depth'] = self.predictor.max_depth
            self.predictor_params['Max leaves'] = self.predictor.max_leaf_nodes
        else:
            ...
        self.predictor_params['Test set'] = float(get_param('Test set'))
        self.predictor_params['Split seed'] = int(get_param('Split seed'))
        self.train, self.test = train_test_split(self.data, test_size=self.predictor_params['Test set'],
                                                 random_state=self.predictor_params['Split seed'])
        self.predictor.fit(self.train.iloc[:, :-1], self.train.iloc[:, -1])
        print('Done')

    def get_param(self, params, param):
        value = params.get(param)
        if value is not None:
            return value
        value = PREDICTORS[self.predictor_name][1].get(param)
        return FIXED_PREDICTOR_PARAMS[param][0] if value is None else value[0]
