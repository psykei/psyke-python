from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from psyke import Extractor
from psyke.extraction.hypercubic import Grid, FixedStrategy
from psyke.gui.model import PREDICTORS, FIXED_PREDICTOR_PARAMS, EXTRACTORS, cast_param
from psyke.utils import Target


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

    def train_predictor(self):
        self.read_predictor_param()

        print(f'Training {self.predictor_name}... ', end='')
        if self.predictor_name == 'K-NN':
            self.predictor = KNeighborsClassifier() if self.task == 'Classification' else KNeighborsRegressor()
            self.predictor.n_neighbors = self.predictor_params['K']
        elif self.predictor_name == 'DT':
            self.predictor = DecisionTreeClassifier() if self.task == 'Classification' else DecisionTreeRegressor()
            self.predictor.max_depth = self.predictor_params['Max depth']
            self.predictor.max_leaf_nodes = self.predictor_params['Max leaves']
        else:
            ...
        self.train, self.test = train_test_split(self.data, test_size=self.predictor_params['Test set'],
                                                 random_state=self.predictor_params['Split seed'])
        self.predictor.fit(self.train.iloc[:, :-1], self.train.iloc[:, -1])
        print('Done')

    def train_extractor(self):
        # CREEPY, ORCHID, ITER -> output
        # GRIDEX, GRIDREX -> strategy

        self.read_extractor_param()

        print(f'Training {self.extractor_name}... ', end='')
        if self.extractor_name == 'REAL':
            self.extractor = Extractor.real(self.predictor)
        elif self.extractor_name == 'Trepan':
            self.extractor = Extractor.trepan(self.predictor, min_examples=self.extractor_params['Min examples'],
                                              max_depth=self.extractor_params['Max depth'])
        elif self.extractor_name == 'CART':
            self.extractor = Extractor.cart(self.predictor, max_depth=self.extractor_params['Max depth'],
                                            max_leaves=self.extractor_params['Max leaves'])
        elif self.extractor_name == 'Iter':
            self.extractor = Extractor.iter(self.predictor, threshold=self.extractor_params['Threshold'],
                                            min_examples=self.extractor_params['Min examples'],
                                            min_update=self.extractor_params['Min update'],
                                            n_points=self.extractor_params['N points'],
                                            max_iterations=self.extractor_params['Max iterations'])
        elif self.extractor_name == 'GridEx':
            self.extractor = Extractor.gridex(self.predictor, threshold=self.extractor_params['Threshold'],
                                              min_examples=self.extractor_params['Min examples'],
                                              grid=Grid(self.extractor_params['Max depth'],
                                                        FixedStrategy(self.extractor_params['Splits'])))
        elif self.extractor_name == 'GridREx':
            self.extractor = Extractor.gridrex(self.predictor, threshold=self.extractor_params['Threshold'],
                                               min_examples=self.extractor_params['Min examples'],
                                               grid=Grid(self.extractor_params['Max depth'],
                                                         FixedStrategy(self.extractor_params['Splits'])))
        elif self.extractor_name == 'CReEPy':
            self.extractor = Extractor.creepy(self.predictor, depth=self.extractor_params['Max depth'],
                                              error_threshold=self.extractor_params['Threshold'],
                                              ignore_threshold=self.extractor_params['Feat threshold'],
                                              gauss_components=self.extractor_params['Max components'],
                                              output=Target.CONSTANT)
        elif self.extractor_name == 'ORCHiD':
            self.extractor = Extractor.orchid(self.predictor, depth=self.extractor_params['Max depth'],
                                              error_threshold=self.extractor_params['Threshold'],
                                              ignore_threshold=self.extractor_params['Feat threshold'],
                                              gauss_components=self.extractor_params['Max components'],
                                              output=Target.CONSTANT)
        else:
            raise NotImplementedError

        self.extractor.extract(self.train)
        print('Done')

    def set_predictor_param(self, key, value):
        self.predictor_params[key] = \
            cast_param(dict(FIXED_PREDICTOR_PARAMS, **PREDICTORS[self.predictor_name][1]), key, value)

    def set_extractor_param(self, key, value):
        self.extractor_params[key] = cast_param(EXTRACTORS[self.extractor_name][1], key, value)

    def read_predictor_param(self):
        all_params = dict(FIXED_PREDICTOR_PARAMS, **PREDICTORS[self.predictor_name][1])
        for name, (default, _) in all_params.items():
            if name not in self.predictor_params.keys():
                self.predictor_params[name] = default

    def read_extractor_param(self):
        params = EXTRACTORS[self.extractor_name][1]
        for name, (default, _) in params.items():
            if name not in self.extractor_params.keys():
                self.extractor_params[name] = default
