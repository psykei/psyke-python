from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from psyke.gui.model import TASKS


class Controller:

    def __init__(self):
        self.data_panel = None
        self.predictor_panel = None
        self.data = None
        self.train = None
        self.test = None
        self.predictor = None
        self.predictor_params = {}

    def set_data_panel(self, data_panel):
        self.data_panel = data_panel

    def set_predictor_panel(self, predictor_panel):
        self.predictor_panel = predictor_panel

    def select_task(self):
        self.data_panel.init_datasets()
        self.predictor_panel.init_predictors()
        self.reset_dataset()

    def reset_dataset(self):
        self.data = None
        self.train = None
        self.test = None
        self.reset_predictor()
        self.data_panel.set_dataset_info()

    def reset_predictor(self):
        self.predictor = None
        self.predictor_params = {}

    def load_dataset(self):
        dataset = self.data_panel.dataset
        print(f'Loading {dataset}... ', end='')
        if dataset == 'Iris':
            x, y = load_iris(return_X_y=True, as_frame=True)
            self.data = (x, y.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}))
        elif dataset == 'Wine':
            self.data = load_wine(return_X_y=True, as_frame=True)
        elif dataset == "House":
            self.data = fetch_california_housing(return_X_y=True, as_frame=True)
        else:
            raise NotImplementedError
        print('Done')
        self.data = self.data[0].join(self.data[1])
        self.data_panel.set_dataset_info()
        self.predictor_panel.enable_predictors()

    def train_predictor(self):
        predictor = self.predictor_panel.predictor
        task = self.data_panel.task
        print(f'Training {predictor}... ', end='')
        if predictor == 'K-NN':
            self.predictor = KNeighborsClassifier() if task == 'Classification' else KNeighborsRegressor()
            self.predictor.n_neighbors = int(self.predictor_panel.get_param('K'))
            self.predictor_params['K'] = self.predictor.n_neighbors
        elif predictor == 'DT':
            self.predictor = DecisionTreeClassifier() if task == 'Classification' else DecisionTreeRegressor()
            self.predictor.max_depth = int(self.predictor_panel.get_param('Max depth'))
            self.predictor.max_leaf_nodes = int(self.predictor_panel.get_param('Max leaves'))
            self.predictor_params['Max depth'] = self.predictor.max_depth
            self.predictor_params['Max leaves'] = self.predictor.max_leaf_nodes
        else:
            ...
        self.predictor_params['Test set'] = float(self.predictor_panel.get_param('Test set'))
        self.predictor_params['Split seed'] = int(self.predictor_panel.get_param('Split seed'))
        self.train, self.test = train_test_split(self.data, test_size=self.predictor_params['Test set'],
                                                 random_state=self.predictor_params['Split seed'])
        self.predictor.fit(self.train.iloc[:, :-1], self.train.iloc[:, -1])
        print('Done')
        self.predictor_panel.set_predictor_info()
