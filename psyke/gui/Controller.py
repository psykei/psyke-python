from sklearn.datasets import load_iris, load_wine, fetch_california_housing

from psyke.gui.model import TASKS


class Controller:

    def __init__(self):
        self.data_panel = None
        self.predictor_panel = None
        self.data = (None, None)
        self.predictor = None

    def set_data_panel(self, data_panel):
        self.data_panel = data_panel

    def set_predictor_panel(self, predictor_panel):
        self.predictor_panel = predictor_panel

    def select_task(self):
        self.data_panel.init_datasets()
        self.predictor_panel.init_predictors()
        self.reset_dataset()

    def reset_dataset(self):
        self.data = (None, None)
        self.reset_predictor()
        self.data_panel.set_dataset_info()

    def reset_predictor(self):
        self.predictor = None

    def load_dataset(self, dataset):
        print(f'Loading {dataset}... ', end='')
        if dataset == 'Iris':
            x, y = load_iris(return_X_y=True, as_frame=True)
            self.data = (x, y.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}))
        elif dataset == 'Wine':
            self.data = load_wine(return_X_y=True, as_frame=True)
        elif dataset == "House":
            self.data = fetch_california_housing(return_X_y=True, as_frame=True)
        else:
            ...
        print('Done')
        self.data_panel.set_dataset_info()

    def train_predictor(self, predictor):
        pass
