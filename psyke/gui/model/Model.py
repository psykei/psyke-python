import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from psyke import Extractor
from psyke.extraction.hypercubic import Grid, FixedStrategy, FeatureRanker
from psyke.gui.model import PREDICTORS, FIXED_PREDICTOR_PARAMS, EXTRACTORS, cast_param, DatasetError, SVMError, \
    PredictorError
from psyke.gui.model.plot import init_plot, plotSamples, create_grid, plot_regions
from psyke.utils import Target
from psyke.utils.dataframe import get_discrete_features_supervised, get_discrete_dataset


class Model:

    def __init__(self):
        self.task = 'Classification'
        self.preprocessing = None
        self.discretization = None
        self.dataset = None
        self.data = None
        self.pruned_data = None
        self.train = None
        self.test = None
        self.predictor_name = None
        self.predictor = None
        self.predictor_params = {}
        self.extractor_name = None
        self.extractor = None
        self.extractor_params = {}
        self.theory = None
        self.data_plot = None
        self.predictor_plot = None
        self.extractor_plot = None

    def select_task(self, task):
        self.task = task

    def select_preprocessing(self, action):
        self.preprocessing = action

    def select_dataset(self, dataset):
        self.dataset = dataset

    def select_predictor(self, predictor):
        self.predictor_name = predictor

    def select_extractor(self, extractor):
        self.extractor_name = extractor

    def reset_preprocessing(self):
        self.preprocessing = None
        self.discretization = None

    def reset_dataset(self, soft=False):
        if not soft:
            self.data = None
        self.pruned_data = None
        self.train = None
        self.test = None
        self.data_plot = None

    def reset_predictor(self):
        self.predictor_name = None
        self.predictor = None
        self.predictor_params = {}
        self.predictor_plot = None

    def reset_extractor(self):
        self.extractor_name = None
        self.extractor = None
        self.extractor_params = {}
        self.theory = None
        self.extractor_plot = None

    def load_dataset(self, ret=False):
        print(f'Loading {self.dataset}... ', end='')
        if self.dataset == 'Arti':
            import os
            print(os.getcwd())
            data = pd.read_csv('../../test/resources/datasets/arti.csv')
        else:
            if self.dataset == 'Iris':
                x, y = load_iris(return_X_y=True, as_frame=True)
                x.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
                y.name = 'iris'
                data = (x, y.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}))
            elif self.dataset == 'Wine':
                x, y = load_wine(return_X_y=True, as_frame=True)
                data = (x, y.apply(str))
            elif self.dataset == "House":
                data = fetch_california_housing(return_X_y=True, as_frame=True)
            else:
                raise DatasetError
            data = data[0].join(data[1])
        print('Done')
        if ret:
            return data
        if self.preprocessing == 'Discretize':
            self.discretization = get_discrete_features_supervised(data)
            self.data = get_discrete_dataset(data.iloc[:, :-1], self.discretization, False).join(data.iloc[:, -1])
        else:
            self.data = data

    def select_features(self, features):
        inputs = [k for k, v in features.items() if v == 'I']
        output = [k for k, v in features.items() if v == 'O'][0]
        if self.discretization is not None:
            inputs = [[list(discretization.admissible_values.keys()) for discretization in self.discretization
                       if discretization.name == variable] for variable in inputs]
            inputs = [item for sublist in inputs for item in sublist[0]]
        self.pruned_data = self.data[inputs].join(self.data[output])

    def train_predictor(self):
        self.read_predictor_param()

        print(f'Training {self.predictor_name}... ', end='')
        if self.predictor_name == 'K-NN':
            self.predictor = KNeighborsClassifier() if self.task == 'Classification' else KNeighborsRegressor()
            self.predictor.n_neighbors = self.predictor_params['K']
        elif self.predictor_name == 'LR':
            self.predictor = LinearRegression()
        elif self.predictor_name in ['DT', 'RF']:
            if self.predictor_name == 'DT':
                self.predictor = DecisionTreeClassifier() if self.task == 'Classification' else DecisionTreeRegressor()
            else:
                self.predictor = RandomForestClassifier() if self.task == 'Classification' else RandomForestRegressor()
                self.predictor.n_estimators = self.predictor_params['N estimators']
            self.predictor.max_depth = self.predictor_params['Max depth']
            self.predictor.max_leaf_nodes = self.predictor_params['Max leaves']
        elif self.predictor_name in ['SVC', 'SVR']:
            if self.predictor_name == 'SVC':
                self.predictor = SVC()
            else:
                self.predictor = SVR()
                self.predictor.epsilon = self.predictor_params['Epsilon']
            self.predictor.C = self.predictor_params['Regularization']
            if self.predictor_params['Kernel'].lower() in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
                self.predictor.kernel = self.predictor_params['Kernel'].lower()
            else:
                raise SVMError
        else:
            raise PredictorError
        self.train, self.test = train_test_split(self.data if self.pruned_data is None else self.pruned_data,
                                                 test_size=self.predictor_params['Test set'],
                                                 random_state=self.predictor_params['Split seed'])
        self.predictor.fit(self.train.iloc[:, :-1], self.train.iloc[:, -1])
        print('Done')

    def train_extractor(self):
        # GRIDEX, GRIDREX -> strategy
        # target regression/constant
        self.read_extractor_param()

        print(f'Training {self.extractor_name}... ', end='')
        if self.extractor_name == 'REAL':
            self.extractor = Extractor.real(self.predictor, discretization=self.discretization)
        elif self.extractor_name == 'Trepan':
            self.extractor = Extractor.trepan(self.predictor, min_examples=self.extractor_params['Min examples'],
                                              max_depth=self.extractor_params['Max depth'],
                                              discretization=self.discretization)
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
        elif self.extractor_name in ['CReEPy', 'ORCHiD']:
            ranked = FeatureRanker(self.data.columns[:-1]).fit(self.predictor, self.data.iloc[:, :-1]).rankings()
            extractor = Extractor.creepy if self.extractor_name == 'CReEPy' else Extractor.orchid
            self.extractor = extractor(self.predictor, depth=self.extractor_params['Max depth'],
                                       error_threshold=self.extractor_params['Threshold'],
                                       ignore_threshold=self.extractor_params['Feat threshold'],
                                       gauss_components=self.extractor_params['Max components'],
                                       output=Target.CONSTANT, ranks=ranked)
        else:
            raise NotImplementedError

        self.theory = self.extractor.extract(self.train)
        print('Done')

    def plot(self, inputs, output):
        x = inputs[0]
        y = inputs[1] if len(inputs) > 1 else output
        z = output if len(inputs) > 1 else None

        #data = self.data if self.pruned_data is None else self.pruned_data
        data = self.load_dataset(True)

        init_plot(data[x], data[y], 'Data set')
        plotSamples(data[x], data[y], data[z if z is not None else y])
        self.data_plot = plt.gcf()
        plt.close()

        if self.predictor is None:
            return

        grid = create_grid(x, y, data)
        discrete_grid = grid if self.discretization is None else get_discrete_dataset(grid, self.discretization, False)

        for model, name in zip([self.predictor, self.extractor], [self.predictor_name, self.extractor_name]):
            if model is None:
                break
            init_plot(data[x], data[y], name)
            grid_data = pd.concat(
                [grid, pd.DataFrame(model.predict(discrete_grid), columns=[data.columns[-1]])], axis=1
            )
            grid_data = grid_data[grid_data.iloc[:, -1].notna()]
            grouped = grid_data.groupby([x, y])[grid_data.columns[-1]]
            outputs = grouped.agg(pd.Series.mode) if isinstance(grid_data.iloc[0, -1], str) else grouped.mean()
            grid_data = grid_data.groupby([x, y])[grid_data.columns[:-1]].mean().join(outputs)
            if z is not None:
                plot_regions(grid_data[x].values, grid_data[y].values, grid_data[z].values)
            plotSamples(data[x], data[y], data[z if z is not None else y])
            if isinstance(model, Extractor):
                self.extractor_plot = plt.gcf()
            else:
                self.predictor_plot = plt.gcf()
            plt.close()

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
