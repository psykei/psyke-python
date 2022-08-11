import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen

from sklearn.datasets import fetch_california_housing, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from psyke import Extractor
from psyke.extraction.hypercubic import Grid
from psyke.extraction.hypercubic.strategy import FixedStrategy
from psyke.gui import TaskBox, DataBox, DataInfoBox, PredictorParameterBox, PredictorInfoBox, PredictorBox, \
    ExtractorBox, ExtractorParameterBox, ExtractorInfoBox, ExtractorPerformanceInfoBox, PredictorPerformanceInfoBox, \
    VerticalBoxLayout, DataManipulationBox
from psyke.utils import Target
from psyke.utils.logic import pretty_theory

import re

kivy.require('2.1.0')  # replace with your current kivy version !

Window.top = 50
Window.left = 10
Window.size = (1400, 750)

CLASSIFICATION_DATA = ['Iris', 'Wine']

REGRESSION_DATA = ['House', 'Artificial10']

MODELS = ['KNN', 'DT']

CLASSIFICATION_EXTRACTORS = ['REAL', 'Trepan', 'CART', 'GridEx', 'CReEPy', 'CREAM']

REGRESSION_EXTRACTORS = ['Iter', 'CART', 'GridEx', 'GridREx', 'CReEPy', 'CREAM']


class PSyKEApp1(App):

    def build(self):
        return PSyKEScreenManager()


class PSyKEScreenManager(ScreenManager):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.main_screen = Screen(name="main_screen")
        self.wait_screen = Screen(name="wait_screen")
        self.theory_screen = Screen(name="theory_screen")

        self.add_widget(self.main_screen)
        self.add_widget(self.wait_screen)
        self.add_widget(self.theory_screen)

        self.main_screen.add_widget(PSyKEMainScreen(self))
        wait_box = BoxLayout()
        wait_box.add_widget(Label(text='Please wait...'))
        self.wait_screen.add_widget(wait_box)
        self.concrete_theory_screen = PSyKETheoryScreen(self.main)
        self.theory_screen.add_widget(self.concrete_theory_screen)

    def wait(self):
        self.current = "wait_screen"

    def main(self, widget=None):
        self.current = "main_screen"

    def show_theory(self, theory):
        self.concrete_theory_screen.set_theory(theory)
        self.current = 'theory_screen'


class PSyKETheoryScreen(VerticalBoxLayout):

    def __init__(self, action, **kwargs):
        super().__init__(**kwargs)
        self.back_to_main = action

    def set_theory(self, theory):
        self.clear_widgets()
        text = pretty_theory(theory)
        self.add_widget(Label(text='Extracted theory:\n\n' + re.sub(r"\.\n", ".\n\n", text)))
        button = Button(text='Back', size_hint_y=None, height=30)
        button.bind(on_press=self.back_to_main)
        self.add_widget(button)
        # self.add_widget(Label())


class PSyKEMainScreen(GridLayout):

    def __init__(self, manager, **kwargs):
        super().__init__(**kwargs)
        self.cols = 4
        self.manager = manager
        self.data = None
        self.train = None
        self.test = None
        self.task = 'C'
        self.model_kind = 'KNN'
        self.model = None
        self.extractor_kind = {'C': 'REAL', 'R': 'Iter'}
        self.extractor = None
        self.theory = None
        self.taskBox = TaskBox(self)
        self.dataBox = DataBox(self, CLASSIFICATION_DATA, REGRESSION_DATA)
        self.extractorParamBox = ExtractorParameterBox(self)
        self.dataInfoBox = DataInfoBox()
        self.dataManipulationBox = DataManipulationBox(self)
        self.predictorBox = PredictorBox(self, MODELS)
        self.predictorParamBox = PredictorParameterBox(self)
        self.predictorInfoBox = PredictorInfoBox()
        self.predictorPerformanceInfoBox = PredictorPerformanceInfoBox()
        self.extractorBox = ExtractorBox(self, CLASSIFICATION_EXTRACTORS, REGRESSION_EXTRACTORS)
        self.extractorInfoBox = ExtractorInfoBox()
        self.extractorPerformanceInfoBox = ExtractorPerformanceInfoBox(action=self.show_theory)
        self.widgets = [
            self.taskBox, self.dataBox, self.dataInfoBox, self.dataManipulationBox,
            self.predictorBox, self.predictorParamBox, self.predictorInfoBox, self.predictorPerformanceInfoBox,
            self.extractorBox, self.extractorParamBox, self.extractorInfoBox, self.extractorPerformanceInfoBox
        ]

        for widget in self.widgets:
            self.add_widget(widget)

    def select_task(self, task):
        self.task = task
        self.dataBox.reset()
        self.dataManipulationBox.reset()
        # self.predictorBox.reset(MODELS)
        self.select_model()
        self.extractorBox.reset(self.task)
        self.select_extractor()
        self.predictorParamBox.disable()
        self.extractorParamBox.disable()

    def select_model(self):
        self.model_kind = self.predictorBox.model
        self.predictorParamBox.reset()
        self.extractorParamBox.disable()

    def select_extractor(self):
        self.extractor_kind[self.task] = self.extractorBox.extractor[self.task]
        self.extractorParamBox.reset()

    def select_dataset(self, widget):
        dataset = self.dataBox.dataset[self.task]
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
        self.data = self.data[0].join(self.data[1])
        self.dataInfoBox.reset(dataset, self.data)
        self.dataManipulationBox.reset()
        self.predictorParamBox.enable()

    def train_model(self, widget):
        print(f'Training {self.model_kind}... ', end='')
        params = self.predictorParamBox.params
        if self.task == 'C':
            if self.model_kind == 'KNN':
                self.model = KNeighborsClassifier(n_neighbors=params.get('neighbors', 5))
            elif self.model_kind == 'DT':
                self.model = DecisionTreeClassifier(max_depth=params.get('depth'), max_leaf_nodes=params.get('leaves'))
            else:
                ...
        else:
            if self.model_kind == 'KNN':
                self.model = KNeighborsRegressor(n_neighbors=params.get('neighbors', 5))
            elif self.model_kind == 'DT':
                self.model = DecisionTreeRegressor(max_depth=params.get('depth'), max_leaf_nodes=params.get('leaves'))
            else:
                ...
        self.train, self.test = train_test_split(self.data, test_size=params.get('test', .5))
        self.model.fit(self.train.iloc[:, :-1], self.train.iloc[:, -1])
        print('Done')
        self.predictorInfoBox.reset(self.model)
        self.predictorPerformanceInfoBox.reset(self.model, self.test, params.get('test', .5))
        self.extractorParamBox.enable()

    def train_extractor(self, widget):
        extractor = self.extractor_kind[self.task]
        print(f'Extracting rules from {self.model_kind} with {self.extractor_kind[self.task]}... ', end='')
        params = self.extractorParamBox.params
        print(params)
        if extractor == 'GridEx':
            self.extractor = Extractor.gridex(
                predictor=self.model, grid=Grid(params.get('depth', 2), FixedStrategy(params.get('splits', 2))),
                min_examples=params.get('examples', 200), threshold=params.get('threshold', 0.1)
            )
        elif extractor == 'GridREx':
            self.extractor = Extractor.gridrex(
                predictor=self.model, grid=Grid(params.get('depth', 2), FixedStrategy(params.get('splits', 2))),
                min_examples=params.get('examples', 200), threshold=params.get('threshold', 0.1)
            )
        elif extractor == 'Iter':
            self.extractor = Extractor.iter(
                predictor=self.model, min_update=params.get('min_update', .1), n_points=params.get('n_points', 1),
                max_iterations=params.get('max_iter', 600), min_examples=params.get('examples', 250),
                threshold=params.get('threshold', 0.1), fill_gaps=True
            )
        elif extractor == 'Trepan':
            self.extractor = Extractor.trepan(
                predictor=self.model, discretization=None, min_examples=params.get('examples', 0),
                max_depth=params.get('depth', 3), split_logic=None
            )
        elif extractor == 'REAL':
            self.extractor = Extractor.real(predictor=self.model, discretization=None)
        elif extractor == 'CART':
            self.extractor = Extractor.cart(
                predictor=self.model, simplify=params.get('simplify', True))
        elif extractor == 'CReEPy':
            output = Target.CLASSIFICATION if self.task == 'C' else \
                Target.CONSTANT if params.get('output', False) else Target.REGRESSION
            self.extractor = Extractor.creepy(
                predictor=self.model, depth=params.get('depth', 3), error_threshold=params.get('threshold', .1),
                output=output, gauss_components=params.get('components', 10)
            )
        elif extractor == 'CREAM':
            output = Target.CLASSIFICATION if self.task == 'C' else \
                Target.CONSTANT if params.get('output', False) else Target.REGRESSION
            self.extractor = Extractor.cream(
                self.model, depth=params.get('depth', 3), error_threshold=params.get('threshold', .1),
                output=output, gauss_components=params.get('components', 10)
            )
        else:
            ...
        print('Done')
        self.theory = self.extractor.extract(self.train)
        self.extractorInfoBox.reset(self.extractor, self.model)
        self.extractorPerformanceInfoBox.reset(self.extractor, self.model, self.test)

    def show_theory(self, widget):
        self.manager.show_theory(self.theory)


if __name__ == '__main__':
    PSyKEApp1().run()
