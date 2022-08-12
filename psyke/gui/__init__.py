import numpy as np
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from functools import partial

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.neighbors._base import NeighborsBase
from sklearn.tree import BaseDecisionTree

from psyke.extraction.cart import Cart
from psyke.extraction.real import REAL
from psyke.extraction.trepan import Trepan
from psyke.clustering.cream import CREAM
from psyke.extraction.hypercubic.creepy import CReEPy
from psyke.extraction.hypercubic.gridex import GridEx
from psyke.extraction.hypercubic.gridrex import GridREx
from psyke.extraction.hypercubic.iter import ITER
from psyke.gui.layout import VerticalBoxLayout, HorizontalBoxLayout
from psyke.utils import Target


def default_action(widget=None, value=None):
    pass


def radio_with_label(group: str, active: bool, label: str, action=default_action) -> GridLayout:
    box = GridLayout(cols=2, padding=0)
    button = CheckBox(group=group, size_hint_x=.047, size_hint_y=.047, active=active)
    box.add_widget(button)
    button.bind(active=action)
    box.add_widget(Label(text=label))
    return box


def button_with_label(group: str, color: list[int, int, int], active: bool,
                      label: str, action=default_action) -> BoxLayout:
    box = GridLayout(cols=2, padding=0)
    button = CheckBox(group=group, size_hint_x=.047, size_hint_y=.047, color=color, active=active)
    box.add_widget(button)
    button.bind(active=action)
    box.add_widget(Label(text=label))
    return box


def checkbox_with_label(color: list[int, int, int], active: bool, label: str, action=default_action) -> BoxLayout:
    box = GridLayout(cols=3, size_hint_y=None, height=25, padding=10)
    box.add_widget(Label(text=label))
    box.add_widget(Label())
    button = CheckBox(color=color, active=active)
    button.bind(active=action)
    box.add_widget(button)
    box.add_widget(Label())
    return box


def text_with_label(label: str, text: str, filter: str, action) -> BoxLayout:
    box = HorizontalBoxLayout(size_hint_y=None, height=30)
    box.add_widget(Label(text=label))
    text = TextInput(text=text, input_filter=filter, multiline=False, size_hint_y=None, height=30)
    text.bind(text=action)
    box.add_widget(text)
    return box





class TitleBox(HorizontalBoxLayout):

    def __init__(self, title: str, action=None, button_name=None, disabled=False, **kwargs):
        super().__init__(**kwargs)
        self.padding = 5
        self.spacing = 15
        self.size_hint_y = None
        self.height = 40
        self.add_widget(Label(text=title))
        if action is not None:
            self.button = Button(text=button_name, size_hint=(None, None), height=30, width=120, disabled=disabled)
            self.button.bind(on_press=action)
            self.add_widget(self.button)


class TaskBox(VerticalBoxLayout):

    def __init__(self, root, **kwargs):
        super().__init__(**kwargs)
        self.padding = 20
        self.root = root
        self.add_widget(TitleBox('Select task'))
        self.add_widget(Label())
        self.add_widget(button_with_label('task', [1, 1, 1], True, 'Classification', partial(self.set_task, 'C')))
        self.add_widget(button_with_label('task', [1, 1, 1], False, 'Regression', partial(self.set_task, 'R')))
        self.add_widget(Label())

    def set_task(self, task, widget, value):
        if value:
            self.root.select_task(task)


class DataBox(VerticalBoxLayout):

    def __init__(self, root, classification: list[str], regression: list[str], **kwargs):
        super().__init__(**kwargs)
        self.padding = 20
        self.root = root
        self.dataset = {'C': classification[0], 'R': regression[0]}
        self.classification = VerticalBoxLayout()
        self.regression = VerticalBoxLayout()
        for (container, datasets, task) in \
                zip([self.classification, self.regression], [classification, regression], ['C', 'R']):
            for i, dataset in enumerate(datasets):
                container.add_widget(button_with_label(f'data_{task}', [1, 1, 1], i == 0, dataset,
                                                       partial(self.set_dataset, task, dataset)))
        self.reset()

    def set_dataset(self, task, dataset, widget, value):
        if value:
            self.dataset[task] = dataset

    def reset(self):
        self.clear_widgets()
        self.add_widget(TitleBox('Select dataset', self.root.select_dataset, 'Load'))
        self.add_widget(Label())
        self.add_widget(self.classification if self.root.task == 'C' else self.regression)
        self.add_widget(Label())


class PredictorBox(VerticalBoxLayout):

    def __init__(self, root, options, **kwargs):
        super().__init__(**kwargs)
        self.padding = 20
        self.model = options[0]
        self.root = root
        self.add_widget(TitleBox('Select predictor'))
        self.add_widget(Label())
        for i, option in enumerate(options):
            self.add_widget(button_with_label('predictor', [1, 1, 1], i == 0, option,
                                              partial(self.set_model, option)))
        self.add_widget(Label())

    def set_model(self, model, widget, value):
        if value:
            self.model = model
            self.root.select_model()


class ExtractorBox(VerticalBoxLayout):

    def __init__(self, root, classifiers: list[str], regressors: list[str],  **kwargs):
        super().__init__(**kwargs)
        self.padding = 20
        #self.spacing = 20
        self.extractor = {'C': classifiers[0], 'R': regressors[0]}
        self.root = root
        self.classifiers = GridLayout(cols=2)
        self.regressors = GridLayout(cols=2)
        for (container, extractors, task) in \
                zip([self.classifiers, self.regressors], [classifiers, regressors], ['C', 'R']):
            for i, extractor in enumerate(extractors):
                container.add_widget(button_with_label(f'extractor_{task}', [1, 1, 1], i == 0, extractor,
                                                       partial(self.set_extractor, task, extractor)))
        self.reset()

    def set_extractor(self, task, extractor, widget, value):
        if value:
            self.extractor[task] = extractor
            self.root.select_extractor()

    def reset(self, task='C'):
        self.clear_widgets()
        self.add_widget(TitleBox('Select extractor', size_hint_y=.2))
        self.add_widget(Label(size_hint_y=None, height=30))
        self.add_widget(self.classifiers if task == 'C' else self.regressors)


class DataInfoBox(VerticalBoxLayout):

    def __init__(self, name=None, df=None, **kwargs):
        super().__init__(**kwargs)
        self.padding = 20
        self.label = Label()
        self.add_widget(self.label)
        self.reset(name, df)

    def reset(self, name, df):
        self.label.text = 'No selected dataset' if name is None else \
            f'Dataset info\n\nDataset: {name}\nInput variables: {len(df.columns) - 1}\nInstances: {len(df)}'
        if df is not None and isinstance(df.iloc[0, -1], str):
            self.label.text += f'\nClasses: {len(np.unique(df.iloc[:, -1]))}'


class ParameterBox(VerticalBoxLayout):

    def __init__(self, root, title, **kwargs):
        super().__init__(**kwargs)
        self.padding = 20
        self.params = {}
        self.root = root
        self.titleBox = title
        self.reset()

    def reset(self):
        self.clear_widgets()
        self.params = {}
        self.add_widget(self.titleBox)

    def enable(self):
        self.titleBox.button.disabled = False

    def disable(self):
        self.titleBox.button.disabled = True

    def set_param(self, key, widget, value):
        if value == '':
            try:
                del self.params[key]
            except KeyError:
                pass
        else:
            try:
                self.params[key] = int(value)
            except ValueError:
                self.params[key] = float(value)

    def set_param_checkbox(self, key, widget, value):
        self.params[key] = bool(value)


class PredictorParameterBox(ParameterBox):

    def __init__(self, root, disabled=True, **kwargs):
        super().__init__(root, TitleBox('Predictor', root.train_model, 'Train', disabled), **kwargs)

    def reset(self):
        super().reset()
        self.add_widget(Label())
        self.add_widget(text_with_label('Test set', '', 'float', partial(self.set_param, 'test')))
        self.add_widget(Label())
        if self.root.model_kind == 'KNN':
            self.add_widget(text_with_label('N neighbors', '', 'int', partial(self.set_param, 'neighbors')))
        elif self.root.model_kind == 'DT':
            self.add_widget(text_with_label('Max depth', '', 'int', partial(self.set_param, 'depth')))
            self.add_widget(text_with_label('Max leaves', '', 'int', partial(self.set_param, 'leaves')))
        self.add_widget(Label())


class ExtractorParameterBox(ParameterBox):

    def __init__(self, root, disabled=True, **kwargs):
        super().__init__(root, TitleBox('Extractor', root.train_extractor, 'Fit', disabled), **kwargs)
        self.simplify = checkbox_with_label([1, 1, 1], True, 'Simplify theory',
                                            partial(self.set_param_checkbox, 'simplify'))
        self.depth = text_with_label('Max depth', '', 'int', partial(self.set_param, 'depth'))
        self.splits = text_with_label('Number of splits', '', 'int', partial(self.set_param, 'splits'))
        self.examples = text_with_label('Min examples', '', 'int', partial(self.set_param, 'min_examples'))
        self.threshold = text_with_label('Threshold', '', 'float', partial(self.set_param, 'threshold'))
        self.output = checkbox_with_label([1, 1, 1], True, 'Constant output',
                                          partial(self.set_param_checkbox, 'output'))
        self.components = text_with_label('N components', '', 'int', partial(self.set_param, 'components'))
        self.update = text_with_label('Min update', '', 'float', partial(self.set_param, 'min_update'))
        self.points = text_with_label('Initial points', '', 'int', partial(self.set_param, 'n_points'))
        self.iterations = text_with_label('Max iterations', '', 'int', partial(self.set_param, 'max_iter'))

    def reset(self):
        super().reset()
        self.add_widget(Label())
        extractor = self.root.extractor_kind[self.root.task]
        if extractor == 'REAL':
            self.add_widget(Label(text='No parameters required'))
        elif extractor == 'CART':
            self.add_widget(self.simplify)
        else:
            if extractor in ['GridEx', 'GridREx', 'Trepan', 'CReEPy', 'CREAM']:
                self.add_widget(self.depth)
            if extractor in ['GridEx', 'GridREx']:
                self.add_widget(self.splits)
            if extractor in ['GridEx', 'GridREx', 'Iter', 'Trepan']:
                self.add_widget(self.examples)
            if extractor in ['GridEx', 'GridREx', 'Iter', 'CReEPy', 'CREAM']:
                self.add_widget(self.threshold)
            if extractor in ['CReEPy', 'CREAM']:
                if self.root.task == 'R':
                    self.add_widget(self.output)
                self.add_widget(self.components)
            if extractor == 'Iter':
                self.add_widget(self.update)
                self.add_widget(self.points)
                self.add_widget(self.iterations)
        self.add_widget(Label())


class InfoBox(VerticalBoxLayout):

    def __init__(self, item=None, **kwargs):
        super().__init__(**kwargs)
        self.padding = 20
        self.label = Label(text='')
        self.add_widget(Label())
        self.add_widget(self.label)
        self.add_widget(Label())
        self.reset(item)


class PredictorInfoBox(InfoBox):

    def __init__(self, model=None, **kwargs):
        super().__init__(model, **kwargs)

    def reset(self, model=None):
        if model is None:
            self.label.text = 'No trained predictor'
        else:
            self.label.text = 'Predictor info\n\n'
            if isinstance(model, BaseDecisionTree):
                self.label.text += f'Predictor: Decision Tree\nMax leaves: {model.max_leaf_nodes}\n' \
                                   f'Max depth: {model.max_depth}'
            elif isinstance(model, NeighborsBase):
                self.label.text += f'Predictor: KNN\nNeighbors: {model.n_neighbors}'
            else:
                ...


class PredictorPerformanceInfoBox(InfoBox):

    def __init__(self, model=None, **kwargs):
        super().__init__(model, **kwargs)

    def reset(self, model=None, test=None, test_quota=None):
        if model is None:
            self.label.text = 'No trained predictor'
        else:
            self.label.text = f'Predictor Performance\n\nTest set: {test_quota}{" (%)" if test_quota < 1 else ""}\n\n'
            true = test.iloc[:, -1]
            predicted = model.predict(test.iloc[:, :-1])
            if isinstance(model, ClassifierMixin):
                self.label.text += f'Accuracy: {accuracy_score(true, predicted):.2f}'
            elif isinstance(model, RegressorMixin):
                self.label.text += f'MAE: {mean_absolute_error(true, predicted):.2f}\n' \
                                   f'MSE: {mean_squared_error(true, predicted):.2f}\n' \
                                   f'R2: {r2_score(true, predicted):.2f}'
            else:
                ...


class ExtractorInfoBox(InfoBox):

    def __init__(self, extractor=None, **kwargs):
        super().__init__(extractor, **kwargs)

    def reset(self, extractor=None, model=None):
        if extractor is None:
            self.label.text = 'No fitted extractor'
        else:
            self.label.text = 'Extractor info\n\n'
            if isinstance(extractor, Trepan):
                self.label.text += f'Extractor: Trepan\n' \
                                   f'Max depth: {extractor.max_depth}\nMin examples: {extractor.min_examples}'
            elif isinstance(extractor, REAL):
                self.label.text += f'Extractor: REAL'
            elif isinstance(extractor, ITER):
                self.label.text += f'Predictor: Iter\nMin Examples: {extractor.min_examples}\n' \
                                   f'Threshold: {extractor.threshold}\nMin update: {extractor.min_update}\n' \
                                   f'Initial points: {extractor.n_points}\nMax iterations: {extractor.max_iterations}'
            elif isinstance(extractor, Cart):
                self.label.text += f'Extractor: CART\nSimplify theory: {extractor._simplify}'
            elif isinstance(extractor, GridEx) or isinstance(extractor, GridREx):
                if isinstance(extractor, GridEx):
                    self.label.text += f'Extractor: GridEx\n'
                else:
                    self.label.text += f'Extractor: GridREx\n'
                self.label.text += f'Max depth: {extractor.grid.iterations}\n' \
                                   f'Number of splits: Fixed\nMin examples: {extractor.min_examples}\n' \
                                   f'Threshold: {extractor.threshold}'
            elif isinstance(extractor, CReEPy) or isinstance(extractor, CREAM):
                if isinstance(extractor, CReEPy):
                    self.label.text += f'Extractor: CReEPy\n'
                else:
                    self.label.text += f'Extractor: CREAM\n'
                self.label.text += f'Max depth: {extractor.depth}\nThreshold: {extractor.error_threshold}\n'
                if isinstance(model, RegressorMixin):
                    self.label.text += f'Constant output: {extractor.output == Target.CONSTANT}\n'
                self.label.text += f'N components: {extractor.gauss_components}'
            else:
                ...


class ExtractorPerformanceInfoBox(InfoBox):

    def __init__(self, extractor=None, action=default_action(), **kwargs):
        button_box = HorizontalBoxLayout()
        button_box.add_widget(Label())
        self.button = Button(text='Show theory', size_hint_y=None, height=30, disabled=True)
        self.button.bind(on_press=action)
        button_box.add_widget(self.button)
        button_box.add_widget(Label())
        super().__init__(extractor, **kwargs)
        self.add_widget(Label())
        self.add_widget(button_box)
        self.add_widget(Label())

    def reset(self, extractor=None, model=None, test=None):
        if extractor is None:
            self.label.text = 'No fitted extractor'
            self.button.disabled = True
        else:
            self.button.disabled = False
            self.label.text = 'Extractor performance\n\n'
            extracted = extractor.predict(test.iloc[:, :-1])
            idx = np.array([e is not None for e in extracted])
            extracted = extracted[idx]
            true = test.iloc[idx, -1]
            predicted = model.predict(test.iloc[idx, :-1])

            self.label.text += f'Extracted rules: {extractor.n_rules}\n\n'
            if isinstance(model, ClassifierMixin):
                self.label.text += f'Accuracy: {accuracy_score(true, predicted):.2f} (model) ' \
                                   f'{accuracy_score(predicted, extracted):.2f} (fidelity)'
            elif isinstance(model, RegressorMixin):
                self.label.text += f'MAE: {mean_absolute_error(true, extracted):.2f} (model) ' \
                                   f'{mean_absolute_error(predicted, extracted):.2f} (fidelity)\n' \
                                   f'MSE: {mean_squared_error(true, extracted):.2f} (model) ' \
                                   f'{mean_squared_error(predicted, extracted):.2f} (fidelity)\n' \
                                   f'R2: {r2_score(true, extracted):.2f} (model) ' \
                                   f'{r2_score(predicted, extracted):.2f} (fidelity)'
            else:
                ...


class DataManipulationBox(VerticalBoxLayout):

    def __init__(self, root, **kwargs):
        super().__init__(**kwargs)
        self.padding = 20
        self.root = root
        self.reset()

    def reset(self):
        self.clear_widgets()
        if self.root.data is None:
            self.add_widget(Label(text='No selected dataset'))
        else:
            self.add_widget(Label(text='Options'))
            self.add_widget(button_with_label('', [1, 1, 1], False, 'Discretise') if self.root.task == 'C' else
                            button_with_label('', [1, 1, 1], False, 'Normalise'))
            self.add_widget(Label())
