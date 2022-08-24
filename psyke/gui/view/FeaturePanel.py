from collections import OrderedDict
from functools import partial
from math import floor

from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout

from psyke.gui.view import NO_DATASET
from psyke.gui.view.layout import FeatureSelectionBoxLayout


class FeaturePanel(RelativeLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.features = {}
        self.feature_panels = {}
        self.plot_features = {}

        self.top_label = Label(text='Feature selection', size_hint=(.15, .17), pos_hint={'x': .01, 'y': .76})
        self.top_button = Button(text='Plot', size_hint=(.15, .17), pos_hint={'x': .2, 'y': .76})
        self.top_button.bind(on_press=self.plot)
        self.alert_label = Label(size_hint=(.5, .17), pos_hint={'x': .4, 'y': .76}, font_size=20, color=(1, 0, 0))

        self.add_widget(self.top_label)
        self.add_widget(self.top_button)
        self.add_widget(self.alert_label)

        self.feature_panel = RelativeLayout(size_hint=(1., .7), pos_hint={'x': 0., 'y': 0.})
        self.add_widget(self.feature_panel)

    def init(self):
        self.features = {}
        self.feature_panels = {}
        self.plot_features = {}
        self.top_button.disabled = True
        self.reset_alert()
        self.feature_panel.clear_widgets()
        self.feature_panel.add_widget(Label(text=NO_DATASET, pos_hint={'x': -.2, 'y': .25}))
        self.feature_panel.add_widget(Label())

    @staticmethod
    def __inverse_mapping(discretization, name):
        original = [feature for feature in discretization if name in feature.admissible_values]
        return original[0].name if len(original) > 0 else name

    @staticmethod
    def __remove_duplicates(list_with_duplicates):
        return list(OrderedDict.fromkeys(list_with_duplicates))

    def set_info(self):
        dataset, pruned_dataset, action, preprocessing = self.controller.get_data_from_model()
        if dataset is not None:
            pruned_columns = dataset.columns if pruned_dataset is None else pruned_dataset.columns
            columns = dataset.columns
            if action == 'Discretize' and preprocessing is not None:
                pruned_columns = self.__remove_duplicates(
                    [self.__inverse_mapping(preprocessing, column) for column in pruned_columns]
                )
                columns = self.__remove_duplicates(
                    [self.__inverse_mapping(preprocessing, column) for column in dataset.columns]
                )
            self.feature_panel.clear_widgets()
            self.top_button.disabled = False
            for i, feature in enumerate(columns):
                self.features[feature] = None if feature not in pruned_columns else \
                    'O' if feature == pruned_columns[-1] else 'I'
                self.plot_features[feature] = feature == dataset.columns[-1]
                self.feature_panels[feature] = FeatureSelectionBoxLayout(
                    feature, self.features[feature], partial(self.set_feature, feature),
                    partial(self.set_plot_feature, feature),
                    pos_hint={'x': .01 + .24 * floor(i / 6.), 'y': 1 - 1. / 7. * (i % 6 + 1)})
                self.feature_panel.add_widget(self.feature_panels[feature])
        else:
            self.init()

    def set_feature(self, feature, button):
        if self.features[feature] == button.text:
            if len([k for k, v in self.features.items() if v == button.text]) > 1:
                self.features[feature] = None
                self.feature_panels[feature].plot_button.state = 'normal'
                self.plot_features[feature] = False
                self.reset_alert()
            else:
                self.set_alert(f'Cannot remove the only selected {"input" if button.text == "I" else "output"} feature')
                self.feature_panels[feature].buttons[button.text].state = 'down'
        else:
            if button.text == 'I':
                if self.features[feature] is None:
                    self.features[feature] = button.text
                    self.reset_alert()
                else:
                    self.set_alert('Cannot remove the only selected output feature')
                    self.feature_panels[feature].buttons['I'].state = 'normal'
                    self.feature_panels[feature].buttons['O'].state = 'down'
            else:
                if len([k for k, v in self.features.items() if v == 'I' and k != feature]) == 0:
                    self.set_alert('Cannot remove the only selected input feature')
                    self.feature_panels[feature].buttons['O'].state = 'normal'
                    self.feature_panels[feature].buttons['I'].state = 'down'
                else:
                    self.features[feature] = button.text
                    to_normal = [k for k, v in self.features.items() if v == 'O' and k != feature]
                    if len(to_normal) > 0:
                        self.feature_panels[to_normal[0]].buttons['O'].state = 'normal'
                        self.features[to_normal[0]] = None
                        self.feature_panels[to_normal[0]].plot_button.state = 'normal'
                        self.plot_features[to_normal[0]] = False
                    self.feature_panels[feature].plot_button.state = 'down'
                    self.plot_features[feature] = True
                    self.reset_alert()
        self.controller.reload_dataset(self.features)

    def set_plot_feature(self, feature, button):
        if self.features[feature] is None:
            self.set_alert('Cannot plot non-selected features')
            self.feature_panels[feature].plot_button.state = 'normal'
        elif self.features[feature] == 'O':
            self.set_alert('Output feature must be plotted')
            self.feature_panels[feature].plot_button.state = 'down'
        else:
            inputs = [k for k, v in self.features.items() if v == 'I']
            active = len([k for k, v in self.plot_features.items() if v and k in inputs])
            if (self.plot_features[feature] and active == 2) or (not self.plot_features[feature] and active < 2):
                self.plot_features[feature] = not self.plot_features[feature]
                self.reset_alert()
            else:
                self.set_alert('Cannot plot less than 1 input feature' if self.plot_features[feature] else
                               'Cannot plot more than 2 input features')
                state = self.feature_panels[feature].plot_button.state
                self.feature_panels[feature].plot_button.state = 'down' if state == 'normal' else 'normal'

    def plot(self, button):
        inputs = len([k for k, v in self.features.items() if v == 'I' and self.plot_features[k]])
        task = self.controller.get_task_from_model()
        if task == 'Classification' and inputs < 2:
            self.set_alert('Classification plots require two input features')
        elif task == 'Regression' and inputs == 0:
            self.set_alert('Regression plots require at least one input feature')
        else:
            self.controller.plot(self.features, [k for k, v in self.plot_features.items() if v])
            self.reset_alert()

    def set_alert(self, text: str = ''):
        self.alert_label.text = text

    def reset_alert(self):
        self.set_alert()
