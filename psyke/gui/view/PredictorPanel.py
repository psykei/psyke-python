from functools import partial

from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, f1_score

from psyke.gui.view import INFO_PREDICTOR_MESSAGE, PREDICTOR_MESSAGE, PREDICTOR_PERFORMANCE_PREFIX, \
    INFO_PREDICTOR_PREFIX
from psyke.gui.view.layout import PanelBoxLayout, TextLabelCoupledRelativeLayout
from psyke.gui.model import PREDICTORS, FIXED_PREDICTOR_PARAMS


class PredictorPanel(PanelBoxLayout):

    def __init__(self, controller, ratio, **kwargs):
        super().__init__(controller, 'Train', INFO_PREDICTOR_MESSAGE, 1, ratio,
                         PREDICTOR_MESSAGE, PREDICTORS, controller.set_predictor_param, **kwargs)

        self.parameter_panel = RelativeLayout(size_hint=(1, .83 / ratio))

        self.add_widget(self.main_panel)
        self.add_widget(self.parameter_panel)
        self.add_widget(self.info_label)

    def select(self, spinner, text):
        if text == PREDICTOR_MESSAGE:
            self.controller.reset_predictor()
        else:
            self.controller.select_predictor(text)
            self.go_button.disabled = False
            params = PREDICTORS[text][1]
            self.parameter_panel.clear_widgets()
            for i, (name, (default, param_type)) in enumerate(dict(FIXED_PREDICTOR_PARAMS, **params).items()):
                self.parameter_panel.add_widget(TextLabelCoupledRelativeLayout(
                    f'{name} ({default})', '', param_type, partial(self.set_param, name), i))
            self.parameter_panel.add_widget(Label())

    def go_action(self, button):
        self.controller.train_predictor()

    def set_info(self):
        predictor_name, predictor, predictor_params = self.controller.get_predictor_from_model()
        if predictor is None:
            self.info_label.text = INFO_PREDICTOR_MESSAGE
        else:
            self.info_label.text = ''
            for name, _ in FIXED_PREDICTOR_PARAMS.items():
                self.info_label.text += f'{name} = {predictor_params[name]}\n'
            self.info_label.text += f'\n{INFO_PREDICTOR_PREFIX}Predictor: {predictor_name}\n'
            for name, _ in PREDICTORS[predictor_name][1].items():
                self.info_label.text += f'{name} = {predictor_params[name]}\n'

            self.info_label.text += '\n' + PREDICTOR_PERFORMANCE_PREFIX

            test = self.controller.get_test_set_from_model()
            true = test.iloc[:, -1]
            predicted = predictor.predict(test.iloc[:, :-1])
            if isinstance(predictor, ClassifierMixin):
                self.info_label.text += f'Accuracy: {accuracy_score(true, predicted):.2f}\n' \
                                        f'F1: {f1_score(true, predicted, average="weighted"):.2f}'
            elif isinstance(predictor, RegressorMixin):
                self.info_label.text += f'MAE: {mean_absolute_error(true, predicted):.2f}\n' \
                                   f'MSE: {mean_squared_error(true, predicted):.2f}\n' \
                                   f'R2: {r2_score(true, predicted):.2f}'
