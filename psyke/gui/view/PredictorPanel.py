from functools import partial

from kivy.uix.relativelayout import RelativeLayout
from sklearn.base import ClassifierMixin, RegressorMixin

from psyke.gui.view import INFO_PREDICTOR_MESSAGE, PREDICTOR_MESSAGE, PREDICTOR_PERFORMANCE_PREFIX, \
    INFO_PREDICTOR_PREFIX
from psyke.gui.view.layout import PanelBoxLayout, create_param_layout
from psyke.gui.model import PREDICTORS, FIXED_PREDICTOR_PARAMS
from psyke.utils.metrics import accuracy, f1, mae, mse, r2


class PredictorPanel(PanelBoxLayout):

    def __init__(self, controller, ratio, **kwargs):
        super().__init__(controller, 'Train', INFO_PREDICTOR_MESSAGE, 1, ratio,
                         PREDICTOR_MESSAGE, PREDICTORS, controller.set_predictor_param, **kwargs)

        self.parameter_panel = RelativeLayout(size_hint=(1, .83))

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
            for i, (name, (default, type, constraint)) in enumerate(dict(FIXED_PREDICTOR_PARAMS, **params).items()):
                if constraint is not None and constraint != self.controller.get_task_from_model():
                    continue
                self.parameter_panel.add_widget(create_param_layout(
                    name, default, type, partial(self.set_param, name), i, self.ratio))

    def go_action(self, button):
        self.controller.train_predictor()

    def set_info(self):
        predictor_name, predictor, predictor_params = self.controller.get_predictor_from_model()
        if predictor is None:
            self.info_label.text = INFO_PREDICTOR_MESSAGE
        else:
            self.info_label.text = ''
            for name, (_, _, constraint) in FIXED_PREDICTOR_PARAMS.items():
                if constraint is not None and constraint != self.controller.get_task_from_model():
                    continue
                self.info_label.text += f'{name} = {predictor_params[name]}\n'
            self.info_label.text += f'\n{INFO_PREDICTOR_PREFIX}Predictor: {predictor_name}\n'
            for name, (_, _, constraint) in PREDICTORS[predictor_name][1].items():
                if constraint is not None and constraint != self.controller.get_task_from_model():
                    continue
                self.info_label.text += f'{name} = {predictor_params[name]}\n'

            self.info_label.text += '\n' + PREDICTOR_PERFORMANCE_PREFIX

            test, action, preprocessing = self.controller.get_test_set_from_model()
            true = test.iloc[:, -1]
            predicted = predictor.predict(test.iloc[:, :-1])
            if action == 'Scale' and preprocessing is not None:
                m, s = preprocessing[test.columns[-1]]
                true = true * s + m
                predicted = predicted * s + m
            if isinstance(predictor, ClassifierMixin):
                self.info_label.text += f'Accuracy: {accuracy(true, predicted):.2f}\n' \
                                        f'F1: {f1(true, predicted):.2f}'
            elif isinstance(predictor, RegressorMixin):
                self.info_label.text += f'MAE: {mae(true, predicted):.2f}\n' \
                                   f'MSE: {mse(true, predicted):.2f}\n' \
                                   f'R2: {r2(true, predicted):.2f}'
