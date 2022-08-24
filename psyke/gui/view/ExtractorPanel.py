from functools import partial

from kivy.uix.relativelayout import RelativeLayout
from sklearn.base import ClassifierMixin, RegressorMixin

from psyke.gui.model import EXTRACTORS
from psyke.gui.view.layout import PanelBoxLayout, create_param_layout
from psyke.gui.view import INFO_EXTRACTOR_MESSAGE, EXTRACTOR_MESSAGE, INFO_EXTRACTOR_PREFIX, \
    EXTRACTOR_PERFORMANCE_PREFIX
from psyke.utils.metrics import accuracy, f1, mae, mse, r2


class ExtractorPanel(PanelBoxLayout):

    def __init__(self, controller, ratio=1, **kwargs):
        super().__init__(controller, 'Extract', INFO_EXTRACTOR_MESSAGE, 1, ratio,
                         EXTRACTOR_MESSAGE, EXTRACTORS, controller.set_extractor_param, **kwargs)

        self.parameter_panel = RelativeLayout(size_hint=(1, .83))

        self.add_widget(self.main_panel)
        self.add_widget(self.parameter_panel)
        self.add_widget(self.info_label)

    def set_info(self):
        extractor_name, extractor, extractor_params = self.controller.get_extractor_from_model()
        if extractor is None:
            self.info_label.text = INFO_EXTRACTOR_MESSAGE
        else:
            self.info_label.text = f'\n{INFO_EXTRACTOR_PREFIX}Predictor: {extractor_name}\n'
            for name, (_, _, constraint) in EXTRACTORS[extractor_name][1].items():
                if constraint is not None and constraint != self.controller.get_task_from_model():
                    continue
                self.info_label.text += f'{name} = {extractor_params[name]}\n'

            self.info_label.text += '\n' + EXTRACTOR_PERFORMANCE_PREFIX
            self.info_label.text += f'N. rules: {extractor.n_rules}\n'

            test, action, preprocessing = self.controller.get_test_set_from_model()
            predictor = self.controller.get_predictor_from_model()[1]
            extracted = extractor.predict(test.iloc[:, :-1])
            true = test.iloc[:, -1]
            predicted = predictor.predict(test.iloc[:, :-1])
            if action == 'Scale' and preprocessing is not None:
                m, s = preprocessing[test.columns[-1]]
                extracted = extracted * s + m
                true = true * s + m
                predicted = predicted * s + m

            if isinstance(predictor, ClassifierMixin):
                labels = ['Acc.', 'F1']
                metrics = [accuracy, f1]
            elif isinstance(predictor, RegressorMixin):
                labels = ['MAE', 'MSE', 'R2']
                metrics = [mae, mse, r2]
            else:
                raise NotImplementedError

            for label, metric in zip(labels, metrics):
                self.info_label.text += \
                    f'{label}: {metric(true, extracted):.2f} (data), {metric(predicted, extracted):.2f} (BB)\n'

    def select(self, spinner, text):
        if text == EXTRACTOR_MESSAGE:
            self.controller.reset_extractor()
        else:
            self.controller.select_extractor(text)
            self.go_button.disabled = False
            params = EXTRACTORS[text][1]
            self.parameter_panel.clear_widgets()
            for i, (name, (default, type, constraint)) in enumerate(params.items()):
                if constraint is not None and constraint != self.controller.get_task_from_model():
                    continue
                self.parameter_panel.add_widget(create_param_layout(
                    name, default, type, partial(self.set_param, name), i, self.ratio))

    def go_action(self, button):
        self.controller.train_extractor()
