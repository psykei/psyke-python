from functools import partial

from kivy.uix.label import Label
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from psyke.gui.model import EXTRACTORS
from psyke.gui.view.layout import PanelBoxLayout, VerticalBoxLayout, SidebarBoxLayout
from psyke.gui.view import INFO_EXTRACTOR_MESSAGE, EXTRACTOR_MESSAGE, text_with_label, INFO_EXTRACTOR_PREFIX, \
    EXTRACTOR_PERFORMANCE_PREFIX


class ExtractorPanel(PanelBoxLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(controller, 'Extract', INFO_EXTRACTOR_MESSAGE, 350,
                         EXTRACTOR_MESSAGE, EXTRACTORS, controller.set_extractor_param, **kwargs)

        self.parameter_panel = VerticalBoxLayout(size_hint_y=None, height=190)

        left_sidebar = SidebarBoxLayout()
        left_sidebar.add_widget(self.main_panel)
        left_sidebar.add_widget(self.parameter_panel)
        left_sidebar.add_widget(Label())

        self.add_widget(left_sidebar)
        self.add_widget(self.info_label)

    def set_info(self):
        extractor_name, extractor, extractor_params = self.controller.get_extractor_from_model()
        if extractor is None:
            self.info_label.text = INFO_EXTRACTOR_MESSAGE
        else:
            self.info_label.text = f'\n{INFO_EXTRACTOR_PREFIX}Predictor: {extractor_name}\n'
            for name, _ in EXTRACTORS[extractor_name][1].items():
                self.info_label.text += f'{name} = {extractor_params[name]}\n'

            self.info_label.text += '\n' + EXTRACTOR_PERFORMANCE_PREFIX
            self.info_label.text += f'N. rules: {extractor.n_rules}\n'

            test = self.controller.get_test_set_from_model()
            predicted = extractor.predict(test.iloc[:, :-1])
            idx = [prediction is not None for prediction in predicted]
            predicted = predicted[idx]

            predictor = self.controller.get_predictor_from_model()[1]
            true = test.iloc[idx, -1]
            predicted_bb = predictor.predict(test.iloc[idx, :-1])

            if isinstance(predictor, ClassifierMixin):
                self.info_label.text += f'Acc.: {accuracy_score(true, predicted):.2f} (data), ' \
                                        f'{accuracy_score(predicted_bb, predicted):.2f} (BB)\n' \
                                        f'F1: {f1_score(true, predicted, average="weighted"):.2f} (data), ' \
                                        f'{f1_score(predicted_bb, predicted, average="weighted"):.2f} (BB)'
            elif isinstance(predictor, RegressorMixin):
                self.info_label.text += f'MAE: {mean_absolute_error(true, predicted):.2f} (data), ' \
                                        f'{mean_absolute_error(predicted_bb, predicted):.2f} (BB)\n' \
                                        f'MSE: {mean_squared_error(true, predicted):.2f} (data), ' \
                                        f'{mean_squared_error(predicted_bb, predicted):.2f} (BB)\n' \
                                        f'R2: {r2_score(true, predicted):.2f} (data), ' \
                                        f'{r2_score(predicted_bb, predicted):.2f} (BB)'

    def select(self, spinner, text):
        if text == EXTRACTOR_MESSAGE:
            self.controller.reset_extractor()
        else:
            self.controller.select_extractor(text)
            self.go_button.disabled = False
            params = EXTRACTORS[text][1]
            self.parameter_panel.clear_widgets()
            for name, (default, param_type) in params.items():
                self.parameter_panel.add_widget(
                    text_with_label(f'{name} ({default})', '', param_type, partial(self.set_param, name))
                )
            self.parameter_panel.add_widget(Label())

    def go_action(self, button):
        self.controller.train_extractor()
