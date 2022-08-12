from functools import partial

from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner

from psyke.gui import text_with_label
from psyke.gui.layout import PanelBoxLayout, SidebarBoxLayout, HorizontalBoxLayout, VerticalBoxLayout
from psyke.gui.model import PREDICTOR_MESSAGE, PREDICTORS, FIXED_PREDICTOR_PARAMS, INFO_PREDICTOR_MESSAGE, \
    INFO_PREDICTOR_PREFIX


class PredictorPanel(PanelBoxLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(controller, 'Train', INFO_PREDICTOR_MESSAGE, **kwargs)
        self.controller.set_predictor_panel(self)

        self.predictor = None
        self.params = {}

        self.parameter_panel = VerticalBoxLayout(size_hint_y=None, height=190)

        left_sidebar = SidebarBoxLayout()
        left_sidebar.add_widget(self.main_panel)
        left_sidebar.add_widget(self.parameter_panel)
        left_sidebar.add_widget(Label())

        self.add_widget(left_sidebar)
        self.add_widget(self.info_label)

        #dataset_option_panel = VerticalBoxLayout(size_hint_x=None, width=500)
        #preprocessing_panel = HorizontalBoxLayout(spacing=25, size_hint_y=None, height=40)
        #preprocessing_panel.add_widget(Label(text='Dataset options'))
        #self.discretize_button = ToggleButton(text='Discretize', group='preprocessing', state='normal')
        #self.scale_button = ToggleButton(text='Scale', group='preprocessing')
        #for btn_proc in [self.discretize_button, self.scale_button]:
        #    btn_proc.bind(state=self.select_preprocessing)
        #    preprocessing_panel.add_widget(btn_proc)
        #dataset_option_panel.add_widget(Label())
        #dataset_option_panel.add_widget(preprocessing_panel)
        #dataset_option_panel.add_widget(Label())

        #self.add_widget(dataset_option_panel)

        self.init_predictors()

    def select(self, spinner, text):
        if text == PREDICTOR_MESSAGE:
            self.predictor = None
        else:
            self.predictor = text
            self.go_button.disabled = False
            params = PREDICTORS[self.predictor][1]
            self.parameter_panel.clear_widgets()
            for name, (default, param_type) in dict(FIXED_PREDICTOR_PARAMS, **params).items():
                self.parameter_panel.add_widget(
                    text_with_label(f'{name} ({default})', '', param_type, partial(self.set_param, name))
                )
            self.parameter_panel.add_widget(Label())

    def go_action(self, button):
        self.controller.train_predictor()
        self.set_predictor_info()

    def set_predictor_info(self):
        predictor = self.controller.predictor
        if predictor is None:
            self.info_label.text = INFO_PREDICTOR_MESSAGE
        else:
            self.info_label.text = ''
            for name, _ in FIXED_PREDICTOR_PARAMS.items():
                self.info_label.text += f'{name} = {self.controller.predictor_params[name]}\n'
            self.info_label.text += f'\n\n{INFO_PREDICTOR_PREFIX}Predictor: {self.predictor}\n'
            for name, _ in PREDICTORS[self.predictor][1].items():
                self.info_label.text += f'{name} = {self.controller.predictor_params[name]}\n'
        self.info_label.text += '\n\n\n\n\n\n\n'

    def init_predictors(self):
        self.spinner_options.text = PREDICTOR_MESSAGE
        self.spinner_options.values = [k for k, v in PREDICTORS.items() if self.controller.data_panel.task in v[0]]
        self.go_button.disabled = True
        self.spinner_options.disabled = True
        self.parameter_panel.clear_widgets()
        #self.discretize_button.state = 'normal'
        #self.scale_button.state = 'normal'
        #self.discretize_button.disabled = True
        #self.scale_button.disabled = True
        #self.controller.reset_dataset()
        #self.set_dataset_info()

    def set_param(self, key, widget, value):
        self.params[key] = value

    def get_param(self, param):
        value = self.params.get(param)
        if value is not None:
            return value
        value = PREDICTORS[self.predictor][1].get(param)
        return FIXED_PREDICTOR_PARAMS[param][0] if value is None else value[0]

    def enable_predictors(self):
        self.spinner_options.disabled = False
