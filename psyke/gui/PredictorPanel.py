import numpy as np
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.togglebutton import ToggleButton

from psyke.gui import HorizontalBoxLayout, VerticalBoxLayout, radio_with_label
from psyke.gui.model import PREDICTOR_MESSAGE, PREDICTORS


class PredictorPanel(HorizontalBoxLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(size_hint_y=None, height=200, **kwargs)
        self.controller = controller
        self.controller.set_predictor_panel(self)

        self.predictor = None

        self.predictor_options = Spinner(
            size_hint_x=None, width=130, pos_hint={'center_x': .5, 'center_y': .5}
        )
        self.predictor_options.bind(text=self.select_predictor)
        self.train_button = Button(text='Train', disabled=True, on_press=self.train_predictor)

        predictor_panel = HorizontalBoxLayout(size_hint=(None, None), size=(130, 40))
        predictor_panel.add_widget(self.predictor_options)
        predictor_panel.add_widget(self.train_button)

        #left_sidebar = VerticalBoxLayout(size_hint_x=None, width=450, padding=15, spacing=15)
        #left_sidebar.add_widget(Label())
        #left_sidebar.add_widget(task_panel)
        #left_sidebar.add_widget(predictor_panel)
        #left_sidebar.add_widget(Label())

        self.add_widget(predictor_panel)

        #dataset_info_panel = VerticalBoxLayout(spacing=15, size_hint_x=None, width=300)
        #dataset_info_panel.add_widget(Label())
        #dataset_info_panel.add_widget(Label(text='Dataset info'))
        #dataset_info_panel.add_widget(self.info_label)
        #dataset_info_panel.add_widget(Label())

        #self.add_widget(dataset_info_panel)

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

    def select_predictor(self, spinner, text):
        self.predictor = text if text != PREDICTOR_MESSAGE else None
        self.train_button.disabled = False

    def train_predictor(self, button):
        self.controller.train_predictor(self.predictor)
        self.set_predictor_info()

    def set_predictor_info(self):
        pass

    def init_predictors(self):
        self.predictor_options.text = PREDICTOR_MESSAGE
        self.predictor_options.values = \
            [entry[0] for entry in PREDICTORS if self.controller.data_panel.task in entry[1]]
        self.train_button.disabled = True
        #self.discretize_button.state = 'normal'
        #self.scale_button.state = 'normal'
        #self.discretize_button.disabled = True
        #self.scale_button.disabled = True
        #self.controller.reset_dataset()
        #self.set_dataset_info()

    def select_task(self, button, value):
        if value == 'down':
            self.task = button.text
            self.init_datasets()

    def select_preprocessing(self, button, value):
        self.preprocessing[button.text] = value == 'down'
