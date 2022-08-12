import numpy as np
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.togglebutton import ToggleButton

from psyke.gui import HorizontalBoxLayout, VerticalBoxLayout
from psyke.gui.model import TASKS, DATASET_MESSAGE, DATASETS, INFO_DATASET_MESSAGE


class DataPanel(HorizontalBoxLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(size_hint_y=None, height=200, **kwargs)
        self.controller = controller
        self.controller.set_data_panel(self)

        self.task = TASKS[0]
        self.dataset = None
        self.info_label = Label(text=INFO_DATASET_MESSAGE)
        self.preprocessing = {}

        task_panel = HorizontalBoxLayout()
        for task in TASKS:
            btn_task = ToggleButton(text=task, group='task', state='down' if self.task == task else 'normal')
            btn_task.bind(state=self.select_task)
            task_panel.add_widget(btn_task)

        self.data_options = Spinner(
            size_hint_x=None, width=130, pos_hint={'center_x': .5, 'center_y': .5}
        )
        self.data_options.bind(text=self.select_dataset)
        self.load_button = Button(text='Load', disabled=True, on_press=self.load_dataset)

        dataset_panel = HorizontalBoxLayout(size_hint=(None, None), size=(130, 40))
        dataset_panel.add_widget(self.data_options)
        dataset_panel.add_widget(self.load_button)

        left_sidebar = VerticalBoxLayout(size_hint_x=None, width=330, padding=15, spacing=15)
        left_sidebar.add_widget(Label())
        left_sidebar.add_widget(task_panel)
        left_sidebar.add_widget(dataset_panel)
        left_sidebar.add_widget(Label())

        self.add_widget(left_sidebar)

        dataset_info_panel = VerticalBoxLayout(spacing=15, size_hint_x=None, width=300)
        dataset_info_panel.add_widget(Label())
        dataset_info_panel.add_widget(Label(text='Dataset info'))
        dataset_info_panel.add_widget(self.info_label)
        dataset_info_panel.add_widget(Label())

        self.add_widget(dataset_info_panel)

        dataset_option_panel = VerticalBoxLayout(size_hint_x=None, width=500)
        preprocessing_panel = HorizontalBoxLayout(spacing=25, size_hint_y=None, height=40)
        preprocessing_panel.add_widget(Label(text='Dataset options'))
        self.discretize_button = ToggleButton(text='Discretize', group='preprocessing', state='normal')
        self.scale_button = ToggleButton(text='Scale', group='preprocessing')
        for btn_proc in [self.discretize_button, self.scale_button]:
            btn_proc.bind(state=self.select_preprocessing)
            preprocessing_panel.add_widget(btn_proc)
        dataset_option_panel.add_widget(Label())
        dataset_option_panel.add_widget(preprocessing_panel)
        dataset_option_panel.add_widget(Label())

        self.add_widget(dataset_option_panel)

        self.init_datasets()

    def select_dataset(self, spinner, text):
        self.dataset = text if text != DATASET_MESSAGE else None
        self.load_button.disabled = False

    def load_dataset(self, button):
        self.controller.load_dataset(self.dataset)
        self.discretize_button.disabled = False
        self.scale_button.disabled = False

    def set_dataset_info(self):
        data, classes = self.controller.data
        self.info_label.text = INFO_DATASET_MESSAGE if data is None else \
            f'Dataset: {self.dataset}\nInput variables: {len(data.columns) - 1}\nInstances: {len(data)}'
        if classes is not None and isinstance(classes[0], str):
            self.info_label.text += f'\nClasses: {len(np.unique(classes))}'

    def init_datasets(self):
        self.data_options.text = DATASET_MESSAGE
        self.data_options.values = [entry[0] for entry in DATASETS if self.task in entry[1]]
        self.load_button.disabled = True
        self.discretize_button.state = 'normal'
        self.scale_button.state = 'normal'
        self.discretize_button.disabled = True
        self.scale_button.disabled = True

    def select_task(self, button, value):
        if value == 'down':
            self.task = button.text
            self.controller.select_task()

    def select_preprocessing(self, button, value):
        self.preprocessing[button.text] = value == 'down'
