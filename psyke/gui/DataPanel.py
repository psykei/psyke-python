import numpy as np
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.togglebutton import ToggleButton

from psyke.gui.layout import PanelBoxLayout, SidebarBoxLayout, HorizontalBoxLayout
from psyke.gui.model import TASKS, DATASET_MESSAGE, DATASETS, INFO_DATASET_MESSAGE, INFO_DATASET_PREFIX


class DataPanel(PanelBoxLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(controller, 'Load', INFO_DATASET_MESSAGE, **kwargs)
        self.controller.set_data_panel(self)

        self.task = TASKS[0]
        self.dataset = None
        self.preprocessing = {}

        task_panel = HorizontalBoxLayout()
        for task in TASKS:
            btn_task = ToggleButton(text=task, group='task', state='down' if self.task == task else 'normal')
            btn_task.bind(state=self.select_task)
            task_panel.add_widget(btn_task)

        processing_panel = HorizontalBoxLayout(size_hint=(None, None), size=(130, 40))
        self.discretize_button = ToggleButton(text='Discretize')
        self.scale_button = ToggleButton(text='Scale')
        for btn_proc in [self.discretize_button, self.scale_button]:
            btn_proc.state = 'normal'
            btn_proc.group = 'preprocessing'
            btn_proc.bind(state=self.select_preprocessing)
            processing_panel.add_widget(btn_proc)

        left_sidebar = SidebarBoxLayout()
        left_sidebar.add_widget(task_panel)
        left_sidebar.add_widget(self.main_panel)
        left_sidebar.add_widget(processing_panel)
        left_sidebar.add_widget(Label())

        self.add_widget(left_sidebar)
        self.add_widget(self.info_label)

        self.init_datasets()

    def select(self, spinner, text):
        self.dataset = text if text != DATASET_MESSAGE else None
        self.go_button.disabled = False

    def go_action(self, button):
        self.controller.load_dataset()
        if self.task == 'Classification':
            self.discretize_button.disabled = False
        else:
            self.scale_button.disabled = False

    def set_dataset_info(self):
        data = self.controller.data
        self.info_label.text = INFO_DATASET_MESSAGE if data is None else \
            INFO_DATASET_PREFIX + \
            f'Dataset: {self.dataset}\nInput variables: {len(data.columns) - 1}\nInstances: {len(data)}'
        if data is not None and isinstance(data.iloc[0, -1], str):
            self.info_label.text += f'\nClasses: {len(np.unique(data.iloc[:, -1]))}'
        self.info_label.text += '\n'

    def init_datasets(self):
        self.spinner_options.text = DATASET_MESSAGE
        self.spinner_options.values = [entry[0] for entry in DATASETS if self.task in entry[1]]
        self.go_button.disabled = True
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
