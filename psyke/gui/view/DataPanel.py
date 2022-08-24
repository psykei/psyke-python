import numpy as np
from psyke.gui.view.layout import PanelBoxLayout, CoupledRelativeLayout, CoupledToggleButton
from psyke.gui.model import TASKS, DATASETS
from psyke.gui.view import DATASET_MESSAGE, INFO_DATASET_MESSAGE, INFO_DATASET_PREFIX


class DataPanel(PanelBoxLayout):

    def __init__(self, controller, ratio=1, **kwargs):
        super().__init__(controller, 'Load', INFO_DATASET_MESSAGE, 2, ratio, **kwargs)

        task_panel = CoupledRelativeLayout(1, ratio)
        for i, task in enumerate(TASKS):
            btn_task = CoupledToggleButton(i, text=task, group='task', state='down'
                                           if self.controller.get_task_from_model() == task else 'normal')
            btn_task.bind(state=self.select_task)
            task_panel.add_widget(btn_task)

        processing_panel = CoupledRelativeLayout(3, ratio)
        self.discretize_button = CoupledToggleButton(0, text='Discretize')
        self.scale_button = CoupledToggleButton(1, text='Scale')
        for btn_proc in [self.discretize_button, self.scale_button]:
            btn_proc.state = 'normal'
            btn_proc.group = 'preprocessing'
            btn_proc.bind(state=self.select_preprocessing)
            processing_panel.add_widget(btn_proc)

        self.add_widget(task_panel)
        self.add_widget(self.main_panel)
        self.add_widget(processing_panel)
        self.add_widget(self.info_label)

    def select(self, spinner, text):
        self.controller.select_dataset(text if text != DATASET_MESSAGE else None)
        self.go_button.disabled = False

    def go_action(self, button):
        self.controller.load_dataset()

    def enable(self):
        self.discretize_button.disabled = False
        self.scale_button.disabled = False

    def disable(self):
        self.discretize_button.disabled = True
        self.scale_button.disabled = True

    def set_info(self):
        data, pruned_data, _, _ = self.controller.get_data_from_model()
        if pruned_data is not None:
            data = pruned_data
        self.info_label.text = INFO_DATASET_MESSAGE if data is None else \
            INFO_DATASET_PREFIX + \
            f'Dataset: {self.controller.get_dataset_from_model()}\n' \
            f'Input variables: {len(data.columns) - 1}\nInstances: {len(data)}'
        if data is not None and isinstance(data.iloc[0, -1], str):
            self.info_label.text += f'\nClasses: {len(np.unique(data.iloc[:, -1]))}'

    def init(self):
        self.spinner_options.text = DATASET_MESSAGE
        self.spinner_options.values = [entry[0] for entry in DATASETS
                                       if self.controller.get_task_from_model() in entry[1]]
        self.go_button.disabled = True
        self.discretize_button.state = 'normal'
        self.scale_button.state = 'normal'
        self.disable()

    def select_task(self, button, value):
        if value == 'down':
            self.controller.select_task(button.text)

    def select_preprocessing(self, button, value):
        self.controller.select_preprocessing(button.text, value == 'down')
