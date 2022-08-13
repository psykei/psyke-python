from kivy.uix.screenmanager import ScreenManager

from psyke.gui.view.View import MainScreen


class Controller:

    def __init__(self, model):
        self.model = model

        self.screen_manager = ScreenManager()
        self.main_screen = MainScreen(name="main")
        self.screen_manager.add_widget(self.main_screen)

        self.view = self.main_screen
        self.view.init(self)

    @property
    def screen(self):
        return self.screen_manager

    def get_task_from_model(self):
        return self.model.task

    def get_dataset_from_model(self):
        return self.model.dataset

    def get_data_from_model(self):
        return self.model.data

    def get_predictor_from_model(self, with_params_and_name=True):
        if with_params_and_name:
            return self.model.predictor_name, self.model.predictor, self.model.predictor_params
        else:
            return self.model.predictor

    def get_test_set_from_model(self):
        return self.model.test

    def select_task(self, task):
        self.model.reset_dataset()
        self.model.select_task(task)
        self.reset_dataset()

    def select_dataset(self, dataset):
        self.model.reset_dataset()
        self.model.reset_predictor()
        self.model.select_dataset(dataset)
        self.reset_predictor()
        self.view.data_panel.set_dataset_info()

    def select_predictor(self, predictor):
        self.model.reset_predictor()
        self.model.reset_extractor()
        self.model.select_predictor(predictor)
        self.reset_extractor()
        self.view.predictor_panel.set_predictor_info()

    def reset_dataset(self):
        self.reset_predictor()
        self.view.data_panel.init()
        self.view.data_panel.set_dataset_info()

    def reset_predictor(self):
        self.model.reset_predictor()
        self.reset_extractor()
        self.view.predictor_panel.init()
        self.view.predictor_panel.set_predictor_info()

    def reset_extractor(self):
        self.model.reset_extractor()
        self.view.extractor_panel.init()
        self.view.extractor_panel.set_extractor_info()

    def load_dataset(self):
        self.model.load_dataset()
        self.view.data_panel.set_dataset_info()
        self.view.predictor_panel.enable()

    def train_predictor(self):
        self.model.train_predictor(self.view.predictor_panel.params)
        self.view.predictor_panel.set_predictor_info()
        self.view.extractor_panel.enable()

    def train_extractor(self):
        # self.model.train_extractor(...)
        self.view.extractor_panel.set_extractor_info()
