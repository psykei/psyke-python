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

    def get_predictor_from_model(self):
        return self.model.predictor_name, self.model.predictor, self.model.predictor_params

    def get_extractor_from_model(self):
        return self.model.extractor_name, self.model.extractor, self.model.extractor_params

    def get_test_set_from_model(self):
        return self.model.test

    def select_task(self, task):
        self.model.reset_dataset()
        self.model.select_task(task)
        self.reset_dataset()

    def select_dataset(self, dataset):
        self.model.reset_dataset()
        self.model.select_dataset(dataset)
        self.reset_predictor()
        self.view.data_panel.set_info()

    def select_predictor(self, predictor):
        self.model.reset_predictor()
        self.model.select_predictor(predictor)
        self.reset_extractor()
        self.view.predictor_panel.set_info()

    def select_extractor(self, extractor):
        self.model.reset_extractor()
        self.model.select_extractor(extractor)
        self.view.extractor_panel.set_info()

    def reset_dataset(self):
        self.model.reset_dataset()
        self.reset_predictor()
        self.view.data_panel.init()
        self.view.data_panel.set_info()

    def reset_predictor(self):
        self.model.reset_predictor()
        self.reset_extractor()
        self.view.predictor_panel.init()
        self.view.predictor_panel.set_info()

    def reset_extractor(self):
        self.model.reset_extractor()
        self.view.extractor_panel.init()
        self.view.extractor_panel.set_info()

    def load_dataset(self):
        self.model.load_dataset()
        self.reset_predictor()
        self.view.data_panel.set_info()
        self.view.predictor_panel.enable()

    def train_predictor(self):
        self.model.train_predictor()
        self.reset_extractor()
        self.view.predictor_panel.set_info()
        self.view.extractor_panel.enable()

    def train_extractor(self):
        self.model.train_extractor()
        self.view.extractor_panel.set_info()

    def set_predictor_param(self, key, value):
        self.model.set_predictor_param(key, value)

    def set_extractor_param(self, key, value):
        self.model.set_extractor_param(key, value)
