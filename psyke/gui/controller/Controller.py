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

    def load_dataset(self):
        self.model.load_dataset(self.view.data_panel.dataset)
        self.view.data_panel.set_dataset_info()
        self.view.predictor_panel.enable_predictors()

    def get_data_from_model(self):
        return self.model.data

    def get_predictor_from_model(self, with_params=True):
        if with_params:
            return self.model.predictor, self.model.predictor_params
        else:
            return self.model.predictor

    def get_test_set_from_model(self):
        return self.model.test

    def get_task_from_model(self):
        return self.model.task

    def select_task(self):
        self.model.reset_dataset()
        self.model.select_task(self.view.data_panel.task)
        self.reset_dataset()

    def reset_dataset(self):
        self.view.data_panel.init_datasets()
        self.view.data_panel.set_dataset_info()
        self.reset_predictor()

    def reset_predictor(self):
        self.model.reset_predictor()
        self.view.predictor_panel.init_predictors()
        self.view.predictor_panel.set_predictor_info()

    def train_predictor(self):
        self.model.train_predictor(self.view.predictor_panel.predictor, self.view.predictor_panel.params)
        self.view.predictor_panel.set_predictor_info()
