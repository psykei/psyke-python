from kivy.uix.screenmanager import ScreenManager

from psyke.gui.model import DatasetError, PredictorError, SVMError
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
        return self.model.data, self.model.pruned_data, self.model.preprocessing_action, self.model.preprocessing

    def get_predictor_from_model(self):
        return self.model.predictor_name, self.model.predictor, self.model.predictor_params

    def get_extractor_from_model(self):
        return self.model.extractor_name, self.model.extractor, self.model.extractor_params

    def get_test_set_from_model(self):
        return self.model.test, self.model.preprocessing_action, self.model.preprocessing

    def get_theory_from_model(self):
        return self.model.theory

    def select_task(self, task):
        self.model.reset_dataset()
        self.model.reset_preprocessing()
        self.model.select_task(task)
        self.reset_dataset()

    def select_preprocessing(self, action, value):
        if value:
            self.model.select_preprocessing(action)
        else:
            self.model.reset_preprocessing()
        self.load_dataset()

    def select_dataset(self, dataset):
        self.model.reset_dataset()
        self.model.select_dataset(dataset)
        self.reset_predictor()
        self.view.data_panel.disable()
        self.view.data_panel.set_info()
        self.view.feature_panel.set_info()
        self.view.plot_panel.clear_data()

    def select_predictor(self, predictor):
        self.model.reset_predictor()
        self.model.select_predictor(predictor)
        self.reset_extractor()
        self.view.predictor_panel.set_info()
        self.view.plot_panel.clear_predictor()

    def select_extractor(self, extractor):
        self.model.reset_extractor()
        self.model.select_extractor(extractor)
        self.view.extractor_panel.set_info()
        self.view.theory_panel.set_info()
        self.view.plot_panel.clear_extractor()

    def reload_dataset(self, features):
        self.model.select_features(features)
        self.reset_predictor()
        self.view.data_panel.set_info()
        self.view.predictor_panel.enable()
        self.view.plot_panel.clear_data()

    def plot(self, features, plot_features):
        inputs = [k for k, v in features.items() if v == 'I' and k in plot_features]
        output = [k for k, v in features.items() if v == 'O' and k in plot_features][0]
        self.model.plot(inputs, output)
        self.view.plot_panel.set_info()

    def get_plots_from_model(self):
        return self.model.data_plot, self.model.predictor_plot, self.model.extractor_plot

    def reset_dataset(self):
        self.model.reset_dataset()
        self.reset_predictor()
        self.view.data_panel.init()
        self.view.feature_panel.init()
        self.view.data_panel.set_info()
        self.view.feature_panel.set_info()
        self.view.plot_panel.clear_data()

    def reset_predictor(self):
        self.model.reset_predictor()
        self.reset_extractor()
        self.view.predictor_panel.init()
        self.view.predictor_panel.set_info()
        self.view.plot_panel.clear_predictor()

    def reset_extractor(self):
        self.model.reset_extractor()
        self.view.extractor_panel.init()
        self.view.theory_panel.init()
        self.view.extractor_panel.set_info()
        self.view.theory_panel.set_info()
        self.view.plot_panel.clear_extractor()

    def load_dataset(self):
        self.model.reset_dataset(True)
        self.reset_predictor()
        self.view.feature_panel.init()
        try:
            self.model.load_dataset()
            self.view.data_panel.enable()
            self.view.predictor_panel.enable()
            self.view.feature_panel.set_info()
        except DatasetError as e:
            self.view.feature_panel.set_alert(e.message)
        self.view.data_panel.set_info()
        self.view.plot_panel.clear_data()

    def train_predictor(self):
        self.reset_extractor()
        self.view.plot_panel.clear_predictor()
        try:
            self.model.train_predictor()
            self.view.extractor_panel.enable()
            self.view.predictor_panel.set_info()
            self.view.feature_panel.reset_alert()
        except (PredictorError, SVMError) as e:
            self.view.feature_panel.set_alert(e.message)

    def train_extractor(self):
        self.view.plot_panel.clear_extractor()
        self.model.train_extractor()
        self.view.extractor_panel.set_info()
        self.view.theory_panel.set_info()

    def set_predictor_param(self, key, value):
        self.model.set_predictor_param(key, value)

    def set_extractor_param(self, key, value):
        self.model.set_extractor_param(key, value)
