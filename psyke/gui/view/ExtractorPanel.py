from functools import partial

from kivy.uix.label import Label

from psyke.gui.model import EXTRACTORS
from psyke.gui.view.layout import PanelBoxLayout, VerticalBoxLayout, SidebarBoxLayout
from psyke.gui.view import INFO_EXTRACTOR_MESSAGE, EXTRACTOR_MESSAGE, text_with_label


class ExtractorPanel(PanelBoxLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(controller, 'Extract', INFO_EXTRACTOR_MESSAGE, **kwargs)

        self.extractor = None
        self.params = {}

        self.parameter_panel = VerticalBoxLayout(size_hint_y=None, height=190)

        left_sidebar = SidebarBoxLayout()
        left_sidebar.add_widget(self.main_panel)
        left_sidebar.add_widget(self.parameter_panel)
        left_sidebar.add_widget(Label())

        self.add_widget(left_sidebar)
        self.add_widget(self.info_label)

    def init(self):
        self.spinner_options.text = EXTRACTOR_MESSAGE
        task = self.controller.get_task_from_model()
        self.spinner_options.values = [k for k, v in EXTRACTORS.items() if task in v[0]]
        self.go_button.disabled = True
        self.spinner_options.disabled = True
        self.parameter_panel.clear_widgets()
        self.set_extractor_info()

    def set_extractor_info(self):
        pass

    def select(self, spinner, text):
        if text == EXTRACTOR_MESSAGE:
            self.extractor = None
        else:
            self.controller.reset_extractor()
            self.extractor = text
            self.go_button.disabled = False
            params = EXTRACTORS[self.extractor][1]
            self.parameter_panel.clear_widgets()
            for name, (default, param_type) in params.items():
                self.parameter_panel.add_widget(
                    text_with_label(f'{name} ({default})', '', param_type, partial(self.set_param, name))
                )
            self.parameter_panel.add_widget(Label())

    def go_action(self, button):
        self.controller.train_extractor()
