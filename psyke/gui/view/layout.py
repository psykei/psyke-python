from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner


class VerticalBoxLayout(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'


class HorizontalBoxLayout(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'


class PanelBoxLayout(HorizontalBoxLayout):

    def __init__(self, controller, button_text, label_text, label_height=250,
                 spinner_text=None, spinner_list=None, param_function=None, **kwargs):
        super().__init__(spacing=10, **kwargs)
        self.controller = controller
        self.info_label = Label(text=label_text, size_hint=(None, None), height=label_height, width=220)
        self.param_f = param_function
        self.spinner_text = spinner_text
        self.spinner_list = spinner_list

        self.spinner_options = Spinner(
            pos_hint={'center_x': .5, 'center_y': .5}
        )
        self.spinner_options.bind(text=self.select)
        self.go_button = Button(text=button_text, disabled=True, on_press=self.go_action)

        self.main_panel = HorizontalBoxLayout(size_hint_y=None, height=40)
        self.main_panel.add_widget(self.spinner_options)
        self.main_panel.add_widget(self.go_button)

    def select(self, spinner, text):
        pass

    def go_action(self, button):
        pass

    def enable(self):
        self.spinner_options.disabled = False

    def set_param(self, key, widget, value):
        self.param_f(key, value)

    def init(self):
        self.spinner_options.text = self.spinner_text
        task = self.controller.get_task_from_model()
        self.spinner_options.values = [k for k, v in self.spinner_list.items() if task in v[0]]
        self.go_button.disabled = True
        self.spinner_options.disabled = True
        self.parameter_panel.clear_widgets()
        self.set_info()

    def set_info(self):
        pass


class SidebarBoxLayout(VerticalBoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(Label())
