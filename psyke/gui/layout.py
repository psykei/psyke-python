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

    def __init__(self, controller, button_text, label_text, **kwargs):
        super().__init__(spacing=10, **kwargs)
        self.controller = controller
        self.info_label = Label()
        self.info_label = Label(text=label_text, size_hint_x=None, width=220)

        self.spinner_options = Spinner(
            size_hint_x=None, width=130, pos_hint={'center_x': .5, 'center_y': .5}
        )
        self.spinner_options.bind(text=self.select)
        self.go_button = Button(text=button_text, disabled=True, on_press=self.go_action)

        self.main_panel = HorizontalBoxLayout(size_hint=(None, None), size=(130, 40))
        self.main_panel.add_widget(self.spinner_options)
        self.main_panel.add_widget(self.go_button)

    def select(self):
        pass

    def go_action(self):
        pass

class SidebarBoxLayout(VerticalBoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(Label())
