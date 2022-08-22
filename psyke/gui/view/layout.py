from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.togglebutton import ToggleButton


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


class FeatureSelectionBoxLayout(HorizontalBoxLayout):

    def __init__(self, feature, role, action, plot_action, **kwargs):
        super().__init__(size_hint=(None, None), size=(225, 20), **kwargs)
        self.buttons = {}
        text = feature if len(feature) < 18 else feature[:15] + '...'
        self.add_widget(Label(text=text, size_hint=(None, None), height=20, width=150))
        for text in ['I', 'O']:
            button = ToggleButton(text=text, height=20, width=25, state='down' if role == text else 'normal',
                                  group=f'feature_{feature}')
            button.bind(on_press=action)
            self.add_widget(button)
            self.buttons[text] = button
        self.plot_button = ToggleButton(text='P', height=20, width=25, state='down' if role == 'O' else 'normal')
        self.plot_button.bind(on_press=plot_action)
        self.add_widget(self.plot_button)


