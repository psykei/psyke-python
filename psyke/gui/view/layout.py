from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton


class LeftButton(Button):

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.pos_hint = {'x': .05 + .45 * index, 'y': 0}


class CoupledToggleButton(ToggleButton):

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.pos_hint = {'x': .05 + .45 * index, 'y': 0}


class CoupledRelativeLayout(RelativeLayout):

    def __init__(self, index, ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (.575, .17 / ratio)
        self.pos_hint = {'x': 0., 'y': .95 - index * (self.size_hint[1] + .02)}


class TextLabelCoupledRelativeLayout(RelativeLayout):

    def __init__(self, label:str, text: str, filter: str, action, index: int, ratio: float = 1.0, **kwargs):
        super().__init__(size_hint=(.575, 1. / 6. / ratio), pos_hint={'x': 0., 'y': .85 - index * 1. / 6.}, **kwargs)
        self.add_widget(Label(text=label, size_hint=(.65, 1), pos_hint={'x': 0., 'y': 0.}))
        text = TextInput(text=text, input_filter=filter, size_hint=(.3, 1), pos_hint={'x': .65, 'y': 0.})
        text.bind(text=action)
        self.add_widget(text)


class PanelBoxLayout(RelativeLayout):

    def __init__(self, controller, button_text, label_text, index=1, ratio=1,
                 spinner_text=None, spinner_list=None, param_function=None, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.info_label = Label(text=label_text, size_hint=(.425, 1), pos_hint={'x': .575, 'y': 0})
        self.param_f = param_function
        self.spinner_text = spinner_text
        self.spinner_list = spinner_list

        self.spinner_options = Spinner(
            pos_hint={'center_x': .275, 'center_y': .5}, size_hint=(.45, 1)
        )
        self.spinner_options.bind(text=self.select)
        self.go_button = LeftButton(1, text=button_text, disabled=True, on_press=self.go_action)

        self.main_panel = CoupledRelativeLayout(index, ratio)
        self.main_panel.add_widget(self.spinner_options)
        self.main_panel.add_widget(self.go_button)
        #self.bind(size=self.draw)

    #def draw(self, w, v):
    #    with self.canvas.before:
    #        Color(1, 0, 0)
    #        Rectangle(pos=self.pos, size=self.size)

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


class FeatureSelectionBoxLayout(RelativeLayout):

    def __init__(self, feature, role, action, plot_action, **kwargs):
        super().__init__(size_hint=(.24, 1. / 7.), **kwargs)
        self.buttons = {}
        text = feature if len(feature) < 18 else feature[:15] + '...'
        self.add_widget(Label(text=text, size_hint=(.7, 1.), pos_hint={'x': 0, 'y': 0}))
        for i, text in enumerate(['I', 'O']):
            button = ToggleButton(text=text, size_hint=(.1, .8), pos_hint={'x': .7 + .1 * i, 'y': .1},
                                  state='down' if role == text else 'normal', group=f'feature_{feature}')
            button.bind(on_press=action)
            self.add_widget(button)
            self.buttons[text] = button
        self.plot_button = ToggleButton(text='P', size_hint=(.1, .8), pos_hint={'x': .9, 'y': .1},
                                        state='down' if role == 'O' else 'normal')
        self.plot_button.bind(on_press=plot_action)
        self.add_widget(self.plot_button)
