from typing import Union

from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget


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
        self.size_hint = (.575, .2 / ratio)
        self.pos_hint = {'x': 0., 'y': .95 - index * (self.size_hint[1] + .02)}


class WidgetLabelCoupledRelativeLayout(RelativeLayout):

    def __init__(self, widget: Widget, action, label: str, index: int, ratio: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (.575, 1. / 6. / ratio)
        self.pos_hint = {'x': 0., 'y': .75 - index / 6. / ratio}
        self.add_widget(Label(text=label, size_hint=(.6, 1), pos_hint={'x': 0., 'y': 0.}))
        widget.size_hint = (.35, 1)
        widget.pos_hint = {'x': .6, 'y': 0.}
        if isinstance(widget, TextInput):
            widget.bind(text=action)
        elif isinstance(widget, CheckBox):
            widget.bind(active=action)
        else:
            raise NotImplementedError
        self.add_widget(widget)


class TextLabelCoupledRelativeLayout(WidgetLabelCoupledRelativeLayout):

    def __init__(self, label: str, text: str, filter: str, action, index: int, ratio: float = 1.0, **kwargs):
        super().__init__(TextInput(text=text, input_filter=filter), action, label, index, ratio, **kwargs)


class RadioLabelCoupledRelativeLayout(WidgetLabelCoupledRelativeLayout):

    def __init__(self, label: str, default: bool, action, index: int, ratio: float = 1.0, **kwargs):
        super().__init__(CheckBox(active=default), action, label, index, ratio, **kwargs)


class SpinnerLabelCoupledRelativeLayout(RelativeLayout):

    def __init__(self, label: str, default: str, options: list, action, index: int, ratio: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (.575, 1. / 6. / ratio)
        self.pos_hint = {'x': 0., 'y': .75 - index / 6. / ratio}
        self.add_widget(Label(text=label, size_hint=(.6, 1), pos_hint={'x': 0., 'y': 0.}))
        spinner = Spinner(text=default, values=options, pos_hint={'center_x': .775, 'center_y': .5}, size_hint=(.35, 1))
        spinner.bind(text=action)
        self.add_widget(spinner)


class PanelBoxLayout(RelativeLayout):

    def __init__(self, controller, button_text, label_text, index=1, ratio=1,
                 spinner_text=None, spinner_list=None, param_function=None, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.info_label = Label(text=label_text, size_hint=(.425, 1), pos_hint={'x': .575, 'y': 0})
        self.param_f = param_function
        self.spinner_text = spinner_text
        self.spinner_list = spinner_list
        self.ratio = ratio

        self.spinner_options = Spinner(
            pos_hint={'center_x': .275, 'center_y': .5}, size_hint=(.45, 1)
        )
        self.spinner_options.bind(text=self.select)
        self.go_button = LeftButton(1, text=button_text, disabled=True, on_press=self.go_action)

        self.main_panel = CoupledRelativeLayout(index, ratio)
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


def create_param_layout(name: str, default: Union[str, bool, int, float], type, action, index: int, ratio: float):
    if isinstance(type, list):
        widget = SpinnerLabelCoupledRelativeLayout(name, default, type, action, index, ratio)
    elif type == 'bool':
        widget = RadioLabelCoupledRelativeLayout(f'{name}', default, action, index, ratio)
    else:
        widget = TextLabelCoupledRelativeLayout(f'{name} ({default})', '', type, action, index, ratio)
    return widget
