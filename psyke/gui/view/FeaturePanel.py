from functools import partial

from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label

from psyke.gui.view import HorizontalBoxLayout
from psyke.gui.view.layout import VerticalBoxLayout, FeatureSelectionBoxLayout


class FeaturePanel(VerticalBoxLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(spacing=10, **kwargs)
        self.controller = controller
        self.features = {}
        self.feature_panels = {}
        self.plot_features = {}

        self.top_label = Label(text='Feature selection')
        self.top_button = Button(text='Reload')
        self.alert_label = Label(size_hint_x=None, width=600)

        top_panel = HorizontalBoxLayout(size_hint=(None, None), size=(900, 40), spacing=-10)
        top_panel.add_widget(self.top_label)
        top_panel.add_widget(self.top_button)
        top_panel.add_widget(self.alert_label)
        self.add_widget(top_panel)

        self.feature_panel = GridLayout(rows=5, size_hint=(None, None), size=(800, 150),
                                        spacing=(15, 2), padding=(15, 0))
        self.add_widget(self.feature_panel)

    def init(self):
        self.top_button.disabled = True
        self.feature_panel.clear_widgets()
        self.feature_panel.add_widget(Label(text='No dataset selected'))
        self.feature_panel.add_widget(Label())

    def set_info(self):
        dataset = self.controller.get_data_from_model()
        if dataset is not None:
            self.feature_panel.clear_widgets()
            self.top_button.disabled = False
            for feature in dataset.columns:
                self.features[feature] = 'O' if feature == dataset.columns[-1] else 'I'
                self.plot_features[feature] = feature == dataset.columns[-1]
                self.feature_panels[feature] = FeatureSelectionBoxLayout(feature, self.features[feature],
                                                                         partial(self.set_feature, feature),
                                                                         partial(self.set_plot_feature, feature))
                self.feature_panel.add_widget(self.feature_panels[feature])
        else:
            self.init()

    def set_feature(self, feature, button):
        if self.features[feature] == button.text:
            if len([k for k, v in self.features.items() if v == button.text]) > 1:
                self.features[feature] = None
                self.feature_panels[feature].plot_button.state = 'normal'
                self.plot_features[feature] = False
                self.alert_label.text = ''
            else:
                self.alert_label.text = \
                    f'Cannot remove the only selected {"input" if button.text == "I" else "output"} feature'
                self.feature_panels[feature].buttons[button.text].state = 'down'
        else:
            if button.text == 'I':
                if self.features[feature] is None:
                    self.features[feature] = button.text
                    self.alert_label.text = ''
                else:
                    self.alert_label.text = f'Cannot remove the only selected output feature'
                    self.feature_panels[feature].buttons['I'].state = 'normal'
                    self.feature_panels[feature].buttons['O'].state = 'down'
            else:
                if len([k for k, v in self.features.items() if v == 'I' and k != feature]) == 0:
                    self.alert_label.text = f'Cannot remove the only selected input feature'
                    self.feature_panels[feature].buttons['O'].state = 'normal'
                    self.feature_panels[feature].buttons['I'].state = 'down'
                else:
                    self.features[feature] = button.text
                    to_normal = [k for k, v in self.features.items() if v == 'O' and k != feature]
                    if len(to_normal) > 0:
                        self.feature_panels[to_normal[0]].buttons['O'].state = 'normal'
                        self.features[to_normal[0]] = None
                        self.feature_panels[to_normal[0]].plot_button.state = 'normal'
                        self.plot_features[to_normal[0]] = False
                    self.feature_panels[feature].plot_button.state = 'down'
                    self.plot_features[feature] = True
                    self.alert_label.text = ''
        self.check()

    def set_plot_feature(self, feature, button):
        if self.features[feature] is None:
            self.alert_label.text = 'Cannot plot non-selected features'
            self.feature_panels[feature].plot_button.state = 'normal'
        elif self.features[feature] == 'O':
            self.alert_label.text = 'Output feature must be plotted'
            self.feature_panels[feature].plot_button.state = 'down'
        else:
            inputs = [k for k, v in self.features.items() if v == 'I']
            active = len([k for k, v in self.plot_features.items() if v and k in inputs])
            if (self.plot_features[feature] and active == 2) or (not self.plot_features[feature] and active < 2):
                self.plot_features[feature] = ~self.plot_features[feature]
                self.alert_label.text = ''
            else:
                self.alert_label.text = 'Cannot plot less than 1 input feature' if self.plot_features[feature] else \
                    'Cannot plot more than 2 input features'
                state = self.feature_panels[feature].plot_button.state
                self.feature_panels[feature].plot_button.state = 'down' if state == 'normal' else 'normal'

        self.check()

    def check(self):
        outputs = [k for k, v in self.features.items() if v == 'O']
        inputs = [k for k, v in self.features.items() if v == 'I']
        input_plots = len([k for k, v in self.plot_features.items() if v and k in inputs])
        output_plots = len([k for k, v in self.plot_features.items() if v and k in outputs])

        # self.alert_label.text += f'{len(inputs)} {len(outputs)} {input_plots} {output_plots}'
