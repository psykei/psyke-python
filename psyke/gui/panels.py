from kivy.properties import StringProperty
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner

from psyke.gui import HorizontalBoxLayout


class DataPanel(HorizontalBoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = StringProperty()
        data_options = Spinner(
            text='Select dataset', values=['Iris', 'Wine', 'Arti', 'House', 'Custom'],
            size_hint=(None, None), size=(150, 35),
            pos_hint={'center_x': .5, 'center_y': .5}
        )
        data_options.bind(text=self.change_dataset)
        self.add_widget(data_options)

    def change_dataset(self, spinner, text):
        self.dataset = text
