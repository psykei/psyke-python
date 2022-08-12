from kivy.uix.label import Label
from psyke.gui.layout import PanelBoxLayout
from psyke.gui.model import INFO_EXTRACTOR_MESSAGE


class ExtractorPanel(PanelBoxLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(controller, 'Train', INFO_EXTRACTOR_MESSAGE, **kwargs)
        self.add_widget(Label(text='ccc'))
        self.add_widget(Label(text='ccc'))
        self.add_widget(Label(text='ccc'))
        self.add_widget(Label(text='ccc'))
        #self.controller.set_predictor_panel(self)
