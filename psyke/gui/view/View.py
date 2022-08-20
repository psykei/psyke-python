from kivy.graphics import Color, Rectangle, Line, RoundedRectangle
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

from psyke.gui.view.DataPanel import DataPanel
from psyke.gui.view.ExtractorPanel import ExtractorPanel
from psyke.gui.view.FeaturePanel import FeaturePanel
from psyke.gui.view.PlotPanel import PlotPanel
from psyke.gui.view.PredictorPanel import PredictorPanel
from psyke.gui.view.TheoryPanel import TheoryPanel


class View(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MainScreen(View):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_panel = None
        self.feature_panel = None
        self.predictor_panel = None
        self.plot_panel = None
        self.extractor_panel = None
        self.theory_panel = None

        with self.canvas:
            Color(1, 1, 1)
            Rectangle(pos=[0, 0], size=Window.size)
            Color(0, 0, 0)
            RoundedRectangle(pos=[4, 4], size=[Window.width - 8, 294])
            RoundedRectangle(pos=[4, 302], size=[514, 276])
            RoundedRectangle(pos=[522, 302], size=[Window.width - 526, 276])
            RoundedRectangle(pos=[4, 582], size=[Window.width - 8, 184])
            Color(1, 1, 1)
            RoundedRectangle(pos=[527, 307], size=[318, 266])
            RoundedRectangle(pos=[850, 307], size=[318, 266])
            RoundedRectangle(pos=[1173, 307], size=[318, 266])
            RoundedRectangle(pos=[850, 710], size=[600, 40])
            Line(points=[300, 0, 300, Window.height])
            Line(points=[520, 0, 520, Window.height])

    def init(self, controller):
        self.data_panel = DataPanel(controller)
        self.feature_panel = FeaturePanel(controller)
        self.predictor_panel = PredictorPanel(controller)
        self.plot_panel = PlotPanel(controller)
        self.extractor_panel = ExtractorPanel(controller)
        self.theory_panel = TheoryPanel(controller)

        self.data_panel.init()
        self.feature_panel.init()
        self.predictor_panel.init()
        self.plot_panel.init()
        self.extractor_panel.init()
        self.theory_panel.init()

        layout = GridLayout(cols=2, spacing=0, rows_minimum={0: 200, 1: 270, 2: 285})
        layout.add_widget(self.data_panel)
        layout.add_widget(self.feature_panel)
        layout.add_widget(self.predictor_panel)
        layout.add_widget(self.plot_panel)
        layout.add_widget(self.extractor_panel)
        layout.add_widget(self.theory_panel)
        self.add_widget(layout)
