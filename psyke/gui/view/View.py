from kivy.graphics import Color, Rectangle, Line, RoundedRectangle
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.screenmanager import Screen

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

        self.l = .33
        self.r = 1 - self.l
        self.t = .27
        self.m = .35
        self.b = 1 - self.t - self.m
        self.pad = 4

    def init(self, controller):
        self.data_panel = DataPanel(controller, pos_hint={'x': 0., 'y': self.b + self.m}, size_hint=(self.l, self.t))
        self.feature_panel = FeaturePanel(controller, pos_hint={'x': self.l, 'y': self.b + self.m},
                                          size_hint=(self.r, self.t))
        self.predictor_panel = PredictorPanel(controller, self.m / self.t, pos_hint={'x': 0., 'y': self.b},
                                              size_hint=(self.l, self.m))
        self.plot_panel = PlotPanel(controller, pos_hint={'x': self.l, 'y': self.b}, size_hint=(self.r, self.m))
        self.extractor_panel = ExtractorPanel(controller, self.b / self.t, pos_hint={'x': 0., 'y': 0},
                                              size_hint=(self.l, self.b))
        self.theory_panel = TheoryPanel(controller, pos_hint={'x': self.l, 'y': 0}, size_hint=(self.r, self.b))

        self.data_panel.init()
        self.feature_panel.init()
        self.predictor_panel.init()
        self.plot_panel.init()
        self.extractor_panel.init()
        self.theory_panel.init()

        self.add_widget(self.data_panel)
        self.add_widget(self.feature_panel)
        self.add_widget(self.predictor_panel)
        self.add_widget(self.plot_panel)
        self.add_widget(self.extractor_panel)
        self.add_widget(self.theory_panel)
        self.bind(size=self.repaint)

    def draw(self):
        with self.canvas.before:
            Color(1, 1, 1)
            Rectangle(pos=[0, 0], size=self.size)
            Color(0, 0, 0)
            RoundedRectangle(pos=[self.pad, self.pad],
                             size=[self.width - self.pad * 2, self.height * self.b - self.pad])
            RoundedRectangle(pos=[self.pad, self.height * self.b + self.pad],
                             size=[self.width * self.l - self.pad, self.height * self.m - self.pad])
            RoundedRectangle(pos=[self.width * self.l + self.pad, self.height * self.b + self.pad],
                             size=[self.width * self.r - self.pad * 2, self.height * self.m - self.pad])
            RoundedRectangle(pos=[self.pad, self.height * (self.b + self.m) + self.pad],
                             size=[self.width - self.pad * 2, self.height * self.t - self.pad])
            Color(1, 1, 1)
            for i in range(3):
                RoundedRectangle(pos=[self.width * (self.l + self.r * i / 3) + self.pad * (2 - i),
                                      self.height * self.b + self.pad * 2],
                                 size=[(self.width * self.r - self.pad * 6) / 3, self.height * self.m - self.pad * 3])
            Line(points=[self.width * self.l * .575, 0, self.width * self.l * .575, self.height])
            Line(points=[self.width * self.l + self.pad / 2, 0, self.width * self.l + self.pad / 2, self.height])

    def repaint(self, widget, size):
        self.draw()


class RedLayout(RelativeLayout):

    def __init__(self, **kw):
        super().__init__(**kw)
        with self.canvas:
            Color(1, 0, 0)
            Rectangle(pos=self.pos, size=self.size)

    def init(self):
        pass

