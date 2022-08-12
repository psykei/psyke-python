from kivy.app import App
from kivy.core.window import Window
from kivy.graphics import Line, Color, Rectangle
from kivy.lang import Builder
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.screenmanager import ScreenManager, Screen

from psyke.gui import VerticalBoxLayout
from psyke.gui.Controller import Controller
from psyke.gui.ExtractorPanel import ExtractorPanel
from psyke.gui.PredictorPanel import PredictorPanel
from psyke.gui.layout import HorizontalBoxLayout
from psyke.gui.DataPanel import DataPanel

Window.top = 50
Window.left = 10
Window.size = (1400, 750)

Builder.load_file('panels.kv')


class MainScreen(Screen):

    def __init__(self, **kw):
        super().__init__(**kw)
        controller = Controller()
        with self.canvas:
            Color(1, 1, 1)
            Rectangle(pos=[0, 0], size=Window.size)
            Color(0, 0, 0)
            Rectangle(pos=[4, 4], size=[Window.width - 8, 244])
            Rectangle(pos=[4, 252], size=[Window.width - 8, 296])
            Rectangle(pos=[4, 552], size=[Window.width - 8, 196])
            Color(1, 1, 1)
            Line(points=[300, 0, 300, Window.height])
            Line(points=[520, 0, 520, Window.height])
            Line(points=[800, 0, 800, Window.height])
        layout = GridLayout(cols=1, spacing=0, rows_minimum={0: 200, 1: 300, 2: 250})
        layout.add_widget(DataPanel(controller))
        layout.add_widget(PredictorPanel(controller))
        layout.add_widget(ExtractorPanel(controller))
        self.add_widget(layout)


class PSyKEApp(App):

    def build(self):
        screen_manager = ScreenManager()
        screen_manager.add_widget(MainScreen(name="main"))
        return screen_manager


if __name__ == '__main__':
    PSyKEApp().run()
