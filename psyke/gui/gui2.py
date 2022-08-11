from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

from psyke.gui import VerticalBoxLayout
from psyke.gui.layout import HorizontalBoxLayout
from psyke.gui.panels import DataPanel

Window.top = 50
Window.left = 10
Window.size = (1400, 750)

Builder.load_file('panels.kv')


class MainScreen(Screen):

    def __init__(self, **kw):
        super().__init__(**kw)
        body = VerticalBoxLayout()
        body.add_widget(DataPanel())
        body.add_widget(HorizontalBoxLayout())
        body.add_widget(HorizontalBoxLayout())
        self.add_widget(body)


class PSyKEApp(App):

    def build(self):
        screen_manager = ScreenManager()
        screen_manager.add_widget(MainScreen(name="main"))
        return screen_manager


if __name__ == '__main__':
    PSyKEApp().run()
