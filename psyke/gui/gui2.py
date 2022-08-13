from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder

from psyke.gui.controller.Controller import Controller
from psyke.gui.model.Model import Model
from psyke.gui.view.View import View

Window.top = 50
Window.left = 10
Window.size = (1400, 750)

Builder.load_file('view/panels.kv')


class PSyKEApp(App):

    model = Model()
    controller = Controller(model)

    def build(self):
        return self.controller.screen


if __name__ == '__main__':
    PSyKEApp().run()
