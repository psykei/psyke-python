from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder

from psyke.gui.controller.Controller import Controller
from psyke.gui.model.Model import Model

Window.top = 50
Window.left = 10
Window.size = (1500, 770)

Builder.load_file('view/style.kv')


class PSyKEApp(App):

    model = Model()
    controller = Controller(model)

    def build(self):
        return self.controller.screen


if __name__ == '__main__':
    PSyKEApp().run()
