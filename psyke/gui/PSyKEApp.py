from kivy.app import App
from kivy import Config
from kivy.lang import Builder

Config.set('graphics', 'resizable', 0)
Config.set('graphics', 'width', 1500)
Config.set('graphics', 'height', 770)

Builder.load_file('view/style.kv')


class PSyKEApp(App):
    from psyke.gui.controller.Controller import Controller
    from psyke.gui.model.Model import Model

    model = Model()
    controller = Controller(model)

    def build(self):
        return self.controller.screen


if __name__ == '__main__':
    PSyKEApp().run()
