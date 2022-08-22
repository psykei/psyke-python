def main(resizable=False, width=1500, height=770):
    from kivy import Config
    from kivy.app import App
    from kivy.lang import Builder

    Config.set('graphics', 'resizable', 1 if resizable else 0)
    Config.set('graphics', 'width', width)
    Config.set('graphics', 'height', height)

    Builder.load_file('view/style.kv')

    class PSyKEApp(App):
        from psyke.gui.controller.Controller import Controller
        from psyke.gui.model.Model import Model

        model = Model()
        controller = Controller(model)

        def build(self):
            return self.controller.screen


    PSyKEApp().run()
