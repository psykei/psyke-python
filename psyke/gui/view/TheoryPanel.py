from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout

from psyke.gui.view import THEORY_PREFIX, THEORY_MESSAGE, THEORY_ERROR_MESSAGE
from psyke.utils.logic import pretty_theory


class TheoryPanel(RelativeLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller

        self.add_widget(Label(text=THEORY_PREFIX, size_hint=(1, .17), pos_hint={'x': -.42, 'y': .77}))
        self.theory_label = Label(size_hint=(1, .83))
        self.add_widget(self.theory_label)

    def init(self):
        self.theory_label.text = THEORY_MESSAGE
        self.theory_label.pos_hint = {'x': -.2, 'y': 0.}

    def set_info(self):
        theory = self.controller.get_theory_from_model()
        if theory is not None:
            n_rules = len(list(theory.clauses))
            theory = pretty_theory(theory)
            if n_rules > 6:
                self.theory_label.text = THEORY_ERROR_MESSAGE['amount']
            else:
                length = len(max(theory.split('\n'), key=len))
                if length > 150:
                    self.theory_label.text = THEORY_ERROR_MESSAGE['length']
                else:
                    self.theory_label.text = theory + '\n' * (6 - n_rules) * 2
                    self.theory_label.pos_hint = {'x': 0., 'y': 0.}
        else:
            self.init()

