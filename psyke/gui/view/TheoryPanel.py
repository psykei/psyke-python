from kivy.uix.label import Label

from psyke.gui.view import THEORY_PREFIX, THEORY_MESSAGE, THEORY_ERROR_MESSAGE
from psyke.gui.view.layout import VerticalBoxLayout
from psyke.utils.logic import pretty_theory


class TheoryPanel(VerticalBoxLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(size_hint_x=None, width=900, **kwargs)
        self.controller = controller

        self.top_label = Label(text=THEORY_PREFIX, size_hint=(None, None), size=(175, 40))
        self.theory_label = Label(text=THEORY_MESSAGE, size_hint=(None, None), size=(950, 250))
        self.add_widget(self.top_label)
        self.add_widget(self.theory_label)

    def init(self):
        self.theory_label.text = THEORY_MESSAGE

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
        else:
            self.theory_label.text = THEORY_MESSAGE

