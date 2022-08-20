from kivy.graphics import Color, Rectangle
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

from psyke.gui.view.layout import HorizontalBoxLayout

DATASET_MESSAGE = 'Select dataset'

INFO_DATASET_PREFIX = 'Dataset info:\n\n'

INFO_DATASET_MESSAGE = INFO_DATASET_PREFIX + 'No dataset selected\n\n'

PREDICTOR_MESSAGE = 'Select predictor'

INFO_PREDICTOR_PREFIX = 'Predictor info:\n'

INFO_PREDICTOR_MESSAGE = f'\n\n{INFO_PREDICTOR_PREFIX}\nNo predictor trained\n\n\n\n\n\n'

PREDICTOR_PERFORMANCE_PREFIX = 'Predictor performance:\n'

EXTRACTOR_MESSAGE = 'Select extractor'

INFO_EXTRACTOR_PREFIX = 'Extractor info:\n'

INFO_EXTRACTOR_MESSAGE = f'\n\n{INFO_EXTRACTOR_PREFIX}\nNo extractor trained\n\n\n\n\n\n'

THEORY_PREFIX = 'Extracted theory'

THEORY_MESSAGE = 'No extractor trained\n\n\n'

EXTRACTOR_PERFORMANCE_PREFIX = 'Extractor performance:\n'

THEORY_ERROR_MESSAGE = {
    'amount': 'Too many extracted rules\n\n\n',
    'length': 'Too long clauses\n\n\n',
}


def text_with_label(label: str, text: str, filter: str, action) -> HorizontalBoxLayout:
    box = HorizontalBoxLayout(size_hint=(None, None), width=260, height=40)
    box.add_widget(Label(text=label, size_hint=(None, None), width=190, height=30))
    text = TextInput(text=text, input_filter=filter)
    text.bind(text=action)
    box.add_widget(text)
    return box
