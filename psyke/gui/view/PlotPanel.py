from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout


class PlotPanel(RelativeLayout):

    def __init__(self, controller, **kwargs):
        super().__init__(size=(1000, 300), **kwargs)
        self.controller = controller

        size = (318, 266)
        y = 27

        self.data_plot_container = RelativeLayout(pos=(7, y), size=size)
        self.predictor_plot_container = RelativeLayout(pos=(330, y), size=size)
        self.extractor_plot_container = RelativeLayout(pos=(653, y), size=size)

        self.data_plot = None
        self.predictor_plot = None
        self.extractor_plot = None

        self.add_widget(self.data_plot_container)
        self.add_widget(self.predictor_plot_container)
        self.add_widget(self.extractor_plot_container)

        self.init()

    def init(self):
        self.clear_data()

    def __add_widgets(self):
        self.data_plot_container.clear_widgets()
        self.data_plot_container.add_widget(self.data_plot)
        self.predictor_plot_container.clear_widgets()
        self.predictor_plot_container.add_widget(self.predictor_plot)
        self.extractor_plot_container.clear_widgets()
        self.extractor_plot_container.add_widget(self.extractor_plot)

    def set_info(self):
        plots = self.controller.get_plots_from_model()
        if plots[0] is not None:
            self.data_plot = FigureCanvasKivyAgg(plots[0], size_hint=(.33, .965))
        if plots[1] is not None:
            self.predictor_plot = FigureCanvasKivyAgg(plots[1], size_hint=(.33, .965))
        if plots[2] is not None:
            self.extractor_plot = FigureCanvasKivyAgg(plots[2], size_hint=(.33, .965))
        self.__add_widgets()

    def clear_data(self):
        self.data_plot = Label(text='Load a dataset\n\n', color=(0, 0, 0), pos=(-335, 0))
        self.clear_predictor()

    def clear_predictor(self):
        self.predictor_plot = Label(text='Train a predictor\n\n', color=(0, 0, 0), pos=(-335, 0))
        self.clear_extractor()

    def clear_extractor(self):
        self.extractor_plot = Label(text='Train an extractor\n\n', color=(0, 0, 0), pos=(-335, 0))
        self.__add_widgets()
