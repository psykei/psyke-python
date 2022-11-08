from array import array
from typing import Callable, Iterable
import numpy as np
import pandas as pd
from matplotlib import colors, pyplot as plt
from matplotlib.lines import Line2D
from tuprolog.solve.prolog import prolog_solver
from tuprolog.theory import Theory, mutable_theory
from psyke.utils.logic import data_to_struct, pretty_theory, get_in_rule, get_not_in_rule


def predict_from_theory(theory: Theory, data: pd.DataFrame) -> list[float or str]:
    solver = prolog_solver(static_kb=mutable_theory(theory).assertZ(get_in_rule()).assertZ(get_not_in_rule()))
    index = data.shape[1] - 1
    y_element = data.iloc[0, -1]
    cast: Callable = lambda x: (str(x) if isinstance(y_element, str) else x)
    substitutions = [solver.solveOnce(data_to_struct(data)) for _, data in data.iterrows()]
    return [cast(query.solved_query.get_arg_at(index)) if query.is_yes else -1 for query in substitutions]


def plot_theory(theory: Theory, data: pd.DataFrame = None, output: str = 'plot.pdf', azimuth: float = 45,
                distance: float = 9, elevation: float = 5, show_theory: bool = True, features: Iterable[str] = None) -> None:
    # Check if the number of common variables in clauses is less or equal to three.
    # If not raise an exception.
    fresh_theory = mutable_theory(theory)
    clauses = fresh_theory.clauses
    variables = sorted(list(set(arg.args[0].name.split('_')[0] for clause in clauses if clause.body_size > 0 and clause.body.is_recursive for arg in clause.body.unfolded)), reverse=True)
    if len(variables) > 3:
        raise Exception("Theory contains too many different features in the body of clauses, maximum is 3.")
    # If data is None, then create synthetic data covering a good portion of the variables space.
    # Just skip for now.
    if data is None:
        raise Exception("Method without data is not implemented yet")

    # Prepare data
    ys = predict_from_theory(fresh_theory, data)
    xs = data[variables].values.tolist()
    for i in range(len(ys)):
        xs[i].append(ys[i])

    # Prepare colors
    if isinstance(ys[0], str):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        class ColorGenerator:

            def __init__(self):
                self.color_list = ['red', 'royalblue', 'green', 'orange', 'pink', 'acqua', 'grey']
                self.counter = 0

            def get_new_color(self) -> str:
                self.counter += 1
                if self.counter > len(self.color_list):
                    raise Exception("Classes exceed the maximum supported number (7)")
                return self.color_list[self.counter - 1]

        classes = set(ys)
        generator = ColorGenerator()
        class_color = {c: generator.get_new_color() for c in classes}
        get_color: Callable = lambda c: class_color[c]
    else:
        def color_fader(v: float = 0., c1: str = 'green', c2: str = 'red'):
            c1 = array(colors.to_rgb(c1))
            c2 = array(colors.to_rgb(c2))
            return colors.to_hex((1 - v) * c1 + v * c2)
        min_value = min(ys)
        max_value = max(ys)
        get_normalized_value: Callable = lambda v: (v - min_value)/(max_value - min_value)
        get_color: Callable = lambda c: color_fader(get_normalized_value(c))

    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(projection='3d' if len(variables) == 3 else '2d')

    for x in xs:
        ax.scatter(*x[:-1], c=get_color(x[-1]), s=14)

    ax.set_xlabel(variables[0], fontsize=18)
    ax.set_ylabel(variables[1], fontsize=18)
    if len(variables) == 3:
        ax.set_zlabel(variables[2], fontsize=18)

    ax.azim = azimuth
    ax.dist = distance
    ax.elev = elevation
    ax.set_title('Predictions according to Prolog theory', fontsize=24)
    if show_theory:
        ax.text2D(0., 0.88, pretty_theory(theory, new_line=False), transform=ax.transAxes, fontsize=8)
    if isinstance(ys[0], str):
        custom_lines = [Line2D([0], [0], marker='o', markerfacecolor=get_color(c), markersize=20, color='w') for c in classes]
        ax.legend(custom_lines, classes, loc='upper left', numpoints=1, ncol=3, fontsize=18, bbox_to_anchor=(0, 0))
    plt.savefig(output, format='pdf')
