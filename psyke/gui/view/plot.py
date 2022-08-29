import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from matplotlib import colors
from numpy.core.defchararray import capitalize

from psyke.gui.view import MARKERS, COLORS, COLOR_MAPS

plt.rcParams.update({'font.size': 9})

size = (3.75, 3.25)


def init_plot(x, y, title):
    def capitalize_label(label):
        return f'{capitalize(label[0])}{label[1:]}'

    plt.figure(figsize=size)
    plt.title(title)
    plt.gcf().patch.set_alpha(0.0)
    plt.gca().set_xlabel(capitalize_label(x.name))
    plt.gca().set_ylabel(capitalize_label(y.name))
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.gca().set_rasterized(True)
    plt.tight_layout()


def plot_regions(x: pd.Series, y: pd.Series, z: pd.Series, colormap):
    idx = [value is not None for value in z]
    if isinstance(z[idx][0], str):
        unique = np.unique(z[idx])
        classes_dict = {c: i for i, c in enumerate(unique)}
        bounds = np.array([i - 0.5 for i in range(len(unique) + 1)])
        norm = colors.BoundaryNorm(bounds, len(unique))
        plt.tricontourf(x[idx], y[idx], [classes_dict[v] for v in z[idx]],
                        cmap=colormap, norm=norm, alpha=.5, levels=bounds)
    else:
        plt.tricontourf(x[idx], y[idx], z[idx], cmap=colormap, alpha=.5, levels=10)


def plotSamples(x: pd.Series, y: pd.Series, z: pd.Series, colormap=None, index=0, name: str = None):
    if isinstance(z[0], str):
        legend_marks = [Line2D([0], [0], color=c, marker=m, lw=0) for c, m in zip(COLORS[index], MARKERS[index])]
        classes = np.unique(z)
        for i, c in enumerate(classes):
            idx = [output == c for output in z]
            plt.scatter(x[idx], y[idx], c=COLORS[index][i], marker=MARKERS[index][i])
        plt.gca().legend(legend_marks, classes)
    else:
        sc = plt.scatter(x, y, c=z, cmap=colormap,
                         edgecolor="k", linewidths=0.05, s=7)
        plt.colorbar(sc, pad=0.03, label=z.name)
    if name is not None:
        plt.savefig(name, dpi=500, bbox_inches='tight')


def create_grid(x_name: str, y_name: str, data: pd.DataFrame):
    n = [100 if feature in [x_name, y_name] else 10 for feature in data.columns[:-1]]
    vectors = [np.linspace(data[feature].min(), data[feature].max(), nn) if feature in [x_name, y_name] else
               data[feature].mean() for feature, nn in zip(data.columns[:-1], n)]
    mesh = np.meshgrid(*vectors)
    df = pd.DataFrame(np.c_[[vector.ravel() for vector in mesh]]).T
    df.columns = data.columns[:-1]
    return df
