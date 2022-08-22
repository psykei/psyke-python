import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from matplotlib import colors
from numpy.core.defchararray import capitalize

plt.rcParams.update({'font.size': 9})

size = (3.75, 3.25)
markers = ['x', '*', '+', 'o', 's', '^', 'D']
color_list = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:olive', 'tab:pink']

cmap = colors.ListedColormap(color_list)
legend_marks = [Line2D([0], [0], color=c, marker=m, lw=0) for c, m in zip(color_list, markers)]


def init_plot(x, y, title):
    plt.figure(figsize=size)
    plt.title(title)
    plt.gcf().patch.set_alpha(0.0)
    plt.gca().set_xlabel(capitalize(x.name))
    plt.gca().set_ylabel(capitalize(y.name))
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.gca().set_rasterized(True)
    plt.tight_layout()


def plot_regions(x: pd.Series, y: pd.Series, z: pd.Series):
    idx = [value is not None for value in z]
    if isinstance(z[idx][0], str):
        unique = np.unique(z[idx])
        classes_dict = {c: i for i, c in enumerate(unique)}
        bounds = np.array([i - 0.5 for i in range(len(unique) + 1)])
        norm = colors.BoundaryNorm(bounds, len(unique))
        plt.tricontourf(x[idx], y[idx], [classes_dict[v] for v in z[idx]],
                        cmap=cmap, norm=norm, alpha=.5, levels=bounds)
    else:
        plt.tricontourf(x[idx], y[idx], z[idx], cmap='cool', alpha=.5, levels=10)


def plotSamples(x: pd.Series, y: pd.Series, z: pd.Series, name: str = None):
    if isinstance(z[0], str):
        classes = np.unique(z)
        for i, c in enumerate(classes):
            idx = [output == c for output in z]
            plt.scatter(x[idx], y[idx], c=color_list[i], marker=markers[i])
        plt.gca().legend(legend_marks, classes)
    else:
        sc = plt.scatter(x, y, c=z, cmap='cool', edgecolor="k", linewidths=0.05, s=7)
        plt.colorbar(sc, pad=0.03, label=z.name)
    # if name is not None:
    #    plt.savefig("CLA/{}.pdf".format(name), dpi=500, bbox_inches='tight')
    #    plt.show()


def create_grid(x_name: str, y_name: str, data: pd.DataFrame):
    n = [100 if feature in [x_name, y_name] else 10 for feature in data.columns[:-1]]
    vectors = [np.linspace(data[feature].min(), data[feature].max(), nn) if feature in [x_name, y_name] else
               data[feature].mean() for feature, nn in zip(data.columns[:-1], n)]
    mesh = np.meshgrid(*vectors)
    df = pd.DataFrame(np.c_[[vector.ravel() for vector in mesh]]).T
    df.columns = data.columns[:-1]
    return df
