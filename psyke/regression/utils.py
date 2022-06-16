import math
from collections import Counter

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.simplefilter("ignore")

Dimension = tuple[float, float]
Dimensions = dict[str, Dimension]


def select_gaussian_mixture(data: pd.DataFrame, max_components) -> tuple[float, int, GaussianMixture]:
    components = range(2, max_components + 1)
    try:
        models = [GaussianMixture(n_components=n).fit(data) for n in components if n <= len(data)]
    except ValueError:
        print(data)
        print(len(data))
    return min([(m.bic(data) / (i + 2), (i + 2), m) for i, m in enumerate(models)])


def select_dbscan_epsilon(data: pd.DataFrame, clusters: int) -> float:
    neighbors = NearestNeighbors(n_neighbors=min(len(data.columns) * 2, len(data))).fit(data)
    distances = sorted(np.mean(neighbors.kneighbors(data)[0], axis=1), reverse=True)
    try:
        kn = KneeLocator([d for d in range(len(distances))], distances, curve='convex', direction='decreasing')
        if kn.knee is None:
            epsilon = max(distances[-1], 1e-3)
        else:
            epsilon = distances[kn.knee] * .9
    except (RuntimeWarning, UserWarning, ValueError):
        epsilon = max(distances[-1], 1e-3)
    k = 1.
    dbscan_pred = DBSCAN(eps=epsilon * k).fit_predict(data.iloc[:, :-1])
    # while Counter(dbscan_pred).most_common(1)[0][0] == -1:
    while len(np.unique(dbscan_pred)) > clusters + 1:
        k += .1
        dbscan_pred = DBSCAN(eps=epsilon * k).fit_predict(data.iloc[:, :-1])
    return epsilon * k


class Expansion:

    def __init__(self, cube, feature: str, direction: str, distance: float = math.nan):
        self.cube = cube
        self.feature = feature
        self.direction = direction
        self.distance = distance

    def __getitem__(self, index: int) -> float:
        return self.cube[self.feature][index]

    def boundaries(self, a: float, b: float) -> (float, float):
        return (self[0], b) if self.direction == '-' else (a, self[1])


class Limit:

    def __init__(self, feature: str, direction: str):
        self.feature = feature
        self.direction = direction

    def __eq__(self, other):
        return (self.feature == other.feature) and (self.direction == other.direction)

    def __hash__(self):
        return hash(self.feature + self.direction)


class MinUpdate:

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value


class ZippedDimension:

    def __init__(self, name: str, this_dimension: Dimension, other_dimension: Dimension):
        self.name = name
        self.this_dimension = this_dimension
        self.other_dimension = other_dimension
