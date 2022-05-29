import math

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

Dimension = tuple[float, float]
Dimensions = dict[str, Dimension]


def select_gaussian_mixture(data: pd.DataFrame, max_components) -> GaussianMixture:
    components = range(2, max_components + 1)
    models = [GaussianMixture(n_components=n).fit(data) for n in components]
    return min([(m.bic(data), i, m) for i, m in enumerate(models)])[2]


def select_dbscan_epsilon(data: pd.DataFrame) -> float:
    neighbors = NearestNeighbors(n_neighbors=min(len(data.columns) * 2, len(data))).fit(data)
    distances = sorted(np.mean(neighbors.kneighbors(data)[0], axis=1), reverse=True)
    kn = KneeLocator([d for d in range(len(distances))], distances, curve='convex', direction='decreasing')
    if kn.knee is None:
        return distances[0]
    return distances[kn.knee] * .9


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
