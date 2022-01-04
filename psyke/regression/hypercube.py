from __future__ import annotations

from typing import Iterable
import pandas as pd
from sklearn.linear_model import LinearRegression

from psyke.regression import Limit, MinUpdate, ZippedDimension, Expansion
from random import Random
import numpy as np
from psyke import get_default_random_seed


class FeatureNotFoundException(Exception):

    def __init__(self, feature: str):
        super().__init__('Feature "' + feature + '" not found.')


class HyperCube:
    """
    A N-dimensional cube holding a numeric value.
    """

    EPSILON = 1.0 / 1000

    def __init__(self, dimension: dict[str, tuple] = None, limits: set[Limit] = None,
                 output: float | LinearRegression = 0.0):
        self._dimension = dimension if dimension is not None else {}
        self.__limits = limits if limits is not None else set()
        self._output = output
        self._diversity = 0.0

    @property
    def dimensions(self) -> dict[str, tuple]:
        return self._dimension

    @property
    def limit_count(self) -> int:
        return len(self.__limits)

    @property
    def output(self) -> float | LinearRegression:
        return self._output

    @property
    def diversity(self) -> float:
        return self._diversity

    def __expand_one(self, update: MinUpdate, surrounding: HyperCube, ratio: float = 1.0):
        self.update_dimension(update.name,
                              (max(self.get_first(update.name) - update.value / ratio,
                                   surrounding.get_first(update.name)),
                               min(self.get_second(update.name) + update.value / ratio,
                                   surrounding.get_second(update.name))))

    def _filter_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        v = np.array([v for _, v in self._dimension.items()])
        ds = dataset.to_numpy(copy=True)
        indices = np.all((ds >= v[:, 0]) & (ds < v[:, 1]), axis=1)
        return dataset[indices]

    def __zip_dimensions(self, hypercube: HyperCube) -> list[ZippedDimension]:
        return [ZippedDimension(dimension, self.get(dimension), hypercube.get(dimension))
                for dimension in self._dimension.keys()]

    def add_limit(self, limit_or_feature: Limit | str, direction: str = None):
        if isinstance(limit_or_feature, Limit):
            self.__limits.add(limit_or_feature)
        else:
            self.add_limit(Limit(limit_or_feature, direction))

    def check_limits(self, feature: str) -> str | None:
        filtered = [limit for limit in self.__limits if limit.feature == feature]
        if len(filtered) == 0:
            return None
        if len(filtered) == 1:
            return filtered[0].direction
        if len(filtered) == 2:
            return '*'
        raise Exception('Too many limits for this feature')

    def create_samples(self, n: int = 1, generator: Random = Random(get_default_random_seed())) -> pd.DataFrame:
        return pd.DataFrame([self.__create_tuple(generator) for _ in range(n)])

    @staticmethod
    def check_overlap(to_check: Iterable[HyperCube], hypercubes: Iterable[HyperCube]) -> bool:
        checked = []
        to_check_copy = list(to_check).copy()
        while len(to_check_copy) > 0:
            cube = to_check_copy.pop()
            checked += [cube]
            for hypercube in hypercubes:
                if (hypercube not in checked) & cube.overlap(hypercube):
                    return True
        return False

    def contains(self, t: dict[str, float]) -> bool:
        return all([(self.get_first(k) <= v < self.get_second(k)) for k, v in t.items()])

    def copy(self) -> HyperCube:
        return HyperCube(self.dimensions.copy(), self.__limits.copy(), self.output)

    def count(self, dataset: pd.DataFrame) -> int:
        return self._filter_dataframe(dataset.iloc[:, :-1]).shape[0]

    @staticmethod
    def create_surrounding_cube(dataset: pd.DataFrame) -> HyperCube:
        return HyperCube({
            column: (min(dataset[column]) - HyperCube.EPSILON ** 2, max(dataset[column]) + HyperCube.EPSILON ** 2)
            for column in dataset.columns[:-1]
        })

    def __create_tuple(self, generator: Random) -> dict:
        return {k: generator.uniform(self.get_first(k), self.get_second(k)) for k in self._dimension.keys()}

    @staticmethod
    def cube_from_point(point: dict) -> HyperCube:
        return HyperCube({k: (v, v) for k, v in list(point.items())[:-1]}, output=list(point.values())[-1])

    def equal(self, hypercubes: Iterable[HyperCube] | HyperCube) -> bool:
        if isinstance(hypercubes, Iterable):
            return any([self.equal(cube) for cube in hypercubes])
        else:
            return all([(abs(dimension.this_cube[0] - dimension.other_cube[0]) < HyperCube.EPSILON)
                        & (abs(dimension.this_cube[1] - dimension.other_cube[1]) < HyperCube.EPSILON)
                        for dimension in self.__zip_dimensions(hypercubes)])

    def expand(self, expansion: Expansion, hypercubes: Iterable[HyperCube]) -> None:
        feature = expansion.feature
        a, b = self.get(feature)
        self.update_dimension(feature, expansion.boundaries(a, b))
        other_cube = self.overlap(hypercubes)
        if isinstance(other_cube, HyperCube):
            self.update_dimension(feature, (other_cube.get_second(feature), b)
                                  if expansion.direction == '-' else (a, other_cube.get_first(feature)))
        if isinstance(self.overlap(hypercubes), HyperCube):
            raise Exception('Overlapping not handled')

    def expand_all(self, updates: Iterable[MinUpdate], surrounding: HyperCube, ratio: float = 1.0):
        for update in updates:
            self.__expand_one(update, surrounding, ratio)

    def get(self, feature: str) -> tuple:
        if feature in self._dimension.keys():
            return self._dimension[feature]
        else:
            raise FeatureNotFoundException(feature)

    def get_first(self, feature: str) -> float:
        return self.get(feature)[0]

    def get_second(self, feature: str) -> float:
        return self.get(feature)[1]

    def has_volume(self) -> bool:
        return all([dimension[1] - dimension[0] > HyperCube.EPSILON for dimension in self._dimension.values()])

    def is_adjacent(self, cube: HyperCube) -> str | None:
        adjacent = None
        for (feature, [a1, b1]) in self._dimension.items():
            if self.get(feature) == cube.get(feature):
                continue
            [a2, b2] = cube.get(feature)
            if (adjacent is not None) or ((b1 != a2) and (b2 != a1)):
                return None
            adjacent = feature
        return adjacent

    def merge_along_dimension(self, cube: HyperCube, feature: str) -> HyperCube:
        new_cube = self.copy()
        (a1, b1) = self.get(feature)
        (a2, b2) = cube.get(feature)
        new_cube.update_dimension(feature, (min(a1, a2), max(b1, b2)))
        return new_cube

    def overlap(self, hypercubes: Iterable[HyperCube] | HyperCube) -> HyperCube | bool | None:
        if isinstance(hypercubes, Iterable):
            for hypercube in hypercubes:
                if (self != hypercube) & self.overlap(hypercube):
                    return hypercube
            return None
        else:
            return all([not ((dimension.other_cube[0] >= dimension.this_cube[1]) |
                             (dimension.this_cube[0] >= dimension.other_cube[1]))
                        for dimension in self.__zip_dimensions(hypercubes)])

    def update_dimension(self, feature: str, lower: float | (float, float), upper: float | None = None) -> None:
        if upper is None:
            self._dimension[feature] = lower
        else:
            self.update_dimension(feature, (lower, upper))

    def update(self, dataset: pd.DataFrame, predictor) -> None:
        filtered = self._filter_dataframe(dataset.iloc[:, :-1])
        predictions = predictor.predict(filtered.to_numpy())
        self._output = np.mean(predictions)
        self._diversity = np.std(predictions)

    def init_std(self, std: float):
        self._diversity = std


class RegressionCube(HyperCube):
    def __init__(self):
        super().__init__(output=LinearRegression())

    def update(self, dataset: pd.DataFrame, predictor):
        filtered = self._filter_dataframe(dataset.iloc[:, :-1])
        if len(filtered > 0):
            predictions = predictor.predict(filtered.values)
            self._output.fit(filtered, predictions)
            self._diversity = (abs(self._output.predict(filtered) - predictions)).mean()
