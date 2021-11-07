from __future__ import annotations
from psyke.regression.feature_not_found_exception import FeatureNotFoundException
from psyke.regression.iter.expansion import Expansion
from psyke.regression.iter.limit import Limit
from psyke.regression.iter.minupdate import MinUpdate
from psyke.regression.iter.zipped_dimension import ZippedDimension
import math as m
import numpy as np
import pandas as pd
import random


class HyperCube:
    """
    A N-dimensional cube holding a numeric value.
    """

    def __init__(self, dimension: dict[str, tuple] = None, limits: set[Limit] = None, output: float = 0.0):
        self.__dimension = dimension if dimension is not None else {}
        self.__limits = limits if limits is not None else set()
        self.__output = output
        self.__epsilon = 1.0 / 1000

    @property
    def dimensions(self):
        return self.__dimension

    @property
    def limit_count(self):
        return len(self.__limits)

    @property
    def mean(self):
        return self.__output

    def __expand_one(self, update: MinUpdate, surrounding: HyperCube):
        self.__dimension[update.name] = \
            (max(self.get_first(update.name) - update.value, surrounding.get_first(update.name)),
             min(self.get_second(update.name) + update.value, surrounding.get_second(update.name)))

    def __filter_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset[dataset.apply(
            lambda row: all([(v[0] <= row[k]) & (row[k] < v[1]) for k, v in self.__dimension.items()]), axis=1)]

    def __zip_dimensions(self, hypercube: HyperCube) -> list[ZippedDimension]:
        return [ZippedDimension(dimension, self.get(dimension), hypercube.get(dimension))
                for dimension in self.__dimension.keys()]

    def add_limit(self, limit_or_feature, direction: str = None):
        if isinstance(limit_or_feature, Limit):
            self.__limits.add(limit_or_feature)
        else:
            self.add_limit(Limit(limit_or_feature, direction))

    def check_limits(self, feature: str) -> str | None:
        filtered = [limit for limit in self.__limits if limit.feature == feature]
        if len(filtered) == 0:
            return None
        else:
            if len(filtered) == 1:
                return filtered[0].direction
            else:
                if len(filtered) == 2:
                    return '*'
                else:
                    raise Exception('Not allowed direction')

    @staticmethod
    def check_overlap(to_check: list[HyperCube], hypercubes: list[HyperCube]) -> bool:
        checked = []
        to_check_copy = to_check.copy()
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
        return HyperCube(self.dimensions.copy(), self.__limits.copy(), self.mean)

    def count(self, dataset: pd.DataFrame) -> int:
        return self.__filter_dataframe(dataset).shape[0]

    @staticmethod
    def create_surrounding_cube(dataset: pd.DataFrame) -> HyperCube:
        return HyperCube({column: (m.floor(min(dataset[column])), m.ceil(max(dataset[column])))
                          for column in dataset.columns[:-1]})

    def create_tuple(self, generator: random.Random = random.Random(1)) -> dict:
        return {k: generator.uniform(self.get_first(k), self.get_second(k)) for k in self.__dimension.keys()}

    @staticmethod
    def cube_from_point(point: dict) -> HyperCube:
        return HyperCube({k: (v, v) for k, v in list(point.items())[:-1]}, output=list(point.values())[-1])

    def equal(self, hypercubes) -> bool:
        if isinstance(hypercubes, list):
            return any([self.equal(cube) for cube in hypercubes])
        else:
            return all([(abs(dimension.this_cube[0] - dimension.other_cube[0]) < self.__epsilon)
                        & (abs(dimension.this_cube[1] - dimension.other_cube[1]) < self.__epsilon)
                        for dimension in self.__zip_dimensions(hypercubes)])

    def expand(self, expansion: Expansion, hypercubes: list[HyperCube]):
        feature, direction = expansion.feature, expansion.direction
        a, b = self.get(feature)
        self.__dimension[feature] = (expansion.get()[0], b) if direction == '-' else (a, expansion.get()[1])
        other_cube = self.overlap(hypercubes)
        if isinstance(other_cube, HyperCube):
            self.__dimension[feature] = (other_cube.get_second(feature), b) \
                if direction == '-' else (a, other_cube.get_first(feature))

    def expand_all(self, updates: list[MinUpdate], surrounding: HyperCube):
        for update in updates:
            self.__expand_one(update, surrounding)

    def get(self, feature: str) -> tuple:
        if feature in self.__dimension.keys():
            return self.__dimension[feature]
        else:
            raise FeatureNotFoundException(feature)

    def get_first(self, feature: str) -> float:
        return self.get(feature)[0]

    def get_second(self, feature: str) -> float:
        return self.get(feature)[1]

    def has_volume(self) -> bool:
        return all([dimension[1] - dimension[0] > self.__epsilon for dimension in self.__dimension.values()])

    def overlap(self, hypercubes) -> HyperCube | bool | None:
        if hasattr(hypercubes, '__iter__'):
            for hypercube in hypercubes:
                if (self != hypercube) & self.overlap(hypercube):
                    return hypercube
            return None
        else:
            return all([not ((dimension.other_cube[0] >= dimension.this_cube[1]) |
                             (dimension.this_cube[0] >= dimension.other_cube[1]))
                        for dimension in self.__zip_dimensions(hypercubes)])

    def update_dimension(self, feature: str, lower, upper=None):
        if upper is None:
            self.__dimension[feature] = lower
        else:
            self.update_dimension(feature, (lower, upper))

    def update_mean(self, dataset: pd.DataFrame, predictor):
        filtered = self.__filter_dataframe(dataset)
        self.__output = np.mean(predictor.predict(filtered))
