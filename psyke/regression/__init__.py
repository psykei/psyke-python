from __future__ import annotations
import random
from typing import Iterable
import numpy as np
import pandas as pd
from psyke import get_default_random_seed
from psyke.regression.utils import Limit, MinUpdate, ZippedDimension, Expansion, Dimensions, Dimension


class FeatureNotFoundException(Exception):

    def __init__(self, feature: str):
        super().__init__('Feature "' + feature + '" not found.')


class HyperCube:
    
    # TODO: this should be configurable by the designer
    EPSILON = 1.0 / 1000  # Precision used when comparing two hypercubes
    
    """
    A N-dimensional cube holding a numeric value.
    """
    def __init__(self, dimensions: Dimensions = None, limits: set[Limit] = None, output: float = 0.0):
        self.__dimensions = dimensions if dimensions is not None else {}
        self.__limits = limits if limits is not None else set()
        self.__output = output

    def __contains__(self, point: dict[str, float]) -> bool:
        """
        Note that a point (dict[str, float]) is inside an hypercube if ALL its dimensions' values satisfy:
            min_dim <= value < max_dim
        :param point: a N-dimensional point
        :return: true if the point is inside the hypercube, false otherwise
        """
        return all([(self.get_first(k) <= v < self.get_second(k)) for k, v in point.items()])

    def __eq__(self, other: HyperCube) -> bool:
        return all([(abs(dimension.this_dimension[0] - dimension.other_dimension[0]) < HyperCube.EPSILON)
                    & (abs(dimension.this_dimension[1] - dimension.other_dimension[1]) < HyperCube.EPSILON)
                    for dimension in self.__zip_dimensions(other)])

    def __getitem__(self, feature: str) -> Dimension:
        if feature in self.__dimensions.keys():
            return self.__dimensions[feature]
        else:
            raise FeatureNotFoundException(feature)

    def __hash__(self) -> int:
        result = [hash(name + str(dimension[0]) + str(dimension[1])) for name, dimension in self.dimensions.items()]
        return sum(result)

    @property
    def dimensions(self) -> Dimensions:
        return self.__dimensions

    @property
    def limit_count(self) -> int:
        return len(self.__limits)

    @property
    def mean(self) -> float:
        return self.__output

    def __expand_one(self, update: MinUpdate, surrounding: HyperCube, ratio: float = 1.0):
        self.__dimensions[update.name] = \
            (max(self.get_first(update.name) - update.value / ratio, surrounding.get_first(update.name)),
             min(self.get_second(update.name) + update.value / ratio, surrounding.get_second(update.name)))

    def __filter_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        v = np.array([v for _, v in self.__dimensions.items()])
        ds = dataset.to_numpy(copy=True)
        indices = np.all((ds >= v[:, 0]) & (ds < v[:, 1]), axis=1)
        return dataset[indices]

    def __zip_dimensions(self, hypercube: HyperCube) -> list[ZippedDimension]:
        return [ZippedDimension(dimension, self[dimension], hypercube[dimension])
                for dimension in self.__dimensions.keys()]

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

    @staticmethod
    def check_overlap(to_check: Iterable[HyperCube], hypercubes: Iterable[HyperCube]) -> bool:
        checked = []
        to_check_copy = list(to_check).copy()
        while len(to_check_copy) > 0:
            cube = to_check_copy.pop()
            for hypercube in hypercubes:
                if hypercube not in checked and cube.overlap(hypercube):
                    return True
            checked += [cube]
        return False

    def copy(self) -> HyperCube:
        return HyperCube(self.dimensions.copy(), self.__limits.copy(), self.mean)

    def count(self, dataset: pd.DataFrame) -> int:
        return self.__filter_dataframe(dataset.iloc[:, :-1]).shape[0]

    @staticmethod
    def create_surrounding_cube(dataset: pd.DataFrame) -> HyperCube:
        return HyperCube({
            column: (min(dataset[column]) - HyperCube.EPSILON ** 2, max(dataset[column]) + HyperCube.EPSILON ** 2)
            for column in dataset.columns[:-1]
        })

    def create_tuple(self, generator: random.Random = random.Random(get_default_random_seed())) -> dict:
        return {k: generator.uniform(self.get_first(k), self.get_second(k)) for k in self.__dimensions.keys()}

    @staticmethod
    def cube_from_point(point: dict) -> HyperCube:
        return HyperCube({k: (v, v) for k, v in list(point.items())[:-1]}, output=list(point.values())[-1])

    def expand(self, expansion: Expansion, hypercubes: Iterable[HyperCube]) -> None:
        feature = expansion.feature
        a, b = self[feature]
        self.__dimensions[feature] = expansion.boundaries(a, b)
        other_cube = self.overlap(hypercubes)
        if isinstance(other_cube, HyperCube):
            self.__dimensions[feature] = (other_cube.get_second(feature), b) \
                if expansion.direction == '-' else (a, other_cube.get_first(feature))
        if isinstance(self.overlap(hypercubes), HyperCube):
            raise Exception('Overlapping not handled')

    def expand_all(self, updates: Iterable[MinUpdate], surrounding: HyperCube, ratio: float = 1.0):
        for update in updates:
            self.__expand_one(update, surrounding, ratio)

    def get_first(self, feature: str) -> float:
        return self[feature][0]

    def get_second(self, feature: str) -> float:
        return self[feature][1]

    def has_volume(self) -> bool:
        return all([dimension[1] - dimension[0] > HyperCube.EPSILON for dimension in self.__dimensions.values()])

    def overlap(self, hypercubes: Iterable[HyperCube] | HyperCube) -> HyperCube | bool | None:
        if isinstance(hypercubes, Iterable):
            for hypercube in hypercubes:
                if (self != hypercube) & self.overlap(hypercube):
                    return hypercube
            return None
        elif self is hypercubes:
            return False
        else:
            return all([not ((dimension.other_dimension[0] >= dimension.this_dimension[1]) |
                             (dimension.this_dimension[0] >= dimension.other_dimension[1]))
                        for dimension in self.__zip_dimensions(hypercubes)])

    def update_dimension(self, feature: str, lower: float | tuple[float, float], upper: float | None = None) -> None:
        if upper is None:
            self.__dimensions[feature] = lower
        else:
            self.update_dimension(feature, (lower, upper))

    def update_mean(self, dataset: pd.DataFrame, predictor) -> None:
        filtered = self.__filter_dataframe(dataset.iloc[:, :-1])
        predictions = predictor.predict(filtered.to_numpy())
        self.__output = np.mean(predictions)
