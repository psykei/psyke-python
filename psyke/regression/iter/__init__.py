from __future__ import annotations
from random import Random
from typing import Iterable
import numpy as np
import pandas as pd
from tuprolog.theory import Theory
from psyke.regression.hypercube import HyperCube
from psyke.regression import HyperCubeExtractor
from psyke.regression.utils import Expansion, MinUpdate
from psyke.utils import get_default_random_seed

DomainProperties = (Iterable[MinUpdate], HyperCube)


class ITER(HyperCubeExtractor):
    """
    Explanator implementing ITER algorithm, doi:10.1007/11823728_26.
    """

    def __init__(self, predictor, min_update, n_points, max_iterations, min_examples, threshold, fill_gaps,
                 seed=get_default_random_seed()):
        super().__init__(predictor)
        self.predictor = predictor
        self.discretization = []
        self.min_update = min_update
        self.n_points = n_points
        self.max_iterations = max_iterations
        self.min_examples = min_examples
        self.threshold = threshold
        self.fill_gaps = fill_gaps
        self.__generator = Random(seed)

    def __best_cube(self, dataframe: pd.DataFrame, cube: HyperCube, cubes: Iterable[Expansion]) -> Expansion | None:
        expansions = []
        for limit in cubes:
            count = limit.cube.count(dataframe)
            # if count == 0:
            #    continue
            dataframe = dataframe.append(limit.cube.create_samples(self.min_examples - count, self.__generator))
            limit.cube.update(dataframe, self.predictor)
            expansions.append(Expansion(
                limit.cube, limit.feature, limit.direction, abs(cube.output - limit.cube.output)
            ))
        if len(expansions) > 0:
            return sorted(expansions, key=lambda e: e.distance)[0]
        return None

    def __calculate_min_updates(self, surrounding: HyperCube) -> Iterable[MinUpdate]:
        return [MinUpdate(name, (interval[1] - interval[0]) * self.min_update) for (name, interval) in
                surrounding.dimensions.items()]

    @staticmethod
    def __create_range(cube: HyperCube, domain: DomainProperties, feature: str, direction: str)\
            -> tuple[HyperCube, tuple[float, float]]:
        min_updates, surrounding = domain
        a, b = cube[feature]
        size = [min_update for min_update in min_updates if min_update.name == feature][0].value
        return (cube.copy(), (max(a - size, surrounding.get_first(feature)), a) if direction == '-' else
        (b, min(b + size, surrounding.get_second(feature))))

    @staticmethod
    def __create_temp_cube(cube: HyperCube, domain: DomainProperties, hypercubes: Iterable[HyperCube], feature: str,
                           direction: str) -> Iterable[Expansion]:
        temp_cube, values = ITER.__create_range(cube, domain, feature, direction)
        temp_cube.update_dimension(feature, values)
        overlap = temp_cube.overlap(hypercubes)
        while (overlap is not None) & (temp_cube.has_volume()):
            overlap = ITER.__resolve_overlap(temp_cube, overlap, hypercubes, feature, direction)
        if (temp_cube.has_volume() & (overlap is None)) & (all(temp_cube != cube for cube in hypercubes)):
            yield Expansion(temp_cube, feature, direction)
        else:
            cube.add_limit(feature, direction)

    @staticmethod
    def __create_temp_cubes(cube: HyperCube, domain: DomainProperties,
                            hypercubes: Iterable[HyperCube]) -> Iterable[Expansion]:
        tmp_cubes = []
        for feature in domain[1].dimensions.keys():
            limit = cube.check_limits(feature)
            if limit == '*':
                continue
            for x in {'-', '+'} - {limit}:
                tmp_cubes += ITER.__create_temp_cube(cube, domain, hypercubes, feature, x)
        return tmp_cubes

    def __cubes_to_update(self, dataframe: pd.DataFrame, to_expand: Iterable[HyperCube],
                          hypercubes: Iterable[HyperCube], domain: DomainProperties) \
            -> Iterable[tuple[HyperCube, Expansion]]:
        results = [(hypercube, self.__best_cube(dataframe, hypercube, self.__create_temp_cubes(
            hypercube, domain, hypercubes))) for hypercube in to_expand]
        return sorted([result for result in results if result[1] is not None], key=lambda x: x[1].distance)

    def __expand_or_create(self, cube: HyperCube, expansion: Expansion, hypercubes: Iterable[HyperCube]) -> None:
        if expansion.distance > self.threshold:
            hypercubes += [expansion.cube]
        else:
            cube.expand(expansion, hypercubes)

    @staticmethod
    def __find_closer_sample(dataframe: pd.DataFrame, output: float) -> dict[str, tuple]:
        difference = abs(dataframe.iloc[:, -1] - output)
        close_sample = dataframe[difference == min(difference)].iloc[0].to_dict()
        return close_sample

    def __generate_starting_points(self, dataframe: pd.DataFrame) -> Iterable[HyperCube]:
        desc = dataframe.iloc[:, -1].describe()
        min_output, max_output = desc["min"], desc["max"]
        if self.n_points <= 0:
            raise (Exception('InvalidAttributeValueException'))
        points: Iterable[float] = [(max_output - min_output) / 2] if self.n_points == 1 else \
            [min_output + (max_output - min_output) / (self.n_points - 1) * index for index in range(self.n_points)]
        return [HyperCube.cube_from_point(ITER.__find_closer_sample(dataframe, point)) for point in points]

    def __init(self, dataframe: pd.DataFrame) -> tuple[Iterable[HyperCube], DomainProperties]:
        self.__fake_dataframe = dataframe.copy()
        surrounding = HyperCube.create_surrounding_cube(dataframe)
        min_updates = self.__calculate_min_updates(surrounding)
        self._hypercubes = self.__init_hypercubes(dataframe, min_updates, surrounding)
        for hypercube in self._hypercubes:
            hypercube.update(dataframe, self.predictor)
        return self._hypercubes, (min_updates, surrounding)

    def __init_hypercubes(
            self,
            dataframe: pd.DataFrame,
            min_updates: Iterable[MinUpdate],
            surrounding: HyperCube
    ) -> Iterable[HyperCube]:
        while True:
            hypercubes = self.__generate_starting_points(dataframe)
            for hypercube in hypercubes:
                hypercube.expand_all(min_updates, surrounding)
            self.n_points = self.n_points - 1
            if not HyperCube.check_overlap(hypercubes, hypercubes):
                break
        return hypercubes

    def __iterate(self, dataframe: pd.DataFrame, hypercubes: Iterable[HyperCube], domain: DomainProperties,
                  left_iteration: int) -> int:
        iterations = 0
        to_expand = [cube for cube in hypercubes if cube.limit_count < (len(dataframe.columns) - 1) * 2]
        while (len(to_expand) > 0) and (iterations < left_iteration):
            updates = list(self.__cubes_to_update(dataframe, to_expand, hypercubes, domain))
            if len(updates) > 0:
                self.__expand_or_create(updates[0][0], updates[0][1], hypercubes)
            iterations += 1
            to_expand = [cube for cube in hypercubes if cube.limit_count < (len(dataframe.columns) - 1) * 2]
        return iterations

    @staticmethod
    def __resolve_overlap(cube: HyperCube, overlapping_cube: HyperCube, hypercubes: Iterable[HyperCube], feature: str,
                          direction: str) -> HyperCube:
        a, b = cube[feature]
        cube.update_dimension(feature, max(overlapping_cube.get_second(feature), a) if direction == '-' else a,
                              min(overlapping_cube.get_first(feature), b) if direction == '+' else b)
        return cube.overlap(hypercubes)

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        self._hypercubes, domain = self.__init(dataframe)
        temp_train = dataframe.copy()
        fake = dataframe.copy()
        iterations = 0
        while temp_train.shape[0] > 0:
            iterations += self.__iterate(fake, self._hypercubes, domain, self.max_iterations - iterations)
            if (iterations >= self.max_iterations) or (not self.fill_gaps):
                break
            temp_train = temp_train.iloc[np.isnan(self.predict(temp_train.iloc[:, :-1]))]
            if temp_train.shape[0] > 0:
                point, ratio, overlap, new_cube = temp_train.iloc[0].to_dict(), 1.0, True, None
                temp_train = temp_train.drop([temp_train.index[0]])
                while overlap is not None:
                    if new_cube is not None:
                        if not new_cube.has_volume():
                            break
                    new_cube = HyperCube.cube_from_point(point)
                    new_cube.expand_all(domain[0], domain[1], ratio)
                    overlap = new_cube.overlap(self._hypercubes)
                    ratio *= 2
                if new_cube.has_volume():
                    self._hypercubes += [new_cube]
        return self._create_theory(dataframe)
