from __future__ import annotations
from random import Random
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from tuprolog.theory import Theory
from psyke import PedagogicalExtractor
from psyke.extraction.hypercubic import HyperCube, HyperCubeExtractor
from psyke.extraction.hypercubic.hypercube import GenericCube
from psyke.extraction.hypercubic.utils import MinUpdate, Expansion
from psyke.utils import get_default_random_seed, Target

DomainProperties = (Iterable[MinUpdate], GenericCube)


class ITER(PedagogicalExtractor, HyperCubeExtractor):
    """
    Explanator implementing ITER algorithm, doi:10.1007/11823728_26.
    """

    def __init__(self, predictor, min_update, n_points, max_iterations, min_examples, threshold, fill_gaps,
                 normalization, output: Target = Target.CONSTANT, seed=get_default_random_seed()):
        super().__init__(predictor, normalization)
        if output is Target.REGRESSION:
            raise NotImplementedError
        self.predictor = predictor
        self.min_update = min_update
        self.n_points = n_points
        self.max_iterations = max_iterations
        self.min_examples = min_examples
        self.threshold = threshold
        self.fill_gaps = fill_gaps
        self._output = Target.CLASSIFICATION if isinstance(predictor, ClassifierMixin) else \
            output if output is not None else Target.CONSTANT
        self.__generator = Random(seed)

    def _best_cube(self, dataframe: pd.DataFrame, cube: GenericCube, cubes: Iterable[Expansion]) -> Expansion | None:
        expansions = []
        for limit in cubes:
            count = limit.cube.count(dataframe)
            dataframe = dataframe.append(limit.cube.create_samples(self.min_examples - count, self.__generator))
            limit.cube.update(dataframe, self.predictor)
            expansions.append(Expansion(
                limit.cube, limit.feature, limit.direction,
                abs(cube.output - limit.cube.output) if self._output is Target.CONSTANT else
                1 - int(cube.output == limit.cube.output)
            ))
        if len(expansions) > 0:
            return sorted(expansions, key=lambda e: e.distance)[0]
        return None

    def _calculate_min_updates(self, surrounding: GenericCube) -> Iterable[MinUpdate]:
        return [MinUpdate(name, (interval[1] - interval[0]) * self.min_update) for (name, interval) in
                surrounding.dimensions.items()]

    @staticmethod
    def _create_range(cube: GenericCube, domain: DomainProperties, feature: str, direction: str)\
            -> tuple[GenericCube, tuple[float, float]]:
        min_updates, surrounding = domain
        a, b = cube[feature]
        size = [min_update for min_update in min_updates if min_update.name == feature][0].value
        return (cube.copy(), (max(a - size, surrounding.get_first(feature)), a)
                if direction == '-' else (b, min(b + size, surrounding.get_second(feature))))

    @staticmethod
    def _create_temp_cube(cube: GenericCube, domain: DomainProperties,
                          hypercubes: Iterable[GenericCube], feature: str,
                          direction: str) -> Iterable[Expansion]:
        temp_cube, values = ITER._create_range(cube, domain, feature, direction)
        temp_cube.update_dimension(feature, values)
        overlap = temp_cube.overlap(hypercubes)
        while (overlap is not None) & (temp_cube.has_volume()):
            overlap = ITER._resolve_overlap(temp_cube, overlap, hypercubes, feature, direction)
        if (temp_cube.has_volume() & (overlap is None)) & (all(temp_cube != cube for cube in hypercubes)):
            yield Expansion(temp_cube, feature, direction)
        else:
            cube.add_limit(feature, direction)

    @staticmethod
    def _create_temp_cubes(cube: GenericCube, domain: DomainProperties,
                           hypercubes: Iterable[GenericCube]) -> Iterable[Expansion]:
        tmp_cubes = []
        for feature in domain[1].dimensions.keys():
            limit = cube.check_limits(feature)
            if limit == '*':
                continue
            for x in {'-', '+'} - {limit}:
                tmp_cubes += ITER._create_temp_cube(cube, domain, hypercubes, feature, x)
        return tmp_cubes

    def _cubes_to_update(self, dataframe: pd.DataFrame, to_expand: Iterable[GenericCube],
                         hypercubes: Iterable[GenericCube], domain: DomainProperties) \
            -> Iterable[tuple[GenericCube, Expansion]]:
        results = [(hypercube, self._best_cube(dataframe, hypercube, self._create_temp_cubes(
            hypercube, domain, hypercubes))) for hypercube in to_expand]
        return sorted([result for result in results if result[1] is not None], key=lambda x: x[1].distance)

    def _expand_or_create(self, cube: GenericCube, expansion: Expansion, hypercubes: Iterable[GenericCube]) -> None:
        if expansion.distance > self.threshold:
            hypercubes += [expansion.cube]
        else:
            cube.expand(expansion, hypercubes)

    @staticmethod
    def _find_closer_sample(dataframe: pd.DataFrame, output: float | str) -> dict[str, tuple]:
        if isinstance(output, str):
            close_sample = dataframe[dataframe.iloc[:, -1] == output].iloc[0].to_dict()
        else:
            difference = abs(dataframe.iloc[:, -1] - output)
            close_sample = dataframe[difference == min(difference)].iloc[0].to_dict()
        return close_sample

    def _generate_starting_points(self, dataframe: pd.DataFrame) -> Iterable[GenericCube]:
        if self.n_points <= 0:
            raise (Exception('InvalidAttributeValueException'))
        points: Iterable[float]
        if isinstance(dataframe.iloc[0, -1], str):
            classes = np.unique(dataframe.iloc[:, -1].values)
            points = [classes[i] for i in range(min(self.n_points, len(classes)))]
        else:
            desc = dataframe.iloc[:, -1].describe()
            min_output, max_output = desc["min"], desc["max"]
            points = [(max_output - min_output) / 2] if self.n_points == 1 else \
                [min_output + (max_output - min_output) / (self.n_points - 1) * index for index in range(self.n_points)]
        return [HyperCube.cube_from_point(ITER._find_closer_sample(dataframe, point), output=self._output)
                for point in points]

    def _initialize(self, dataframe: pd.DataFrame) -> tuple[Iterable[GenericCube], DomainProperties]:
        self._fake_dataframe = dataframe.copy()
        surrounding = HyperCube.create_surrounding_cube(dataframe, output=self._output)
        min_updates = self._calculate_min_updates(surrounding)
        self._hypercubes = self._init_hypercubes(dataframe, min_updates, surrounding)
        for hypercube in self._hypercubes:
            hypercube.update(dataframe, self.predictor)
        return self._hypercubes, (min_updates, surrounding)

    def _init_hypercubes(
            self,
            dataframe: pd.DataFrame,
            min_updates: Iterable[MinUpdate],
            surrounding: GenericCube
    ) -> Iterable[GenericCube]:
        while True:
            hypercubes = self._generate_starting_points(dataframe)
            for hypercube in hypercubes:
                hypercube.expand_all(min_updates, surrounding)
            self.n_points = self.n_points - 1
            if not HyperCube.check_overlap(hypercubes, hypercubes):
                break
        return hypercubes

    def _iterate(self, dataframe: pd.DataFrame, hypercubes: Iterable[GenericCube], domain: DomainProperties,
                 left_iteration: int) -> int:
        iterations = 0
        to_expand = [cube for cube in hypercubes if cube.limit_count < (len(dataframe.columns) - 1) * 2]
        while (len(to_expand) > 0) and (iterations < left_iteration):
            updates = list(self._cubes_to_update(dataframe, to_expand, hypercubes, domain))
            if len(updates) > 0:
                self._expand_or_create(updates[0][0], updates[0][1], hypercubes)
            iterations += 1
            to_expand = [cube for cube in hypercubes if cube.limit_count < (len(dataframe.columns) - 1) * 2]
        return iterations

    @staticmethod
    def _resolve_overlap(cube: GenericCube, overlapping_cube: GenericCube, hypercubes: Iterable[GenericCube],
                         feature: str, direction: str) -> GenericCube:
        a, b = cube[feature]
        cube.update_dimension(feature, max(overlapping_cube.get_second(feature), a) if direction == '-' else a,
                              min(overlapping_cube.get_first(feature), b) if direction == '+' else b)
        return cube.overlap(hypercubes)

    def _extract(self, dataframe: pd.DataFrame, mapping: dict[str: int] = None) -> Theory:
        self._hypercubes, domain = self._initialize(dataframe)
        temp_train = dataframe.copy()
        fake = dataframe.copy()
        iterations = 0
        while temp_train.shape[0] > 0:
            iterations += self._iterate(fake, self._hypercubes, domain, self.max_iterations - iterations)
            if (iterations >= self.max_iterations) or (not self.fill_gaps):
                break
            temp_train = temp_train.iloc[[p is None for p in self.predict(temp_train.iloc[:, :-1])]]
            if temp_train.shape[0] > 0:
                point, ratio, overlap, new_cube = temp_train.iloc[0].to_dict(), 1.0, True, None
                temp_train = temp_train.drop([temp_train.index[0]])
                while overlap is not None:
                    if new_cube is not None:
                        if not new_cube.has_volume():
                            break
                    new_cube = HyperCube.cube_from_point(point, self._output)
                    new_cube.expand_all(domain[0], domain[1], ratio)
                    overlap = new_cube.overlap(self._hypercubes)
                    ratio *= 2
                if new_cube.has_volume():
                    self._hypercubes += [new_cube]
        return self._create_theory(dataframe)
