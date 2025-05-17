from __future__ import annotations
from itertools import product
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from tuprolog.theory import Theory
from psyke import get_default_random_seed
from psyke.utils import Target
from psyke.extraction.hypercubic import HyperCubeExtractor, Grid, HyperCube


class GridEx(HyperCubeExtractor):
    """
    Explanator implementing GridEx algorithm, doi:10.1007/978-3-030-82017-6_2.
    """

    def __init__(self, predictor, grid: Grid, min_examples: int, threshold: float, output: Target = Target.CONSTANT,
                 discretization=None, normalization=None, seed: int = get_default_random_seed()):
        super().__init__(predictor, Target.CLASSIFICATION if isinstance(predictor, ClassifierMixin) else output,
                         discretization, normalization)
        self.grid = grid
        self.min_examples = min_examples
        self.threshold = threshold
        np.random.seed(seed)

    def _extract(self, dataframe: pd.DataFrame) -> Theory:
        self._hypercubes = []
        self._surrounding = HyperCube.create_surrounding_cube(dataframe, output=self._output)
        self._surrounding.init_diversity(2 * self.threshold)
        self._iterate(dataframe)
        return self._create_theory(dataframe)

    def _create_ranges(self, cube, iteration):
        ranges = {}
        for (feature, (a, b)) in cube.dimensions.items():
            n_bins = self.grid.get(feature, iteration)
            if n_bins == 1:
                ranges[feature] = [(a, b)]
                self._dimensions_to_ignore.add(feature)
            else:
                size = (b - a) / n_bins
                ranges[feature] = [(a + size * i, a + size * (i + 1)) for i in range(n_bins)]
        return ranges

    def _cubes_to_split(self, cube, iteration, dataframe, fake, keep_empty=False):
        to_split = []
        for p in product(*self._create_ranges(cube, iteration).values()):
            cube = self._default_cube()
            for i, f in enumerate(dataframe.columns[:-1]):
                cube.update_dimension(f, p[i])
            n = cube.count(dataframe)
            if n > 0 or keep_empty:
                fake = pd.concat([fake, cube.create_samples(self.min_examples - n)])
                cube.update(fake, self.predictor)
                to_split.append(cube)
        return to_split, fake

    def _iterate(self, dataframe: pd.DataFrame):
        fake = dataframe.copy()
        prev = [self._surrounding]
        next_iteration = []

        for iteration in self.grid.iterate():
            next_iteration = []
            for cube in prev:
                if cube.count(dataframe) == 0:
                    continue
                if cube.diversity < self.threshold:
                    self._hypercubes += [cube]
                    continue
                to_split, fake = self._cubes_to_split(cube, iteration, dataframe, fake)
                next_iteration += [c for c in self._merge(to_split, fake)]
            prev = next_iteration.copy()
        self._hypercubes += [cube for cube in next_iteration]

    @staticmethod
    def _find_couples(to_split: Iterable[HyperCube], not_in_cache: Iterable[HyperCube],
                      adjacent_cache: dict[tuple[HyperCube, HyperCube], str | None]) -> \
            Iterable[tuple[HyperCube, HyperCube, str]]:
        checked = []
        eligible = []
        for cube in to_split:
            checked.append(cube)
            for other_cube in [c for c in to_split if c not in checked]:
                if (cube in not_in_cache) or (other_cube in not_in_cache):
                    adjacent_cache[(cube, other_cube)] = cube.is_adjacent(other_cube)
                adjacent_feature = adjacent_cache[(cube, other_cube)]
                eligible.append((cube, other_cube, adjacent_feature))
        return [couple for couple in eligible if couple[2] is not None]

    def _evaluate_merge(self, not_in_cache: Iterable[HyperCube],
                        dataframe: pd.DataFrame, feature: str,
                        cube: HyperCube, other_cube: HyperCube,
                        merge_cache: dict[(HyperCube, HyperCube), HyperCube | None]) -> bool:
        if (cube in not_in_cache) or (other_cube in not_in_cache):
            merged_cube = cube.merge_along_dimension(other_cube, feature)
            merged_cube.update(dataframe, self.predictor)
            merge_cache[(cube, other_cube)] = merged_cube
        return cube.output == other_cube.output if self._output == Target.CLASSIFICATION else \
            merge_cache[(cube, other_cube)].diversity < self.threshold

    def _merge(self, to_split: Iterable[HyperCube], dataframe: pd.DataFrame) -> Iterable[HyperCube]:
        not_in_cache = [cube for cube in to_split]
        adjacent_cache = {}
        merge_cache = {}
        cont = True
        while cont:
            to_merge = [([cube, other_cube], merge_cache[(cube, other_cube)]) for cube, other_cube, feature in
                        GridEx._find_couples(to_split, not_in_cache, adjacent_cache) if
                        self._evaluate_merge(not_in_cache, dataframe, feature, cube, other_cube, merge_cache)]
            if len(to_merge) == 0:
                cont = False
            else:
                sorted(to_merge, key=lambda c: c[1].diversity)
                best = to_merge[0]
                to_split = [cube for cube in to_split if cube not in best[0]] + [best[1]]
                not_in_cache = [best[1]]
        return to_split

    def make_fair(self, features: Iterable[str]):
        self.grid.make_fair(features)
