from __future__ import annotations
import random as rnd
from itertools import product
from typing import Iterable
import numpy as np
import pandas as pd
from tuprolog.theory import Theory
from psyke import get_default_random_seed, PedagogicalExtractor
from psyke.utils import Target
from psyke.extraction.hypercubic import HyperCubeExtractor, Grid, HyperCube


class GridEx(PedagogicalExtractor, HyperCubeExtractor):
    """
    Explanator implementing GridEx algorithm, doi:10.1007/978-3-030-82017-6_2.
    """

    def __init__(self, predictor, grid: Grid, min_examples: int, threshold: float, normalization,
                 seed=get_default_random_seed()):
        super().__init__(predictor, normalization)
        self.grid = grid
        self.min_examples = min_examples
        self.threshold = threshold
        self.__generator = rnd.Random(seed)

    def _extract(self, dataframe: pd.DataFrame, mapping: dict[str: int] = None) -> Theory:
        if isinstance(np.array(self.predictor.predict(dataframe.iloc[0:1, :-1])).flatten()[0], str):
            self._output = Target.CLASSIFICATION
        surrounding = HyperCube.create_surrounding_cube(dataframe, output=self._output)
        surrounding.init_std(2 * self.threshold)
        self._iterate(surrounding, dataframe)
        return self._create_theory(dataframe)

    def _ignore_dimensions(self) -> Iterable[str]:
        cube = self._hypercubes[0]
        return [d for d in cube.dimensions if all(c[d] == cube[d] for c in self._hypercubes)]

    def _iterate(self, surrounding: HyperCube, dataframe: pd.DataFrame):
        fake = dataframe.copy()
        prev = [surrounding]
        next_iteration = []

        for iteration in self.grid.iterate():
            next_iteration = []
            for cube in prev:
                to_split = []
                if cube.count(dataframe) == 0:
                    continue
                if cube.diversity < self.threshold:
                    self._hypercubes += [cube]
                    continue
                ranges = {}
                for (feature, (a, b)) in cube.dimensions.items():
                    bins = []
                    n_bins = self.grid.get(feature, iteration)
                    size = (b - a) / n_bins
                    for i in range(n_bins):
                        bins.append((a + size * i, a + size * (i + 1)))
                    ranges[feature] = bins
                for (pn, p) in enumerate(list(product(*ranges.values()))):
                    cube = self._default_cube()
                    for i, f in enumerate(dataframe.columns[:-1]):
                        cube.update_dimension(f, p[i])
                    n = cube.count(dataframe)
                    if n > 0:
                        fake = pd.concat([fake, cube.create_samples(self.min_examples - n, self.__generator)])
                        cube.update(fake, self.predictor)
                        to_split += [cube]
                to_split = self._merge(to_split, fake)
                next_iteration += [cube for cube in to_split]
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
        # TODO: refactor this. A while true with a break is as ugly as hunger.
        while True:
            to_merge = [([cube, other_cube], merge_cache[(cube, other_cube)]) for cube, other_cube, feature in
                        GridEx._find_couples(to_split, not_in_cache, adjacent_cache) if
                        self._evaluate_merge(not_in_cache, dataframe, feature, cube, other_cube, merge_cache)]
            if len(to_merge) == 0:
                break
            sorted(to_merge, key=lambda c: c[1].diversity)
            best = to_merge[0]
            to_split = [cube for cube in to_split if cube not in best[0]] + [best[1]]
            not_in_cache = [best[1]]
        return to_split
