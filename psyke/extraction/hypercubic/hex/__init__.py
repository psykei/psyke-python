from itertools import product

import pandas as pd
from sklearn.linear_model import LinearRegression

from psyke import get_default_random_seed
from psyke.extraction.hypercubic import Grid, HyperCube
from psyke.extraction.hypercubic.gridex import GridEx


class HEx(GridEx):
    """
    Explanator implementing HEx algorithm.
    """

    def __init__(self, predictor, grid: Grid, min_examples: int, threshold: float, normalization,
                 seed=get_default_random_seed()):
        super().__init__(predictor, grid, min_examples, threshold, normalization, seed)
        self._default_surrounding_cube = True

    def _different_output(self, this_cube, other_cube):
        if isinstance(this_cube.output, str) and this_cube.output == other_cube.output:
            return False
        if isinstance(this_cube.output, float) and abs(this_cube.output - other_cube.output) < self.threshold:
            return False
        if isinstance(this_cube.output, LinearRegression):
            raise NotImplementedError
        return True

    def _iterate(self, surrounding: HyperCube, dataframe: pd.DataFrame):
        fake = dataframe.copy()
        surrounding.update(dataframe, self.predictor)
        prev = [surrounding]
        next_iteration = []

        for iteration in self.grid.iterate():
            next_iteration = []
            for cube in prev:
                # subcubes =
                # [c for c in self._merge(self._cubes_to_split(cube, iteration, dataframe, fake, True), fake)]
                subcubes = [c for c in self._cubes_to_split(cube, iteration, dataframe, fake, True)]
                cleaned = [c for c in subcubes if c.count(dataframe) > 0 and self._different_output(cube, c)]
                if len(subcubes) > len(cleaned):
                    self._hypercubes = [cube] + self._hypercubes
                next_iteration += cleaned
            prev = next_iteration.copy()
        self._hypercubes = [cube for cube in next_iteration] + self._hypercubes
