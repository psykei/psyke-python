import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from psyke import get_default_random_seed, Target
from psyke.extraction.hypercubic import Grid, HyperCube, GenericCube, ClassificationCube, RegressionCube
from psyke.extraction.hypercubic.gridex import GridEx


class HEx(GridEx):
    """
    Explanator implementing HEx algorithm.
    """

    def __init__(self, predictor, grid: Grid, min_examples: int, threshold: float, output: Target = Target.CONSTANT,
                 discretization=None, normalization=None, seed: int = get_default_random_seed()):
        super().__init__(predictor, grid, min_examples, threshold, output, discretization, normalization, seed)
        self._default_surrounding_cube = True

    def _different_output(self, this_cube: GenericCube, other_cube: GenericCube) -> bool:
        if isinstance(this_cube.output, str) and this_cube.output == other_cube.output:
            return False
        if isinstance(this_cube.output, float) and abs(this_cube.output - other_cube.output) < self.threshold:
            return False
        if isinstance(this_cube.output, LinearRegression):
            raise NotImplementedError
        return True

    def _gain(self, parent_cube: GenericCube, new_cube: GenericCube) -> float:
        if isinstance(parent_cube, ClassificationCube):
            return parent_cube.output != new_cube.output
        return parent_cube.error - new_cube.error > self.threshold * .6

    def _iterate(self, surrounding: HyperCube, dataframe: pd.DataFrame):
        fake = dataframe.copy()
        surrounding.update(dataframe, self.predictor)
        prev = [(None, surrounding, True)]

        for iteration in self.grid.iterate():
            next_iteration = []
            for (parent, cube, gain) in prev:
                subcubes, fake = self._cubes_to_split(cube, surrounding, iteration, dataframe, fake, True)
                parent_idx = cube.filter_indices(fake.iloc[:, :-1])
                cube = cube if gain else parent
                cleaned = [(c, self._gain(cube, c)) for c in subcubes if c.count(dataframe) > 0]
                eligible_idx = np.any([c.filter_indices(fake.iloc[:, :-1]) for c, g in cleaned if g], axis=0)
                if sum(g for _, g in cleaned) > 0 and sum(parent_idx) > sum(eligible_idx) and gain:
                    cube.update(fake[parent_idx & ~eligible_idx], self.predictor)
                    if parent and self._gain(parent, cube):
                        self._hypercubes = [cube] + self._hypercubes
                next_iteration += [(cube, c, self._gain(cube, c)) for c in self._merge([c for c, _ in cleaned], fake)]
            prev = next_iteration.copy()
        self._hypercubes = [cube for (_, cube, gain) in prev if gain] + self._hypercubes
        if min(np.any([c.filter_indices(dataframe.iloc[:, :-1]) for c in self._hypercubes[:-1]], axis=0)):
            self._hypercubes = self._hypercubes[:-1]
