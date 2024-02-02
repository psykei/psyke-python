import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from psyke import get_default_random_seed, Target
from psyke.extraction.hypercubic import Grid, HyperCube, GenericCube, ClassificationCube
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
        return parent_cube.diversity - new_cube.diversity > self.threshold / 3.0

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
                subcubes, fake = self._cubes_to_split(cube, surrounding, iteration, dataframe, fake, True)
                cleaned = [c for c in subcubes if c.count(dataframe) > 0 and self._gain(cube, c)]
                if len(subcubes) > len(cleaned):
                    if len(cleaned) > 0:
                        idx = np.any([c.filter_indices(fake.iloc[:, :-1]) for c in cleaned], axis=0)
                        cube.update(fake[~idx], self.predictor)
                    self._hypercubes = [cube] + self._hypercubes
                next_iteration += self._merge(cleaned, fake)
            prev = next_iteration.copy()
        self._hypercubes = [cube for cube in next_iteration] + self._hypercubes
