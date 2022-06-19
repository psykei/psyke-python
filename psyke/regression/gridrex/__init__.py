from psyke import get_default_random_seed
from psyke.regression import Grid
from psyke.regression.gridex import GridEx
from psyke.regression.hypercube import RegressionCube


class GridREx(GridEx):
    """
    Explanator implementing GridREx algorithm.
    """

    def __init__(self, predictor, grid: Grid, min_examples: int, threshold: float, seed=get_default_random_seed()):
        super().__init__(predictor, grid, min_examples, threshold, seed)

    def _default_cube(self) -> RegressionCube:
        return RegressionCube()
